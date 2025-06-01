import functools
from typing import List, Type, Union
from inspect import isclass

import torch
import cuda.bindings.driver as cuda
import torch.distributed as dist

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
from cutlass.torch import dtype as torch_dtype

from flashinfer.cute_dsl.moe_utils import MoEParam
from flashinfer.cute_dsl.dist_utils import ProcessGroupInfo
import flashinfer.cute_dsl.kernel.dsl_ptx_wrapper as inline_ptx

"""
A intra-node dispatch kernel for the MoE model with cute DSL on blackwell (SM100).
TODO(Zhihao): 
  1. Have a dist_buffer class
  2. Support generalized number of tokens and experts
"""

class IntraDispatchKernel:
    def __init__(
        self,
        moe_param: MoEParam,
        dist_param: ProcessGroupInfo,
    ):
        self.moe_param = moe_param
        self.dist_param = dist_param
        
        self.local_rank: cutlass.Constexpr[int] = dist_param.local_rank
        self.num_local_ranks: cutlass.Constexpr[int] = dist_param.world_local_size

        self.num_warp: cutlass.Constexpr[int] = 16
        self.threads_per_cta: cutlass.Constexpr[int] = 32 * self.num_warp
        self.num_smem_capacity = sm100_utils.SMEM_CAPACITY["sm100"]
        self.num_local_experts: cutlass.Constexpr[int] = int(moe_param.num_experts / (dist_param.world_size))
        self.hidden_dim: cutlass.Constexpr[int] = moe_param.hidden_dim
        # NOTE(Zhihao): assume the in_dtype and out_dtype are the same
        self.hidden_dim_in_bytes: cutlass.Constexpr[int] = moe_param.hidden_dim * torch_dtype(moe_param.in_dtype).itemsize
        self.num_tokens_per_rank: cutlass.Constexpr[int] = moe_param.num_tokens_per_rank
        self.max_num_tokens: cutlass.Constexpr[int] = self.num_tokens_per_rank * self.num_local_ranks
        self.combine_token_stride: cutlass.Constexpr[int] = cutlass.cute.round_up(self.hidden_dim_in_bytes, 16) # align to 16 bytes for 128b data transfer
        self.dispatch_token_stride: cutlass.Constexpr[int] = cutlass.cute.round_up(self.hidden_dim_in_bytes + 4, 16) # additional 4 bytes for meta data, align to 16 bytes for 128b data transfer
        if cutlass.const_expr(self.combine_token_stride % 16 != 0 or self.dispatch_token_stride % 16 != 0):
            raise TypeError(f"dispatch_token_stride {self.dispatch_token_stride} and combine_token_stride {self.combine_token_stride} should be divisible by 16")
        
        self.buffer_size_in_bytes: cutlass.Constexpr[int] = max(self.get_combine_buffer_size(), self.get_dispatch_buffer_size())
    
    def get_dispatch_buffer_size(self):
        size = 0
        meta_size = 4

        size += 16
        size += cutlass.cute.round_up(self.num_local_experts * 4, 16)
        size += self.num_local_experts * self.max_num_tokens * self.dispatch_token_stride

        return int(size)
    
    def get_combine_buffer_size(self):
        size = 0

        size += 16
        size += cutlass.cute.round_up(self.num_local_experts * 4, 16)
        size += self.num_local_experts * self.max_num_tokens * self.combine_token_stride

        return int(size)


    def _setup_attributes(self):
        self.dispatch_buffer_offset_in_bytes: cutlass.Constexpr[int] = 0
        self.combine_buffer_offset_in_bytes: cutlass.Constexpr[int] = 4
        self.count_buffer_offset_in_bytes: cutlass.Constexpr[int] = 16
        self.token_buffer_offset_in_bytes: cutlass.Constexpr[int] = 16 + cutlass.cute.round_up(self.num_local_experts * 4, 16)

        self.thr_tile_shape = (1, self.hidden_dim//self.threads_per_cta)
    
    @cute.jit
    def __call__(
        self,
        # input tensors
        rank_input_tensor: cute.Tensor,
        rank_input_topk_indices: cute.Tensor,
        # output tensor
        num_tokens_per_local_expert_recv: cute.Tensor,
        local_token_send_count_per_expert: cute.Tensor,
        rank_token_count: cute.Tensor,
        rank_token_index: cute.Tensor,
        recv_token_tensor: cute.Tensor,
        # buffer ptr
        local_buffer_ptr: cute.Tensor,
        remote_buffer_ptr: cute.Tensor,
        count_buffer_ptr: cute.Tensor,
    ):

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            send_index_buffer: cute.struct.MemRange[
                cutlass.Int32, 1
            ]

        self.shared_storage = SharedStorage
        
        
        # Get launch parameters
        sm_count = utils.HardwareInfo(torch.cuda.current_device()).get_device_multiprocessor_count()
        grid_dim = [sm_count, 1, 1]
        block_dim = [self.threads_per_cta, 1, 1]
        smem_size = 96 * 1024

        assert self.num_tokens_per_rank < sm_count, "The number of tokens per rank should be less than the number of SMs."
        assert self.num_local_experts * self.num_local_ranks < sm_count, "The number of local experts should be less than the number of SMs."
        assert self.hidden_dim % self.threads_per_cta == 0, "The hidden dimension should be divisible by the number of threads per CTA."

        self._setup_attributes()
        
        self.kernel(
            rank_input_tensor=rank_input_tensor,
            rank_input_topk_indices=rank_input_topk_indices,
            num_tokens_per_local_expert_recv=num_tokens_per_local_expert_recv,
            local_token_send_count_per_expert=local_token_send_count_per_expert,
            rank_token_count=rank_token_count,
            rank_token_index=rank_token_index,
            recv_token_tensor=recv_token_tensor,
            local_buffer_ptr=local_buffer_ptr,
            remote_buffer_ptr=remote_buffer_ptr,
            count_buffer_ptr=count_buffer_ptr,
        ).launch(
            grid=grid_dim,
            block=block_dim,
            smem=smem_size,
        )

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        # input tensors
        rank_input_tensor: cute.Tensor,
        rank_input_topk_indices: cute.Tensor,
        # output tensor
        num_tokens_per_local_expert_recv: cute.Tensor,
        local_token_send_count_per_expert: cute.Tensor,
        rank_token_count: cute.Tensor,
        rank_token_index: cute.Tensor,
        recv_token_tensor: cute.Tensor,
        # buffer ptr
        local_buffer_ptr: cute.Tensor,
        remote_buffer_ptr: cute.Tensor,
        count_buffer_ptr: cute.Tensor,
    ):
        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        block_dim, _, _ = cute.arch.block_dim()

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        send_index_buffer = storage.send_index_buffer.get_tensor(
            cute.make_layout((1), stride=(1))
        )

        # dispatch send

        if (block_idx * block_dim + thread_idx < self.num_local_ranks):
            sync_tensor = self.get_dispatch_sync_buffer(remote_buffer_ptr, block_idx * block_dim + thread_idx)
            sync_tensor[0] = cutlass.Uint32(1) # TODO(Zhihao): need to store in volatile mode

        # warp_idx: cutlass.Constexpr[int] = thread_idx // 32
        # lane_idx: cutlass.Constexpr[int] = thread_idx % 32

        if (block_idx < self.num_tokens_per_rank):

            thr_tiled_rank_input_tensor = cute.zipped_divide(rank_input_tensor, self.thr_tile_shape)
            thr_src_vec = thr_tiled_rank_input_tensor[(None, (block_idx, thread_idx))]

            for topk_idx in cutlass.range_constexpr(0, self.moe_param.num_topk, 1):

                # Get the local expert index
                expert_idx = rank_input_topk_indices[block_idx, topk_idx]
                
                # Get the synchronized index for sending tokens
                if (thread_idx == 0):
                    recv_index = inline_ptx.atomic_add(local_token_send_count_per_expert[expert_idx, None], 1)
                    send_index_buffer[0] = recv_index
                cute.arch.sync_threads()
                remote_index = send_index_buffer[0]

                remote_rank = expert_idx // self.num_local_experts
                remote_expert_idx = expert_idx % self.num_local_experts

                remote_tensor = self.get_dispatch_token_ptr_buffer(
                    remote_buffer_ptr,
                    remote_rank,
                    remote_expert_idx,
                    remote_index,
                )

                meta_tensor = self.get_dispatch_meta_ptr_buffer(
                    remote_buffer_ptr,
                    remote_rank,
                    remote_expert_idx,
                    remote_index,
                )

                if (thread_idx == 0):
                    # Store the meta data
                    meta_tensor[0] = cutlass.Int32(block_idx + self.num_tokens_per_rank * self.local_rank)  # token index


                thr_tiled_rank_recv_tensor = cute.zipped_divide(remote_tensor, self.thr_tile_shape)
                thr_dst_vec = thr_tiled_rank_recv_tensor[(None, (0, thread_idx))]
                    
                thr_dst_vec.store(thr_src_vec.load())

                cute.arch.sync_threads()

        # grid_sync



        # send token count to remote buffer

        if (block_idx * block_dim + thread_idx < self.moe_param.num_experts):
            expert_idx = block_idx * block_dim + thread_idx
            remote_rank = expert_idx // self.num_local_experts
            remote_expert_idx = expert_idx % self.num_local_experts
            sync_tensor = self.get_count_buffer_ptr(remote_buffer_ptr, remote_rank, remote_expert_idx)
            sync_tensor[0] = local_token_send_count_per_expert[expert_idx] + 1  # TODO(Zhihao): need to store in release mode

        # dispatch recv

        # 1. use ld_acquire to wait for the token to be sent and collect meta info

        if (block_idx < self.num_local_experts * self.num_local_ranks):
            local_expert_idx = block_idx % self.num_local_experts
            local_rank = block_idx // self.num_local_experts

            # token_count = self.get_count_buffer_ptr(local_buffer_ptr, local_rank, local_expert_idx)[0] - 1 # TODO(Zhihao): use ld_acquire here
            token_count = cutlass.Int32(2) # temporary for testing
            if thread_idx == 0:
                write_index = inline_ptx.atomic_add(rank_token_count, token_count)
                for i in cutlass.range_dynamic(0, token_count, 1, unroll=1):
                    rank_token_index[write_index+i] = i + block_idx * self.max_num_tokens

        # 2. cp from local buffer to output tensor (token parallel)
        if (block_idx < rank_token_count[0]):
            token_abs_index = rank_token_index[block_idx]
            local_rank_idx = token_abs_index // (self.max_num_tokens * self.num_local_experts)
            local_expert_idx = (token_abs_index % (self.max_num_tokens * self.num_local_experts)) // self.max_num_tokens
            token_rel_idx = token_abs_index % self.max_num_tokens

            local_buffer_tensor = self.get_dispatch_token_ptr_buffer(
                    local_buffer_ptr,
                    local_rank_idx,
                    local_expert_idx,
                    token_rel_idx,
                )
            tiled_src_tensor = cute.zipped_divide(local_buffer_tensor, self.thr_tile_shape)
            thr_src_vec = tiled_src_tensor[(None, (0, thread_idx))]

            dst_tensor = recv_token_tensor[(local_rank_idx, local_expert_idx, token_rel_idx, None, None)]
            tiled_dst_tensor = cute.zipped_divide(dst_tensor, self.thr_tile_shape)
            thr_dst_vec = tiled_dst_tensor[(None, (0, thread_idx))]
            thr_dst_vec.store(thr_src_vec.load())

        

    @cute.jit
    def make_global_tensor_from_buffer_ptr(
        self,
        dtype: Type[cutlass.Numeric],
        offset: cutlass.Int64,
        layout: cutlass.cute.typing.Layout,
        ptr_i64: cutlass.Int64,
    ):
        """
        Create a global tensor from a buffer pointer.
        Args:
            dtype (Type[cutlass.Numeric]): The data type of the tensor.
            offset (cutlass.Int64): The offset in bytes of the tensor in the buffer.
            layout (cutlass.cute.typing.Layout): The layout of the tensor.
            ptr_i64 (cutlass.Int64): The pointer to the buffer.
        Returns:
            cute.Tensor: The global tensor.
        """
        if cutlass.const_expr(
            not isclass(dtype) or not issubclass(dtype, cutlass.Numeric)
        ):
            raise TypeError(
                f"dtype must be a type of cutlass.Numeric, got {type(dtype)}"
            )
        tensor_gmem_ptr = cute.make_ptr(
            dtype, ptr_i64+offset, cute.AddressSpace.gmem, assumed_align=16
        )
        tensor = cute.make_tensor(tensor_gmem_ptr, layout)
        return tensor
    

    def get_dispatch_sync_buffer(
        self,
        buffer_ptr_tensor: cute.Tensor,
        rank: cutlass.Int32,
    ):
        """
        Get the dispatch sync buffer from the buffer pointer.
        Args:
            ptr_i64 (cutlass.Int64): The pointer to the buffer.
            rank (cutlass.Int32): The rank of the process.
        Returns:
            cute.Tensor: The dispatch sync buffer.
        """
        return self.make_global_tensor_from_buffer_ptr(
                dtype=cutlass.Uint32,
                offset=self.dispatch_buffer_offset_in_bytes,
                layout=cute.make_layout((1,), stride=(1,)),
                ptr_i64=buffer_ptr_tensor[rank],
            )
    
    def get_combine_sync_buffer(
        self,
        buffer_ptr_tensor: cute.Tensor,
        rank: cutlass.Int32,
    ):
        """
        Get the dispatch sync buffer from the buffer pointer.
        Args:
            ptr_i64 (cutlass.Int64): The pointer to the buffer.
            rank (cutlass.Int32): The rank of the process.
        Returns:
            cute.Tensor: The dispatch sync buffer.
        """
        return self.make_global_tensor_from_buffer_ptr(
                dtype=cutlass.Uint32,
                offset=self.combine_buffer_offset_in_bytes,
                layout=cute.make_layout((1,), stride=(1,)),
                ptr_i64=buffer_ptr_tensor[rank],
            )
    
    def get_dispatch_token_ptr_buffer(
        self,
        buffer_ptr_tensor: cute.Tensor,
        rank: cutlass.Int32,
        expert_idx: cutlass.Int32,
        recv_token_idx: cutlass.Int64,
    ):
        """
        Get the token pointer buffer from the buffer pointer.
        Args:
            ptr_i64 (cutlass.Int64): The pointer to the buffer.
            rank (cutlass.Int32): The rank of the process.
        Returns:
            cute.Tensor: The token pointer buffer.
        """

        ptr_offset = self.token_buffer_offset_in_bytes
        ptr_offset += (expert_idx * self.max_num_tokens + recv_token_idx) * self.dispatch_token_stride

        # cute.printf(">?? rank: {}, expert_idx: {}, recv_token_idx: {}, offset: {}", rank, expert_idx, recv_token_idx, offset)
        # cute.printf(">?? token_buffer_offset_in_bytes: {}", self.token_buffer_offset_in_bytes)

        return self.make_global_tensor_from_buffer_ptr(
                dtype=self.moe_param.in_dtype,
                offset=ptr_offset,
                layout=cute.make_layout((1, self.hidden_dim), stride=(self.hidden_dim, 1)),
                ptr_i64=buffer_ptr_tensor[rank],
            )

    def get_dispatch_meta_ptr_buffer(
        self,
        buffer_ptr_tensor: cute.Tensor,
        rank: cutlass.Int32,
        expert_idx: cutlass.Int32,
        recv_token_idx: cutlass.Int64,
    ):
        """
        Get the meta pointer buffer from the buffer pointer.
        Args:
            ptr_i64 (cutlass.Int64): The pointer to the buffer.
            rank (cutlass.Int32): The rank of the process.
        Returns:
            cute.Tensor: The meta pointer buffer.
        """
        ptr_offset = self.token_buffer_offset_in_bytes
        ptr_offset += (expert_idx * self.max_num_tokens + recv_token_idx) * self.dispatch_token_stride

        return self.make_global_tensor_from_buffer_ptr(
                dtype=cutlass.Int32,
                offset=ptr_offset + self.hidden_dim_in_bytes,
                layout=cute.make_layout((1), stride=(1)),
                ptr_i64=buffer_ptr_tensor[rank],
            )
    
    def get_combine_token_ptr_buffer(
        self,
        buffer_ptr_tensor: cute.Tensor,
        rank: cutlass.Int32,
        expert_idx: cutlass.Int64,
        recv_token_idx: cutlass.Int64,
    ):
        """
        Get the token pointer buffer from the buffer pointer.
        Args:
            ptr_i64 (cutlass.Int64): The pointer to the buffer.
            rank (cutlass.Int32): The rank of the process.
        Returns:
            cute.Tensor: The token pointer buffer.
        """
        ptr_offset = self.token_buffer_offset_in_bytes
        ptr_offset += (expert_idx * self.max_num_tokens + recv_token_idx) * self.combine_token_stride

        return self.make_global_tensor_from_buffer_ptr(
                dtype=self.moe_param.out_dtype,
                offset=ptr_offset,
                layout=cute.make_layout((1,self.hidden_dim), stride=(self.hidden_dim,1)),
                ptr_i64=buffer_ptr_tensor[rank],
            )
    
    def get_count_buffer_ptr(
        self,
        buffer_ptr_tensor: cute.Tensor,
        rank: cutlass.Int32,
        expert_idx: cutlass.Int64 = 0,
    ):
        """
        Get the count buffer pointer from the buffer pointer.
        Args:
            ptr_i64 (cutlass.Int64): The pointer to the buffer.
            rank (cutlass.Int32): The rank of the process.
        Returns:
            cute.Tensor: The count buffer pointer.
        """
        offset = self.count_buffer_offset_in_bytes + expert_idx * 4
        return self.make_global_tensor_from_buffer_ptr(
                dtype=cutlass.Int32,
                offset=offset,
                layout=cute.make_layout((1), stride=(1)),
                ptr_i64=buffer_ptr_tensor[rank],
            )