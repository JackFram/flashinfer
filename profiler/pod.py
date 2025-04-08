"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse

import torch

import flashinfer
from flashinfer.profiler import export_to_perfetto_trace_pod


def profile_pod_decode(
    # Prefill params
    kv_len_p,
    qo_len_p,
    causal,
    # Decode params
    batch_size_d,
    kv_len_d,
    page_size_d,
    kv_layout_d,
    # Shared params
    num_kv_heads,
    num_qo_heads,
    head_dim,
    pos_encoding_mode,
    q_dtype,
    kv_dtype,
    contiguous_kv,
    profiler_buffer_size,
):
    if causal and qo_len_p > kv_len_p:
        raise ValueError("Causal prefill with qo_len_p > kv_len_p is not supported")
    return_lse = False
    # Prefill inputs
    kv_layout_p = "NHD"
    q_p = torch.randn(
        qo_len_p, num_qo_heads, head_dim, device="cuda:0", dtype=torch.float16
    )
    k_p = torch.randn(
        kv_len_p, num_kv_heads, head_dim, device="cuda:0", dtype=torch.float16
    )
    v_p = torch.randn(
        kv_len_p, num_kv_heads, head_dim, device="cuda:0", dtype=torch.float16
    )

    # Decode inputs
    q_d = torch.randn(
        batch_size_d, num_qo_heads, head_dim, device="cuda:0", dtype=torch.float16
    )
    num_pages_per_seq = (kv_len_d + page_size_d - 1) // page_size_d
    total_num_pages = num_pages_per_seq * batch_size_d
    if kv_layout_d == "HND":
        kv_shape = [total_num_pages, 2, num_kv_heads, page_size_d, head_dim]
    else:
        kv_shape = [total_num_pages, 2, page_size_d, num_kv_heads, head_dim]
    if not contiguous_kv:
        tmp = [kv_shape[0]]
        for v_d in kv_shape[1:]:
            tmp.append(2)
            tmp.append(v_d)
        kv_shape = tmp
        kv_data_fp32 = torch.randn(*kv_shape, device="cuda:0", dtype=torch.float32)
        kv_data = kv_data_fp32.to(kv_dtype)
        kv_data = kv_data[:, 1, :, 1, :, 1, :, 1, :]
        kv_data_fp32 = kv_data_fp32[:, 1, :, 1, :, 1, :, 1, :]
        # actual data is stored in non-contiguous memory
        assert (
            kv_data.stride(-4)
            != kv_data.shape[-3] * kv_data.shape[-2] * kv_data.shape[-1]
        )
    else:
        kv_data_fp32 = torch.randn(*kv_shape, device="cuda:0", dtype=torch.float32)
        kv_data = kv_data_fp32.to(kv_dtype)
    kv_indptr_d = (
        torch.arange(0, batch_size_d + 1, device="cuda:0", dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices_d = torch.arange(0, total_num_pages, device="cuda:0", dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size_d,),
        (kv_len_d - 1) % page_size_d + 1,
        device="cuda:0",
        dtype=torch.int32,
    )
    
    workspace_buffer = torch.empty(32 * 1024 * 1024, device="cuda:0", dtype=torch.int8)
    pod_wrapper = flashinfer.PODWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout_d,
    )
    pod_wrapper.plan(
        kv_indptr_d,
        kv_indices_d,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size_d,
        pos_encoding_mode=pos_encoding_mode,
        data_type=kv_dtype,
        q_data_type=q_dtype,
        use_profiler=True,
    )
    
    profiler_buffer = torch.zeros(
        (profiler_buffer_size,), dtype=torch.uint64, device="cuda"
    )
    # warmup run
    o = pod_wrapper.run(
        q_p,
        k_p,
        v_p,
        q_d,
        kv_data,
        pos_encoding_mode_p=pos_encoding_mode,
        causal_p=causal,
        profiler_buffer=profiler_buffer,
    )
    profiler_buffer.zero_()
    # run

    o_p, o_d = pod_wrapper.run(
        q_p,
        k_p,
        v_p,
        q_d,
        kv_data,
        pos_encoding_mode_p=pos_encoding_mode,
        causal_p=causal,
        profiler_buffer=profiler_buffer,
    )

    export_to_perfetto_trace_pod(
        profiler_buffer,
        [
            "prefill",
            "decode",
            "schedule",
        ],
        f"pod-{kv_len_p}-{qo_len_p}-{batch_size_d}-{kv_len_d}-{page_size_d}-{kv_layout_d}-{num_qo_heads}-{num_kv_heads}-{head_dim}.perfetto-trace",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Intra-kernel profiling for FlashInfer POD kernels"
    )

    # Prefill params
    parser.add_argument("--kv-len-p", type=int, default=12288)
    parser.add_argument("--qo-len-p", type=int, default=12288)
    
    # Decode params
    parser.add_argument("--batch-size-d", type=int, default=80)
    parser.add_argument("--kv-len-d", type=int, default=12288)
    parser.add_argument("--page-size-d", type=int, default=16)
    parser.add_argument("--kv-layout-d", type=str, default="NHD")
    
    # Shared params
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--num-qo-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)

    # Profile params
    parser.add_argument("--profiler-buffer-size", type=int, default=1024 * 1024)
    args = parser.parse_args()
    
    profile_pod_decode(
        # Prefill params
        args.kv_len_p,
        args.qo_len_p,
        True,
        # Decode params
        args.batch_size_d,
        args.kv_len_d,
        args.page_size_d,
        args.kv_layout_d,
        # Shared params
        args.num_kv_heads,
        args.num_qo_heads,
        args.head_dim,
        "NONE",
        torch.float16,
        torch.float16,
        True,
        args.profiler_buffer_size
    )
