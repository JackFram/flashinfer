/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_PERSISTENT_WORKER_CUH_
#define FLASHINFER_PERSISTENT_WORKER_CUH_

#include <cuda_runtime.h>

#include <cstdint>

#include "mask.cuh"
#include "prefill.cuh"

namespace flashinfer {

template <typename KTraits_, typename Params_>
struct PersistentWorker {
  using KTraits = KTraits_;
  using Params = Params_;
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  using DTypeQKAccum = typename KTraits::DTypeQKAccum;
  using AttentionVariant = typename KTraits::AttentionVariant;
  [[maybe_unused]] static constexpr uint32_t NUM_MMA_Q = KTraits::NUM_MMA_Q;
  [[maybe_unused]] static constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  [[maybe_unused]] static constexpr uint32_t NUM_MMA_D_QK = KTraits::NUM_MMA_D_QK;
  [[maybe_unused]] static constexpr uint32_t NUM_MMA_D_VO = KTraits::NUM_MMA_D_VO;
  [[maybe_unused]] static constexpr uint32_t HEAD_DIM_QK = KTraits::HEAD_DIM_QK;
  [[maybe_unused]] static constexpr uint32_t HEAD_DIM_VO = KTraits::HEAD_DIM_VO;
  [[maybe_unused]] static constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;
  [[maybe_unused]] static constexpr uint32_t UPCAST_STRIDE_K = KTraits::UPCAST_STRIDE_K;
  [[maybe_unused]] static constexpr uint32_t UPCAST_STRIDE_V = KTraits::UPCAST_STRIDE_V;
  [[maybe_unused]] static constexpr uint32_t UPCAST_STRIDE_O = KTraits::UPCAST_STRIDE_O;
  [[maybe_unused]] static constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  [[maybe_unused]] static constexpr uint32_t NUM_WARPS_KV = KTraits::NUM_WARPS_KV;
  [[maybe_unused]] static constexpr SwizzleMode SWIZZLE_MODE_Q = KTraits::SWIZZLE_MODE_Q;
  [[maybe_unused]] static constexpr SwizzleMode SWIZZLE_MODE_KV = KTraits::SWIZZLE_MODE_KV;
  [[maybe_unused]] static constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  [[maybe_unused]] static constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
  [[maybe_unused]] static constexpr bool CAUSAL = KTraits::MASK_MODE == MaskMode::kCausal;
  [[maybe_unused]] static constexpr uint32_t NUM_STAGES = KTraits::NUM_STAGES;

  float s_frag[NUM_MMA_Q][NUM_MMA_KV][8];
  alignas(16) float o_frag[NUM_MMA_Q][NUM_MMA_D_VO][8];
  float m[NUM_MMA_Q][2];
  float d[NUM_MMA_Q][2];

  const uint32_t lane_idx = threadIdx.x % 32;
  const uint32_t warp_idx = threadIdx.x / 32;

  uint32_t q_smem_offset_r = get_permuted_offset<SWIZZLE_MODE_Q, UPCAST_STRIDE_Q>(
      get_warp_idx_q<KTraits>(warp_idx) * NUM_MMA_Q * 16 + lane_idx % 16, lane_idx / 16);
  uint32_t k_smem_offset_r = get_permuted_offset<SWIZZLE_MODE_KV, UPCAST_STRIDE_K>(
               get_warp_idx_kv<KTraits>(warp_idx) * NUM_MMA_KV * 16 + 8 * (lane_idx / 16) +
                   lane_idx % 8,
               (lane_idx % 16) / 8),
           v_smem_offset_r = get_permuted_offset<SWIZZLE_MODE_KV, UPCAST_STRIDE_V>(
               get_warp_idx_kv<KTraits>(warp_idx) * NUM_MMA_KV * 16 + lane_idx % 16, lane_idx / 16);
  uint32_t k_smem_offset_w = get_permuted_offset<SWIZZLE_MODE_KV, UPCAST_STRIDE_K>(
               warp_idx * KTraits::KV_THR_LAYOUT_ROW + lane_idx / KTraits::KV_THR_LAYOUT_COL,
               lane_idx % KTraits::KV_THR_LAYOUT_COL),
           v_smem_offset_w = get_permuted_offset<SWIZZLE_MODE_KV, UPCAST_STRIDE_V>(
               warp_idx * KTraits::KV_THR_LAYOUT_ROW + lane_idx / KTraits::KV_THR_LAYOUT_COL,
               lane_idx % KTraits::KV_THR_LAYOUT_COL);

  size_t thr_local_kv_offset[NUM_MMA_KV * KTraits::KV_THR_LAYOUT_COL / 2 / KTraits::NUM_WARPS_Q];

  __device__ __forceinline__ PersistentWorker(const Params& params)
      : q(params.q),
        k(params.k),
        v(params.v),
        kv_indices(params.kv_indices),
        partial_o(params.partial_o),
        partial_lse(params.partial_lse),
        final_o(params.final_o),
        final_lse(params.final_lse),
        work_indptr(params.work_indptr),
        gqa_group_size(params.gqa_group_size),
        block_size(params.page_size),
        q_stride_n(params.q_stride_n),
        q_stride_h(params.q_stride_h),
        k_stride_page(params.k_stride_page),
        k_stride_h(params.k_stride_h),
        k_stride_n(params.k_stride_n),
        v_stride_page(params.v_stride_page),
        v_stride_h(params.v_stride_h),
        v_stride_n(params.v_stride_n),
        o_stride_n(params.o_stride_n),
        o_stride_h(params.o_stride_h),
        cluster_tile_q(gridDim.x * CTA_TILE_Q),
        variant(params, /*batch_idx=*/0, nullptr) {}

  __device__ __forceinline__ void set_work_tile_info(const Params& params,
                                                     const uint32_t work_idx) {
    batch_idx = 0;  // here we assume batch_idx is always 0
    q_indptr = params.q_indptr[work_idx];
    kv_indptr = params.kv_indptr[work_idx];
    partial_indptr = params.partial_indptr[work_idx];
    q_len = params.q_len[work_idx];
    kv_len = params.kv_len[work_idx];
    packed_qo_start = params.q_start[work_idx];
    kv_start = params.kv_start[work_idx];
    kv_end = params.kv_end[work_idx];
    kv_head_idx = params.kv_head_idx_arr[work_idx];

    init_states<KTraits>(variant, o_frag, m, d);

    qo_packed_idx_base = packed_qo_start + blockIdx.x * CTA_TILE_Q +
                         get_warp_idx_q<KTraits>(warp_idx) * NUM_MMA_Q * 16;

    qo_upperbound = min(q_len, ceil_div(qo_packed_idx_base + CTA_TILE_Q, gqa_group_size));
  }

  __device__ __forceinline__ void load_q_global_smem(smem_t<SWIZZLE_MODE_Q>* q_smem) {
    DTypeQ* q_ptr_base = q + q_indptr * q_stride_n + (kv_head_idx * gqa_group_size) * q_stride_h;
    const uint32_t warp_idx_x = get_warp_idx_q<KTraits>(warp_idx);
    if (get_warp_idx_kv<KTraits>(warp_idx) == 0) {
      uint32_t q_smem_offset_w = q_smem->get_permuted_offset<UPCAST_STRIDE_Q>(
          warp_idx_x * NUM_MMA_Q * 16 + lane_idx / 8, lane_idx % 8);

#pragma unroll
      for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
        for (uint32_t j = 0; j < 2 * 2; ++j) {
          uint32_t q, r;
          gqa_group_size.divmod(qo_packed_idx_base + lane_idx / 8 + mma_q * 16 + j * 4, q, r);
          const uint32_t q_idx = q;
          DTypeQ* q_ptr =
              q_ptr_base + q * q_stride_n + r * q_stride_h + (lane_idx % 8) * upcast_size<DTypeQ>();
#pragma unroll
          for (uint32_t mma_do = 0; mma_do < NUM_MMA_D_QK / 4; ++mma_do) {
            // load q fragment from gmem to smem
            q_smem->load_128b_async<SharedMemFillMode::kNoFill>(q_smem_offset_w, q_ptr,
                                                                q_idx < qo_upperbound);
            q_smem_offset_w = q_smem->template advance_offset_by_column<8>(q_smem_offset_w, mma_do);
            q_ptr += 8 * upcast_size<DTypeQ>();
          }
          q_smem_offset_w =
              q_smem->template advance_offset_by_row<4, UPCAST_STRIDE_Q>(q_smem_offset_w) -
              2 * NUM_MMA_D_QK;
        }
      }
    }
  }

  __device__ __forceinline__ void init_kv_info() {
    kv_tile_idx =
        ceil_div((CAUSAL ? min(kv_end,
                               kv_len - q_len + (packed_qo_start + cluster_tile_q) / gqa_group_size)
                         : kv_end),
                 CTA_TILE_KV) -
        1 - (kv_start / CTA_TILE_KV);

    mask_tile_idx =
        (CAUSAL ? min(kv_end, kv_len - q_len + packed_qo_start / gqa_group_size) : kv_end) /
            CTA_TILE_KV -
        (kv_start / CTA_TILE_KV);

    block_iter_base = kv_indptr * block_size + kv_start;
    packed_kv_bound = kv_indptr * block_size + kv_len;
  }

  __device__ __forceinline__ void prefetch_offset(const uint32_t prefetch_offset) {
    constexpr uint32_t KV_THR_LAYOUT_ROW = KTraits::KV_THR_LAYOUT_ROW;
    constexpr uint32_t KV_THR_LAYOUT_COL = KTraits::KV_THR_LAYOUT_COL;
#pragma unroll
    for (uint32_t i = 0;
         i < NUM_MMA_KV * (SWIZZLE_MODE_KV == SwizzleMode::k128B ? 4 : 2) / NUM_WARPS_Q; ++i) {
      uint32_t page_iter, entry_idx;
      uint32_t packed_block_iter = block_iter_base + (kv_tile_idx - prefetch_offset) * CTA_TILE_KV +
                                   warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL +
                                   KV_THR_LAYOUT_ROW * NUM_WARPS_Q * NUM_WARPS_KV * i;
      block_size.divmod(packed_block_iter, page_iter, entry_idx);
      thr_local_kv_offset[i] =
          (packed_block_iter < packed_kv_bound ? kv_indices[page_iter] : 0) * k_stride_page +
          entry_idx * k_stride_n + kv_head_idx * k_stride_h +
          (lane_idx % KV_THR_LAYOUT_COL) * upcast_size<DTypeKV>();
    }
  }

  __device__ __forceinline__ void page_load_k(smem_t<SWIZZLE_MODE_KV>* k_smem,
                                              const uint32_t kv_tile_offset) {
    constexpr SharedMemFillMode fill_mode = SharedMemFillMode::kNoFill;
    constexpr uint32_t NUM_MMA_D = NUM_MMA_D_QK;
    constexpr uint32_t NUM_WARPS = KTraits::NUM_WARPS;
    constexpr uint32_t UPCAST_STRIDE = UPCAST_STRIDE_K;
    const uint32_t kv_idx_base = kv_start + (kv_tile_idx - kv_tile_offset) * CTA_TILE_KV;
    if constexpr (SWIZZLE_MODE_KV == SwizzleMode::k128B) {
      uint32_t kv_idx = kv_idx_base + warp_idx * 4 + lane_idx / 8;
      // NOTE: NUM_MMA_KV * 4 / NUM_WARPS_Q = NUM_WARPS_KV * NUM_MMA_KV * 4 / num_warps
      static_assert(NUM_MMA_KV * 4 % NUM_WARPS_Q == 0);
      
#pragma unroll
      for (uint32_t i = 0; i < NUM_MMA_KV * 4 / NUM_WARPS_Q; ++i) {
        DTypeKV* kv = k + thr_local_kv_offset[i];
#pragma unroll
        for (uint32_t j = 0; j < NUM_MMA_D / (8 / sizeof(DTypeKV)); ++j) {
          k_smem->load_128b_async<fill_mode>(k_smem_offset_w, kv, kv_idx < kv_end);
          k_smem_offset_w = k_smem->template advance_offset_by_column<8>(k_smem_offset_w, j);
          kv += 8 * upcast_size<DTypeKV>();
        }
        kv_idx += NUM_WARPS * 4;
        k_smem_offset_w =
            k_smem->template advance_offset_by_row<NUM_WARPS * 4, UPCAST_STRIDE>(k_smem_offset_w) -
            sizeof(DTypeKV) * NUM_MMA_D;
      }
      k_smem_offset_w -= CTA_TILE_KV * UPCAST_STRIDE;
    } else {
      uint32_t kv_idx = kv_idx_base + warp_idx * 8 + lane_idx / 4;
      // NOTE: NUM_MMA_KV * 2 / NUM_WARPS_Q = NUM_WARPS_KV * NUM_MMA_KV * 2 / num_warps
      static_assert(NUM_MMA_KV * 2 % NUM_WARPS_Q == 0);
#pragma unroll
      for (uint32_t i = 0; i < NUM_MMA_KV * 2 / NUM_WARPS_Q; ++i) {
        DTypeKV* kv = k + thr_local_kv_offset[i];
        k_smem->load_128b_async<fill_mode>(k_smem_offset_w, kv, kv_idx < kv_end);
        kv_idx += NUM_WARPS * 8;
        k_smem_offset_w =
            k_smem->template advance_offset_by_row<NUM_WARPS * 8, UPCAST_STRIDE>(k_smem_offset_w);
      }
      k_smem_offset_w -= CTA_TILE_KV * UPCAST_STRIDE;
    }
  }

  __device__ __forceinline__ void page_load_v(smem_t<SWIZZLE_MODE_KV>* v_smem,
                                              const uint32_t kv_tile_offset) {
    using DType = typename KTraits::DTypeKV;
    constexpr SharedMemFillMode fill_mode = SharedMemFillMode::kFillZero;
    constexpr uint32_t NUM_MMA_D = KTraits::NUM_MMA_D_VO;
    constexpr uint32_t NUM_WARPS = KTraits::NUM_WARPS;
    constexpr uint32_t UPCAST_STRIDE = KTraits::UPCAST_STRIDE_V;
    const uint32_t kv_idx_base = kv_start + (kv_tile_idx - kv_tile_offset) * CTA_TILE_KV;
    if constexpr (SWIZZLE_MODE_KV == SwizzleMode::k128B) {
      uint32_t kv_idx = kv_idx_base + warp_idx * 4 + lane_idx / 8;
      // NOTE: NUM_MMA_KV * 4 / NUM_WARPS_Q = NUM_WARPS_KV * NUM_MMA_KV * 4 / num_warps
      static_assert(NUM_MMA_KV * 4 % NUM_WARPS_Q == 0);
#pragma unroll
      for (uint32_t i = 0; i < NUM_MMA_KV * 4 / NUM_WARPS_Q; ++i) {
        DType* kv = v + thr_local_kv_offset[i];
#pragma unroll
        for (uint32_t j = 0; j < NUM_MMA_D / (8 / sizeof(DType)); ++j) {
          v_smem->load_128b_async<fill_mode>(v_smem_offset_w, kv, kv_idx < kv_end);
          v_smem_offset_w = v_smem->template advance_offset_by_column<8>(v_smem_offset_w, j);
          kv += 8 * upcast_size<DType>();
        }
        kv_idx += NUM_WARPS * 4;
        v_smem_offset_w =
            v_smem->template advance_offset_by_row<NUM_WARPS * 4, UPCAST_STRIDE>(v_smem_offset_w) -
            sizeof(DType) * NUM_MMA_D;
      }
      v_smem_offset_w -= CTA_TILE_KV * UPCAST_STRIDE;
    } else {
      uint32_t kv_idx = kv_idx_base + warp_idx * 8 + lane_idx / 4;
      // NOTE: NUM_MMA_KV * 2 / NUM_WARPS_Q = NUM_WARPS_KV * NUM_MMA_KV * 2 / num_warps
      static_assert(NUM_MMA_KV * 2 % NUM_WARPS_Q == 0);
#pragma unroll
      for (uint32_t i = 0; i < NUM_MMA_KV * 2 / NUM_WARPS_Q; ++i) {
        DType* kv = k + thr_local_kv_offset[i];
        v_smem->load_128b_async<fill_mode>(v_smem_offset_w, kv, kv_idx < kv_end);
        kv_idx += NUM_WARPS * 8;
        v_smem_offset_w =
            v_smem->template advance_offset_by_row<NUM_WARPS * 8, UPCAST_STRIDE>(v_smem_offset_w);
      }
      v_smem_offset_w -= CTA_TILE_KV * UPCAST_STRIDE;
    }
  }

  __device__ __forceinline__ void gemm_qk(smem_t<SWIZZLE_MODE_Q>* q_smem,
                                          smem_t<SWIZZLE_MODE_KV>* k_smem) {
    uint32_t a_frag[NUM_MMA_Q][4], b_frag[4];
    // compute q*k^T
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_QK; ++mma_d) {
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
        q_smem->ldmatrix_m8n8x4(q_smem_offset_r, a_frag[mma_q]);
        q_smem_offset_r =
            q_smem->template advance_offset_by_row<16, UPCAST_STRIDE_Q>(q_smem_offset_r);
      }

      q_smem_offset_r = q_smem->template advance_offset_by_column<2>(q_smem_offset_r, mma_d) -
                        NUM_MMA_Q * 16 * UPCAST_STRIDE_Q;

#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
        if constexpr (sizeof(DTypeKV) == 1) {
          uint32_t b_frag_f8[2];
          if (mma_d % 2 == 0) {
            k_smem->ldmatrix_m8n8x4_left_half(k_smem_offset_r, b_frag_f8);
          } else {
            k_smem->ldmatrix_m8n8x4_right_half(k_smem_offset_r, b_frag_f8);
          }
          b_frag_f8[0] = frag_layout_swizzle_16b_to_8b(b_frag_f8[0]);
          b_frag_f8[1] = frag_layout_swizzle_16b_to_8b(b_frag_f8[1]);
          vec_cast<DTypeQ, DTypeKV>::cast<8>((DTypeQ*)b_frag,
                                                               (DTypeKV*)b_frag_f8);
        } else {
          k_smem->ldmatrix_m8n8x4(k_smem_offset_r, b_frag);
        }
        k_smem_offset_r =
            k_smem->template advance_offset_by_row<16, UPCAST_STRIDE_K>(k_smem_offset_r);

#pragma unroll
        for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
          if constexpr (std::is_same_v<DTypeQKAccum, float>) {
            if (mma_d == 0) {
              mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeQ, MMAMode::kInit>(
                  s_frag[mma_q][mma_kv], a_frag[mma_q], b_frag);
            } else {
              mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeQ>(s_frag[mma_q][mma_kv],
                                                                         a_frag[mma_q], b_frag);
            }
          } else if (std::is_same_v<DTypeQKAccum, half>) {
            if (mma_d == 0) {
              mma::mma_sync_m16n16k16_row_col_f16f16f16<MMAMode::kInit>(
                  (uint32_t*)s_frag[mma_q][mma_kv], a_frag[mma_q], b_frag);
            } else {
              mma::mma_sync_m16n16k16_row_col_f16f16f16((uint32_t*)s_frag[mma_q][mma_kv],
                                                        a_frag[mma_q], b_frag);
            }
          }
        }
      }
      if constexpr (sizeof(DTypeKV) == 1) {
        if (mma_d % 2 == 1) {
          k_smem_offset_r =
              k_smem->template advance_offset_by_column<2>(k_smem_offset_r, mma_d / 2);
        }
        k_smem_offset_r -= NUM_MMA_KV * 16 * UPCAST_STRIDE_K;
      } else {
        k_smem_offset_r = k_smem->template advance_offset_by_column<2>(k_smem_offset_r, mma_d) -
                          NUM_MMA_KV * 16 * UPCAST_STRIDE_K;
      }
    }
    q_smem_offset_r -= NUM_MMA_D_QK * 2;
    k_smem_offset_r -= NUM_MMA_D_QK * sizeof(DTypeKV);
  }

  __device__ __forceinline__ void gemm_pv(smem_t<SWIZZLE_MODE_KV>* v_smem) {
    DTypeQ p_frag[NUM_MMA_Q][NUM_MMA_KV][8];
    if constexpr (std::is_same_v<DTypeQKAccum, float>) {
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
        for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
          vec_cast<DTypeQ, float>::cast<8>(p_frag[mma_q][mma_kv], s_frag[mma_q][mma_kv]);
        }
      }
    }

    if constexpr (KTraits::AttentionVariant::use_softmax) {
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
        for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
          if constexpr (std::is_same_v<DTypeQKAccum, float>) {
            mma::m16k16_rowsum_f16f16f32(d[mma_q], p_frag[mma_q][mma_kv]);
          } else {
            mma::m16k16_rowsum_f16f16f32(d[mma_q], s_frag[mma_q][mma_kv]);
          }
        }
      }
    }

#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_VO; ++mma_d) {
        uint32_t b_frag[4];
        if constexpr (sizeof(DTypeKV) == 1) {
          uint32_t b_frag_f8[2];
          if (mma_d % 2 == 0) {
            v_smem->ldmatrix_m8n8x4_trans_left_half(v_smem_offset_r, b_frag_f8);
          } else {
            v_smem->ldmatrix_m8n8x4_trans_right_half(v_smem_offset_r, b_frag_f8);
          }
          b_frag_f8[0] = frag_layout_swizzle_16b_to_8b_trans(b_frag_f8[0]);
          b_frag_f8[1] = frag_layout_swizzle_16b_to_8b_trans(b_frag_f8[1]);
          vec_cast<DTypeQ, DTypeKV>::cast<8>((DTypeQ*)b_frag, (DTypeKV*)b_frag_f8);
          swap(b_frag[1], b_frag[2]);
        } else {
          v_smem->ldmatrix_m8n8x4_trans(v_smem_offset_r, b_frag);
        }
#pragma unroll
        for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
          if constexpr (std::is_same_v<DTypeQKAccum, float>) {
            mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeQ>(
                o_frag[mma_q][mma_d], (uint32_t*)p_frag[mma_q][mma_kv], b_frag);
          } else {
            mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeQ>(
                o_frag[mma_q][mma_d], (uint32_t*)s_frag[mma_q][mma_kv], b_frag);
          }
        }
        if constexpr (sizeof(DTypeKV) == 1) {
          if (mma_d % 2 == 1) {
            v_smem_offset_r =
                v_smem->template advance_offset_by_column<2>(v_smem_offset_r, mma_d / 2);
          }
        } else {
          v_smem_offset_r = v_smem->template advance_offset_by_column<2>(v_smem_offset_r, mma_d);
        }
      }
      v_smem_offset_r =
          v_smem->template advance_offset_by_row<16, UPCAST_STRIDE_V>(v_smem_offset_r) -
          sizeof(DTypeKV) * NUM_MMA_D_VO;
    }
    v_smem_offset_r -= 16 * NUM_MMA_KV * UPCAST_STRIDE_V;
  }

  __device__ __forceinline__ void logits_mask(const Params& params) {
    constexpr MaskMode MASK_MODE = KTraits::MASK_MODE;
    const uint32_t kv_idx_base =
        kv_start +
        (kv_tile_idx * NUM_WARPS_KV + get_warp_idx_kv<KTraits>(warp_idx)) * NUM_MMA_KV * 16;
    uint32_t q_reg[NUM_MMA_Q][2], r_reg[NUM_MMA_Q][2];
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        gqa_group_size.divmod(qo_packed_idx_base + mma_q * 16 + lane_idx / 4 + 8 * j,
                              q_reg[mma_q][j], r_reg[mma_q][j]);
      }
    }

#pragma unroll
    for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
#pragma unroll
        for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
          const uint32_t q_idx = q_reg[mma_q][(reg_id % 4) / 2],
                         kv_idx = kv_idx_base + mma_kv * 16 + 2 * (lane_idx % 4) +
                                  8 * (reg_id / 4) + reg_id % 2;
          const uint32_t qo_head_idx =
              kv_head_idx * gqa_group_size + r_reg[mma_q][(reg_id % 4) / 2];
          const bool mask =
              (!(MASK_MODE == MaskMode::kCausal
                     ? (kv_idx + q_len > kv_len + q_idx || (kv_idx >= kv_end))
                     : kv_idx >= kv_end)) &&
              variant.LogitsMask(params, batch_idx, q_idx, kv_idx, qo_head_idx, kv_head_idx);

          s_frag[mma_q][mma_kv][reg_id] =
              (mask) ? s_frag[mma_q][mma_kv][reg_id] : (KTraits::MaskFillValue);
        }
      }
    }
  }

  __device__ __forceinline__ void update_mdo_states() {
    using AttentionVariant = typename KTraits::AttentionVariant;
    constexpr bool use_softmax = AttentionVariant::use_softmax;

    if constexpr (use_softmax) {
      const float sm_scale = variant.sm_scale_log2;
      if constexpr (std::is_same_v<DTypeQKAccum, float>) {
#pragma unroll
        for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
          for (uint32_t j = 0; j < 2; ++j) {
            float m_prev = m[mma_q][j];
#pragma unroll
            for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
              float m_local =
                  max(max(s_frag[mma_q][mma_kv][j * 2 + 0], s_frag[mma_q][mma_kv][j * 2 + 1]),
                      max(s_frag[mma_q][mma_kv][j * 2 + 4], s_frag[mma_q][mma_kv][j * 2 + 5]));
              m[mma_q][j] = max(m[mma_q][j], m_local);
            }
            m[mma_q][j] = max(m[mma_q][j], math::shfl_xor_sync(m[mma_q][j], 0x2));
            m[mma_q][j] = max(m[mma_q][j], math::shfl_xor_sync(m[mma_q][j], 0x1));

            float o_scale = math::ptx_exp2(m_prev * sm_scale - m[mma_q][j] * sm_scale);
            d[mma_q][j] *= o_scale;
#pragma unroll
            for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_VO; ++mma_d) {
              o_frag[mma_q][mma_d][j * 2 + 0] *= o_scale;
              o_frag[mma_q][mma_d][j * 2 + 1] *= o_scale;
              o_frag[mma_q][mma_d][j * 2 + 4] *= o_scale;
              o_frag[mma_q][mma_d][j * 2 + 5] *= o_scale;
            }
#pragma unroll
            for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
              s_frag[mma_q][mma_kv][j * 2 + 0] = math::ptx_exp2(
                  s_frag[mma_q][mma_kv][j * 2 + 0] * sm_scale - m[mma_q][j] * sm_scale);
              s_frag[mma_q][mma_kv][j * 2 + 1] = math::ptx_exp2(
                  s_frag[mma_q][mma_kv][j * 2 + 1] * sm_scale - m[mma_q][j] * sm_scale);
              s_frag[mma_q][mma_kv][j * 2 + 4] = math::ptx_exp2(
                  s_frag[mma_q][mma_kv][j * 2 + 4] * sm_scale - m[mma_q][j] * sm_scale);
              s_frag[mma_q][mma_kv][j * 2 + 5] = math::ptx_exp2(
                  s_frag[mma_q][mma_kv][j * 2 + 5] * sm_scale - m[mma_q][j] * sm_scale);
            }
          }
        }
      } else if constexpr (std::is_same_v<DTypeQKAccum, half>) {
        const half2 sm_scale = __float2half2_rn(variant.sm_scale_log2);
#pragma unroll
        for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
          half m_prev[2];
#pragma unroll
          for (uint32_t j = 0; j < 2; ++j) {
            m_prev[j] = m[mma_q][j];
#pragma unroll
            for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
              half2 m_local = __hmax2(*(half2*)&s_frag[mma_q][mma_kv][j * 2],
                                      *(half2*)&s_frag[mma_q][mma_kv][j * 2 + 4]);
              m[mma_q][j] = __hmax(m[mma_q][j], __hmax(m_local.x, m_local.y));
            }
          }
          *(half2*)&m[mma_q] =
              __hmax2(*(half2*)&m[mma_q], math::shfl_xor_sync(*(half2*)&m[mma_q], 0x2));
          *(half2*)&m[mma_q] =
              __hmax2(*(half2*)&m[mma_q], math::shfl_xor_sync(*(half2*)&m[mma_q], 0x1));
#pragma unroll
          for (uint32_t j = 0; j < 2; ++j) {
            float o_scale =
                math::ptx_exp2(float(m_prev[j] * sm_scale.x - m[mma_q][j] * sm_scale.x));
            d[mma_q][j] *= o_scale;
#pragma unroll
            for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_VO; ++mma_d) {
              o_frag[mma_q][mma_d][j * 2 + 0] *= o_scale;
              o_frag[mma_q][mma_d][j * 2 + 1] *= o_scale;
              o_frag[mma_q][mma_d][j * 2 + 4] *= o_scale;
              o_frag[mma_q][mma_d][j * 2 + 5] *= o_scale;
            }
            half2 m2 = make_half2(m[mma_q][j], m[mma_q][j]);
#pragma unroll
            for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
              *(half2*)&s_frag[mma_q][mma_kv][j * 2] =
                  math::ptx_exp2(*(half2*)&s_frag[mma_q][mma_kv][j * 2] * sm_scale - m2 * sm_scale);
              *(half2*)&s_frag[mma_q][mma_kv][j * 2 + 4] = math::ptx_exp2(
                  *(half2*)&s_frag[mma_q][mma_kv][j * 2 + 4] * sm_scale - m2 * sm_scale);
            }
          }
        }
      }
    }
  }


  __device__ __forceinline__ void epilogue(float* smem_o, float2* smem_md, smem_t<SWIZZLE_MODE_Q>* o_smem) {
    using AttentionVariant = typename KTraits::AttentionVariant;

    // finalize m
    if constexpr (variant.use_softmax) {
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          if (m[mma_q][j] != DTypeQKAccum(-math::inf)) {
            m[mma_q][j] *= variant.sm_scale_log2;
          }
        }
      }
    }

    // threadblock allreduce
    // only necessary when blockDim.z > 1
    if constexpr (NUM_WARPS_KV > 1) {
      // o: [num_warps, NUM_MMA_Q, NUM_MMA_D_VO, WARP_SIZE(32), 8]
      // md: [num_warps, NUM_MMA_Q, 16, 2 (m/d)]
  #pragma unroll
      for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
  #pragma unroll
        for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_VO; ++mma_d) {
          vec_t<float, 8>::memcpy(
              smem_o + (((warp_idx * NUM_MMA_Q + mma_q) * NUM_MMA_D_VO + mma_d) *
                            WARP_SIZE +
                        lane_idx) *
                           8,
              o_frag[mma_q][mma_d]);
        }
      }
  
      if constexpr (AttentionVariant::use_softmax) {
  #pragma unroll
        for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
  #pragma unroll
          for (uint32_t j = 0; j < 2; ++j) {
            smem_md[((warp_idx * NUM_MMA_Q + mma_q) * 2 + j) * 8 + lane_idx / 4] =
                make_float2(float(m[mma_q][j]), d[mma_q][j]);
          }
        }
  
        // synchronize m,d first
        __syncthreads();
  #pragma unroll
        for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
          float o_scale[2][NUM_WARPS_KV];
  #pragma unroll
          for (uint32_t j = 0; j < 2; ++j) {
            float m_new = -math::inf, d_new = 1.f;
  #pragma unroll
            for (uint32_t i = 0; i < NUM_WARPS_KV; ++i) {
              float2 md = smem_md[(((i * NUM_WARPS_Q + get_warp_idx_q<KTraits>(warp_idx)) *
                                        NUM_MMA_Q +
                                    mma_q) *
                                       2 +
                                   j) *
                                      8 +
                                  lane_idx / 4];
              float m_prev = m_new, d_prev = d_new;
              m_new = max(m_new, md.x);
              d_new = d_prev * math::ptx_exp2(m_prev - m_new) + md.y * math::ptx_exp2(md.x - m_new);
            }
  
  #pragma unroll
            for (uint32_t i = 0; i < NUM_WARPS_KV; ++i) {
              float2 md = smem_md[(((i * NUM_WARPS_Q + get_warp_idx_q<KTraits>(warp_idx)) *
                                        NUM_MMA_Q +
                                    mma_q) *
                                       2 +
                                   j) *
                                      8 +
                                  lane_idx / 4];
              float mi = md.x;
              o_scale[j][i] = math::ptx_exp2(float(mi - m_new));
            }
            m[mma_q][j] = DTypeQKAccum(m_new);
            d[mma_q][j] = d_new;
          }
  
  #pragma unroll
          for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_VO; ++mma_d) {
            vec_t<float, 8> o_new;
            o_new.fill(0.f);
  #pragma unroll
            for (uint32_t i = 0; i < NUM_WARPS_KV; ++i) {
              vec_t<float, 8> oi;
              oi.load(smem_o + ((((i * NUM_WARPS_Q + get_warp_idx_q<KTraits>(warp_idx)) *
                                      NUM_MMA_Q +
                                  mma_q) *
                                     NUM_MMA_D_VO +
                                 mma_d) *
                                    WARP_SIZE +
                                lane_idx) *
                                   8);
  
  #pragma unroll
              for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
                o_new[reg_id] += oi[reg_id] * o_scale[(reg_id % 4) / 2][i];
              }
            }
            o_new.store(o_frag[mma_q][mma_d]);
          }
        }
      } else {
        // synchronize m,d first
        __syncthreads();
  #pragma unroll
        for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
  #pragma unroll
          for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_VO; ++mma_d) {
            vec_t<float, 8> o_new;
            o_new.fill(0.f);
  #pragma unroll
            for (uint32_t i = 0; i < NUM_WARPS_KV; ++i) {
              vec_t<float, 8> oi;
              oi.load(smem_o + ((((i * NUM_WARPS_Q + get_warp_idx_q<KTraits>(warp_idx)) *
                                      NUM_MMA_Q +
                                  mma_q) *
                                     NUM_MMA_D_VO +
                                 mma_d) *
                                    WARP_SIZE +
                                lane_idx) *
                                   8);
  #pragma unroll
              for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
                o_new[reg_id] += oi[reg_id];
              }
            }
            o_new.store(o_frag[mma_q][mma_d]);
          }
        }
      }
    }

    // normalize d
    if constexpr (AttentionVariant::use_softmax) {
      float d_rcp[NUM_MMA_Q][2];
      // compute reciprocal of d
  #pragma unroll
      for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
  #pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          d_rcp[mma_q][j] = (m[mma_q][j] != DTypeQKAccum(-math::inf))
                                ? math::ptx_rcp(d[mma_q][j])
                                : 0.f;
        }
      }
  
  #pragma unroll
      for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
  #pragma unroll
        for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_VO; ++mma_d) {
  #pragma unroll
          for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
            o_frag[mma_q][mma_d][reg_id] =
                o_frag[mma_q][mma_d][reg_id] * d_rcp[mma_q][(reg_id % 4) / 2];
          }
        }
      }
    }

    //write_o_reg_gmem

    DTypeO* o_ptr_base =
          final_o + q_indptr * o_stride_n + (kv_head_idx * gqa_group_size) * o_stride_h;

    const uint32_t warp_idx_x = get_warp_idx_q<KTraits>(warp_idx),
                   warp_idx_z = get_warp_idx_kv<KTraits>(warp_idx);
  
    if constexpr (sizeof(DTypeO) == 4) {
  #pragma unroll
      for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
  #pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          uint32_t q, r;
          gqa_group_size.divmod(qo_packed_idx_base + lane_idx / 4 + mma_q * 16 + j * 8, q, r);
          const uint32_t o_idx = q;
  #pragma unroll
          for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_VO; ++mma_d) {
            if (o_idx < qo_upperbound) {
              *reinterpret_cast<float2*>(o_ptr_base + q * o_stride_n + r * o_stride_h + mma_d * 16 +
                                         (lane_idx % 4) * 2) =
                  *reinterpret_cast<float2*>(&o_frag[mma_q][mma_d][j * 2]);
              *reinterpret_cast<float2*>(o_ptr_base + q * o_stride_n + r * o_stride_h + mma_d * 16 +
                                         8 + (lane_idx % 4) * 2) =
                  *reinterpret_cast<float2*>(&o_frag[mma_q][mma_d][4 + j * 2]);
            }
          }
        }
      }
    } else {
      if (warp_idx_z == 0) {
  #pragma unroll
        for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
  #pragma unroll
          for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_VO; ++mma_d) {
            uint32_t o_frag_f16[8 / 2];
            vec_cast<DTypeO, float>::cast<8>((DTypeO*)o_frag_f16, o_frag[mma_q][mma_d]);
  
  #ifdef FLASHINFER_STMATRIX_M8N8X4_ENABLED
            uint32_t o_smem_offset_w = o_smem->get_permuted_offset<UPCAST_STRIDE_O>(
                (warp_idx_x * NUM_MMA_Q + mma_q) * 16 + lane_idx % 16,
                mma_d * 2 + lane_idx / 16);
            o_smem->stmatrix_m8n8x4(o_smem_offset_w, o_frag_f16);
  #else
            uint32_t o_smem_offset_w = o_smem->get_permuted_offset<UPCAST_STRIDE_O>(
                (warp_idx_x * NUM_MMA_Q + mma_q) * 16 + lane_idx / 4, mma_d * 2);
            ((uint32_t*)(o_smem->base + o_smem_offset_w))[lane_idx % 4] = o_frag_f16[0];
            ((uint32_t*)(o_smem->base + o_smem_offset_w + 8 * UPCAST_STRIDE_O))[lane_idx % 4] =
                o_frag_f16[1];
            ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1)))[lane_idx % 4] = o_frag_f16[2];
            ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1) +
                         8 * UPCAST_STRIDE_O))[lane_idx % 4] = o_frag_f16[3];
  #endif
          }
        }
  
        uint32_t o_smem_offset_w = o_smem->get_permuted_offset<UPCAST_STRIDE_O>(
            warp_idx_x * NUM_MMA_Q * 16 + lane_idx / 8, lane_idx % 8);
  
  #pragma unroll
        for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
  #pragma unroll
          for (uint32_t j = 0; j < 2 * 2; ++j) {
            uint32_t q, r;
            gqa_group_size.divmod(qo_packed_idx_base + lane_idx / 8 + mma_q * 16 + j * 4, q, r);
            const uint32_t o_idx = q;
            DTypeO* o_ptr =
                o_ptr_base + q * o_stride_n + r * o_stride_h + (lane_idx % 8) * upcast_size<DTypeO>();
  #pragma unroll
            for (uint32_t mma_do = 0; mma_do < NUM_MMA_D_VO / 4; ++mma_do) {
              if (o_idx < qo_upperbound) {
                o_smem->store_128b(o_smem_offset_w, o_ptr);
              }
              o_ptr += 8 * upcast_size<DTypeO>();
              o_smem_offset_w = o_smem->template advance_offset_by_column<8>(o_smem_offset_w, mma_do);
            }
            o_smem_offset_w =
                o_smem->template advance_offset_by_row<4, UPCAST_STRIDE_O>(o_smem_offset_w) -
                2 * NUM_MMA_D_VO;
          }
        }
      }
    }

  }

  DTypeQ* q = nullptr;
  DTypeKV* k = nullptr;
  DTypeKV* v = nullptr;
  IdType* kv_indices = nullptr;
  DTypeO* partial_o = nullptr;
  float* partial_lse = nullptr;
  DTypeO* final_o = nullptr;
  float* final_lse = nullptr;
  IdType* work_indptr = nullptr;

  const uint_fastdiv& gqa_group_size;
  const uint_fastdiv& block_size;
  const uint32_t q_stride_n;
  const uint32_t q_stride_h;
  const uint32_t k_stride_page;
  const uint32_t k_stride_h;
  const uint32_t k_stride_n;
  const uint32_t v_stride_page;
  const uint32_t v_stride_h;
  const uint32_t v_stride_n;
  const uint32_t o_stride_n;
  const uint32_t o_stride_h;
  const uint32_t cluster_tile_q = gridDim.x * CTA_TILE_Q;

  AttentionVariant variant;

  int batch_idx, q_indptr, kv_indptr, partial_indptr, q_len, kv_len, packed_qo_start, kv_start,
      kv_end, kv_head_idx;

  int kv_tile_idx, mask_tile_idx;

  uint32_t qo_packed_idx_base, qo_upperbound, block_iter_base, packed_kv_bound;
};

};  // namespace flashinfer

#endif  // FLASHINFER_PERSISTENT_WORKER_CUH_