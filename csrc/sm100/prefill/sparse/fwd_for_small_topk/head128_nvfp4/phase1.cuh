#pragma once
#include "phase1.h"

#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/arch.h>

#include "params.h"
#include "utils.h"
#include "sm100/prefill/sparse/common_subroutine.h"
#include "sm100/helpers.h"

#include "config.h"

namespace sm100::fwd_for_small_topk::head128_nvfp4 {

using namespace cute;
using FwdMode = SparseAttnFwdMode;

template<FwdMode FWD_MODE, int D_QK>
__device__ void
KernelTemplate<FWD_MODE, D_QK>::sparse_attn_fwd_kernel_devfunc(const ArgT &params, const TmaParams &tma_params) {
#ifdef KERUTILS_ENABLE_SM100A
    // Grid shape: [2*s_q, 1, 1] for prefilling, [2*s_q, num_sm_parts, 1] for decoding
    // Cluster shape: [2, 1, 1]
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = threadIdx.x % 32;
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();
    const int idx_in_warpgroup = threadIdx.x % 128;
    const int cta_idx = block_id_in_cluster().x;

    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &smem = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);

    if (warp_idx == 0 && elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tma_params.tensor_map_q);
        cute::prefetch_tma_descriptor(&tma_params.tensor_map_o);
        if constexpr (IS_DECODE) {
            cute::prefetch_tma_descriptor(&tma_params.tensor_map_kv_nope);
            cute::prefetch_tma_descriptor(&tma_params.tensor_map_kv_rope);
        } else {
            cute::prefetch_tma_descriptor(&tma_params.tensor_map_kv);
        }
    } else if (warp_idx == 1 && elect_one_sync()) {
        smem.bar_sQ_full.init(1);
        smem.bar_tQ_empty.init(1);
        smem.bar_tQ_full.init(1);
        smem.bar_tOut_full.init(1);
        smem.bar_tOut_empty.init(256);
        smem.bar_P_empty.init(256);
        smem.bar_QK_done.init(1);
        smem.bar_SV_done.init(1);
        smem.bar_S_O_full.init(256);
        smem.bar_li_full.init(H_Q/2);
        smem.bar_li_empty.init(128);
        if constexpr (FWD_MODE != FwdMode::DecodeWithSplitKV) {
            smem.bar_clc_full.init(1);
            smem.bar_clc_empty.init(NUM_WORKER_THREADS);
        }
        fence_barrier_init();
    } else if (warp_idx == 2) {
        cute::TMEM::Allocator2Sm().allocate(512, smem.tmem_start_addr.data());
        KU_TRAP_ONLY_DEVICE_ASSERT(smem.tmem_start_addr.data()[0] == 0);
        cute::TMEM::Allocator2Sm().release_allocation_lock();
    } else if (warp_idx == 3 && elect_one_sync()) {
        CUTE_UNROLL
        for (int i = 0; i < NUM_K_BUFS; ++i) {
            smem.bar_KV_full[i].init(IS_PREFILL ? 1 : (128/32)*2+1);
            smem.bar_KV_empty[i].init(1);
        }
        CUTE_UNROLL
        for (int i = 0; i < NUM_INDEX_BUFS; ++i) {
            smem.bar_valid_coord_scales_full[i].init(IS_PREFILL ? B_TOPK/8 : 32);
            smem.bar_valid_coord_scales_empty[i].init(IS_PREFILL ? 128 : (128 + (cta_idx==1) + 2 + 128));
        }
        if constexpr (IS_DECODE) {
            CUTE_UNROLL
            for (int i = 0; i < NUM_RAW_K_BUFS; ++i) {
                smem.bar_raw_KV_full[i].init(1);
                smem.bar_raw_KV_empty[i].init(128);
            }
        }
        fence_barrier_init();
    }

    ku::barrier_cluster_arrive_relaxed();
    ku::barrier_cluster_wait_acquire();

    struct OuterloopArgs {
        bool outer_loop_phase;
        int batch_idx, s_q_idx;
        int start_block_idx, end_block_idx;
        int topk_length;

        int extra_topk_length, num_orig_kv_blocks;  // extra-KV related
        bool is_no_split; int n_split_idx;  // splitkv related
    };

    auto run_outer_loop = [&](auto loop_body) -> bool {
        int outer_loop_phase = false;
        if constexpr (FWD_MODE == FwdMode::DecodeWithSplitKV) {
            int s_q_idx = blockIdx.x / 2;
            DecodingSchedMeta sched_meta;
            KU_LDG_256(
                params.tile_scheduler_metadata_ptr + blockIdx.y,
                &sched_meta,
                ".nc",
                "no_allocate",
                "evict_normal",
                "256B"
            );
            if (sched_meta.begin_req_idx >= params.b) {
                return 0;
            }

            #pragma unroll 1
            for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
                int topk_length = params.topk_length ? __ldg(params.topk_length + batch_idx) : params.topk;
                int orig_topk_padded = max(ku::ceil(topk_length, (int)B_TOPK), (int)B_TOPK);
                int extra_topk_length = params.extra_topk_length ? __ldg(params.extra_topk_length + batch_idx) : params.extra_topk;
                int total_topk_padded = orig_topk_padded + ku::ceil(extra_topk_length, (int)B_TOPK);    // % B_TOPK == 0
                int start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
                int end_block_idx = batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx : total_topk_padded / B_TOPK;
                bool is_split = batch_idx == sched_meta.begin_req_idx ? sched_meta.is_first_req_splitted : (batch_idx == sched_meta.end_req_idx ? sched_meta.is_last_req_splitted : false);
                int n_split_idx = batch_idx == sched_meta.begin_req_idx ? (__ldg(params.num_splits_ptr+batch_idx) + sched_meta.begin_split_idx) : __ldg(params.num_splits_ptr+batch_idx);

                    // start_block_idx = 0;
                    // end_block_idx = total_topk_padded / B_TOPK;
                    // is_split = false;
                    // n_split_idx = 0;

                OuterloopArgs args = {
                    (bool)outer_loop_phase,
                    batch_idx, s_q_idx,
                    start_block_idx, end_block_idx,
                    topk_length,

                    extra_topk_length, orig_topk_padded / B_TOPK,
                    !is_split, n_split_idx
                };

                loop_body(args);
                outer_loop_phase ^= 1;
            }
        } else {
            // Prefill mode. Use CLC to allocate different s_q (for decoding, different batches + s_q) to different workers
            ku::CLCResult next_job = {true, (int)blockIdx.x, IS_PREFILL ? 0 : (int)blockIdx.y, 0};
            CUTE_NO_UNROLL
            while (next_job.is_valid) {
                int s_q_idx = next_job.x / 2;
                int batch_idx = IS_PREFILL ? 0 : next_job.y;
                int topk_length = params.topk_length != nullptr ? __ldg(params.topk_length + (IS_PREFILL?s_q_idx:batch_idx)) : params.topk;
                
                if constexpr (IS_PREFILL) {
                    int num_k_blocks = max(cute::ceil_div(topk_length, (int)B_TOPK), 1);  // num_k_blocks always >= 1
                    OuterloopArgs args = {
                        (bool)outer_loop_phase,
                        0, s_q_idx,
                        0, num_k_blocks,
                        topk_length
                    };
                    loop_body(args);
                } else {
                    int orig_topk_padded = max(ku::ceil(topk_length, (int)B_TOPK), (int)B_TOPK);
                    int extra_topk_length = params.extra_topk_length ? __ldg(params.extra_topk_length + batch_idx) : params.extra_topk;
                    int total_topk_padded = orig_topk_padded + ku::ceil(extra_topk_length, (int)B_TOPK);    // % B_TOPK == 0

                    OuterloopArgs args = {
                        (bool)outer_loop_phase,
                        batch_idx, s_q_idx,
                        0, total_topk_padded / B_TOPK,
                        topk_length,

                        extra_topk_length, orig_topk_padded / B_TOPK,
                        false, 0
                    };
                    loop_body(args);
                }

                smem.bar_clc_full.wait(outer_loop_phase);
                next_job = ku::get_clc_query_response<true>(smem.clc_response_obj);
                smem.bar_clc_empty.arrive(0u);

                outer_loop_phase ^= 1;
            }
        }
        return outer_loop_phase;
    };

    if (warpgroup_idx == 0) {
        // Q fetching and O writing back warpgroup
        cutlass::arch::warpgroup_reg_alloc<176>();

        bf16* sO_addrs[B_EPI/8];
        CUTE_UNROLL
        for (int i = 0; i < B_EPI/8; ++i) {
            Tensor sO = make_tensor(make_smem_ptr(smem.Q.data()), ku::make_umma_canonical_k_major_layout<H_Q/2, D_V, 128>());
            sO_addrs[i] = &sO(idx_in_warpgroup%64, (idx_in_warpgroup/64)*(D_V/2) + i*8);
        }

        float* sO_accum_addrs[B_EPI_SPLITKV/4];
        if constexpr (FWD_MODE == FwdMode::DecodeWithSplitKV) {
            // If split-KV is enabled, we need to store back O in float32
            // We view Q buffer (with shape 64 x 512, bf16) as 4 buffers with shape (H_Q/2) x (B_EPI_SPLITKV*2), float32
            Tensor sO_accum = make_tensor(make_smem_ptr((float*)smem.Q.data()), ku::make_umma_canonical_k_major_layout<H_Q/2, D_V, 128, float>());
            CUTE_UNROLL
            for (int i = 0; i < B_EPI_SPLITKV/4; ++i) {
                sO_accum_addrs[i] = &sO_accum(idx_in_warpgroup%64, i*4) + (idx_in_warpgroup >= 64 ? (H_Q/2)*B_EPI_SPLITKV : 0);
            }
        }

        auto perform_o_copy_out = [&](const OuterloopArgs &args, bool is_last_o) {
            // outer_loop_phase is the loop phase corresponding to s_q_idx

            // Get li (output_scale actually)
            smem.bar_li_full.wait(args.outer_loop_phase);
            float output_scale = smem.rowwise_li_buf[idx_in_warpgroup%64];
            float2 output_scale_float2 = float2 {output_scale, output_scale};
            smem.bar_li_empty.arrive();

            // Retrieve and store O, and calculate delta := sum(O*dO, dim=-1) if FWD_MODE is Recompute
            smem.bar_tOut_full.wait(args.outer_loop_phase);
            if (is_last_o && elect_one_sync()) {
                cudaTriggerProgrammaticLaunchCompletion();
            }

            if (FWD_MODE != FwdMode::DecodeWithSplitKV || args.is_no_split) {
                CUTE_UNROLL
                for (int k = 0; k < (D_V/2)/B_EPI; ++k) {
                    float2 o[B_EPI/2];
                    ku::tmem_ld_32dp32bNx<B_EPI>(tmem_cols::O + k*B_EPI, o);
                    cutlass::arch::fence_view_async_tmem_load();
                    if (k == (D_V/2)/B_EPI-1) {
                        smem.bar_tOut_empty.arrive(0u);
                    }
                    CUTE_UNROLL
                    for (int i = 0; i < B_EPI/8; ++i) {
                        nv_bfloat162 o_bf16[4];
                        CUTE_UNROLL
                        for (int j = 0; j < 4; ++j) {
                            o[i*4+j] = ku::float2_mul(o[i*4+j], output_scale_float2);
                            o_bf16[j] = __float22bfloat162_rn(o[i*4+j]);
                        }
                        bf16* o_do_addr = sO_addrs[i] + k*B_EPI*(H_Q/2);
                            if (k == 0 && i == 0) {
                                smem.bar_tQ_full.wait(args.outer_loop_phase^1^is_last_o);    // Wait for sQ's availability
                            }
                        ku::st_shared(o_do_addr, *(__int128_t*)o_bf16);
                    }
                }

                fence_view_async_shared();
                NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);
                if (warp_idx == 0 && elect_one_sync()) {
                    SM90_TMA_STORE_5D::copy(
                        &tma_params.tensor_map_o, 
                        smem.Q.data(),
                        0, cta_idx*(H_Q/2), 0, args.s_q_idx, IS_DECODE ? args.batch_idx : 0
                    );
                    cute::tma_store_arrive();
                }
            } else {
                CUTE_UNROLL
                for (int k = 0; k < (D_V/2)/B_EPI_SPLITKV; ++k) {
                    int cur_buf_idx = k % NUM_EPI_SPLITKV_BUFS;
                    if (k == 0) {
                        cute::tma_store_wait<0>();
                    } else {
                        cute::tma_store_wait<NUM_EPI_SPLITKV_BUFS-1>();
                    }
                    NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);

                    float o[B_EPI_SPLITKV];
                    ku::tmem_ld_32dp32bNx<B_EPI_SPLITKV>(tmem_cols::O + k*B_EPI_SPLITKV, o);
                    cutlass::arch::fence_view_async_tmem_load();
                    if (k == (D_V/2)/B_EPI_SPLITKV-1) {
                        smem.bar_tOut_empty.arrive(0u);
                    }
                    CUTE_UNROLL
                    for (int i = 0; i < B_EPI_SPLITKV/4; ++i) {
                        CUTE_UNROLL
                        for (int j = 0; j < 4; j += 2) {
                            *(float2*)(o + i*4 + j) = ku::float2_mul(float2 {o[i*4+j], o[i*4+j+1]}, output_scale_float2);
                        }
                        if (k == 0 && i == 0) {
                            smem.bar_tQ_full.wait(args.outer_loop_phase^1^is_last_o);    // Wait for sQ's availability
                        }
                        ku::st_shared(
                            sO_accum_addrs[i] + cur_buf_idx*((H_Q/2)*B_EPI_SPLITKV*2),
                            *(__int128_t*)(o + i*4)
                        );
                    }

                    fence_view_async_shared();
                    NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);
                    if constexpr (IS_DECODE) {  // Otherwise nvcc complains about `tma_params` doesn't have `tensor_map_o_accum`
                        float* cur_buf_base = (float*)smem.Q.data() + cur_buf_idx*((H_Q/2)*B_EPI_SPLITKV*2);
                        if (warp_idx == 0 && elect_one_sync()) {
                            SM90_TMA_STORE_5D::copy(
                                &tma_params.tensor_map_o_accum, 
                                cur_buf_base,
                                0, cta_idx*(H_Q/2), k*(B_EPI_SPLITKV/32), args.s_q_idx, args.n_split_idx
                            );
                            cute::tma_store_arrive();
                        } else if (warp_idx == 1 && elect_one_sync()) {
                            SM90_TMA_STORE_5D::copy(
                                &tma_params.tensor_map_o_accum, 
                                cur_buf_base + (H_Q/2)*B_EPI_SPLITKV,
                                0, cta_idx*(H_Q/2), k*(B_EPI_SPLITKV/32) + (D_V/2)/32, args.s_q_idx, args.n_split_idx
                            );
                            cute::tma_store_arrive();
                        }
                    }
                }
            }
        };

        OuterloopArgs last_args;
        last_args.batch_idx = -1;

        bool final_outer_loop_phase = \
        run_outer_loop([&](const OuterloopArgs &args) {
            // Copy Q for this round
            if constexpr (FWD_MODE == FwdMode::DecodeWithSplitKV) {
                cute::tma_store_wait<0>();
                NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);  // Since we use two warps to issue TMA during FwdMode::DecodeWithSplitKV
            }
            if (warp_idx == 0 && elect_one_sync()) {
                // Wait for sQ to become empty, and issue G -> S copy for Q
                if constexpr (FWD_MODE != FwdMode::DecodeWithSplitKV) {
                    cute::tma_store_wait<0>();  // This thread must be the same one as o copy out thread (since `elect_one_sync()` always returns the same thread for the same `mask`, according to PTX document)
                }
                int stride_q_b_div_stride_q_s_q = 0;
                if constexpr (IS_DECODE) {
                    stride_q_b_div_stride_q_s_q = params.stride_q_b / params.stride_q_s_q;
                }
                SM100_TMA_2SM_LOAD_5D_NOSPLIT::copy(
                    &tma_params.tensor_map_q,
                    (uint64_t*)&smem.bar_sQ_full,
                    (uint64_t)TMA::CacheHintSm90::EVICT_FIRST,
                    smem.Q.data(),
                    0, cta_idx*(H_Q/2), 0, 0, (IS_DECODE ? args.batch_idx*stride_q_b_div_stride_q_s_q : 0) + args.s_q_idx
                );

                // Wait for sQ to be ready, and issue S -> T copy for Q
                if (cta_idx == 0) {
                    smem.bar_sQ_full.arrive_and_expect_tx(H_Q*D_Q*sizeof(bf16));
                    smem.bar_sQ_full.wait(args.outer_loop_phase);

                    smem.bar_tQ_empty.wait(args.outer_loop_phase^1);
                    ku::tcgen05_after_thread_sync();
                    UMMA::SmemDescriptor sQ_desc = UMMA::make_umma_desc<UMMA::Major::K>(
                        make_tensor(
                            make_smem_ptr(smem.Q.data()),
                            ku::make_umma_canonical_k_major_layout<(H_Q/2)*2, 64, 128>()
                        )
                    );
                    CUTE_UNROLL
                    for (int tile_idx = 0; tile_idx < D_Q/64/2; ++tile_idx) {
                        // A tile is 128 rows * 64 cols in UTCCP's view, or 64 rows * 128 cols in our view
                        CUTE_UNROLL
                        for (int subtile_idx = 0; subtile_idx < 4; ++subtile_idx) {
                            // A subtile is 128 rows * 16 cols (256b, 32B) (in UTCCP's view), or 64 rows * 16 cols * 2 (in our view)
                            // NOTE Using `sQ_desc+((tile_idx*((H_Q/2)*128*2) + subtile_idx*32) >> 4)` leads to IMA, doesn't know why
                            UMMA::SmemDescriptor cur_sQ_desc = sQ_desc;
                            cur_sQ_desc.lo += ((tile_idx*((H_Q/2)*128*2) + subtile_idx*32) >> 4);
                            // uint64_t cur_sQ_desc = sQ_desc;
                            // cur_sQ_desc += ((tile_idx*((H_Q/2)*128*2) + subtile_idx*32) >> 4);
                            SM100_UTCCP_128dp256bit_2cta::copy(
                                cur_sQ_desc,
                                tmem_cols::Q + tile_idx*32 + subtile_idx*8
                            );
                        }
                    }
                    ku::umma_arrive_multicast_2x1SM_noelect(smem.bar_tQ_full, 1|2);
                }
            }

            if (last_args.batch_idx != -1) {
                perform_o_copy_out(last_args, false);
            } else {
                smem.bar_tQ_full.wait(args.outer_loop_phase);   // To prevent double arrive
            }
            last_args = args;
        });
        if (last_args.batch_idx != -1) {
            cute::tma_store_wait<0>();
            NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);
            perform_o_copy_out(last_args, true);
        }

        if (warp_idx == 0) {
            cute::TMEM::Allocator2Sm().free(0, 512);
        }
    } else if (warpgroup_idx == 1) {
        // KV fetching threads for prefill, dequant threads for decoding
        cutlass::arch::warpgroup_reg_dealloc<80>();
        RingBufferState rs;

        if constexpr (!IS_DECODE) {
            const int warp_idx = cutlass::canonical_warp_idx();    // Using `warp_idx` without `__shfl_sync` is faster
            if (elect_one_sync()) {
                // KV fetching threads
                run_outer_loop([&](const OuterloopArgs &args) {
                    int* gIndices = params.indices + args.s_q_idx*params.stride_indices_s_q;
                    int64_t cache_hint = ku::create_simple_cache_policy<ku::CacheHint::EVICT_LAST>();

                    static constexpr int NUM_ROWS_PER_THREAD = B_TOPK / 4;

                    CUTE_NO_UNROLL
                    for (int k = args.start_block_idx; k < args.end_block_idx; ++k) {
                        auto [k_buf_idx, k_bar_phase] = rs.get<NUM_K_BUFS>();

                        int cur_indices[NUM_ROWS_PER_THREAD];
                        CUTE_UNROLL
                        for (int local_row = 0; local_row < NUM_ROWS_PER_THREAD/8; local_row += 1) {
                            int row = local_row*(4*8) + (warp_idx-4)*8;
                            KU_LDG_256(
                                gIndices + k*B_TOPK + row, 
                                cur_indices + local_row*8, 
                                ".nc", 
                                "no_allocate", 
                                "evict_first", 
                                "256B"
                            );
                        }
                        smem.bar_KV_empty[k_buf_idx].wait(k_bar_phase^1);

                        CUTE_UNROLL
                        for (int local_row = 0; local_row < NUM_ROWS_PER_THREAD/4; local_row += 1) {
                            int row = (warp_idx-4)*8 + (local_row/2)*(4*8) + (local_row%2)*4;
                            int4 indices = *(int4*)(cur_indices+local_row*4);
                            static_assert(D_K == 512);
                            CUTE_UNROLL
                            for (int local_col = 0; local_col < (D_K/64)/2; ++local_col) {
                                ku::tma_gather4_cta_group_2<true>(
                                    &tma_params.tensor_map_kv,
                                    smem.bar_KV_full[k_buf_idx],
                                    smem.K[k_buf_idx].data() + row*64 + local_col*64*B_TOPK,
                                    local_col*64 + cta_idx*(D_K/2),
                                    indices,
                                    cache_hint
                                );
                            }
                        }
                        rs.update();
                    }
                });
            }
                
        } else {
            // 8 threads per token
            struct IsCTA0 {};
            struct IsCTA1 {};

            auto launch_dequant_wg = [&](auto cta_id_t) {
                static constexpr bool IS_CTA1 = std::is_same<decltype(cta_id_t), IsCTA1>::value;
                constexpr int GROUP_SIZE = 8, NUM_GROUPS = 128/8, ROWS_PER_GROUP = B_TOPK / NUM_GROUPS, COLS_PER_GROUP = (IS_CTA1 ? 256-64 : 256) / (GROUP_SIZE*8);
                int group_idx = idx_in_warpgroup/GROUP_SIZE, idx_in_group = idx_in_warpgroup%GROUP_SIZE;
                Tensor nope0 = make_tensor(make_smem_ptr(smem.K[0].data()), ku::make_umma_canonical_k_major_layout<B_TOPK, D_K/2, 128>());
                bf16* nope0_base = &nope0(group_idx, idx_in_group*8);
                // NVFP4 v1: K_raw is uint8 packed FP4 (half the bytes of FP8).
                uint8_t* raw_nope0_base = smem.K_raw[0].data() + group_idx*(D_K/4) + idx_in_group*4;
                run_outer_loop([&](const OuterloopArgs &args) {
                    CUTE_NO_UNROLL
                    for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
                        auto [k_buf_idx, k_bar_phase] = rs.get<NUM_K_BUFS>();
                        auto [raw_k_buf_idx, raw_k_bar_phase] = rs.get<NUM_RAW_K_BUFS>();
                        auto [index_buf_idx, index_bar_phase] = rs.get<NUM_INDEX_BUFS>();
                        uint8_t* raw_nope_base = raw_nope0_base + raw_k_buf_idx * (B_TOPK*(D_K/4));
                        // FP4 packed: 4 bytes = 8 nibbles = 8 K elements per iter (matches BF16 output count).
                        // Renamed but keeping the FP8 baseline's lambda name for diff-readability.
                        auto get_raw_fp8 = [&](int local_row_idx, int local_col_idx) -> uint32_t {
                            return *(uint32_t*)(raw_nope_base + local_row_idx*NUM_GROUPS*(D_K/4) + local_col_idx*(GROUP_SIZE*4));
                        };
                        bf16* nope_base = nope0_base + k_buf_idx * (B_TOPK*(D_K/2));
                        uint32_t cur_nope_base_uint_addr = cute::cast_smem_ptr_to_uint(nope_base);
                        auto st_128b = [&](int local_row_idx, int local_col_idx, __int128_t &data) {
                            asm volatile ("st.weak.shared::cta.b128 [%0], %1;\n" 
                                : 
                                : "r"(cur_nope_base_uint_addr + 2*(local_row_idx*NUM_GROUPS*64 + local_col_idx*B_TOPK*64)), "q"(data)   // 2 for sizeof(bf16)
                            );  // We have this `asm volatile` here, otherwise the compiler generates ST.E instead of STS
                        };

                        smem.bar_valid_coord_scales_full[index_buf_idx].wait(index_bar_phase);
                        smem.bar_raw_KV_full[raw_k_buf_idx].wait(raw_k_bar_phase);

                        CUTE_UNROLL
                        for (int local_row_idx = 0; local_row_idx < ROWS_PER_GROUP; ++local_row_idx) {
                            int row_idx = local_row_idx*NUM_GROUPS + group_idx;
                            // ─────────────────────────────────────────────
                            // NVFP4 dequant: read 4-byte (8 e2m1 nibble) packed
                            // K + E4M3 per-16-element-block scale, decode via
                            // SM_100 cvt.rn.bf16x2.e2m1x2 PTX intrinsic, multiply
                            // by E4M3 scale, store as BF16 into smem.K[k_buf_idx]
                            // for the existing BF16 UMMA pipeline downstream.
                            //
                            // Trade-off vs the v2 native-UMMA path: we still
                            // dequant in SMEM (vs feeding packed FP4 + E4M3
                            // scales directly to a block-scaled tensor-core
                            // op). v1 is correctness-equivalent to the FP8
                            // baseline at half the HBM K-read bandwidth and
                            // matches the existing UMMA atom; v2 follow-up
                            // additionally eliminates the SMEM dequant pass.
                            // ─────────────────────────────────────────────
                            // NVFP4 KV: read 32 E4M3 scales per row (one per 16-elem block of D_K=512).
                            __nv_bfloat16 scales_bf16[NUM_SCALES_EACH_TOKEN];
                            CUTE_UNROLL
                            for (int s = 0; s < NUM_SCALES_EACH_TOKEN; ++s) {
                                float scale_f = float(smem.scales[index_buf_idx][row_idx][s]);
                                scales_bf16[s] = __float2bfloat16_rn(scale_f);
                            }

                            // Each inner iter: 4 bytes -> 8 e2m1 nibbles -> 8 bf16 (one cache line).
                            uint32_t cur_data_fp4 = get_raw_fp8(local_row_idx, 0);
                            CUTE_UNROLL
                            for (int local_col_idx = 0; local_col_idx < COLS_PER_GROUP; ++local_col_idx) {
                                __nv_bfloat16 data_bf16[8];
                                uint32_t next_data_fp4 = (local_col_idx + 1 < COLS_PER_GROUP)
                                    ? get_raw_fp8(local_row_idx, local_col_idx + 1)
                                    : 0u;

                                // PTX dequant via f16x2 path (CTK 12.9 doesn't have bf16x2.e2m1x2):
                                // 4 bytes of packed FP4 -> 4 f16x2 -> 8 f16 -> 8 bf16 (precision cast).
                                uint32_t f16x2_packed[4];
                                asm volatile(
                                    "{\n"
                                    " .reg .b8 byte0, byte1, byte2, byte3;\n"
                                    " mov.b32 {byte0, byte1, byte2, byte3}, %4;\n"
                                    " cvt.rn.f16x2.e2m1x2 %0, byte0;\n"
                                    " cvt.rn.f16x2.e2m1x2 %1, byte1;\n"
                                    " cvt.rn.f16x2.e2m1x2 %2, byte2;\n"
                                    " cvt.rn.f16x2.e2m1x2 %3, byte3;\n"
                                    "}"
                                    : "=r"(f16x2_packed[0]),
                                      "=r"(f16x2_packed[1]),
                                      "=r"(f16x2_packed[2]),
                                      "=r"(f16x2_packed[3])
                                    : "r"(cur_data_fp4));

                                // Apply E4M3 scale. Each iter spans 8 elements; with block_size=16,
                                // two consecutive col iters share one scale. The global K position
                                // for this thread + this iter is local_col_idx*GROUP_SIZE*8
                                // + idx_in_group*8; scale_idx is that / 16.
                                int global_k_pos = local_col_idx * (GROUP_SIZE * 8) + idx_in_group * 8;
                                int scale_idx = global_k_pos / 16;
                                __nv_bfloat16 scale = scales_bf16[scale_idx];
                                __nv_bfloat162 scale_x2 = {scale, scale};

                                CUTE_UNROLL
                                for (int b = 0; b < 4; ++b) {
                                    __half2 h2 = *reinterpret_cast<__half2*>(&f16x2_packed[b]);
                                    // f16 -> bf16 (lossy for the 3 dropped mantissa bits, but FP4
                                    // values are in [-6, 6] so bf16's 7-bit mantissa is plenty).
                                    __nv_bfloat162 bf16x2 = {
                                        __float2bfloat16_rn(__half2float(h2.x)),
                                        __float2bfloat16_rn(__half2float(h2.y))
                                    };
                                    __nv_bfloat162 result = __hmul2(bf16x2, scale_x2);
                                    data_bf16[b * 2 + 0] = result.x;
                                    data_bf16[b * 2 + 1] = result.y;
                                }

                                if (local_row_idx == 0 && local_col_idx == 0) {
                                    smem.bar_KV_empty[k_buf_idx].wait(k_bar_phase ^ 1);
                                }
                                st_128b(local_row_idx, local_col_idx, *(__int128_t*)data_bf16);
                                cur_data_fp4 = next_data_fp4;
                            }
                        }

                        fence_view_async_shared();  // NOTE Should we use shared::cluster here?
                        __syncwarp();
                        smem.bar_valid_coord_scales_empty[index_buf_idx].arrive();
                        smem.bar_raw_KV_empty[raw_k_buf_idx].arrive();
                        if (elect_one_sync()) {
                            smem.bar_KV_full[k_buf_idx].arrive(0u);
                        }
                        rs.update();
                    }
                });
            };
            if (cta_idx == 0) {
                launch_dequant_wg(IsCTA0{});
            } else {
                launch_dequant_wg(IsCTA1{});
            }
        }
    } else if (warpgroup_idx == 2) {
        cutlass::arch::warpgroup_reg_dealloc<80>();

        RingBufferState rs;
        if (warp_idx == 8 && cta_idx == 0 && elect_one_sync()) {
            // UMMA thread
            TiledMMA tiled_mma_P = TiledMMA_P{};
            TiledMMA tiled_mma_O = TiledMMA_O{};
            Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<H_Q/2>, Int<B_TOPK*2>>{});
            Tensor tO = partition_fragment_C(tiled_mma_O, Shape<Int<H_Q/2>, Int<D_V>>{});
            Tensor tQ = tiled_mma_P.get_slice(_0{}).make_fragment_A(
                partition_shape_A(tiled_mma_P, Shape<Int<H_Q/2>, Int<D_Q/2>>{})
            );
            tP.data().get() = tmem_cols::P;
            tO.data().get() = tmem_cols::O;
            tQ.data().get() = tmem_cols::Q;
            
            run_outer_loop([&](const OuterloopArgs &args) {
                smem.bar_tQ_full.wait(args.outer_loop_phase);

                // Issue P = Q K^T
                auto issue_P = [&](int k, int rs_offset) {
                    auto [k_buf_idx, k_bar_phase] = rs.offset_by(rs_offset).get<NUM_K_BUFS>();
                    auto [_, bar_phase] = rs.offset_by(rs_offset).get<1>();
                    smem.bar_P_empty.wait(bar_phase^1);
                    if constexpr (IS_PREFILL) {
                        smem.bar_KV_full[k_buf_idx].arrive_and_expect_tx(B_TOPK*D_K*sizeof(bf16));
                    } else {
                        // RoPE only
                        smem.bar_KV_full[k_buf_idx].arrive_and_expect_tx(B_TOPK*D_ROPE*sizeof(bf16));
                    }
                    smem.bar_KV_full[k_buf_idx].wait(k_bar_phase);
                    ku::tcgen05_after_thread_sync();
                    Tensor sK = make_tensor(
                        make_smem_ptr(smem.K[k_buf_idx].data()),
                        ku::make_umma_canonical_k_major_layout<B_TOPK, D_K/2, 128>()
                    );
                    ku::utcmma_ts(tiled_mma_P, tQ, sK, tP, true);
                    ku::umma_arrive_multicast_2x1SM_noelect(smem.bar_QK_done, 1|2);
                };

                // Issue O += S V
                auto issue_O = [&](int k, int rs_offset) {
                    auto [k_buf_idx, k_bar_phase] = rs.offset_by(rs_offset).get<NUM_K_BUFS>();
                    auto [_, bar_phase] = rs.offset_by(rs_offset).get<1>();
                    smem.bar_S_O_full.wait(bar_phase);
                    if (k == args.start_block_idx) {
                        smem.bar_tOut_empty.wait(args.outer_loop_phase^1);
                    }
                    ku::tcgen05_after_thread_sync();
                    Tensor sS = make_tensor(
                        make_smem_ptr(smem.S.data()),
                        ku::make_umma_canonical_k_major_layout<H_Q/2, B_TOPK, 0>()
                    );
                    Tensor sV = make_tensor(
                        make_smem_ptr(smem.K[k_buf_idx].data()),
                        ku::make_umma_canonical_mn_major_layout<D_V/2, B_TOPK, 128>()
                    );
                    ku::utcmma_ss(tiled_mma_O, sS, sV, tO, k == args.start_block_idx);
                    ku::umma_arrive_multicast_2x1SM_noelect(smem.bar_SV_done, 1|2);
                    ku::umma_arrive_multicast_2x1SM_noelect(smem.bar_KV_empty[k_buf_idx], 1|2);
                };

                CUTE_NO_UNROLL
                for (int k = args.start_block_idx; k < args.end_block_idx+1; ++k) {
                    if (k < args.end_block_idx) {
                        issue_P(k, 0);
                    }
                    if (k == args.end_block_idx-1) {
                        ku::umma_arrive_2x1SM_noelect(smem.bar_tQ_empty);
                    }

                    if (k > args.start_block_idx) {
                        issue_O(k-1, -1);
                    }
                    
                    if (k != args.end_block_idx) {
                        rs.update();
                    }
                }
                ku::tcgen05_before_thread_sync();
                ku::umma_arrive_multicast_2x1SM_noelect(smem.bar_tOut_full, 1|2);
            });
        } else if (warp_idx == 8 && cta_idx == 1 && elect_one_sync()) {
            if constexpr (IS_DECODE) {
                // KV RoPE fetching warp
                run_outer_loop([&](const OuterloopArgs &args) {
                    CUTE_NO_UNROLL
                    for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
                        auto [index_buf_idx, index_bar_phase] = rs.get<NUM_INDEX_BUFS>();
                        auto [k_buf_idx, k_bar_phase] = rs.get<NUM_K_BUFS>();
                        smem.bar_valid_coord_scales_full[index_buf_idx].wait(index_bar_phase);
                        smem.bar_KV_empty[k_buf_idx].wait(k_bar_phase^1);
                        CUTE_UNROLL
                        for (int row = 0; row < B_TOPK; row += 4) {
                            int4 cur_indices = *(int4*)(smem.tma_coord[index_buf_idx] + row);
                            ku::tma_gather4_cta_group_2<true>(
                                block_idx >= args.num_orig_kv_blocks ? &tma_params.tensor_map_extra_kv_rope : &tma_params.tensor_map_kv_rope,
                                smem.bar_KV_full[k_buf_idx],
                                smem.K[k_buf_idx].data() + (D_NOPE-D_K/2)*B_TOPK + row*D_ROPE,
                                0,
                                cur_indices,
                                (int64_t)TMA::CacheHintSm90::EVICT_LAST
                            );
                        }
                        smem.bar_valid_coord_scales_empty[index_buf_idx].arrive();
                        rs.update();
                    }
                });
            }
        } else if (warp_idx == 9) {
            // KV validness loading warp (for prefill), Indices transformation warp (for decode, Responsible for generating: TMA coordinates, scale factors, and valid masks)
            if constexpr (IS_PREFILL) {
                if (lane_idx < B_TOPK/8) {
                    run_outer_loop([&](const OuterloopArgs &args) {
                        int* gIndices = params.indices + args.s_q_idx*params.stride_indices_s_q;
                        CUTE_NO_UNROLL
                        for (int k = args.start_block_idx; k < args.end_block_idx; ++k) {
                            char k_validness_mask = load_indices_and_generate_mask(
                                lane_idx,
                                gIndices + k*B_TOPK,
                                params.s_kv,
                                k*B_TOPK,
                                args.topk_length
                            );
                            
                            auto [indices_buf_idx, indices_bar_phase] = rs.get<NUM_INDEX_BUFS>();
                            smem.bar_valid_coord_scales_empty[indices_buf_idx].wait(indices_bar_phase^1);
                            smem.is_k_valid[indices_buf_idx][lane_idx] = k_validness_mask;
                            smem.bar_valid_coord_scales_full[indices_buf_idx].arrive();
                            
                            rs.update();
                        }
                    });
                }
            } else {
                static_assert(B_TOPK == 64);
                // Each thread is responsible for 2 tokens
                static constexpr int tma_coords_step_per_token = 352/TMA_K_STRIDE_FOR_DECODING;  // NVFP4 layout
                int tma_coords_step_per_block = params.stride_kv_block / TMA_K_STRIDE_FOR_DECODING; // must < 2G since k_batch_stride < 1T and TMA_K_STRIDE_FOR_DECODING > 512
                int tma_coords_step_per_extra_block = params.stride_extra_kv_block / TMA_K_STRIDE_FOR_DECODING;
                // NVFP4: scales live in a separate buffer (different alignment requirement
                // than nope+rope packed inline). params.kv_scales is set by the host-side
                // sparse_attn_decode_nvfp4_interface (see csrc/api/sparse_decode_nvfp4.cu).
                uint8_t* k_scales_ptr = params.kv_scales;
                uint8_t* extra_k_scales_ptr = nullptr;  // NVFP4 v1: no extra KV pool
                
                run_outer_loop([&](const OuterloopArgs &args) {
                    int* indices = (int*)params.indices + params.stride_indices_b*args.batch_idx + params.stride_indices_s_q*args.s_q_idx;
                    int* extra_indices = (int*)params.extra_indices + params.stride_extra_indices_b*args.batch_idx + params.stride_extra_indices_s_q*args.s_q_idx;
                    
                    struct IsOrigBlock {};
                    struct IsExtraBlock {};
                    auto process_one_block = [&](int block_idx, auto is_extra_block_t) {
                        auto [index_buf_idx, index_bar_phase] = rs.get<NUM_INDEX_BUFS>();
                        static constexpr bool IS_EXTRA_BLOCK = std::is_same_v<decltype(is_extra_block_t), IsExtraBlock>;
                        int cur_block_size = IS_EXTRA_BLOCK ? params.extra_page_block_size : params.page_block_size;
                        // NVFP4: scales block stride is independent of nope+rope block stride.
                        // (FP8 path: both come from the same packed buffer with one stride.)
                        int64_t cur_k_block_stride = IS_EXTRA_BLOCK ? params.stride_extra_kv_block : params.stride_kv_scales_block;
                        [[maybe_unused]] int cur_k_row_stride = IS_EXTRA_BLOCK ? params.stride_extra_kv_row : params.stride_kv_scales_row;
                        uint8_t* cur_k_scales_ptr = IS_EXTRA_BLOCK ? extra_k_scales_ptr : k_scales_ptr;
                        int cur_tma_coords_step_per_block = IS_EXTRA_BLOCK ? tma_coords_step_per_extra_block : tma_coords_step_per_block;

                        int abs_pos, my_indices[2];
                        if (!IS_EXTRA_BLOCK) {
                            abs_pos = block_idx*B_TOPK + lane_idx*2;
                            *(int2*)my_indices = __ldg((int2*)(indices + abs_pos));
                        } else {
                            abs_pos = (block_idx-args.num_orig_kv_blocks)*B_TOPK + lane_idx*2;
                            *(int2*)my_indices = __ldg((int2*)(extra_indices + abs_pos));
                        }
                        smem.bar_valid_coord_scales_empty[index_buf_idx].wait(index_bar_phase^1);

                        int tma_coords[2];
                        // NVFP4: 32 E4M3 scales/token (vs FP8: 8 E8M0). Per CTA-half: 16 scales = 16 bytes.
                        fp8_e4m3 scales[2*(NUM_SCALES_EACH_TOKEN/2)];  // 2 tokens * 16 scales/half = 32 bytes
                        char valid_mask = 0;
                        CUTE_UNROLL
                        for (int i = 0; i < 2; ++i) {
                            int block_idx, idx_in_block;
                            block_idx = (unsigned int)my_indices[i] / cur_block_size;
                            idx_in_block = (unsigned int)my_indices[i] % cur_block_size;
                            bool is_token_valid = my_indices[i] != -1 && (abs_pos+i < (IS_EXTRA_BLOCK?args.extra_topk_length:args.topk_length));
                            valid_mask |= is_token_valid << i;
                            tma_coords[i] = is_token_valid ? block_idx*cur_tma_coords_step_per_block + idx_in_block*tma_coords_step_per_token : -1;

                            // NVFP4 scales offset: 32 bytes/token, 16 bytes per CTA-half
                            int64_t offset = block_idx*cur_k_block_stride + (idx_in_block*NUM_SCALES_EACH_TOKEN + (cta_idx == 1 ? NUM_SCALES_EACH_TOKEN/2 : 0));
                            uint4 scalesx16 = is_token_valid ? __ldg((uint4*)(cur_k_scales_ptr + offset)) : uint4{0, 0, 0, 0};
                            *(uint4*)(scales + i*(NUM_SCALES_EACH_TOKEN/2)) = scalesx16;
                        }
                        valid_mask <<= lane_idx%4*2;
                        valid_mask |= __shfl_xor_sync(0xFFFFFFFF, valid_mask, 0x1);
                        valid_mask |= __shfl_xor_sync(0xFFFFFFFF, valid_mask, 0x2);
                        // 32 bytes per lane, lanes interleaved by 8 bytes; use uint4 (16B) store x2.
                        *(uint4*)(smem.scales[index_buf_idx] + lane_idx*(NUM_SCALES_EACH_TOKEN/2)) = ((uint4*)scales)[0];
                        *(uint4*)(smem.scales[index_buf_idx] + lane_idx*(NUM_SCALES_EACH_TOKEN/2) + 16) = ((uint4*)scales)[1];
                        *(int2*)(smem.tma_coord[index_buf_idx] + lane_idx*2) = *(int2*)tma_coords;
                        if (lane_idx%4 == 0)
                            smem.is_k_valid[index_buf_idx][lane_idx/4] = valid_mask;
                        
                        smem.bar_valid_coord_scales_full[index_buf_idx].arrive();
                        rs.update();
                    };

                    CUTE_NO_UNROLL
                    for (int block_idx = args.start_block_idx; block_idx < min(args.num_orig_kv_blocks, args.end_block_idx); ++block_idx) {
                        process_one_block(block_idx, IsOrigBlock{});
                    }

                    CUTE_NO_UNROLL
                    for (int block_idx = max(args.start_block_idx, args.num_orig_kv_blocks); block_idx < args.end_block_idx; ++block_idx) {
                        process_one_block(block_idx, IsExtraBlock{});
                    }
                });
            }
        } else if (warp_idx >= 10 && elect_one_sync()) {
            if constexpr (IS_PREFILL) {
                if (warp_idx == 10) {
                    // CLC Producer thread
                    run_outer_loop([&](const OuterloopArgs &args) {
                        if (cta_idx == 0) {
                            smem.bar_clc_empty.wait(args.outer_loop_phase^1);
                            ku::issue_clc_query_multicast_cluster_all(smem.bar_clc_full, smem.clc_response_obj);
                        }
                        smem.bar_clc_full.arrive_and_expect_tx(sizeof(smem.clc_response_obj));
                    });
                }
            } else {
                // Raw KV NoPE Producer thread
                run_outer_loop([&](const OuterloopArgs &args) {
                    CUTE_NO_UNROLL
                    for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
                        auto [raw_k_buf_idx, raw_k_bar_phase] = rs.get<NUM_RAW_K_BUFS>();
                        auto [index_buf_idx, index_bar_phase] = rs.get<NUM_INDEX_BUFS>();
                        smem.bar_valid_coord_scales_full[index_buf_idx].wait(index_bar_phase);
                        smem.bar_raw_KV_empty[raw_k_buf_idx].wait(raw_k_bar_phase^1);

                        int4 nxt_indices = *(int4*)(smem.tma_coord[index_buf_idx] + (warp_idx == 10 ? 0 : 4));
                        CUTE_UNROLL
                        for (int row = (warp_idx == 10 ? 0 : 4); row < B_TOPK; row += 8) {
                            int4 cur_indices = nxt_indices;
                            if (row+8 < B_TOPK)
                                nxt_indices = *(int4*)(smem.tma_coord[index_buf_idx] + row + 8);
                            ku::tma_gather4(
                                block_idx >= args.num_orig_kv_blocks ? &tma_params.tensor_map_extra_kv_nope : &tma_params.tensor_map_kv_nope,
                                smem.bar_raw_KV_full[raw_k_buf_idx],
                                smem.K_raw[raw_k_buf_idx].data() + row*(D_K/4),  // NVFP4: packed bytes
                                cta_idx*(D_K/4),  // NVFP4: half stride
                                cur_indices,
                                (int64_t)TMA::CacheHintSm90::EVICT_LAST
                            );
                        }
                        if (warp_idx == 10) {
                            smem.bar_raw_KV_full[raw_k_buf_idx].arrive_and_expect_tx(B_TOPK*(D_K/4)*sizeof(uint8_t));  // NVFP4 packed
                        }
                        smem.bar_valid_coord_scales_empty[index_buf_idx].arrive();
                        rs.update();
                    }
                });
            }
        }
    } else {
        // Scale & Exp threads
        cutlass::arch::warpgroup_reg_alloc<176>();

        int local_warp_idx = warp_idx - 12;
        bf16* sS_base = smem.S.data() + (local_warp_idx >= 2 ? (H_Q/2)*(B_TOPK/2) : 0) + (idx_in_warpgroup%64)*8;

        RingBufferState rs;
        run_outer_loop([&](const OuterloopArgs &args) {
            // For definition and consistency about `mi`, `li`, and `real_mi`, plz refer to head64 prefill
            float mi = MAX_INIT_VAL;
            float li = 0.0f;
            float real_mi = -CUDART_INF_F;
            static constexpr int NUM_ELEMS_PER_THREAD = B_TOPK / 2;

            CUTE_NO_UNROLL
            for (int k = args.start_block_idx; k < args.end_block_idx; ++k) {
                auto [k_buf_idx, k_bar_phase] = rs.get<NUM_K_BUFS>();
                auto [indices_buf_idx, indices_bar_phase] = rs.get<NUM_INDEX_BUFS>();
                auto [_, bar_phase] = rs.get<1>();
                // NOTE We don't need to sync for Prefill mode, since we have two synchronizations inside the loop body (one for p_exchange_buf sync, another one for rowwise_max_buf sync). The latter one guarantees the emptyness of p_exchange_buf and the former one guarantees the emptyness of rowwise_max_buf
                smem.bar_valid_coord_scales_full[indices_buf_idx].wait(indices_bar_phase);

                // Get P from TMEM
                float p[NUM_ELEMS_PER_THREAD];
                smem.bar_QK_done.wait(bar_phase);
                ku::tcgen05_after_thread_sync();
                retrieve_mask_and_reduce_p<
                    NUM_ELEMS_PER_THREAD,
                    tmem_cols::P,
                    barrier_ids::WG2_WARP02_SYNC,
                    barrier_ids::WG2_WARP13_SYNC,
                    false
                >(
                    smem.is_k_valid[indices_buf_idx],
                    local_warp_idx,
                    lane_idx,
                    [&]() {smem.bar_P_empty.arrive(0u);},
                    smem.P_exchange,
                    p
                );

                // Get rowwise max of P
                float cur_pi_max = get_max<NUM_ELEMS_PER_THREAD>(p);
                cur_pi_max *= params.sm_scale_div_log2;

                smem.rowwise_max_buf[idx_in_warpgroup] = cur_pi_max;
                NamedBarrier::arrive_and_wait(64, barrier_ids::WG2_WARP02_SYNC + (local_warp_idx&1));
                cur_pi_max = max(cur_pi_max, smem.rowwise_max_buf[idx_in_warpgroup^64]);
                real_mi = max(real_mi, cur_pi_max);
                bool should_scale_o = __any_sync(0xffffffff, cur_pi_max - mi > 6.0f);


                // Calc scale factor, and scale li
                float new_max, scale_for_old;
                if (!should_scale_o) {
                    // Don't scale O
                    scale_for_old = 1.0f;
                    new_max = mi;
                } else {
                    new_max = max(cur_pi_max, mi);
                    scale_for_old = exp2f(mi - new_max);
                }
                mi = new_max;   // mi is still identical within each row

                // Calculate S
                nv_bfloat162 s[NUM_ELEMS_PER_THREAD/2];
                float cur_sum = get_s_from_p<NUM_ELEMS_PER_THREAD>(s, p, params.sm_scale_div_log2, new_max);
                li = fmaf(li, scale_for_old, cur_sum);

                // Store S
                smem.bar_SV_done.wait(bar_phase^1);
                CUTE_UNROLL
                for (int i = 0; i < NUM_ELEMS_PER_THREAD/8; ++i) {
                    ku::st_shared(sS_base + i*8*(H_Q/2), *(__int128_t*)(s + i*4));
                }

                // Rescale O
                if (k > 0 && should_scale_o) {
                    ku::tcgen05_after_thread_sync();
                    rescale_O<D_V, 32, tmem_cols::O>(scale_for_old);
                    ku::tcgen05_before_thread_sync();
                }

                fence_view_async_shared();
                smem.bar_S_O_full.arrive(0u);
                smem.bar_valid_coord_scales_empty[indices_buf_idx].arrive();

                rs.update();
            }

            if (real_mi == -CUDART_INF_F) {
                // real_mi == -CUDART_INF_F <=> No valid TopK indices
                // We set li to 0 to fit the definition that li := exp(x[i] - mi)
                li = 0.0f;
                mi = -CUDART_INF_F;
            }

            // Reduce li
            smem.bar_li_empty.wait(args.outer_loop_phase^1);
            smem.rowwise_li_buf[idx_in_warpgroup^64] = li;
            NamedBarrier::arrive_and_wait(128, barrier_ids::WG2_SYNC);
            li += smem.rowwise_li_buf[idx_in_warpgroup];

            if (idx_in_warpgroup < H_Q/2) {
                // Calculate output_scale and save
                int head_idx = cta_idx*(H_Q/2) + idx_in_warpgroup;
                float attn_sink = params.attn_sink == nullptr ? -CUDART_INF_F : __ldg(params.attn_sink + head_idx);
                float output_scale;
                if (FWD_MODE != FwdMode::DecodeWithSplitKV || args.is_no_split) {
                    output_scale = __fdividef(1.0f, li + exp2f(fmaf(attn_sink, CUDART_L2E_F, -mi)));
                } else {
                    output_scale = __fdividef(1.0f, li);
                }
                smem.rowwise_li_buf[idx_in_warpgroup] = li == 0.0f ? 0.0f : output_scale;
                smem.bar_li_full.arrive();

                float cur_lse = fmaf(mi, CUDART_LN2_F, logf(li));
                cur_lse = cur_lse == -CUDART_INF_F ? +CUDART_INF_F : cur_lse;
                if constexpr (IS_PREFILL) {
                    int global_index = args.s_q_idx*params.h_q + head_idx;
                    params.max_logits[global_index] = real_mi*CUDART_LN2_F;
                    params.lse[global_index] = cur_lse;
                } else {
                    if (FWD_MODE != FwdMode::DecodeWithSplitKV || args.is_no_split) {
                        params.lse[args.batch_idx*params.stride_lse_b + args.s_q_idx*params.stride_lse_s_q + head_idx] = cur_lse;
                    } else {
                        float cur_lse_2base = log2f(li) + mi;
                        params.lse_accum[args.n_split_idx*params.stride_lse_accum_split + args.s_q_idx*params.stride_lse_accum_s_q + head_idx] = cur_lse_2base;
                    }
                }

            }
        });
    }

    ku::barrier_cluster_arrive_relaxed();
    ku::barrier_cluster_wait_acquire();

#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm100");
    }
#endif
}

// We have two launchers with different kernel names to distinguish prefill and decode

template<typename Kernel>
static __global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, 2)
sparse_attn_fwd_for_small_topk_kernel(__grid_constant__ const typename Kernel::ArgT params, __grid_constant__ const typename Kernel::TmaParams tma_params) {
    Kernel::sparse_attn_fwd_kernel_devfunc(params, tma_params);
}

template<typename Kernel>
static __global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, 2)
flash_fwd_splitkv_mla_fp8_sparse_kernel(__grid_constant__ const typename Kernel::ArgT params, __grid_constant__ const typename Kernel::TmaParams tma_params) {
    Kernel::sparse_attn_fwd_kernel_devfunc(params, tma_params);
}

template<FwdMode FWD_MODE, int D_QK>
void KernelTemplate<FWD_MODE, D_QK>::run(const ArgT& params) {
    static_assert(D_QK == 576 || D_QK == 512);

    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.topk % B_TOPK == 0);   // To save some boundry checkings
    KU_ASSERT(params.h_q == H_Q);  // To save some calculation
    KU_ASSERT(params.d_qk == D_QK);

    static_assert(D_Q == 512);
    CUtensorMap tensor_map_q;
    if constexpr (IS_DECODE) {
        KU_ASSERT(params.stride_q_b % params.stride_q_s_q == 0, "In decode mode for MODEL1 sparse fp8 decoding on sm100f, q.stride(0) (on the batch dimension) must be divisible by q.stride(1) (on the sequence dimension).");
        tensor_map_q = ku::make_tensor_map(
            {64ul, H_Q, 2ul, (D_Q/64ul)/2ul, (unsigned long)params.b * (params.stride_q_b / params.stride_q_s_q)},
            ku::make_stride_helper<int>({params.stride_q_h_q, D_Q/2, 64, params.stride_q_s_q}, sizeof(bf16)),
            {64, H_Q/2, 2, (D_Q/64)/2, 1},
            params.q,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        );
    } else {
        tensor_map_q = ku::make_tensor_map(
            {64ul, H_Q, 2ul, (D_Q/64ul)/2ul, (unsigned long)params.s_q},
            ku::make_stride_helper<int>({params.stride_q_h_q, D_Q/2, 64, params.stride_q_s_q}, sizeof(bf16)),
            {64, H_Q/2, 2, (D_Q/64)/2, 1},
            params.q,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        );  // We use this layout to group Q[0:64] and Q[256:256+64] together, for UTCCP for dual gemm
    }

    CUtensorMap tensor_map_kv;
    CUtensorMap tensor_map_kv_nope, tensor_map_kv_rope, tensor_map_extra_kv_nope = {}, tensor_map_extra_kv_rope = {};
    if constexpr (IS_DECODE) {
        auto get_kv_tensormap = [&](bool is_extra, void* k_ptr, int num_blocks, int64_t stride_kv_block, int64_t stride_kv_row) -> std::pair<CUtensorMap, CUtensorMap> {
            KU_ASSERT((int64_t)k_ptr % 16 == 0, "The base address of %sk_ptr (%p) must be 16B aligned for sparse fp8 attention on sm100f", is_extra?"extra_":"", k_ptr);
            KU_ASSERT(stride_kv_block % TMA_K_STRIDE_FOR_DECODING == 0, "%sk_cache.stride(0) (%ld) must be a multiple of %d. Padding might be necessary", is_extra?"extra_":"", stride_kv_block, TMA_K_STRIDE_FOR_DECODING);
            CUtensorMap tensor_map_kv_nope = ku::make_tensor_map(
                {D_NOPE/2 + D_ROPE*2, (uint64_t)num_blocks * (stride_kv_block/TMA_K_STRIDE_FOR_DECODING)},  // NVFP4 packed
                {TMA_K_STRIDE_FOR_DECODING},
                {D_K/4, 1},  // NVFP4: half tile width vs FP8
                k_ptr,
                CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
                CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B
            );  // NOTE: Here we use `D_NOPE+D_ROPE*2` as the box shape instead of D_NOPE because it's actually faster. I think that's because, if we use `D_NOPE+D_ROPE*2`, we can prefetch part of the RoPE part of the selected tokens.
            CUtensorMap tensor_map_kv_rope = ku::make_tensor_map(
                {D_ROPE, (uint64_t)num_blocks * (stride_kv_block/TMA_K_STRIDE_FOR_DECODING)},
                {TMA_K_STRIDE_FOR_DECODING},
                {64, 1},
                (uint8_t*)k_ptr + D_NOPE/2,  // NVFP4: rope starts after packed nope (224 bytes)
                CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
                CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B
            );
            return {tensor_map_kv_nope, tensor_map_kv_rope};
        };
        std::tie(tensor_map_kv_nope, tensor_map_kv_rope) = get_kv_tensormap(false, params.kv, params.num_blocks, params.stride_kv_block, params.stride_kv_row);
        if (params.extra_topk > 0)
            std::tie(tensor_map_extra_kv_nope, tensor_map_extra_kv_rope) = get_kv_tensormap(true, params.extra_kv, params.extra_num_blocks, params.stride_extra_kv_block, params.stride_extra_kv_row);
    } else {
        tensor_map_kv = ku::make_tensor_map(
            {D_QK, (unsigned long)params.s_kv}, 
            {(unsigned long)params.stride_kv_s_kv*sizeof(bf16)},
            {64, 1},
            params.kv,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        );
    }

    CUtensorMap tensor_map_o;
    if constexpr (IS_DECODE) {
        tensor_map_o = ku::make_tensor_map(
            {64, H_Q, D_V/64, (unsigned long)params.s_q, (unsigned long)params.b},
            ku::make_stride_helper<int>({params.stride_o_h_q, 64, params.stride_o_s_q, params.stride_o_b}, sizeof(bf16)),
            {64, H_Q/2, D_V/64, 1, 1},
            params.out,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        );
    } else {
        tensor_map_o = ku::make_tensor_map(
            {64, H_Q, D_V/64, (unsigned long)params.s_q, 1ul},
            ku::make_stride_helper<int>({D_V, 64, H_Q*D_V, H_Q*D_V}, sizeof(bf16)),
            {64, H_Q/2, D_V/64, 1, 1},
            params.out,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        );
    }


    CUtensorMap tensor_map_o_accum = {};
    if constexpr (FWD_MODE == FwdMode::DecodeWithSplitKV) {
        tensor_map_o_accum = ku::make_tensor_map(
            {32, H_Q, D_V/32, (unsigned long)params.s_q, (unsigned long)params.num_sm_parts + params.b},
            ku::make_stride_helper<int>({params.stride_o_accum_h_q, 32, params.stride_o_accum_s_q, params.stride_o_accum_split}, sizeof(float)),
            {32, H_Q/2, B_EPI_SPLITKV/32, 1, 1},
            params.o_accum,
            CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        );
    }

    TmaParams tma_params;
    if constexpr (IS_DECODE) {
        tma_params = {
            tensor_map_q,
            tensor_map_o,
            tensor_map_o_accum,
            tensor_map_kv_nope,
            tensor_map_kv_rope,
            tensor_map_extra_kv_nope,
            tensor_map_extra_kv_rope
        };
    } else {
        tma_params = {
            tensor_map_q,
            tensor_map_kv,
            tensor_map_o
        };
    }
    
    auto kernel = IS_PREFILL ? &sparse_attn_fwd_for_small_topk_kernel<KernelTemplate<FWD_MODE, D_QK>> : &flash_fwd_splitkv_mla_fp8_sparse_kernel<KernelTemplate<FWD_MODE, D_QK>>;
    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 grid_shape;
    if constexpr (IS_DECODE) {
        grid_shape = dim3(2*params.s_q, FWD_MODE == FwdMode::DecodeWithSplitKV ? params.num_sm_parts : params.b, 1);
    } else {
        grid_shape = dim3(2*params.s_q, 1, 1);
    }

    cutlass::ClusterLaunchParams launch_params = {
        grid_shape,
        dim3(NUM_THREADS, 1, 1),
        dim3(2, 1, 1),
        smem_size,
        params.stream
    };
    KU_CUTLASS_CHECK(cutlass::launch_kernel_on_cluster(
        launch_params, (void*)kernel, params, tma_params
    ));
}

template<FwdMode FWD_MODE, int D_QK>
void run_fwd_for_small_topk_phase1_kernel(const SparseFwdArgT<FWD_MODE>& params) {
    using Kernel = KernelTemplate<FWD_MODE, D_QK>;
    Kernel::run(params);
}

}
