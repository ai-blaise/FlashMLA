#pragma once

#include "kernel.h"

#include <cuda_fp8.h>
#include <cutlass/barrier.h>
#include <cute/tensor.hpp>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>

#include <kerutils/kerutils.cuh>

#include "defines.h"
#include "params.h"

namespace sm100::decode::head64_nvfp4 {

using cutlass::arch::fence_view_async_shared;
using cutlass::arch::NamedBarrier;
using e8m0 = __nv_fp8_e8m0;
using e4m3 = cutlass::float_e4m3_t;
using ue4m3 = cutlass::float_ue4m3_t;   // NVFP4 scale-factor type (unsigned E4M3)
using e2m1  = cutlass::float_e2m1_t;     // NVFP4 element type
using namespace cute;

enum NamedBarriers : uint32_t {
    main_loop_sync = 0,
    wg0_sync = 1,
    wg0_warp02_sync = 2,
    wg0_warp13_sync = 3,
    everyone_sync = 4
};

template<ModelType MODEL_TYPE>
struct KernelTemplate {

static constexpr int D_Q = MODEL_TYPE == ModelType::V32 ? 576 : 512;
static constexpr int D_K = D_Q;
static constexpr int D_V = 512;
static constexpr int D_NOPE = MODEL_TYPE == ModelType::V32 ? 512 : 448;
static constexpr int D_ROPE = 64;
static constexpr int QUANT_TILE_SIZE = MODEL_TYPE == ModelType::V32 ? 16 : 16;  // NVFP4 block size
static constexpr bool V_HAVE_ROPE = MODEL_TYPE == ModelType::V32 ? false : true;
static constexpr int NUM_SCALES_EACH_TOKEN = MODEL_TYPE == ModelType::V32 ? 32 : 32;    // NVFP4 block_size=16 -> 32 scales per token for D_NOPE=512 (V32) or 28 padded to 32 (MODEL1)
// NVFP4 V32: D_NOPE/2 (=256) packed FP4 + 2*D_ROPE (=128 BF16) + NUM_SCALES (=32 E4M3)
// = 416 B/token. Inline layout (scales follow rope per token), matching FP8 V32 architecture.
// MODEL1: D_NOPE/2 (=224) + 2*D_ROPE (=128) + 32 scales = 384.
static constexpr int TMA_K_STRIDE = MODEL_TYPE == ModelType::V32 ? (D_NOPE/2)+2*D_ROPE+NUM_SCALES_EACH_TOKEN : (D_NOPE/2)+2*D_ROPE+NUM_SCALES_EACH_TOKEN;   // Stride of K's tensormap. This stride must 1) be a factor of the actual stride between tokens 2) large enough to cover the entire KV cache. Since TMA copy's coordinate can only be 32bit signed integers, this number must >= 128, perferrably >= 256. So we set this to 656 for V32 and 576 for MODEL1. Extra padding may be necessary for KV blocks.
static_assert(D_NOPE + D_ROPE == D_Q);
static_assert(V_HAVE_ROPE ? (D_NOPE + D_ROPE == D_V) : (D_NOPE == D_V));

static constexpr int B_H = 64;
static constexpr int B_TOPK = 64;
static constexpr int NUM_BUFS = 2;
static constexpr int NUM_INDEX_BUFS = 3;  // NVFP4: reduced from 4 to fit SMEM (32 scales/token expands SMEM)
static constexpr int NUM_THREADS = 128*3;  // 128 exp + 1/32 utcmma + 1/32 raw KV producer + 1/32 rope producer + 32 index+scale+valid_mask producer + 128 dequant
static constexpr float MAX_INIT_VAL = -1e30f;  // To avoid (-inf) - (-inf) = NaN

static constexpr int D_Q_SW128 = 512;
static constexpr int D_Q_SW64 = MODEL_TYPE == ModelType::V32 ? 64 : 0;
static_assert(D_Q_SW128 + D_Q_SW64 == D_Q);
static constexpr int K_ROPE_SW = MODEL_TYPE == ModelType::V32 ? 64 : 128; // RoPE part stored in SW64 (for V32) or SW128 (for MODEL1), in bytes

template<
    typename Shape_Q_SW128, typename TMA_Q_SW128,
    typename Shape_O, typename TMA_O
>
struct TmaParams {
    Shape_Q_SW128 shape_Q_SW128; TMA_Q_SW128 tma_Q_SW128;
    Shape_O shape_O; TMA_O tma_O;
    CUtensorMap tensor_map_q_sw64;  // Invalid if D_Q_SW64 == 0
    CUtensorMap tensor_map_kv_nope;
    CUtensorMap tensor_map_kv_rope;
    CUtensorMap tensor_map_extra_kv_nope;
    CUtensorMap tensor_map_extra_kv_rope;
};

// Tensor memory columns
struct tmem_cols {
    //   0 ~ 256: output
    // 256 ~ 256 + B_H*D_NOPE/2/128: Q (used by BF16 NoPE path — still allocated for MODEL1;
    //                                  V32 after Phase 1 doesn't use this region since NoPE
    //                                  reads from FP4 SMEM, but layout reserved for binary
    //                                  compatibility across MODEL_TYPE branches).
    // Q_Tail .. 400: BF16 RoPE Q-TMEM (still in use, RoPE stays BF16 in Phase 1)
    // 400 ~ 464: P (S_p softmax accumulator)
    // 464 ~ 496: SFA_Q (Q-side SF, ~32 cols — actual footprint via find_tmem_tensor_col_offset)
    // 496 ~ 512: SFB_K (K-side SF, fits ~16 cols at the tail of TMEM)
    // If runtime TMEM overflow surfaces, reclaim Q region for V32 via per-MODEL_TYPE specialization.
    static constexpr int O = 0;
    static constexpr int Q = 256;
    static constexpr int Q_Tail = 256 + B_H*D_NOPE/2/128;
    static constexpr int P = 400;
    static constexpr int SFA_Q = 464;       // Q-side scale-factor TMEM region (one-shot per batch)
    static constexpr int SFB_K = 496;       // K-side scale-factor TMEM region (moved from 468 to 496 to avoid SFA overlap)
    static_assert(SFB_K <= 512, "SFB_K base exceeds 512-col TMEM budget");
};

template<int NUM_TILES>
using SmemLayoutQTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<NUM_TILES*64>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutQ_SW128 = SmemLayoutQTiles<D_Q_SW128/64>;

// FP4-packed Q SMEM layout for the MXF4 atom (commit 3+4: SW128-swizzle Q + K).
// e2m1 = 4 bits, so this stores D_NOPE FP4 elems = D_NOPE/2 bytes per head row.
// SW128 atom with e2m1 specializes via cute::upcast<sizeof_bits<e2m1>::value>.
// SW128 swizzle is required to satisfy SM100_MMA_MXF4_SS smem-descriptor canonical
// UMMA_K stride check. SW64/INTER variants fail "Not a canonical UMMA_K Layout".
template<int NUM_TILES>
using SmemLayoutQ_FP4_Tiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<e2m1>{},
    Shape<Int<B_H>, Int<NUM_TILES*128>>{},   // 128 e2m1 elems per atom row = 64B SW128 atom
    Step<_1, _2>{}
), Shape<_1, _1>{}));
// Use V32 dim (512) so the layout is always well-defined.
using SmemLayoutQ_FP4 = SmemLayoutQ_FP4_Tiles<512/128>;

// FP4-packed K SMEM layout for the MXF4 atom (dual-gemm packed M=B_H*2=128).
// Mirrors SmemLayoutKTiles_DualGemm_SW128 but for e2m1 + halved K-stride bytes.
template<int NUM_TILES>
using SmemLayoutK_FP4_Tiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<e2m1>{},
    Shape<Int<B_H*2>, Int<NUM_TILES*128>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));
using SmemLayoutK_FP4 = SmemLayoutK_FP4_Tiles<512/128>;

using SmemLayoutOBuf = decltype(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<D_V>>{}
));

using SmemLayoutOBuf_TMA = decltype(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<64>>{}
)); // A TMA tile

static_assert(D_V == 512);
using SmemLayoutOAccumBuf = Layout<
    Shape<Int<B_H>, Int<D_V>>,
    Stride<Int<520>, _1>	// We use stride = 520 here to avoid bank conflict
>;

using SmemLayoutS = decltype(tile_to_shape(
    UMMA::Layout_K_INTER_Atom<bf16>{},
    Shape<Int<B_H>, Int<B_TOPK>>{},
    Step<_1, _2>{}
));

template<int NUM_TILES>
using SmemLayoutKTiles_SW128 = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutKTiles_DualGemm_SW128 = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H*2>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutKTilesTransposed_SW128 = decltype(composition(
    SmemLayoutKTiles_SW128<NUM_TILES>{},
    Layout<
        Shape<Int<64*NUM_TILES>, Int<B_TOPK>>,
        Stride<Int<B_TOPK>, _1>
    >{}
));

template<int NUM_TILES>
using SmemLayoutKTiles_SW64 = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_H>, Int<32*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutKTiles_DualGemm_SW64 = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_H*2>, Int<32*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutKTilesTransposed_SW64 = decltype(composition(
    SmemLayoutKTiles_SW64<NUM_TILES>{},
    Layout<
        Shape<Int<32*NUM_TILES>, Int<B_TOPK>>,
        Stride<Int<B_TOPK>, _1>
    >{}
));

struct SharedMemoryPlan {
    union {
        struct {
            array_aligned<bf16, cosize_v<SmemLayoutQ_SW128>> q;
            bf16 q_sw64[B_H*D_Q_SW64];  // NOTE D_Q_SW64 may be 0 but array_aligned<bf16, 0> will have a size of 16, so we use array here. The former tensor (`q`) promises its alignment.
            // FP4 scaffolding (commit 1: present but unused; commit 2 will activate).
            // q_fp4 + q_scales nested in a union with `o` so they don't enlarge the qo arm.
            // Once the FP4 path is live (commit 2), o.o_accum_buf is still populated AFTER
            // QK has consumed q_fp4 -- matches the existing q-then-o lifetime pattern.
            // q_fp4 sized for V32 (512 nope dim) -- MODEL1 is wired in commit 2.
            union {
                struct {
                    array_aligned<uint8_t, B_H*512/2> q_fp4;
                    CUTE_ALIGNAS(16) ue4m3 q_scales[B_H][NUM_SCALES_EACH_TOKEN];
                } fp4;
                array_aligned<bf16, cosize_v<SmemLayoutOBuf>> o_buf;
                array_aligned<float, cosize_v<SmemLayoutOAccumBuf>> o_accum_buf;
            } o;
        } qo;
        struct {
            struct {
                array_aligned<bf16, B_H*D_NOPE> nope; // NoPE part, dequantized
                array_aligned<bf16, B_H*D_ROPE> rope; // RoPE part, dequantized. SW64 in v32 mode, SW128 in MODEL1 mode
            } dequant[NUM_BUFS];
            static_assert(sizeof(dequant) >= sizeof(bf16) * (B_H*D_Q)); // So that Q does not covers raw_nope
            // NVFP4: packed e2m1, half the byte count vs FP8 raw_nope
            array_aligned<uint8_t, B_H*D_NOPE/2> raw_nope[NUM_BUFS];  // Raw FP4-packed NoPE
        } kv;
    } u;
    union {
        float4 p_exchange_buf[4][16 * B_TOPK / 4];
        array_aligned<bf16, cosize_v<SmemLayoutS>> s;
    } s_p;
    CUTE_ALIGNAS(16) float rowwise_max_buf[128];
    char is_token_valid[NUM_INDEX_BUFS][B_TOPK/8];
    CUTE_ALIGNAS(16) int tma_coord[NUM_INDEX_BUFS][B_TOPK];
    CUTE_ALIGNAS(16) e4m3 scales[NUM_INDEX_BUFS][B_TOPK][NUM_SCALES_EACH_TOKEN];  // NVFP4 E4M3 per-block scales
    array_aligned<uint32_t, 1> tmem_start_addr;
    transac_bar_t bar_last_store_done;
    transac_bar_t bar_q_tma, bar_q_utccp;
    transac_bar_t bar_rope_ready[NUM_BUFS];
    transac_bar_t bar_nope_ready[NUM_BUFS];
    transac_bar_t bar_raw_ready[NUM_BUFS], bar_raw_free[NUM_BUFS];
    transac_bar_t bar_valid_coord_scale_ready[NUM_INDEX_BUFS], bar_valid_coord_scale_free[NUM_INDEX_BUFS];
    transac_bar_t bar_qk_done[NUM_BUFS], bar_so_ready[NUM_BUFS], bar_sv_done[NUM_BUFS];
};

using TiledMMA_P = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_TS_NOELECT<bf16, bf16, float, B_H, B_TOPK*2, UMMA::Major::K, UMMA::Major::K>{}
)); // *2 for dual gemm

using TiledMMA_O = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_H, 256, UMMA::Major::K, UMMA::Major::MN>{}
));

// Block-scaled FP4 MMA for S = Q @ K^T (NoPE path). 1-CTA SM100_MMA_MXF4_SS variant.
// M=B_H*2=128 (REQUIRED: SM100_MMA_MXF4_SS hard-asserts M==128 for 1-CTA cluster, per
//   mma_sm100_umma.hpp:1360). N=B_TOPK*2=128 (dual-gemm pack on N side).
// 1-CTA M=64 attempt failed at static_assert; either use 2x1SM_SS variant for 2-CTA cluster
//   (= M=128 distributed as 64/CTA, kernel architectural change), OR keep M=128 and ensure
//   raw_nope + q_fp4 SMEM hold 2 tiles each (current allocation is 1 tile → undersized for MMA).
// Phase 4-blocker: raw_nope is B_H*D_NOPE/2 = 16K bytes per buf; need 2x for M=128 dual-gemm.
// Scale-factor type is ue4m3 (unsigned E4M3, the type required by CUTLASS NVFP4 traits).
using TiledMMA_S_NVFP4 = decltype(make_tiled_mma(
    cute::SM100_MMA_MXF4_SS<
        e2m1,       // A: NVFP4
        e2m1,       // B: NVFP4
        float,      // C: FP32
        ue4m3,      // SF: UE4M3
        B_H*2,      // M = 128 (HW-required for 1-CTA MXF4_SS)
        B_TOPK*2,   // N = 128 (dual-gemm pack)
        16,         // VS = NVFP4 vector size
        UMMA::Major::K, UMMA::Major::K
    >{}
));

// Canonical SF SMEM layouts via Sm1xxBlockScaledConfig (commit 3+4).
// Replaces the flat layouts that hit "Expected an MMA-SF partitioned tensor".
// SFVecSize=16 matches NVFP4 QUANT_TILE_SIZE.
using Sm100BlockScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<16>;
// MMA tile shape for QK NoPE GEMM: M=B_H*2=128, N=B_TOPK*2=128, K=D_NOPE=512.
using TileShape_QK_FP4 = Shape<Int<B_H*2>, Int<B_TOPK*2>, Int<D_NOPE>>;
using SmemLayoutAtomSFA_QK = decltype(Sm100BlockScaledConfig::deduce_smem_layoutSFA(
    TiledMMA_S_NVFP4{}, TileShape_QK_FP4{}));
using SmemLayoutAtomSFB_QK = decltype(Sm100BlockScaledConfig::deduce_smem_layoutSFB(
    TiledMMA_S_NVFP4{}, TileShape_QK_FP4{}));

template<typename TmaParam>
static __device__ void
flash_fwd_splitkv_mla_fp8_sparse_kernel_devfunc(const SparseAttnDecodeParams &params, const TmaParam &tma_params);

static void run(const SparseAttnDecodeParams &params);

};

}