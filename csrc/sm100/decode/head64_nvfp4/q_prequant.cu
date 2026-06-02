#include "sm100/decode/head64_nvfp4/q_prequant.h"

#include <cuda_bf16.h>
#include <cutlass/numeric_conversion.h>

#include "kerutils/kerutils.cuh"

namespace sm100::decode::head64_nvfp4 {
namespace {

using bf16 = cutlass::bfloat16_t;
using e2m1 = cutlass::float_e2m1_t;
using ue4m3 = cutlass::float_ue4m3_t;

__global__ void v32_q_prequant_nvfp4_kernel(V32QPrequantParams params) {
#if defined(KERUTILS_ENABLE_SM100A)
    constexpr int kRows = V32_Q_NATIVE_ROWS;
    constexpr int kValidRows = V32_Q_NATIVE_VALID_ROWS;
    constexpr int kCols = V32_Q_NATIVE_D_QK;
    constexpr int kScaleVec = 16;
    constexpr int kScalesPerRow = V32_Q_NATIVE_NUM_SCALES;
    constexpr int kNumHeadBlocks = 2;
    static_assert(kRows == 128);
    static_assert(kValidRows == 64);
    static_assert(kCols == 576);
    static_assert(kScalesPerRow == 36);

    int tile_idx = blockIdx.x;
    int head_block = tile_idx % kNumHeadBlocks;
    int sq_tmp = tile_idx / kNumHeadBlocks;
    int s_q_idx = sq_tmp % params.s_q;
    int batch_idx = sq_tmp / params.s_q;
    int tid = threadIdx.x;

    uint8_t* q_native = params.q_native + batch_idx * params.stride_q_native_b +
        s_q_idx * params.stride_q_native_s_q +
        head_block * params.stride_q_native_hblock;
    uint8_t* q_scales = params.q_native_scales + batch_idx * params.stride_q_native_scales_b +
        s_q_idx * params.stride_q_native_scales_s_q +
        head_block * params.stride_q_native_scales_hblock;

    using FP4Converter = cutlass::NumericConverter<e2m1, float>;
    using ScaleConverter = cutlass::NumericConverter<ue4m3, float>;

    constexpr int kTotalScaleBlocks = kRows * kScalesPerRow;
    for (int scale_block = tid; scale_block < kTotalScaleBlocks; scale_block += blockDim.x) {
        int row = scale_block / kScalesPerRow;
        int col_block = scale_block % kScalesPerRow;
        int col_base = col_block * kScaleVec;
        bool valid_row = row < kValidRows;
        int q_head = head_block * kValidRows + row;

        float values[kScaleVec];
        float absmax = 0.0f;
        CUTE_UNROLL
        for (int i = 0; i < kScaleVec; ++i) {
            float value = 0.0f;
            if (valid_row) {
                const bf16* q_ptr = params.q + batch_idx * params.stride_q_b +
                    s_q_idx * params.stride_q_s_q + q_head * params.stride_q_h_q + col_base + i;
                value = static_cast<float>(*q_ptr);
                absmax = fmaxf(absmax, fabsf(value));
            }
            values[i] = value;
        }

        float scale = valid_row ? fmaxf(absmax / 6.0f, 0x1p-9f) : 1.0f;
        float inv_scale = 1.0f / scale;
        uint8_t* row_base = q_native + row * V32_Q_NATIVE_SCORE_BYTES + col_base / 2;
        CUTE_UNROLL
        for (int i = 0; i < kScaleVec; i += 2) {
            e2m1 lo = valid_row ? FP4Converter::convert(values[i] * inv_scale) : e2m1::bitcast(0);
            e2m1 hi = valid_row ? FP4Converter::convert(values[i + 1] * inv_scale) : e2m1::bitcast(0);
            row_base[i / 2] = (lo.raw() & 0x0f) | ((hi.raw() & 0x0f) << 4);
        }
        ue4m3 scale_e4m3 = ScaleConverter::convert(scale);
        q_scales[row * kScalesPerRow + col_block] = *reinterpret_cast<uint8_t*>(&scale_e4m3);
    }
#endif
}

}  // namespace

void run_v32_q_prequant_nvfp4_kernel(const V32QPrequantParams& params) {
    KU_ASSERT(params.h_q == 128, "V3.2 native-Q prequant requires h_q == 128");
    KU_ASSERT(params.q != nullptr);
    KU_ASSERT(params.q_native != nullptr);
    KU_ASSERT(params.q_native_scales != nullptr);
    dim3 grid(params.b * params.s_q * 2);
    dim3 block(128);
    v32_q_prequant_nvfp4_kernel<<<grid, block, 0, params.stream>>>(params);
    KU_CHECK_KERNEL_LAUNCH();
}

}  // namespace sm100::decode::head64_nvfp4
