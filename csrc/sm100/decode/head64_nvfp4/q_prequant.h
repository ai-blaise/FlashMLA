#pragma once

#include <cuda_runtime_api.h>

#include <cutlass/bfloat16.h>

namespace sm100::decode::head64_nvfp4 {

static constexpr int V32_Q_NATIVE_ROWS = 128;
static constexpr int V32_Q_NATIVE_VALID_ROWS = 64;
static constexpr int V32_Q_NATIVE_D_QK = 576;
static constexpr int V32_Q_NATIVE_SCORE_BYTES = V32_Q_NATIVE_D_QK / 2;
static constexpr int V32_Q_NATIVE_NUM_SCALES = V32_Q_NATIVE_D_QK / 16;
static constexpr int V32_Q_NATIVE_BYTES = V32_Q_NATIVE_ROWS * V32_Q_NATIVE_SCORE_BYTES;
static constexpr int V32_Q_NATIVE_SCALE_BYTES = V32_Q_NATIVE_ROWS * V32_Q_NATIVE_NUM_SCALES;

struct V32QPrequantParams {
    int b;
    int s_q;
    int h_q;
    const cutlass::bfloat16_t* __restrict__ q;
    uint8_t* __restrict__ q_native;
    uint8_t* __restrict__ q_native_scales;
    int stride_q_b;
    int stride_q_s_q;
    int stride_q_h_q;
    int stride_q_native_b;
    int stride_q_native_s_q;
    int stride_q_native_hblock;
    int stride_q_native_scales_b;
    int stride_q_native_scales_s_q;
    int stride_q_native_scales_hblock;
    cudaStream_t stream;
};

void run_v32_q_prequant_nvfp4_kernel(const V32QPrequantParams& params);

}  // namespace sm100::decode::head64_nvfp4
