// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek-AI + ai-blaise. Apache-2.0
//
// NVFP4 KV sparse-MLA decode interface (2-tensor: kv + kv_scales).

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "sparse_decode.h"
#include "sm100/prefill/sparse/fwd_for_small_topk/head128_nvfp4/phase1.h"
#include "params.h"
#include "smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.h"
#include "smxx/decode/combine/combine.h"

class Decode_Sm100_Head128_NVFP4_Impl : public DecodeImplBase {
    DECLARE_SUPPORTED_FEATURES(
        DecodeFeatures::HEAD_128,
        DecodeFeatures::HEAD_DIM_512,
        DecodeFeatures::MODEL1_KVCACHE_FORMAT,
        DecodeFeatures::ATTN_SINK,
        DecodeFeatures::TOPK_LENGTH
    )

public:
    DecodeImplMeta get_meta(int h_q, int s_q) override {
        Arch arch = Arch();
        return {std::max(arch.num_sms / s_q / 2, 1), 3, 64};
    }

protected:
    void run_(const SparseAttnDecodeParams &params, const std::vector<FeatureT> &required_features) override {
        sm100::fwd_for_small_topk::head128_nvfp4::run_fwd_for_small_topk_phase1_kernel<SparseAttnFwdMode::DecodeWithSplitKV, 512>(params);
    }
};

std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>>
sparse_attn_decode_nvfp4_interface(
    const at::Tensor &q,
    const at::Tensor &kv,
    const at::Tensor &kv_scales,
    const at::Tensor &indices,
    const std::optional<at::Tensor> &topk_length,
    const std::optional<at::Tensor> &attn_sink,
    std::optional<at::Tensor> &tile_scheduler_metadata,
    std::optional<at::Tensor> &num_splits,
    int d_v,
    float sm_scale
) {
    using bf16 = cutlass::bfloat16_t;

    Arch arch = Arch();
    TORCH_CHECK(arch.is_sm100f(), "sparse_attn_decode_nvfp4 requires sm100f (Blackwell)");

    KU_CHECK_NDIM(q, 4);
    KU_CHECK_NDIM(kv, 4);
    KU_CHECK_NDIM(kv_scales, 4);
    KU_CHECK_NDIM(indices, 3);

    int b = q.size(0);
    int s_q = q.size(1);
    int h_q = q.size(2);
    int d_qk = q.size(3);
    int num_blocks = kv.size(0);
    int page_block_size = kv.size(1);
    int h_kv = kv.size(2);
    int topk = indices.size(2);

    constexpr int NVFP4_NOPE_ROPE_BYTES = 352;  // 224 nope + 128 rope
    constexpr int NVFP4_SCALES_BYTES = 32;      // 28 e4m3 padded to 32

    bool have_topk_length = topk_length.has_value();
    bool have_attn_sink = attn_sink.has_value();

    TORCH_CHECK(b > 0 && s_q > 0 && h_q > 0);
    TORCH_CHECK(h_kv == 1, "MLA requires h_kv == 1 (got ", h_kv, ")");
    TORCH_CHECK(h_q == 128, "head128_nvfp4 requires h_q == 128 (got ", h_q, ")");
    TORCH_CHECK(d_qk == 512, "head128_nvfp4 requires d_qk == 512; got ", d_qk);
    TORCH_CHECK(d_v == 512, "head128_nvfp4 requires d_v == 512; got ", d_v);
    TORCH_CHECK(topk > 0);

    KU_CHECK_DEVICE(q);
    KU_CHECK_DEVICE(kv);
    KU_CHECK_DEVICE(kv_scales);
    KU_CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(topk_length);
    KU_CHECK_DEVICE(attn_sink);

    KU_CHECK_DTYPE(q, torch::kBFloat16);
    TORCH_CHECK(kv.dtype() == torch::kUInt8 || kv.dtype() == torch::kInt8 || kv.dtype() == torch::kFloat8_e4m3fn);
    TORCH_CHECK(kv_scales.dtype() == torch::kUInt8 || kv_scales.dtype() == torch::kInt8 || kv_scales.dtype() == torch::kFloat8_e4m3fn);
    KU_CHECK_DTYPE(indices, torch::kInt32);
    KU_CHECK_DTYPE(topk_length, torch::kInt32);
    KU_CHECK_DTYPE(attn_sink, torch::kFloat32);

    KU_CHECK_LAST_DIM_CONTIGUOUS(q);
    KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(kv_scales);
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    KU_CHECK_CONTIGUOUS(topk_length);
    KU_CHECK_CONTIGUOUS(attn_sink);

    KU_CHECK_SHAPE(q, b, s_q, h_q, d_qk);
    KU_CHECK_SHAPE(kv, num_blocks, page_block_size, h_kv, NVFP4_NOPE_ROPE_BYTES);
    KU_CHECK_SHAPE(kv_scales, num_blocks, page_block_size, h_kv, NVFP4_SCALES_BYTES);
    TORCH_CHECK(kv.stride(1) == NVFP4_NOPE_ROPE_BYTES, "kv tokens must be contiguous; stride(1)=", kv.stride(1));
    TORCH_CHECK(kv_scales.stride(1) == NVFP4_SCALES_BYTES, "kv_scales tokens must be contiguous; stride(1)=", kv_scales.stride(1));
    KU_CHECK_SHAPE(indices, b, s_q, topk);
    KU_CHECK_SHAPE(topk_length, b);
    KU_CHECK_SHAPE(attn_sink, h_q);

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto opts = q.options();
    at::Tensor out = torch::empty({b, s_q, h_q, d_v}, opts);
    at::Tensor lse = torch::empty({b, s_q, h_q}, opts.dtype(at::kFloat));

    std::vector<DecodeFeatures> features;
    features.push_back(DecodeFeatures::HEAD_128);
    features.push_back(DecodeFeatures::HEAD_DIM_512);
    features.push_back(DecodeFeatures::MODEL1_KVCACHE_FORMAT);
    if (have_attn_sink) features.push_back(DecodeFeatures::ATTN_SINK);
    if (have_topk_length) features.push_back(DecodeFeatures::TOPK_LENGTH);

    DecodeImplBase* impl = new Decode_Sm100_Head128_NVFP4_Impl();
    DecodeImplMeta impl_meta = impl->get_meta(h_q, s_q);

    SparseAttnDecodeParams params = {
        b, s_q, h_q, h_kv, d_qk, d_v,
        sm_scale, sm_scale * LOG_2_E,
        num_blocks, page_block_size, topk,
        ModelType::MODEL1,

        (bf16*)q.data_ptr(),
        (bf16*)kv.data_ptr(),
        (int*)indices.data_ptr(),
        ku::get_optional_tensor_ptr<int>(topk_length),
        ku::get_optional_tensor_ptr<float>(attn_sink),
        (float*)lse.data_ptr(),
        (bf16*)out.data_ptr(),

        // No extra KV pool for NVFP4 v1
        0, 0, 0,
        nullptr, nullptr, nullptr,

        // NVFP4: kv_scales buffer
        (uint8_t*)kv_scales.data_ptr(),
        int64_stride_to_int(kv_scales.stride(0)),
        int64_stride_to_int(kv_scales.stride(1)),

        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)), int64_stride_to_int(q.stride(2)),
        int64_stride_to_int(kv.stride(0)), int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),
        int64_stride_to_int(lse.stride(0)), int64_stride_to_int(lse.stride(1)),
        int64_stride_to_int(out.stride(0)), int64_stride_to_int(out.stride(1)), int64_stride_to_int(out.stride(2)),

        0, 0, 0, 0,

        at::cuda::getCurrentCUDAStream().stream()
    };

    at::Tensor o_accum, lse_accum;
    if (!tile_scheduler_metadata.has_value()) {
        tile_scheduler_metadata = torch::empty({impl_meta.num_sm_parts, sizeof(DecodingSchedMeta)/4}, opts.dtype(torch::kInt32));
        num_splits = torch::empty({b+1}, opts.dtype(torch::kInt32));

        GetDecodeSchedMetaParams get_sched_meta_params = {
            b, s_q,
            impl_meta.block_size_topk,
            impl_meta.fixed_overhead_num_blocks,
            topk, 0,
            ku::get_optional_tensor_ptr<int>(topk_length),
            nullptr, nullptr,
            (DecodingSchedMeta*)tile_scheduler_metadata->data_ptr(),
            num_splits->data_ptr<int>(),
            impl_meta.num_sm_parts,
            at::cuda::getCurrentCUDAStream().stream()
        };
        smxx::decode::run_get_decoding_sched_meta_kernel(get_sched_meta_params);
    }
    params.tile_scheduler_metadata_ptr = (DecodingSchedMeta*)tile_scheduler_metadata->data_ptr();
    params.num_splits_ptr = num_splits->data_ptr<int>();
    params.num_sm_parts = impl_meta.num_sm_parts;

    const int total_num_splits = b + impl_meta.num_sm_parts;
    lse_accum = torch::empty({total_num_splits, s_q, h_q}, opts.dtype(at::kFloat));
    o_accum = torch::empty({total_num_splits, s_q, h_q, d_v}, opts.dtype(at::kFloat));
    params.lse_accum = lse_accum.data_ptr<float>();
    params.o_accum = o_accum.data_ptr<float>();
    params.stride_lse_accum_split = int64_stride_to_int(lse_accum.stride(0));
    params.stride_lse_accum_s_q = int64_stride_to_int(lse_accum.stride(1));
    params.stride_o_accum_split = int64_stride_to_int(o_accum.stride(0));
    params.stride_o_accum_s_q = int64_stride_to_int(o_accum.stride(1));
    params.stride_o_accum_h_q = int64_stride_to_int(o_accum.stride(2));

    impl->run(params, features);

    CombineParams combine_params = {
        b, s_q, h_q, d_v,
        params.lse, params.out,
        params.stride_lse_b, params.stride_lse_s_q,
        params.stride_o_b, params.stride_o_s_q, params.stride_o_h_q,
        params.lse_accum, params.o_accum,
        params.stride_lse_accum_split, params.stride_lse_accum_s_q,
        params.stride_o_accum_split, params.stride_o_accum_s_q, params.stride_o_accum_h_q,
        params.tile_scheduler_metadata_ptr, params.num_splits_ptr, params.num_sm_parts,
        ku::get_optional_tensor_ptr<float>(attn_sink),
        at::cuda::getCurrentCUDAStream().stream()
    };
    smxx::decode::run_flash_mla_combine_kernel<bf16>(combine_params);

    delete impl;
    return {out, lse.transpose(1, 2), tile_scheduler_metadata, num_splits};
}
