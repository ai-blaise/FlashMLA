// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek-AI. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// NVFP4 KV variant of the SM100 sparse-MLA decode kernel. Mirrors the FP8
// instantiation under head128/instantiations/phase1_decode_k512.cu — same
// SparseAttnFwdMode + D_QK, just a different namespace + KV format.
#include "../phase1.h"
#include "../phase1.cuh"

namespace sm100::fwd_for_small_topk::head128_nvfp4 {

template void run_fwd_for_small_topk_phase1_kernel<SparseAttnFwdMode::DecodeWithSplitKV, 512>(const SparseAttnDecodeParams& params);

}
