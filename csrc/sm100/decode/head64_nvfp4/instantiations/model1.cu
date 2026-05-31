#include "../kernel.cuh"

namespace sm100::decode::head64_nvfp4 {

template
void run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::MODEL1>(const SparseAttnDecodeParams &params);

}
