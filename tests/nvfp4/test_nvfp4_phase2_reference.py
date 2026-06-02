"""Full-NVFP4 V32 reference correctness for sparse decode."""

import torch

from full_nvfp4_utils import (
    DEVICE, D_QK, assert_reference_close, make_random_full_nvfp4,
    reference_attention, run_kernel,
)


torch.manual_seed(456)
b, s_q, h_q = 1, 1, 128
topk = 128
page_block_size = 64
num_blocks = 4
sm_scale = 1.0 / (D_QK ** 0.5)

q = torch.randn(b, s_q, h_q, D_QK, dtype=torch.bfloat16, device=DEVICE) * 0.05
kv, kv_scales = make_random_full_nvfp4(num_blocks, page_block_size)
indices = torch.randperm(num_blocks * page_block_size, device=DEVICE)[:topk]
indices = indices.int().expand(b, s_q, topk).contiguous()
topk_length = torch.full((b,), topk, dtype=torch.int32, device=DEVICE)

print("Phase 2 reference test: kernel vs Python full-NVFP4 reference")
print(f"  q={tuple(q.shape)} kv={tuple(kv.shape)} indices={tuple(indices.shape)}")

out_kernel, _ = run_kernel(q, kv, kv_scales, indices, topk_length, sm_scale)
out_ref = reference_attention(q, kv, kv_scales, indices, sm_scale)
kernel = out_kernel.float()
ref = out_ref.float()
passed, abs_diff, rms, cos = assert_reference_close(kernel, ref)
print(f"Kernel mean/std: {kernel.mean():.4f} / {kernel.std():.4f}")
print(f"Ref mean/std: {ref.mean():.4f} / {ref.std():.4f}")
print(f"Abs error: max={abs_diff.max():.6f} mean={abs_diff.mean():.6f} median={abs_diff.median():.6f}")
print(f"RMS error: {rms:.6f}")
print(f"Cosine similarity: {cos:.6f}")
print(f"Sample [0,0,0,:4] kernel: {kernel[0,0,0,:4].tolist()}")
print(f"Sample [0,0,0,:4] ref:    {ref[0,0,0,:4].tolist()}")
print(f"=== {'PASS' if passed else 'FAIL'} (max abs < 0.5, RMS < 0.065, cosine > 0.994) ===")
assert passed
