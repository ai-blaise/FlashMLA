"""Deterministic full-NVFP4 dequant correctness for sparse decode."""

import torch

from full_nvfp4_utils import (
    DEVICE, D_QK, make_constant_full_nvfp4, run_kernel,
)


torch.manual_seed(0)
b, s_q, h_q = 1, 1, 128
topk, page_block_size, num_blocks = 256, 64, 16
sm_scale = 1.0 / (D_QK ** 0.5)

q = torch.randn(b, s_q, h_q, D_QK, dtype=torch.bfloat16, device=DEVICE) * 0.01
kv, kv_scales = make_constant_full_nvfp4(num_blocks, page_block_size)
indices = torch.arange(topk, dtype=torch.int32, device=DEVICE).expand(b, s_q, topk).contiguous()
topk_length = torch.full((b,), topk, dtype=torch.int32, device=DEVICE)

out, _ = run_kernel(q, kv, kv_scales, indices, topk_length, sm_scale)
of = out.float()
has_nan = torch.isnan(out).any().item()
has_inf = torch.isinf(out).any().item()
err = abs(of.mean().item() - 1.0)
print("K/V dequant should equal 1.0 for FP4 0x2 with E4M3 scale 0x38.")
print(f"out: NaN={has_nan} Inf={has_inf}")
print(f"mean/std/min/max: {of.mean():.6f} / {of.std():.6f} / {of.min():.4f} / {of.max():.4f}")
print(f"sample [0,0,0,:8]: {of[0,0,0,:8].tolist()}")
print(f"abs err from expected 1.0: {err:.6f}")
passed = (not has_nan) and (not has_inf) and err < 0.05
print(f"=== {'PASS' if passed else 'FAIL'} ===")
assert passed
