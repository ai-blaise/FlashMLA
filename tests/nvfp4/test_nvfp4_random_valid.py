"""Random finite full-NVFP4 sparse-decode input smoke."""

import torch

from full_nvfp4_utils import (
    DEVICE, D_QK, make_random_full_nvfp4, run_kernel,
)


torch.manual_seed(0)
b, s_q, h_q = 1, 1, 128
topk, page_block_size, num_blocks = 256, 64, 16
sm_scale = 1.0 / (D_QK ** 0.5)

q = torch.randn(b, s_q, h_q, D_QK, dtype=torch.bfloat16, device=DEVICE) * 0.1
kv = make_random_full_nvfp4(num_blocks, page_block_size)
indices = torch.arange(topk, dtype=torch.int32, device=DEVICE).expand(b, s_q, topk).contiguous()
topk_length = torch.full((b,), topk, dtype=torch.int32, device=DEVICE)

out, lse = run_kernel(q, kv, indices, topk_length, sm_scale)
out_f = out.float()
has_nan = torch.isnan(out).any().item()
has_inf = torch.isinf(out).any().item()
print("Test: V32 head64x2_nvfp4 with random finite full-NVFP4 bytes")
print(f"out: NaN={has_nan} Inf={has_inf}")
print(f"out mean/std/min/max: {out_f.mean():.4f} / {out_f.std():.4f} / {out_f.min():.4f} / {out_f.max():.4f}")
print(f"sample [0,0,0,:4]: {out_f[0,0,0,:4].tolist()}")
print(f"lse: NaN={torch.isnan(lse).any().item()} mean={lse.float().mean():.4f}")
passed = not has_nan and not has_inf and not torch.isnan(lse).any().item()
print(f"=== {'PASS' if passed else 'FAIL'} ===")
assert passed
