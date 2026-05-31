"""Deterministic NVFP4 dequant correctness:
- nope FP4 = 0x2 (= 1.0 in e2m1) in every nibble
- scales E4M3 = 0x38 (= 1.0 in e4m3)
- rope = 0
- expected K_dequant = 1.0; V = K nope = 1.0
- Q.K = sum(Q) = small random sum; softmax ~ uniform; output ~ mean(V) = 1.0
"""
import torch
import flash_mla.cuda as fc

torch.manual_seed(0)
device = "cuda"

b, s_q, h_q, d_qk, d_v = 1, 1, 128, 576, 512
topk, page_block_size, num_blocks = 256, 64, 16
BYTES = 416

q = torch.randn(b, s_q, h_q, d_qk, dtype=torch.bfloat16, device=device) * 0.01  # very small for near-uniform softmax
kv = torch.empty(num_blocks, page_block_size, 1, BYTES, dtype=torch.uint8, device=device)
kv[:, :, :, 0:256] = 0x22  # FP4 nibbles 0x2 = 1.0
kv[:, :, :, 256:288] = 0x38  # E4M3 0x38 = 1.0
kv[:, :, :, 288:416] = 0    # rope zeros
kv_scales_unused = torch.zeros(num_blocks, page_block_size, 1, 32, dtype=torch.uint8, device=device)
indices = torch.arange(topk, dtype=torch.int32, device=device).expand(b, s_q, topk).contiguous()
topk_length = torch.full((b,), topk, dtype=torch.int32, device=device)
sm_scale = 1.0 / (d_qk ** 0.5)

out, lse, _, _ = fc.sparse_decode_fwd_nvfp4(
    q, kv, kv_scales_unused, indices,
    topk_length, None, None, None,
    d_v, sm_scale,
)
torch.cuda.synchronize()
out_f = out.float()
print(f"K dequant should = 1.0 (FP4 0x2 = 1.0 * E4M3 0x38 = 1.0).")
print(f"V = K nope = 1.0 everywhere. Q.K logits ~ small; softmax ~ uniform.")
print(f"Expected output ~ 1.0 everywhere.")
print()
print(f"actual out [0,0,0,:8]: {out_f[0,0,0,:8].tolist()}")
print(f"actual out mean: {out_f.mean():.6f}")
print(f"actual out std:  {out_f.std():.6f}")
expected = 1.0
err = abs(out_f.mean().item() - expected)
print(f"abs error from 1.0: {err:.6f}")
print(f"=== {'PASS' if err < 0.05 else 'FAIL'} (threshold 0.05) ===")
