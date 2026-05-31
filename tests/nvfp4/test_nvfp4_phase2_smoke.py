"""Phase 2 smoke test: hand-crafted 324 B/token NVFP4 layout.

Per-token layout:
  0..255   : packed FP4 nope (256 B)
  256..287 : 32 E4M3 nope scales (32 B)
  288..319 : packed FP4 rope (32 B)
  320..323 : 4 E4M3 rope scales (4 B)
"""
import torch
import flash_mla.cuda as fc

torch.manual_seed(0)
device = "cuda"

b, s_q, h_q, d_qk, d_v = 1, 1, 128, 576, 512
topk, page_block_size, num_blocks = 256, 64, 16
BYTES = 336  # 324 payload + 12 padding (16B TMA stride alignment)

q = torch.randn(b, s_q, h_q, d_qk, dtype=torch.bfloat16, device=device) * 0.01

kv = torch.empty(num_blocks, page_block_size, 1, BYTES, dtype=torch.uint8, device=device)
# nope FP4: 0x22 nibbles = 1.0
kv[:, :, :, 0:256] = 0x22
# nope E4M3 scales: 0x38 = 1.0
kv[:, :, :, 256:288] = 0x38
# rope FP4: 0x22 = 1.0
kv[:, :, :, 288:320] = 0x22
# rope E4M3 scales: 0x38 = 1.0
kv[:, :, :, 320:324] = 0x38

kv_scales_unused = torch.zeros(num_blocks, page_block_size, 1, 32, dtype=torch.uint8, device=device)
indices = torch.arange(topk, dtype=torch.int32, device=device).expand(b, s_q, topk).contiguous()
topk_length = torch.full((b,), topk, dtype=torch.int32, device=device)
sm_scale = 1.0 / (d_qk ** 0.5)

print(f"Phase 2 Test: 324 B/token NVFP4 layout")
print(f"  K nope dequant = 1.0 * 1.0 = 1.0; K rope dequant = 1.0 * 1.0 = 1.0")
print(f"  K = all 1.0 (576-d); Q.K = ~small, softmax uniform; out = mean(V) = 1.0")
print()

try:
    out, lse, _, _ = fc.sparse_decode_fwd_nvfp4(
        q, kv, kv_scales_unused, indices,
        topk_length, None, None, None,
        d_v, sm_scale,
    )
    torch.cuda.synchronize()
    of = out.float()
    has_nan = torch.isnan(out).any().item()
    has_inf = torch.isinf(out).any().item()
    print(f"out: NaN={has_nan} Inf={has_inf}")
    print(f"mean/std/min/max: {of.mean():.6f} / {of.std():.6f} / {of.min():.4f} / {of.max():.4f}")
    print(f"sample [0,0,0,:8]: {of[0,0,0,:8].tolist()}")
    err = abs(of.mean().item() - 1.0)
    print(f"abs err from expected 1.0: {err:.6f}")
    print(f"=== {'PASS' if (not has_nan) and (not has_inf) and err < 0.05 else 'FAIL'} ===")
except Exception as e:
    print(f"=== EXCEPTION: {type(e).__name__}: {e} ===")
    raise
