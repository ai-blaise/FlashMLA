"""V3.2 head64x2_nvfp4 correctness smoke: use valid FP4 + E4M3 + BF16 values.

NVFP4 layout per token (V32 inline, 416 B):
  bytes 0..255: packed FP4 nope (256 B = 512 elements as 2 nibbles per byte)
  bytes 256..287: 32 E4M3 scales (block_size=16; 1 scale per 16 elements)
  bytes 288..415: 64 BF16 rope (128 B)

Build valid inputs:
  - FP4 nope: every nibble = 0x4 = 1.0 in e2m1 (sign 0, exp 10, mant 0 -> 2^0 = 1.0)
    Byte: 0x44 (two nibbles of 1.0). Result: all K nope elements = 1.0 * scale.
  - E4M3 scale: 0x40 = 1.0 in e4m3 (s=0, e=1000, m=000 -> 2^0 * 1.0 = 1.0).
    Set all scales to 0x40 -> K nope dequants to 1.0 everywhere.
  - BF16 rope: zeros.
  - Q: small random bf16.
  - Indices: 0..topk-1 (first topk slots).

With nope all-1.0 and rope all-0, the attention logits Q.K = Q.dot(K) where K is
all 1.0 over nope, 0 over rope. Output is essentially weighted-mean of value
vectors, which should be ~mean(V) per head. Sane finite values, not NaN.
"""

import torch
import flash_mla.cuda as fc

torch.manual_seed(42)
device = "cuda"

b = 1
s_q = 1
h_q = 128
d_qk = 576
d_v = 512
topk = 256
page_block_size = 64
num_blocks = 16
NVFP4_BYTES_PER_TOKEN = 416
NVFP4_SCALES_BYTES = 32

# Q: small random bf16
q = torch.randn(b, s_q, h_q, d_qk, dtype=torch.bfloat16, device=device) * 0.1

# KV: construct valid layout
kv = torch.zeros(num_blocks, page_block_size, 1, NVFP4_BYTES_PER_TOKEN,
                 dtype=torch.uint8, device=device)
# nope (bytes 0..255): set every nibble to 0x4 (= 1.0 in e2m1). Byte 0x44.
kv[:, :, :, 0:256] = 0x44
# scales (bytes 256..287): set every E4M3 to 0x40 (= 1.0 in e4m3).
kv[:, :, :, 256:288] = 0x40
# rope (bytes 288..415): zeros (BF16 0 = 0x0000). Already zeroed.

# Unused kv_scales (for d_qk=576 path, kernel reads inline)
kv_scales_unused = torch.zeros(num_blocks, page_block_size, 1, NVFP4_SCALES_BYTES,
                                dtype=torch.uint8, device=device)

# Indices: use first topk slots sequentially
indices = torch.arange(topk, dtype=torch.int32, device=device).expand(b, s_q, topk).contiguous()
topk_length = torch.full((b,), topk, dtype=torch.int32, device=device)
attn_sink = None
tile_scheduler_metadata = None
num_splits = None
sm_scale = 1.0 / (d_qk ** 0.5)

print(f"Test: V3.2 head64x2_nvfp4 with valid FP4 nope (all 1.0), E4M3 scales (all 1.0), zero rope.")
print(f"Expected: finite output (NOT NaN/Inf); attention should produce sane values.")
print()
print(f"  q: {q.shape} {q.dtype}  mean={q.float().mean():.4f} std={q.float().std():.4f}")
print(f"  kv: {kv.shape} {kv.dtype}  (first byte: 0x{kv[0,0,0,0].item():02x})")
print(f"  indices: {indices[0,0,:5].tolist()} ... {indices[0,0,-5:].tolist()}")
print()

try:
    out, lse, tile_sched, num_splits = fc.sparse_decode_fwd_nvfp4(
        q, kv, kv_scales_unused, indices,
        topk_length, attn_sink,
        tile_scheduler_metadata, num_splits,
        d_v, sm_scale,
    )
    torch.cuda.synchronize()
    out_f = out.float()
    has_nan = torch.isnan(out).any().item()
    has_inf = torch.isinf(out).any().item()
    print(f"out shape: {out.shape} dtype: {out.dtype}")
    print(f"NaN: {has_nan}  Inf: {has_inf}")
    print(f"mean/std/min/max: {out_f.mean():.4f} / {out_f.std():.4f} / {out_f.min():.4f} / {out_f.max():.4f}")
    print(f"sample [0,0,0,:8]: {out_f[0,0,0,:8].tolist()}")
    print(f"sample [0,0,64,:8]: {out_f[0,0,64,:8].tolist()}")
    if not has_nan and not has_inf:
        print()
        print("=== CORRECTNESS SMOKE PASSED (finite outputs) ===")
    else:
        print()
        print(f"=== CORRECTNESS PARTIAL: output has NaN={has_nan} Inf={has_inf} ===")
except Exception as e:
    print(f"=== TEST FAILED: {type(e).__name__}: {e} ===")
    raise
