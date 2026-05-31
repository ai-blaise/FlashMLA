"""Random VALID NVFP4 data test: ensure all bytes are valid (no NaN scales).

E4M3 NaN is 0xFF (and 0x7F for negative-zero-NaN). Avoid these in scales.
E2M1 has NO NaN encoding — all 16 nibble values are valid.
BF16 NaN: avoid 0x7F8x.. or 0xFF8x.. exponent patterns. Use small values.
"""
import torch
import flash_mla.cuda as fc

torch.manual_seed(0)
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

q = torch.randn(b, s_q, h_q, d_qk, dtype=torch.bfloat16, device=device) * 0.1

# Build valid kv
kv = torch.empty(num_blocks, page_block_size, 1, NVFP4_BYTES_PER_TOKEN,
                 dtype=torch.uint8, device=device)
# nope: random bytes (all FP4 nibbles valid -> finite K in [-6, 6])
kv[:, :, :, 0:256] = torch.randint(0, 256, (num_blocks, page_block_size, 1, 256),
                                    dtype=torch.uint8, device=device)
# scales: random uint8 but EXCLUDE NaN patterns. E4M3 NaN is 0xFF. Use 0..0x7E (excludes neg).
kv[:, :, :, 256:288] = torch.randint(0x30, 0x50, (num_blocks, page_block_size, 1, 32),
                                      dtype=torch.uint8, device=device)
# rope: small random bf16. View uint8 as bf16 in chunks of 2 bytes.
rope_bf16 = torch.randn(num_blocks, page_block_size, 1, 64, dtype=torch.bfloat16, device=device) * 0.1
kv[:, :, :, 288:416] = rope_bf16.view(torch.uint8).view(num_blocks, page_block_size, 1, 128)

kv_scales_unused = torch.zeros(num_blocks, page_block_size, 1, NVFP4_SCALES_BYTES,
                                dtype=torch.uint8, device=device)

indices = torch.arange(topk, dtype=torch.int32, device=device).expand(b, s_q, topk).contiguous()
topk_length = torch.full((b,), topk, dtype=torch.int32, device=device)
sm_scale = 1.0 / (d_qk ** 0.5)

print(f"Test: V3.2 head64x2_nvfp4 with RANDOM valid FP4 + bounded E4M3 + random BF16 rope")
print()

out, lse, _, _ = fc.sparse_decode_fwd_nvfp4(
    q, kv, kv_scales_unused, indices,
    topk_length, None, None, None,
    d_v, sm_scale,
)
torch.cuda.synchronize()
out_f = out.float()
has_nan = torch.isnan(out).any().item()
has_inf = torch.isinf(out).any().item()
print(f"out: NaN={has_nan} Inf={has_inf}")
print(f"out mean/std/min/max: {out_f.mean():.4f} / {out_f.std():.4f} / {out_f.min():.4f} / {out_f.max():.4f}")
print(f"sample [0,0,0,:4]: {out_f[0,0,0,:4].tolist()}")
print(f"sample [0,0,127,508:512]: {out_f[0,0,127,508:512].tolist()}")
print(f"lse: NaN={torch.isnan(lse).any().item()} mean={lse.float().mean():.4f}")

if not has_nan and not has_inf:
    print()
    print("=== RANDOM VALID INPUTS: kernel produces finite output ===")
else:
    print()
    print(f"=== ISSUE: NaN={has_nan} Inf={has_inf} ===")
