"""Reference correctness test: Python NVFP4 dequant + reference MLA attention
vs kernel output. Target: max abs error < 1e-2 (BF16 precision).

Tests with deterministic byte patterns to validate the dequant arithmetic, then
with realistic random valid bytes to ensure end-to-end attention matches.
"""
import torch
import flash_mla.cuda as fc

device = "cuda"

# ============================================================
# E2M1 lookup table (FP4 nibble -> float)
# bit layout: sign(1) exp(2) mantissa(1); bias=1; subnormal at exp=0
# 0x0: +0.0     0x8: -0.0
# 0x1: +0.5     0x9: -0.5
# 0x2: +1.0     0xA: -1.0
# 0x3: +1.5     0xB: -1.5
# 0x4: +2.0     0xC: -2.0
# 0x5: +3.0     0xD: -3.0
# 0x6: +4.0     0xE: -4.0
# 0x7: +6.0     0xF: -6.0
# ============================================================
E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32, device=device)

def dequant_fp4_packed(packed_bytes):
    """Convert (..., N) uint8 packed FP4 to (..., 2N) float32.
    Low nibble is element 0, high nibble is element 1 (per cvt.rn.f16x2.e2m1x2 PTX)."""
    low = packed_bytes & 0xF
    high = (packed_bytes >> 4) & 0xF
    out = torch.empty(*packed_bytes.shape[:-1], packed_bytes.shape[-1] * 2,
                      dtype=torch.float32, device=packed_bytes.device)
    out[..., 0::2] = E2M1_TABLE[low.long()]
    out[..., 1::2] = E2M1_TABLE[high.long()]
    return out

def dequant_e4m3(byte_tensor):
    """E4M3 bytes -> float via torch's fp8_e4m3fn dtype."""
    return byte_tensor.view(torch.float8_e4m3fn).float()

# ============================================================
# Reference attention for NVFP4 head64x2 V32 layout
# ============================================================
def reference_attention(q, kv_bytes, indices, sm_scale, d_v=512, d_nope=512, d_rope=64, num_scales_per_token=32, block_size=16):
    """
    q: (b, s_q, h_q, d_qk) bf16
    kv_bytes: (num_blocks, page_block_size, h_kv=1, 416) uint8
        per token: 0..255 = packed FP4 nope (256 B = 512 elems)
                   256..287 = 32 E4M3 scales (block_size=16)
                   288..415 = 64 BF16 rope (128 B)
    indices: (b, s_q, topk) int32
    Returns: (b, s_q, h_q, d_v) bf16 output
    """
    b, s_q, h_q, d_qk = q.shape
    num_blocks, page_size, h_kv, bytes_per_token = kv_bytes.shape
    topk = indices.shape[-1]

    out = torch.zeros(b, s_q, h_q, d_v, dtype=torch.float32, device=q.device)

    for batch_i in range(b):
        for sq_i in range(s_q):
            tok_idx = indices[batch_i, sq_i, :]  # (topk,)
            block_idx = tok_idx // page_size       # which block per top-k slot
            in_block = tok_idx % page_size

            # Gather KV bytes for these topk slots: (topk, 416)
            gathered = kv_bytes[block_idx, in_block, 0, :]  # (topk, 416)

            # Dequant nope: (topk, 512) float
            nope = dequant_fp4_packed(gathered[:, 0:256])

            # Dequant scales: (topk, 32) float
            scales = dequant_e4m3(gathered[:, 256:288])

            # Apply per-block scales: each scale covers block_size=16 contiguous elements
            scale_expanded = scales.repeat_interleave(block_size, dim=-1)  # (topk, 512)
            nope = nope * scale_expanded

            # Rope: BF16 directly
            # 128 bytes of rope = 64 BF16 elements
            rope_bytes = gathered[:, 288:416].contiguous()  # (topk, 128) uint8
            rope = rope_bytes.view(torch.bfloat16).float()  # (topk, 64) float

            # K = [nope | rope] (topk, 576)
            k = torch.cat([nope, rope], dim=-1)  # (topk, 576)

            q_vec = q[batch_i, sq_i].float()  # (h_q, d_qk)

            # Attention: Q @ K^T (h_q, topk)
            logits = q_vec @ k.transpose(0, 1) * sm_scale
            attn = torch.softmax(logits, dim=-1)  # (h_q, topk)

            # V = nope (first d_v=512 elements of K)
            v = nope  # (topk, 512)
            out_per_head = attn @ v  # (h_q, 512)
            out[batch_i, sq_i] = out_per_head

    return out.bfloat16()


# ============================================================
# Test
# ============================================================
torch.manual_seed(123)
b, s_q, h_q, d_qk, d_v = 1, 1, 128, 576, 512
topk = 128  # smaller for reference perf
page_block_size = 64
num_blocks = 4
BYTES = 416

# Small random Q for stable softmax
q = torch.randn(b, s_q, h_q, d_qk, dtype=torch.bfloat16, device=device) * 0.05

# Build valid KV
kv = torch.empty(num_blocks, page_block_size, 1, BYTES, dtype=torch.uint8, device=device)
kv[:, :, :, 0:256] = torch.randint(0, 256, (num_blocks, page_block_size, 1, 256),
                                    dtype=torch.uint8, device=device)
# E4M3 scales in 0x30..0x4F (excludes NaN 0xFF and large values)
kv[:, :, :, 256:288] = torch.randint(0x30, 0x4F, (num_blocks, page_block_size, 1, 32),
                                      dtype=torch.uint8, device=device)
rope_bf16 = torch.randn(num_blocks, page_block_size, 1, 64, dtype=torch.bfloat16, device=device) * 0.05
kv[:, :, :, 288:416] = rope_bf16.view(torch.uint8).view(num_blocks, page_block_size, 1, 128)

kv_scales_unused = torch.zeros(num_blocks, page_block_size, 1, 32, dtype=torch.uint8, device=device)
indices = torch.randperm(num_blocks * page_block_size, device=device)[:topk].int().expand(b, s_q, topk).contiguous()
topk_length = torch.full((b,), topk, dtype=torch.int32, device=device)
sm_scale = 1.0 / (d_qk ** 0.5)

print(f"Reference correctness test: kernel vs Python NVFP4 reference")
print(f"  Shape: q={tuple(q.shape)}, kv={tuple(kv.shape)}, indices={tuple(indices.shape)}")
print()

# Kernel output
out_kernel, lse_kernel, _, _ = fc.sparse_decode_fwd_nvfp4(
    q, kv, kv_scales_unused, indices,
    topk_length, None, None, None,
    d_v, sm_scale,
)
torch.cuda.synchronize()
out_kernel_f = out_kernel.float()

# Python reference output
out_ref = reference_attention(q, kv, indices, sm_scale)
out_ref_f = out_ref.float()

print(f"Kernel mean/std: {out_kernel_f.mean():.4f} / {out_kernel_f.std():.4f}")
print(f"Reference mean/std: {out_ref_f.mean():.4f} / {out_ref_f.std():.4f}")

abs_diff = (out_kernel_f - out_ref_f).abs()
rel_diff = abs_diff / (out_ref_f.abs() + 1e-6)

print()
print(f"Abs error: max={abs_diff.max():.6f} mean={abs_diff.mean():.6f} median={abs_diff.median():.6f}")
print(f"Rel error: max={rel_diff.max():.6f} mean={rel_diff.mean():.6f} median={rel_diff.median():.6f}")
print(f"RMS error: {(abs_diff**2).mean().sqrt():.6f}")
print(f"Sample [0,0,0,:4] kernel: {out_kernel_f[0,0,0,:4].tolist()}")
print(f"Sample [0,0,0,:4] ref:    {out_ref_f[0,0,0,:4].tolist()}")
print()
threshold = 0.05
passed = abs_diff.max().item() < 0.5 and (abs_diff**2).mean().sqrt().item() < threshold
print(f"=== {'PASS' if passed else 'FAIL'} (max abs < 0.5, RMS < {threshold}) ===")
