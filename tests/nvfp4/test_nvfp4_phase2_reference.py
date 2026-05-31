"""Phase 2 reference correctness: Python NVFP4 (FP4 nope + FP4 rope) vs kernel.

Per-token layout (336 B padded from 324):
  0..255   : FP4 nope (512 elems packed in 256 B)
  256..287 : 32 E4M3 nope scales
  288..319 : FP4 rope (64 elems packed in 32 B)
  320..323 : 4 E4M3 rope scales
  324..335 : padding
"""
import torch
import flash_mla.cuda as fc

device = "cuda"

E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32, device=device)

def dequant_fp4_packed(packed_bytes):
    low = packed_bytes & 0xF
    high = (packed_bytes >> 4) & 0xF
    out = torch.empty(*packed_bytes.shape[:-1], packed_bytes.shape[-1] * 2,
                      dtype=torch.float32, device=packed_bytes.device)
    out[..., 0::2] = E2M1_TABLE[low.long()]
    out[..., 1::2] = E2M1_TABLE[high.long()]
    return out

def dequant_e4m3(byte_tensor):
    return byte_tensor.view(torch.float8_e4m3fn).float()

def reference_attention(q, kv_bytes, indices, sm_scale, d_v=512, d_nope=512, d_rope=64,
                         num_scales=32, num_rope_scales=4, block_size=16):
    b, s_q, h_q, d_qk = q.shape
    num_blocks, page_size, h_kv, _ = kv_bytes.shape
    topk = indices.shape[-1]
    out = torch.zeros(b, s_q, h_q, d_v, dtype=torch.float32, device=q.device)
    for bi in range(b):
        for sqi in range(s_q):
            tok = indices[bi, sqi, :]
            block_idx = tok // page_size
            in_block = tok % page_size
            gathered = kv_bytes[block_idx, in_block, 0, :]  # (topk, 336)

            # nope dequant
            nope = dequant_fp4_packed(gathered[:, 0:256])
            nope_scales = dequant_e4m3(gathered[:, 256:288])
            nope_scaled = nope * nope_scales.repeat_interleave(block_size, dim=-1)

            # rope dequant
            rope = dequant_fp4_packed(gathered[:, 288:320])  # (topk, 64)
            rope_scales = dequant_e4m3(gathered[:, 320:324])  # (topk, 4)
            rope_scaled = rope * rope_scales.repeat_interleave(block_size, dim=-1)

            k = torch.cat([nope_scaled, rope_scaled], dim=-1)  # (topk, 576)
            q_vec = q[bi, sqi].float()
            logits = q_vec @ k.transpose(0, 1) * sm_scale
            attn = torch.softmax(logits, dim=-1)
            v = nope_scaled
            out[bi, sqi] = attn @ v
    return out.bfloat16()


torch.manual_seed(456)
b, s_q, h_q, d_qk, d_v = 1, 1, 128, 576, 512
topk = 128
page_block_size = 64
num_blocks = 4
BYTES = 336

q = torch.randn(b, s_q, h_q, d_qk, dtype=torch.bfloat16, device=device) * 0.05

kv = torch.empty(num_blocks, page_block_size, 1, BYTES, dtype=torch.uint8, device=device)
# nope FP4 random
kv[:, :, :, 0:256] = torch.randint(0, 256, (num_blocks, page_block_size, 1, 256),
                                    dtype=torch.uint8, device=device)
# nope E4M3 scales bounded (avoid 0xFF NaN)
kv[:, :, :, 256:288] = torch.randint(0x30, 0x48, (num_blocks, page_block_size, 1, 32),
                                      dtype=torch.uint8, device=device)
# rope FP4 random
kv[:, :, :, 288:320] = torch.randint(0, 256, (num_blocks, page_block_size, 1, 32),
                                      dtype=torch.uint8, device=device)
# rope E4M3 scales bounded
kv[:, :, :, 320:324] = torch.randint(0x30, 0x48, (num_blocks, page_block_size, 1, 4),
                                      dtype=torch.uint8, device=device)
# padding 324..335 (don't care)
kv[:, :, :, 324:336] = 0

kv_scales_unused = torch.zeros(num_blocks, page_block_size, 1, 32, dtype=torch.uint8, device=device)
indices = torch.randperm(num_blocks * page_block_size, device=device)[:topk].int().expand(b, s_q, topk).contiguous()
topk_length = torch.full((b,), topk, dtype=torch.int32, device=device)
sm_scale = 1.0 / (d_qk ** 0.5)

print(f"Phase 2 reference test: kernel vs Python NVFP4 (FP4 nope + FP4 rope)")
print(f"  Shape: q={tuple(q.shape)}, kv={tuple(kv.shape)}, indices={tuple(indices.shape)}")
print()

out_kernel, lse_kernel, _, _ = fc.sparse_decode_fwd_nvfp4(
    q, kv, kv_scales_unused, indices,
    topk_length, None, None, None,
    d_v, sm_scale,
)
torch.cuda.synchronize()
out_kernel_f = out_kernel.float()

out_ref = reference_attention(q, kv, indices, sm_scale)
out_ref_f = out_ref.float()

print(f"Kernel mean/std: {out_kernel_f.mean():.4f} / {out_kernel_f.std():.4f}")
print(f"Ref mean/std: {out_ref_f.mean():.4f} / {out_ref_f.std():.4f}")

abs_diff = (out_kernel_f - out_ref_f).abs()
rms = (abs_diff**2).mean().sqrt()

print()
print(f"Abs error: max={abs_diff.max():.6f} mean={abs_diff.mean():.6f} median={abs_diff.median():.6f}")
print(f"RMS error: {rms:.6f}")
print(f"Sample [0,0,0,:4] kernel: {out_kernel_f[0,0,0,:4].tolist()}")
print(f"Sample [0,0,0,:4] ref:    {out_ref_f[0,0,0,:4].tolist()}")
print()
passed = abs_diff.max().item() < 0.5 and rms.item() < 0.05
print(f"=== {'PASS' if passed else 'FAIL'} (max abs < 0.5, RMS < 0.05) ===")
