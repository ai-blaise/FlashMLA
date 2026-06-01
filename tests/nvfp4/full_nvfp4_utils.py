"""Shared helpers for V32 full-NVFP4 sparse decode tests."""

import torch
import flash_mla.cuda as fc

DEVICE = "cuda"
D_QK = 576
D_V = 512
BYTES_PER_TOKEN = 336
PACKED_SCORE_BYTES = 288
SCALE_BYTES = 36
PADDING_OFFSET = PACKED_SCORE_BYTES + SCALE_BYTES
BLOCK_SIZE = 16

E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32, device=DEVICE)


def dequant_fp4_packed(packed_bytes):
    low = packed_bytes & 0xF
    high = (packed_bytes >> 4) & 0xF
    out = torch.empty(
        *packed_bytes.shape[:-1],
        packed_bytes.shape[-1] * 2,
        dtype=torch.float32,
        device=packed_bytes.device,
    )
    out[..., 0::2] = E2M1_TABLE[low.long()]
    out[..., 1::2] = E2M1_TABLE[high.long()]
    return out


def dequant_e4m3(byte_tensor):
    return byte_tensor.view(torch.float8_e4m3fn).float()


def reference_attention(q, kv_bytes, indices, sm_scale):
    b, s_q, h_q, _ = q.shape
    _, page_size, _, _ = kv_bytes.shape
    out = torch.zeros(b, s_q, h_q, D_V, dtype=torch.float32, device=q.device)
    for batch_i in range(b):
        for query_i in range(s_q):
            tok = indices[batch_i, query_i]
            block_idx = tok // page_size
            in_block = tok % page_size
            gathered = kv_bytes[block_idx, in_block, 0, :]
            score = dequant_fp4_packed(gathered[:, :PACKED_SCORE_BYTES])
            scales = dequant_e4m3(gathered[:, PACKED_SCORE_BYTES:PADDING_OFFSET])
            k = score * scales.repeat_interleave(BLOCK_SIZE, dim=-1)
            logits = q[batch_i, query_i].float() @ k.transpose(0, 1) * sm_scale
            attn = torch.softmax(logits, dim=-1)
            out[batch_i, query_i] = attn @ k[:, :D_V]
    return out.bfloat16()


def make_random_full_nvfp4(num_blocks, page_block_size):
    kv = torch.empty(num_blocks, page_block_size, 1, BYTES_PER_TOKEN,
                     dtype=torch.uint8, device=DEVICE)
    kv[:, :, :, :PACKED_SCORE_BYTES] = torch.randint(
        0, 256, (num_blocks, page_block_size, 1, PACKED_SCORE_BYTES),
        dtype=torch.uint8, device=DEVICE)
    kv[:, :, :, PACKED_SCORE_BYTES:PADDING_OFFSET] = torch.randint(
        0x30, 0x48, (num_blocks, page_block_size, 1, SCALE_BYTES),
        dtype=torch.uint8, device=DEVICE)
    kv[:, :, :, PADDING_OFFSET:] = 0
    return kv


def make_constant_full_nvfp4(num_blocks, page_block_size, fp4_byte=0x22, scale_byte=0x38):
    kv = torch.empty(num_blocks, page_block_size, 1, BYTES_PER_TOKEN,
                     dtype=torch.uint8, device=DEVICE)
    kv[:, :, :, :PACKED_SCORE_BYTES] = fp4_byte
    kv[:, :, :, PACKED_SCORE_BYTES:PADDING_OFFSET] = scale_byte
    kv[:, :, :, PADDING_OFFSET:] = 0
    return kv


def run_kernel(q, kv, indices, topk_length, sm_scale):
    kv_scales_unused = torch.zeros(kv.shape[0], kv.shape[1], 1, 32,
                                   dtype=torch.uint8, device=DEVICE)
    out, lse, _, _ = fc.sparse_decode_fwd_nvfp4(
        q, kv, kv_scales_unused, indices,
        topk_length, None, None, None,
        D_V, sm_scale,
    )
    torch.cuda.synchronize()
    return out, lse


def assert_reference_close(kernel, ref):
    abs_diff = (kernel - ref).abs()
    rms = abs_diff.square().mean().sqrt()
    cos = torch.nn.functional.cosine_similarity(kernel.flatten(), ref.flatten(), dim=0)
    passed = abs_diff.max().item() < 0.5 and rms.item() < 0.065 and cos.item() > 0.994
    return passed, abs_diff, rms, cos
