"""Benchmark NVFP4 Phase 2 (336 B/slot) vs FP8 baseline (656 B/slot) kernel latency.

Measures per-call kernel time at production-like shapes:
  - b * s_q = 64 (batch * query tokens; we use b=64, s_q=1 for decode)
  - h_q=128, d_qk=576, d_v=512
  - topk = 1024 (DSA sparse-MLA window)
  - page_block_size = 64

Bandwidth math (per-call HBM reads for K):
  - topk * bytes_per_slot
  - FP8:    1024 * 656 = 671 KB / call
  - NVFP4:  1024 * 336 = 344 KB / call
  - Reduction: 48.8%

Latency expectation (memory-bound): 1 - 344/671 = 48.8% speedup on K-load portion.
Whole-kernel speedup will be smaller (compute portion doesn't speed up).
"""
import torch
import flash_mla.cuda as fc
import time

device = "cuda"
torch.manual_seed(0)

# Production shape
b, s_q, h_q, d_qk, d_v = 64, 1, 128, 576, 512
topk = 1024
page_block_size = 64
num_blocks = (b * s_q * topk + page_block_size - 1) // page_block_size * 4  # plenty of headroom

# ===== NVFP4 Phase 2 setup (336 B/slot) =====
print(f"Configuration:")
print(f"  b={b}, s_q={s_q}, h_q={h_q}, d_qk={d_qk}, d_v={d_v}, topk={topk}")
print(f"  page_block_size={page_block_size}, num_blocks={num_blocks}")
print()

NVFP4_BYTES = 336
FP8_BYTES   = 656

q_nvfp4 = torch.randn(b, s_q, h_q, d_qk, dtype=torch.bfloat16, device=device) * 0.01
kv_nvfp4 = torch.randint(0x10, 0x40, (num_blocks, page_block_size, 1, NVFP4_BYTES),
                          dtype=torch.uint8, device=device)
kv_scales_nvfp4 = torch.zeros(num_blocks, page_block_size, 1, 32, dtype=torch.uint8, device=device)
indices = torch.randint(0, num_blocks*page_block_size, (b, s_q, topk), dtype=torch.int32, device=device)
topk_length = torch.full((b,), topk, dtype=torch.int32, device=device)
sm_scale = 1.0 / (d_qk ** 0.5)

# Warmup
for _ in range(3):
    out, _, _, _ = fc.sparse_decode_fwd_nvfp4(
        q_nvfp4, kv_nvfp4, kv_scales_nvfp4, indices,
        topk_length, None, None, None,
        d_v, sm_scale,
    )
torch.cuda.synchronize()

# Time NVFP4 Phase 2
N_ITERS = 50
t0 = time.perf_counter()
for _ in range(N_ITERS):
    out, _, _, _ = fc.sparse_decode_fwd_nvfp4(
        q_nvfp4, kv_nvfp4, kv_scales_nvfp4, indices,
        topk_length, None, None, None,
        d_v, sm_scale,
    )
torch.cuda.synchronize()
t_nvfp4 = (time.perf_counter() - t0) / N_ITERS * 1e6  # us per call

# ===== FP8 baseline setup =====
kv_fp8 = torch.randint(0x10, 0x40, (num_blocks, page_block_size, 1, FP8_BYTES),
                       dtype=torch.uint8, device=device)
for _ in range(3):
    try:
        result = fc.sparse_decode_fwd(
            q_nvfp4, kv_fp8, indices,
            topk_length, None, None, None,
            None, None, None,
            d_v, sm_scale,
        )
    except Exception as e:
        print(f"FP8 path error: {e}")
        result = None
        break
torch.cuda.synchronize()

if result is not None:
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        result = fc.sparse_decode_fwd(
            q_nvfp4, kv_fp8, indices,
            topk_length, None, None, None,
            None, None, None,
            d_v, sm_scale,
        )
    torch.cuda.synchronize()
    t_fp8 = (time.perf_counter() - t0) / N_ITERS * 1e6
else:
    t_fp8 = None

# ===== Report =====
hbm_nvfp4 = topk * NVFP4_BYTES * b  # K bytes read per call
hbm_fp8   = topk * FP8_BYTES   * b

print(f"Results (mean over {N_ITERS} iters):")
print(f"  NVFP4 Phase 2: {t_nvfp4:8.2f} us/call  ({hbm_nvfp4/1024/1024:6.2f} MB K-reads/call)")
if t_fp8 is not None:
    print(f"  FP8 baseline:  {t_fp8:8.2f} us/call  ({hbm_fp8/1024/1024:6.2f} MB K-reads/call)")
    print(f"  Speedup:       {t_fp8/t_nvfp4:6.2f}x  (HBM ratio: {hbm_nvfp4/hbm_fp8:.3f})")
print()

# Theoretical peak: B200 HBM ~8 TB/s
peak_bw_GBs = 8000
t_hbm_nvfp4_us = hbm_nvfp4 / (peak_bw_GBs * 1e9) * 1e6
print(f"Theoretical HBM-bound time NVFP4 (8 TB/s): {t_hbm_nvfp4_us:.2f} us/call")
print(f"Achieved/theoretical NVFP4: {t_hbm_nvfp4_us/t_nvfp4*100:.1f}% of HBM-bound limit")
