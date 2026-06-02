"""Bench iter17 with pre-allocated tile_scheduler_metadata + num_splits.

The API allocates fresh tile_scheduler_metadata and num_splits per call when
they are not provided. These torch::empty calls contribute substantial per-call
overhead. Production code typically pre-allocates these and reuses them.
"""

import argparse
import json
from pathlib import Path

import torch
import flash_mla.cuda as fc

from full_nvfp4_utils import DEVICE, D_QK, D_V, make_random_full_nvfp4


H_Q = 128
PAGE_BLOCK_SIZE = 64
FP8_BYTES_PER_TOKEN = 656
NVFP4_BYTES_PER_TOKEN = 336


def _time_cuda(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    values = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        values.append(start.elapsed_time(end) * 1000.0 / iters)
    values.sort()
    return values[0], values[len(values) // 2], values


def _make_common_inputs(batch_size, topk):
    num_blocks = max((batch_size * topk + PAGE_BLOCK_SIZE - 1) // PAGE_BLOCK_SIZE * 2, 64)
    q = torch.randn(batch_size, 1, H_Q, D_QK, dtype=torch.bfloat16, device=DEVICE) * 0.01
    indices = torch.randint(
        0, num_blocks * PAGE_BLOCK_SIZE,
        (batch_size, 1, topk), dtype=torch.int32, device=DEVICE,
    )
    topk_length = torch.full((batch_size,), topk, dtype=torch.int32, device=DEVICE)
    return num_blocks, q, indices, topk_length, 1.0 / (D_QK ** 0.5)


def _make_fp8_cache(num_blocks):
    kv = torch.zeros(num_blocks, PAGE_BLOCK_SIZE, 1, FP8_BYTES_PER_TOKEN, dtype=torch.uint8, device=DEVICE)
    kv[:, :, :, :512] = 0x38
    kv[:, :, :, 512:528].view(torch.float32).fill_(1.0)
    kv[:, :, :, 528:] = 0
    return kv


def time_case(batch_size, topk, warmup, iters):
    num_blocks, q, indices, topk_length, sm_scale = _make_common_inputs(batch_size, topk)
    nvfp4_kv, nvfp4_scales = make_random_full_nvfp4(num_blocks, PAGE_BLOCK_SIZE)
    fp8_kv = _make_fp8_cache(num_blocks)
    kv_scales_unused = torch.zeros(num_blocks, PAGE_BLOCK_SIZE, 1, 32, dtype=torch.uint8, device=DEVICE)

    # === Baseline: NO pre-allocated metadata ===
    def run_nvfp4_noprealloc():
        fc.sparse_decode_fwd_nvfp4(
            q, nvfp4_kv, nvfp4_scales, indices, topk_length,
            None, None, None, D_V, sm_scale)

    def run_fp8_noprealloc():
        fc.sparse_decode_fwd(
            q, fp8_kv, indices, topk_length, None,
            None, None, None, None, None, D_V, sm_scale)

    # === Pre-allocate metadata once and reuse ===
    # First call to get the meta shape and trigger num_splits computation
    _out, _lse, nv_sched_meta, nv_splits = fc.sparse_decode_fwd_nvfp4(
        q, nvfp4_kv, nvfp4_scales, indices, topk_length,
        None, None, None, D_V, sm_scale)

    def run_nvfp4_prealloc():
        fc.sparse_decode_fwd_nvfp4(
            q, nvfp4_kv, nvfp4_scales, indices, topk_length,
            None, nv_sched_meta, nv_splits, D_V, sm_scale)

    # Time NVFP4 noprealloc vs prealloc
    nv_noprealloc_min, nv_noprealloc_med, _ = _time_cuda(run_nvfp4_noprealloc, warmup, iters)
    nv_prealloc_min, nv_prealloc_med, _ = _time_cuda(run_nvfp4_prealloc, warmup, iters)
    fp8_noprealloc_min, fp8_noprealloc_med, _ = _time_cuda(run_fp8_noprealloc, warmup, iters)

    return {
        "batch_size": batch_size,
        "topk": topk,
        "nvfp4_noprealloc_median_us": nv_noprealloc_med,
        "nvfp4_prealloc_median_us": nv_prealloc_med,
        "fp8_noprealloc_median_us": fp8_noprealloc_med,
        "prealloc_speedup_nvfp4": nv_noprealloc_med / nv_prealloc_med,
        "prealloc_nvfp4_vs_fp8_noprealloc": fp8_noprealloc_med / nv_prealloc_med,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="32,64,128")
    parser.add_argument("--topk", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    rows = []
    for batch_size in [int(x) for x in args.batch_sizes.split(",") if x]:
        iters = min(args.iters, 50) if batch_size >= 128 else args.iters
        row = time_case(batch_size, args.topk, args.warmup, iters)
        rows.append(row)
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
