"""Benchmark V32 full-NVFP4 sparse MLA decode with optional CUDA Graph capture.

CUDA Graph mode amortizes per-call PyTorch dispatch + per-call torch::empty
allocations (tile_scheduler_metadata, num_splits, o_accum, lse_accum), which
dominate the non-graph wall time. With graphs, only the kernel launches contribute
to per-iter wall.
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


def _time_graph(graph, warmup, iters):
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    values = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        values.append(start.elapsed_time(end) * 1000.0 / iters)
    values.sort()
    return values[0], values[len(values) // 2], values


def _make_common_inputs(batch_size, topk):
    num_blocks = max((batch_size * topk + PAGE_BLOCK_SIZE - 1) // PAGE_BLOCK_SIZE * 2, 64)
    q = torch.randn(batch_size, 1, H_Q, D_QK, dtype=torch.bfloat16, device=DEVICE) * 0.01
    indices = torch.randint(
        0,
        num_blocks * PAGE_BLOCK_SIZE,
        (batch_size, 1, topk),
        dtype=torch.int32,
        device=DEVICE,
    )
    topk_length = torch.full((batch_size,), topk, dtype=torch.int32, device=DEVICE)
    return num_blocks, q, indices, topk_length, 1.0 / (D_QK ** 0.5)


def _make_fp8_cache(num_blocks):
    kv = torch.zeros(
        num_blocks,
        PAGE_BLOCK_SIZE,
        1,
        FP8_BYTES_PER_TOKEN,
        dtype=torch.uint8,
        device=DEVICE,
    )
    kv[:, :, :, :512] = 0x38
    scale_view = kv[:, :, :, 512:528].view(torch.float32)
    scale_view.fill_(1.0)
    kv[:, :, :, 528:] = 0
    return kv


def time_case(batch_size, topk, warmup, iters):
    num_blocks, q, indices, topk_length, sm_scale = _make_common_inputs(batch_size, topk)
    nvfp4_kv = make_random_full_nvfp4(num_blocks, PAGE_BLOCK_SIZE)
    fp8_kv = _make_fp8_cache(num_blocks)
    kv_scales_unused = torch.zeros(num_blocks, PAGE_BLOCK_SIZE, 1, 32, dtype=torch.uint8, device=DEVICE)

    def run_nvfp4():
        fc.sparse_decode_fwd_nvfp4(
            q, nvfp4_kv, kv_scales_unused, indices, topk_length,
            None, None, None, D_V, sm_scale)

    def run_fp8():
        fc.sparse_decode_fwd(
            q, fp8_kv, indices, topk_length, None,
            None, None, None, None, None, D_V, sm_scale)

    nv_min, nv_median, nv_samples = _time_cuda(run_nvfp4, warmup, iters)
    fp8_min, fp8_median, fp8_samples = _time_cuda(run_fp8, warmup, iters)

    # CUDA Graph mode: capture the kernel sequence and replay.
    # We pre-warm + capture into a graph, then replay N times.
    graph_nv_min = graph_nv_median = float('nan')
    graph_fp8_min = graph_fp8_median = float('nan')

    try:
        # Pre-warm to ensure all autotuned paths + lazy allocators are settled.
        for _ in range(5):
            run_nvfp4()
        torch.cuda.synchronize()

        graph_nv = torch.cuda.CUDAGraph()
        # Use a side stream for capture (PyTorch convention).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):  # extra warmup on capture stream
                run_nvfp4()
            s.synchronize()
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(graph_nv):
            fc.sparse_decode_fwd_nvfp4(
                q, nvfp4_kv, kv_scales_unused, indices, topk_length,
                None, None, None, D_V, sm_scale)
        graph_nv_min, graph_nv_median, _ = _time_graph(graph_nv, warmup, iters)

        # FP8 graph
        for _ in range(5):
            run_fp8()
        torch.cuda.synchronize()
        graph_fp8 = torch.cuda.CUDAGraph()
        s2 = torch.cuda.Stream()
        s2.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s2):
            for _ in range(3):
                run_fp8()
            s2.synchronize()
        torch.cuda.current_stream().wait_stream(s2)
        with torch.cuda.graph(graph_fp8):
            fc.sparse_decode_fwd(
                q, fp8_kv, indices, topk_length, None,
                None, None, None, None, None, D_V, sm_scale)
        graph_fp8_min, graph_fp8_median, _ = _time_graph(graph_fp8, warmup, iters)
    except Exception as e:
        print(f"# graph capture failed: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()

    return {
        "batch_size": batch_size,
        "topk": topk,
        "d_qk": D_QK,
        "d_v": D_V,
        "nvfp4_bytes_per_token": NVFP4_BYTES_PER_TOKEN,
        "fp8_bytes_per_token": FP8_BYTES_PER_TOKEN,
        "nvfp4_min_us": nv_min,
        "nvfp4_median_us": nv_median,
        "fp8_min_us": fp8_min,
        "fp8_median_us": fp8_median,
        "graph_nvfp4_min_us": graph_nv_min,
        "graph_nvfp4_median_us": graph_nv_median,
        "graph_fp8_min_us": graph_fp8_min,
        "graph_fp8_median_us": graph_fp8_median,
        "nvfp4_vs_fp8_speedup_median": fp8_median / nv_median,
        "graph_nvfp4_vs_fp8_speedup_median": graph_fp8_median / graph_nv_median if graph_nv_median == graph_nv_median else float('nan'),
        "graph_speedup_over_nongraph_nvfp4": nv_median / graph_nv_median if graph_nv_median == graph_nv_median else float('nan'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1,8,32,64")
    parser.add_argument("--topk", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    rows = []
    for batch_size in [int(x) for x in args.batch_sizes.split(",") if x]:
        iters = min(args.iters, 50) if batch_size >= 32 else args.iters
        row = time_case(batch_size, args.topk, args.warmup, iters)
        rows.append(row)
        print(json.dumps(row, sort_keys=True))

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
