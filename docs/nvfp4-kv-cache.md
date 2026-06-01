# Full-NVFP4 Sparse MLA KV Cache

This branch contains the V3.2 full-NVFP4 sparse MLA decode storage bridge. It is a verified intermediate state, not the final native NVFP4 tensor-core QK implementation.

## Layout

V3.2 full-NVFP4 KV stores every QK score dimension, including RoPE, in the same packed record:

| Field | Bytes | Notes |
| --- | ---: | --- |
| Packed score FP4 | 288 | 576 e2m1 elements, low nibble first |
| Inline scales | 36 | E4M3 scale per 16 score elements |
| Padding | 12 | Aligns the token record to 16 bytes |
| Total | 336 | `kv` stride for `d_qk == 576` |

The V3.2 path ignores the legacy `kv_scales` tensor. MODEL1 keeps the older split layout with a separate 32-byte scale tensor.

## Current Decode Path

The production bridge gathers the 336-byte token records, dequantizes packed FP4 score data to BF16 shared memory, and then uses the existing BF16 FlashMLA QK/PV pipeline. This validates the full-NVFP4 cache representation and API plumbing, but it does not yet use native NVFP4 tensor-core QK.

The separate CuTe scaffold `flash_mla/cute_dsl/test_nvfp4_mla_qk_full_nvfp4_scaffold.py` validates the intended native-QK primitive at the target shape: `M=128`, `N=1024`, `K=576`, with RoPE included in NVFP4.

## Verification

B200 verification was run on `a4-us-002-rl9` in `local/dynamo-trtllm-optrt-custom:hisa-buildtools-20260531` with CUDA 13.1 and the CCCL include path set for the build.

Correctness commands:

```bash
python3 tests/nvfp4/test_nvfp4_phase2_smoke.py
python3 tests/nvfp4/test_nvfp4_dequant_correctness.py
python3 tests/nvfp4/test_nvfp4_random_valid.py
python3 tests/nvfp4/test_nvfp4_v32_correctness.py
python3 tests/nvfp4/test_nvfp4_phase2_reference.py
python3 tests/nvfp4/test_nvfp4_reference.py
python3 flash_mla/cute_dsl/test_nvfp4_mla_qk_full_nvfp4_scaffold.py --topk 1024 --bench
```

Observed correctness:

| Check | Result |
| --- | --- |
| All-ones dequant smoke | Exact output mean `1.0` |
| Random finite smoke | No NaN/Inf |
| Python reference seed 456 | `max_abs=0.28125`, `rms=0.054973`, `cos=0.994642` |
| Python reference seed 123 | `max_abs=0.296875`, `rms=0.055695`, `cos=0.994641` |
| Native CuTe QK scaffold | Exact score tile and exact 512-dim value contract |

## Performance Snapshot

Command:

```bash
python3 tests/nvfp4/bench_nvfp4_vs_fp8.py \
  --batch-sizes 1,8,32,64 \
  --topk 1024 \
  --warmup 20 \
  --iters 120
```

| Batch | Full-NVFP4 min us | FP8 min us | Full-NVFP4 vs FP8 |
| ---: | ---: | ---: | ---: |
| 1 | 59.5200 | 59.5197 | 1.0000x |
| 8 | 65.3957 | 65.1928 | 0.9969x |
| 32 | 82.6054 | 81.9123 | 0.9916x |
| 64 | 104.8224 | 104.7213 | 0.9990x |

IKP SASS metrics showed the original bridge was dominated by scalar byte scale gathering in `csrc/sm100/decode/head64_nvfp4/kernel.cuh`. The current bridge therefore vector-loads the 36 inline E4M3 scale bytes and writes them directly into the shared scale buffer. That removes the local temporary copy and brings the BF16 bridge close to FP8 while preserving the 336-byte full-NVFP4 cache layout.

The bridge reduces cache bytes from 656 bytes/token in the FP8 V3.2 layout to 336 bytes/token. The remaining gap is now small but still exists at batch 8 and 32, so the next required optimization remains native NVFP4 tensor-core QK using the validated CuTe/CZS scaffold layout rather than more BF16 bridge polishing.

### Native QK Cache-Row Scaffold

After the bridge checkpoint, the CuTe full-NVFP4 QK scaffold was tightened to support `--cache-layout`. In that mode the K operand is read from the real KV cache row layout, with a configurable row stride that skips the inline scales and padding rather than repacking K into a dense matrix.

```bash
python3 flash_mla/cute_dsl/test_nvfp4_mla_qk_full_nvfp4_scaffold.py \
  --topk 1024 \
  --cache-layout \
  --bench
```

Observed on B200 after recheck: exact QK score tile, exact 512-dim value contract, and `29.73 us` for `M=128`, `N=1024`, `K=576` with a 352-byte row-stride candidate. The same cold sweep measured `30.67 us` at 336 bytes, `32.01 us` at 384 bytes, and `32.01 us` at 512 bytes.

A follow-up same-container repeat showed `336` and `352` byte rows converge after warmup, with `336` slightly ahead on the final repeat (`24.48 us` versus `25.03 us`). The 352-byte production bridge also regressed batch 8 and batch 32 and costs more memory. The production row stride therefore remains 336 bytes until a native decode integration proves a stable end-to-end win for a wider row.


### Native QK Integration Checkpoints

The native tensor-core rewrite is being landed in compile-safe slices before execution is switched over:

| Checkpoint | Commit | Status |
| --- | --- | --- |
| Native layout aliases | `36f6cb6` | Adds the SM100 MXF4/NVF4 MMA atom, K-major Q/K shared-memory layouts, and canonical SFA/SFB layouts. Execution-neutral. |
| Native staging storage | `1e48178` | Reserves Q/K packed FP4 and scale-factor scratch in existing shared-memory lifetimes. Execution-neutral. |

The staged buffers intentionally do not change the active bridge path yet. The next execution step is to populate the canonical packed K layout from the 336-byte cache rows, quantize the full 576-dim Q tile, move SFA/SFB into TMEM with the block-scaled path, and then replace only V3.2 QK with native NVFP4 tensor cores. PV should keep dequantizing only the first 512 value dimensions after QK consumes the packed K tile.

Latest post-staging bridge regression check:

| Batch | Full-NVFP4 min us | FP8 min us | Full-NVFP4 vs FP8 |
| ---: | ---: | ---: | ---: |
| 1 | 59.5672 | 59.5716 | 1.0001x |
| 8 | 65.3588 | 65.1344 | 0.9966x |
| 32 | 82.7296 | 81.9552 | 0.9906x |
| 64 | 104.8634 | 104.9018 | 1.0004x |

## Rejected Candidates

| Candidate | Result |
| --- | --- |
| Adjacent-lane scale sharing with `shfl` | Correct but slower at every measured batch |
| `GROUP_SIZE=4` dequant ownership | Failed all-ones correctness with output mean about `0.5` |
| 48-byte padded shared scale rows with three index buffers | Built, but failed launch because dynamic shared memory exceeded the practical B200 limit |
| 48-byte padded shared scale rows with two index buffers | Correct but slower: batch 32 `83.00 us` and batch 64 `106.16 us` median |
| 352-byte production KV row stride | Native-QK scaffold was slightly faster in one cold sweep but tied after warmup; production bridge regressed batch 8 and 32, so 336 bytes remains the production layout |

These candidates should not be retried unless the surrounding dequant layout changes substantially.
