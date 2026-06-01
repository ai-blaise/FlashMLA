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
  --warmup 10 \
  --iters 100
```

| Batch | Full-NVFP4 min us | FP8 min us | Full-NVFP4 vs FP8 |
| ---: | ---: | ---: | ---: |
| 1 | 59.5709 | 59.5363 | 0.9994x |
| 8 | 67.7034 | 65.0851 | 0.9613x |
| 32 | 90.1830 | 82.0845 | 0.9102x |
| 64 | 121.3690 | 104.7392 | 0.8630x |

The bridge reduces cache bytes from 656 bytes/token in the FP8 V3.2 layout to 336 bytes/token, but the BF16 dequant bridge dominates at useful batch sizes. The next required optimization is native NVFP4 tensor-core QK using the validated CuTe/CZS scaffold layout rather than additional BF16 bridge polishing.


### Native QK Cache-Row Scaffold

After the bridge checkpoint, the CuTe full-NVFP4 QK scaffold was tightened to support `--cache-layout`. In that mode the K operand is read from the real 336-byte KV cache row layout, with a 672-FP4-element row stride that skips the inline scales and padding rather than repacking K into a dense matrix.

```bash
python3 flash_mla/cute_dsl/test_nvfp4_mla_qk_full_nvfp4_scaffold.py \
  --topk 1024 \
  --cache-layout \
  --bench
```

Observed on B200 after recheck: exact QK score tile, exact 512-dim value contract, and `39.13 us` for `M=128`, `N=1024`, `K=576`. This is slower than the same-run dense packed scaffold (`31.35 us`) but is the correct native-QK baseline because it exercises the production KV row stride.

## Rejected Candidates

| Candidate | Result |
| --- | --- |
| Adjacent-lane scale sharing with `shfl` | Correct but slower at every measured batch |
| `GROUP_SIZE=4` dequant ownership | Failed all-ones correctness with output mean about `0.5` |

These candidates should not be retried unless the surrounding dequant layout changes substantially.
