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

The separate CuTe scaffold `flash_mla/cute_dsl/test_nvfp4_mla_qk_full_nvfp4_scaffold.py` validates the intended native-QK primitive at the target shape: `M=128`, `N=1024`, `K=576`, with RoPE included in NVFP4. The in-kernel head64 path uses the legal production tile shape `M=128`, `N=64`, `K=576`; rows `64..127` are padded in Q because Blackwell `SM100_MMA_MXF4_SS` requires `M=128`, while `N=64` is legal.

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
  --iters 240
```

| Batch | Full-NVFP4 min us | FP8 min us | Full-NVFP4 vs FP8 | Full-NVFP4 bytes/token | FP8 bytes/token |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 59.4823 | 59.4723 | 0.9998x | 336 | 656 |
| 8 | 64.0575 | 65.0521 | 1.0155x | 336 | 656 |
| 32 | 82.6195 | 81.9123 | 0.9914x | 336 | 656 |
| 64 | 104.7814 | 105.2602 | 1.0046x | 336 | 656 |

IKP SASS metrics showed the original bridge was dominated by scalar byte scale gathering in `csrc/sm100/decode/head64_nvfp4/kernel.cuh`. The current bridge therefore vector-loads the 36 inline E4M3 scale bytes and writes them directly into the shared scale buffer. That removes the local temporary copy and brings the BF16 bridge close to FP8 while preserving the 336-byte full-NVFP4 cache layout.

The V3.2 bridge now uses a single readiness barrier for the combined NoPE/RoPE dequant pass. The dequant producer writes both regions together, so the consumer waits on `bar_nope_ready` once, runs RoPE QK first, then immediately runs NoPE QK without a second barrier wait. This preserves the existing BF16 FlashMLA QK/PV pipeline while removing redundant synchronization from the full-NVFP4 storage bridge.

The bridge reduces cache bytes from 656 bytes/token in the FP8 V3.2 layout to 336 bytes/token. It now beats FP8 at batch 8 and 64, ties at batch 1, and remains slightly behind at batch 32. The next required optimization remains native NVFP4 tensor-core QK using the validated CuTe/CZS scaffold layout rather than more BF16 bridge polishing.

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
| Native-Q prequant scaffold | `720a21f` | Adds the standalone V3.2 Q prequant kernel for the legal 128-row native Q operand, fixes native QK `N=64`, and proves the dedicated native-QK SMEM plan fits B200 (`231680` bytes). Execution-neutral. |

The staged buffers intentionally do not change the active bridge path yet. The next execution step is to populate the canonical packed K layout from the 336-byte cache rows, move SFA/SFB into TMEM with the block-scaled path, and then replace only V3.2 QK with native NVFP4 tensor cores. PV should keep dequantizing only the first 512 value dimensions after QK consumes the packed K tile.

The native-Q prequant scaffold writes one 128-row Q operand per head64 block: rows `0..63` are quantized from BF16 Q, rows `64..127` are explicit zero padding, and all `128*36` E4M3 scale bytes are initialized. B200 smoke command used during this slice compiled `/tmp/qprequant_smoke.cu` against `q_prequant.cu` and reported `nonzero_real=36864`, `nonzero_pad=0`, `untouched_pad=0`, `scale_untouched=0`.

The dedicated native-QK shared-memory plan is intentionally separate from the BF16 bridge overlay. Its compile-time size is `231680` bytes, under the observed B200 opt-in limit of `232448` bytes. The existing bridge plan remains `231632` bytes.

Latest post-staging bridge regression check from 2026-06-01:

| Batch | Full-NVFP4 min us | FP8 min us | Full-NVFP4 vs FP8 |
| ---: | ---: | ---: | ---: |
| 1 | 59.5232 | 59.5123 | 0.9998x |
| 8 | 65.6304 | 65.5869 | 0.9993x |
| 32 | 82.5702 | 82.1062 | 0.9944x |
| 64 | 104.8762 | 104.7885 | 0.9992x |

An execution-enabled native QK attempt was compiled and tested after these
checkpoints, but it was not promoted. The attempt prequantized Q once and staged
native Q/K tiles in the existing split-K kernel. It exposed a structural shared
memory lifetime conflict: the old kernel overlays Q, K/V raw bytes, dequantized
V, and output scratch in the same union because BF16 Q is moved to TMEM before
K/V staging. Native NVFP4 QK needs Q and packed K resident at the same time as
the value path, so the next implementation should use an explicit native-QK
scratch layout instead of reusing the BF16 bridge overlay. A direct
`cvt.rn.bf16x2.e4m3x2` scale conversion variant was also rejected by CUDA 13.1
ptxas for this V32 path.

## Rejected Candidates

| Candidate | Result |
| --- | --- |
| Adjacent-lane scale sharing with `shfl` | Correct but slower at every measured batch |
| `GROUP_SIZE=4` dequant ownership | Failed all-ones correctness with output mean about `0.5` |
| 48-byte padded shared scale rows with three index buffers | Built, but failed launch because dynamic shared memory exceeded the practical B200 limit |
| 48-byte padded shared scale rows with two index buffers | Correct but slower: batch 32 `83.00 us` and batch 64 `106.16 us` median |
| 352-byte production KV row stride | Native-QK scaffold was slightly faster in one cold sweep but tied after warmup; production bridge regressed batch 8 and 32, so 336 bytes remains the production layout |
| Native QK over the BF16 bridge shared-memory overlay | Compiled after canonical layout fixes but deadlocked or overwrote live scratch; not production-safe without a dedicated native-QK scratch plan |
| Direct `cvt.rn.bf16x2.e4m3x2` scale conversion in the V32 bridge | Rejected by CUDA 13.1 ptxas with unexpected instruction type errors |
| FP16 E4M3 scale decode with FP16 product before BF16 store | Correct, but slower than the BF16 bridge at batch 8, 32, and 64 |
| FP16 E4M3 scale decode with BF16 multiply/store | Correct, but slower than the BF16 bridge at batch 8, 32, and 64 |

These candidates should not be retried unless the surrounding dequant layout changes substantially.
