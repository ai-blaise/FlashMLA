# NVFP4 head64_nvfp4 IKP Loop - Tried & Failed (TRAPS.md)

Working state: tag `nvfp4-beats-fp8-iter11` (commit `d0a3b31`).
Verified perf: **NVFP4 = 101.63 us, FP8 = 105.72 us, +4.02% faster** at
production shape (b=64, s_q=1, h_q=128, d_qk=576, topk=1024, page=64).

## Architecture as of iter 11

- Phase 1 layout: 416 B/slot (256 FP4 nope + 32 E4M3 scales + 128 BF16 rope)
- Dequant warp pipeline:
  - get_raw_fp4 (uint32 SMEM load, 4 bytes / 8 elems per inner iter)
  - cvt.rn.bf16x2.e2m1x2 PTX x4 (compiles to F2FP.BF16.E2M1.UNPACK_B native SASS)
  - LAZY per-iter scale load + cvt.rn.bf16x2.e4m3x2 + pair pick
  - __hmul2 (HMUL2.BF16_V2) x4
  - st.weak.shared::cta.b128 (__int128 store via inline asm)
- Build: nvcc 12.9 + ptxas 13.3 hybrid via `tools/ptxas_wrapper.sh`
- Reg usage: 168/thread (vs FP8's 128/thread)
- SMEM: ~230 KB/block (1 block/SM, near 232 KB limit)

## Things tried that didn't help (do NOT retry without new insight)

### Phase 2 (336 B/slot, FP4 rope dequant)
- Reduced HBM 18% more vs Phase 1, but rope dequant overhead = 14 us > 1us HBM win
- Element-wise CUTE SW64-swizzle writes for rope cost ~12 us alone
- Vectorized swizzled writes via manual XOR (iter 2) broke correctness
  with RMS 0.07. CUTE Layout SW64 has hidden state I couldn't manually
  replicate even after deriving the Swizzle<2,4,3> XOR pattern correctly.
  Even no-permute (xor_mask=0) writes failed - suggests stride or layout
  function is not pure r*32+c as I thought.
- Phase 2 with all iter 4+6+7 opts (iter 10): 118 us, worse than Phase 1

### CTK 13.3 nvcc (full toolchain swap)
- 40x slowdown across the board (4170 us vs 105 us baseline)
- Some codegen regression in nvcc 13.3 for sm_100f
- HYBRID (nvcc 12.9 + ptxas 13.3 via wrapper) is the ONLY working CTK 13 config

### Instruction-level PTX attempts
- LUT-based FP4 decode (iter 3): slower than HW cvt (LUT serializes loads)
- Fused cvt+mul inline asm (iter 9): marginally slower (register pressure)
- 2-col unroll (iter 14): no win - compiler already pipelines via CUTE_UNROLL
- Split cvt asm into 4 single-cvt blocks (iter 16): build error
- Split cvt asm 2x2 interleaved with scale cvt (iter 17): same perf as iter 11
- Removed K prefetch (iter 13): same perf
- Branchless scale pick (iter 12): slightly slower
- uint64 scale loads (iter 8): no improvement over uint32 (iter 7)
- Upfront pair load + lazy cvt (iter 15): slightly worse than full lazy

### Occupancy / SMEM tactics
- NUM_BUFS=3 (more pipelining): SMEM exceeded 232KB limit. NVFP4 SMEM
  is dominated by dequant.nope (B_H*D_NOPE*bf16 = 64KB/buf) which scales
  directly with NUM_BUFS. NUM_INDEX_BUFS=2 only saves 2.3KB - not enough.
- __launch_bounds__(NUM_THREADS, 2, 1) to force 2 blocks/SM: build fails
  (SMEM × 2 > 256KB SM SMEM limit).
- Constant memory for scales: scales are per-token-dynamic, can't go to const.

### PTX instructions that DON'T EXIST in CTK 12.9 or 13.3
- cvt.rn.bf16.e4m3 (scalar) - "Unexpected instruction types"
- cvt.rn.f16.e4m3 (scalar) - same
- cvt.rn.bf16x4.e2m1x4 - "Unknown modifier '.bf16x4'"
- All multi-element cvt variants > x2

## Things that DID help (cumulative path 130 us -> 101.63 us, 21.5% kernel speedup)

| Iter | Change | NVFP4 us | vs FP8 |
|------|--------|----------|--------|
| 0 | Phase 1 baseline (f16x2 PTX + cast) | 130 | 1.24x slower |
| 4 | bf16x2.e2m1x2 PTX (hybrid CTK build for ptxas 13.3) | 115 | 1.12x slower |
| 6 | Vectorized scale conv (cvt.bf16x2.e4m3x2 instead of float-roundtrip) | 103 | tied |
| 7 | uint32 batched scale loads (2 cvts per uint32) | 104 | 1% faster |
| 11 | LAZY scale cvt (load+cvt per inner iter, not upfront) | 101.6 | 4.02% faster |

## To go past 4% (multi-week scopes)

1. Native tcgen05 FP4 tensor cores (`tcgen05.mma.kind::f8f6f4.block_scale_vec`)
   eliminates SMEM dequant entirely. Major kernel architecture rewrite.
2. Fix Phase 2 SW64 swizzle for rope writes. Requires solving the CUTE
   Layout puzzle - needs interactive ncu / CUTE layout printing to debug.
3. Reduce reg count 168 -> 128 via algorithmic restructure that lets B_H
   shrink or NUM_BUFS rebalance. Probably needs new SMEM layout.

## Tools / Builds

- `tools/ptxas_wrapper.sh`: enables bf16x2.e2m1x2 + bf16x2.e4m3x2 PTX on
  CTK 12.9 nvcc by patching `.version 8.8` -> `.version 9.2` before
  invoking ptxas 13.3 binary.
- `pip install nvidia-cuda-nvcc` installs CTK 13.3 to
  /usr/local/lib/python3.12/dist-packages/nvidia/cu13/.
- Build: `unset CUDA_HOME && NVCC_THREADS=8 FLASH_MLA_DISABLE_SM90=1
  pip install -e . --no-build-isolation`.

## References

- AKO4ALL (https://github.com/TongmingLAIC/AKO4ALL): Claude-skill protocol,
  not a codegen tool. Doesn't generate FP4 PTX, doesn't help here.
- OSCAR, RotateKV, HIGGS: all 3 2-bit KV alternatives REJECTED earlier in
  this project (Triton-bound, no Blackwell FP4, no MLA support).
- Subagent transcripts in memory entries (:3811 `mem_20260531T*`).
