"""Validate NVFP4 tensor cores at FlashMLA QK tile shape.

We are using the dimensions from the head64_nvfp4 kernel:
  Q: (b_h*h_q_chunk = 64 rows, d_qk_latent = 512 cols)  -- A operand
  K: (b_topk = 64 rows, d_qk_latent = 512 cols)         -- B operand
  Acc: (64, 64) FP32

Both A and B are NVFP4 (e2m1 + e4m3 block scale, vec_size=16).
The test:
  1) Runs a single CTA block-scaled FP4 GEMM at this shape (no batching).
  2) Computes the BF16 reference: (Q_bf16 @ K_bf16.T) in FP32.
  3) Compares MMA result to reference, prints RMS + max diff.
  4) Optionally benchmarks (latency for many iterations).

Usage:
    python3.11 /tmp/test_nvfp4_mla_qk.py
    python3.11 /tmp/test_nvfp4_mla_qk.py --bench
"""

import argparse
import os
import sys
from typing import Tuple

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.torch as cutlass_torch
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils

# MLA QK tile dims
B_H = 64
B_TOPK = 64
D_QK_LATENT = 512
SF_VEC = 16

# MMA tile (M, N) — we pick the smallest legal tile that covers our (64, 64)
# block-scale-vec atom instruction-K is 64 elements for MXF4NVF4
MMA_INST_K = 64
MMA_TILER_MN = (128, 128)  # min tile that B200 BlockScaled MMA atom supports
MMA_INST_TILE_K = 8         # 8 inner K iters -> mma_tiler_k = 64 * 8 = 512 = d_qk_latent
MMA_TILER_K = MMA_INST_K * MMA_INST_TILE_K

# Block-scale ref types
AB_DTYPE = cutlass.Float4E2M1FN
SF_DTYPE = cutlass.Float8E4M3FN
C_DTYPE = cutlass.Float32  # FP32 accumulator output (matches MLA's QK acc)


class MlaQkValidator:
    def __init__(self):
        self.threads_per_cta = 128
        self.num_tmem_alloc_cols = 512
        self.num_acc_stage = 1
        self.num_ab_stage = 1  # no pipelining for the validator

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        problem_size: tuple,
        stream: cuda.CUstream,
    ):
        m, n, k, l = problem_size
        self.mma_tiler = (MMA_TILER_MN[0], MMA_TILER_MN[1], MMA_TILER_K)
        self.cta_tile_shape_mnk = self.mma_tiler

        a_tensor = cute.make_tensor(
            a_ptr,
            cute.make_layout(
                (m, cute.assume(k, 32), l),
                stride=(cute.assume(k, 32), 1, cute.assume(m * k, 32)),
            ),
        )
        b_tensor = cute.make_tensor(
            b_ptr,
            cute.make_layout(
                (n, cute.assume(k, 32), l),
                stride=(cute.assume(k, 32), 1, cute.assume(n * k, 32)),
            ),
        )
        c_tensor = cute.make_tensor(
            c_ptr,
            cute.make_layout(
                (cute.assume(m, 32), cute.assume(n, 16), l),
                stride=(cute.assume(n, 16), 1, cute.assume(m * n, 512)),
            ),
        )

        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, SF_VEC)
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b_tensor.shape, SF_VEC)
        sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)

        mma_op = tcgen05.MmaMXF4NVF4Op(
            SF_DTYPE,
            (*MMA_TILER_MN, MMA_INST_K),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
        )
        tiled_mma = cute.make_tiled_mma(mma_op)

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
        )

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, AB_DTYPE, self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, AB_DTYPE, self.num_ab_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, SF_VEC, self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, SF_VEC, self.num_ab_stage,
        )

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # TMA atoms
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            a_tensor, a_smem_layout, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            b_tensor, b_smem_layout, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            sfa_tensor, sfa_smem_layout, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape, internal_type=cutlass.Int16,
        )
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            sfb_tensor, sfb_smem_layout, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape, internal_type=cutlass.Int16,
        )

        a_copy_size = cute.size_in_bytes(AB_DTYPE, a_smem_layout)
        b_copy_size = cute.size_in_bytes(AB_DTYPE, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(SF_DTYPE, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(SF_DTYPE, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        grid = (
            cute.ceil_div(c_tensor.shape[0], self.cta_tile_shape_mnk[0]),
            cute.ceil_div(c_tensor.shape[1], self.cta_tile_shape_mnk[1]),
            c_tensor.shape[2],
        )

        self.kernel(
            tiled_mma, tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b,
            tma_atom_sfa, tma_tensor_sfa, tma_atom_sfb, tma_tensor_sfb, c_tensor,
            self.a_smem_layout_staged, self.b_smem_layout_staged,
            self.sfa_smem_layout_staged, self.sfb_smem_layout_staged,
        ).launch(
            grid=grid, block=[self.threads_per_cta, 1, 1],
            cluster=(1, 1, 1), stream=stream,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom, mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom, mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom, mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom, mSFB_nkl: cute.Tensor,
        mC_mnl: cute.Tensor,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        cta_coord = (bidx, bidy, bidz)
        mma_tile_coord_mnl = (
            cta_coord[0] // cute.size(tiled_mma.thr_id.shape),
            cta_coord[1], cta_coord[2],
        )

        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sA = smem.allocate_tensor(
            element_type=AB_DTYPE, layout=a_smem_layout_staged.outer,
            byte_alignment=128, swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=AB_DTYPE, layout=b_smem_layout_staged.outer,
            byte_alignment=128, swizzle=b_smem_layout_staged.inner,
        )
        sSFA = smem.allocate_tensor(
            element_type=SF_DTYPE, layout=sfa_smem_layout_staged, byte_alignment=128,
        )
        sSFB = smem.allocate_tensor(
            element_type=SF_DTYPE, layout=sfb_smem_layout_staged, byte_alignment=128,
        )

        ab_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        ab_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_producer_group,
            consumer_group=ab_consumer_group,
            tx_count=self.num_tma_load_bytes,
        ).make_participants()
        acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=ab_producer_group,
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.threads_per_cta,
            ),
        ).make_participants()

        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None),
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None),
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None),
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None),
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None),
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        thr_mma = tiled_mma.get_slice(0)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        tCgSFB = thr_mma.partition_B(gSFB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a, 0, cute.make_layout(1),
            cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, 0, cute.make_layout(1),
            cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3),
        )
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa, 0, cute.make_layout(1),
            cute.group_modes(sSFA, 0, 3), cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb, 0, cute.make_layout(1),
            cute.group_modes(sSFB, 0, 3), cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.threads_per_cta,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_alloc_barrier,
        )
        tmem.allocate(self.num_tmem_alloc_cols)
        tmem.wait_for_alloc()
        acc_tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
        tCtAcc = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

        sfa_tmem_ptr = cute.recast_ptr(
            acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc),
            dtype=SF_DTYPE,
        )
        tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma, self.mma_tiler, SF_VEC,
            cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
        )
        tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
        sfb_tmem_ptr = cute.recast_ptr(
            acc_tmem_ptr
            + tcgen05.find_tmem_tensor_col_offset(tCtAcc)
            + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
            dtype=SF_DTYPE,
        )
        tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma, self.mma_tiler, SF_VEC,
            cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
        )
        tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), SF_DTYPE,
        )
        tCsSFA_compact = cute.filter_zeros(sSFA)
        tCtSFA_compact = cute.filter_zeros(tCtSFA)
        tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFA_compact)
        thr_copy_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
        tCsSFA_compact_s2t_ = thr_copy_s2t_sfa.partition_S(tCsSFA_compact)
        tCsSFA_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t_sfa, tCsSFA_compact_s2t_,
        )
        tCtSFA_compact_s2t = thr_copy_s2t_sfa.partition_D(tCtSFA_compact)

        tCsSFB_compact = cute.filter_zeros(sSFB)
        tCtSFB_compact = cute.filter_zeros(tCtSFB)
        tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFB_compact)
        thr_copy_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
        tCsSFB_compact_s2t_ = thr_copy_s2t_sfb.partition_S(tCsSFB_compact)
        tCsSFB_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t_sfb, tCsSFB_compact_s2t_,
        )
        tCtSFB_compact_s2t = thr_copy_s2t_sfb.partition_D(tCtSFB_compact)

        tAgA = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
        tBgB = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
        tAgSFA = tAgSFA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
        tBgSFB = tBgSFB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

        if warp_idx == 0:
            acc_empty = acc_producer.acquire_and_advance()
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_tile in cutlass.range(
                k_tile_cnt, prefetch_stages=max(0, self.num_ab_stage - 2),
            ):
                ab_empty = ab_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_a, tAgA[(None, ab_empty.count)],
                    tAsA[(None, ab_empty.index)], tma_bar_ptr=ab_empty.barrier,
                )
                cute.copy(
                    tma_atom_b, tBgB[(None, ab_empty.count)],
                    tBsB[(None, ab_empty.index)], tma_bar_ptr=ab_empty.barrier,
                )
                cute.copy(
                    tma_atom_sfa, tAgSFA[(None, ab_empty.count)],
                    tAsSFA[(None, ab_empty.index)], tma_bar_ptr=ab_empty.barrier,
                )
                cute.copy(
                    tma_atom_sfb, tBgSFB[(None, ab_empty.count)],
                    tBsSFB[(None, ab_empty.index)], tma_bar_ptr=ab_empty.barrier,
                )

                ab_full = ab_consumer.wait_and_advance()
                s2t_stage_coord = (None, None, None, None, ab_full.index)
                tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
                cute.copy(
                    tiled_copy_s2t_sfa, tCsSFA_compact_s2t_staged, tCtSFA_compact_s2t,
                )
                cute.copy(
                    tiled_copy_s2t_sfb, tCsSFB_compact_s2t_staged, tCtSFB_compact_s2t,
                )

                num_kblocks = cute.size(tCrA, mode=[2])
                for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                    kblock_coord = (None, None, kblock_idx, ab_full.index)
                    sf_kblock_coord = (None, None, kblock_idx)
                    tiled_mma.set(
                        tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator,
                    )
                    tiled_mma.set(
                        tcgen05.Field.SFB, tCtSFB[sf_kblock_coord].iterator,
                    )
                    cute.gemm(
                        tiled_mma, tCtAcc,
                        tCrA[kblock_coord], tCrB[kblock_coord], tCtAcc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                ab_full.release()
            acc_empty.commit()

        # Epilogue: copy TMEM acc → registers → GMEM
        op = tcgen05.Ld32x32bOp(tcgen05.Repetition.x128, tcgen05.Pack.NONE)
        copy_atom_t2r = cute.make_copy_atom(op, cutlass.Float32)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc)
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc)
        tTR_gC = thr_copy_t2r.partition_D(tCgC)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[None, None, None, None, 0, 0, 0].shape, cutlass.Float32,
        )
        tTR_rC = cute.make_rmem_tensor(
            tTR_gC[None, None, None, None, 0, 0, 0].shape, C_DTYPE,
        )
        simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), C_DTYPE)
        tTR_gC = tTR_gC[(None, None, None, None, *mma_tile_coord_mnl)]

        tmem.relinquish_alloc_permit()
        acc_full = acc_consumer.wait_and_advance()
        cute.copy(tiled_copy_t2r, tTR_tAcc, tTR_rAcc)
        acc_vec = tTR_rAcc.load().to(C_DTYPE)
        tTR_rC.store(acc_vec)
        cute.copy(simt_atom, tTR_rC, tTR_gC)
        acc_full.release()

        cute.arch.barrier()
        tmem.free(acc_tmem_ptr)
        return


def nvfp4_quantize(t_bf16: torch.Tensor, sf_vec: int = 16):
    """Quantize a BF16 tensor along the last dim to NVFP4 + e4m3 block scales."""
    assert t_bf16.dtype == torch.bfloat16
    *batch, K = t_bf16.shape
    assert K % sf_vec == 0
    nblocks = K // sf_vec

    # Compute per-block absmax in fp32
    blocks = t_bf16.float().view(*batch, nblocks, sf_vec)
    absmax = blocks.abs().max(dim=-1, keepdim=False).values  # (.., nblocks)

    # NVFP4: e4m3 scales encode (per-block-absmax / 6.0) as e4m3.
    # Effective fp4 max value = 6.0 (e2m1 max).
    # scale = absmax / 6.0, quantized to e4m3 (clip + roundeven)
    scale_fp32 = absmax / 6.0
    # Clip to e4m3 representable range [2^-9, 448] roughly
    scale_fp32 = scale_fp32.clamp(min=2 ** -9, max=448.0)
    sf_e4m3 = scale_fp32.to(torch.float8_e4m3fn).to(torch.float32)

    # Quantize blocks: q = round(b / scale) clipped to fp4 grid
    blocks_scaled = blocks / sf_e4m3.unsqueeze(-1)
    # FP4 e2m1 representable values: ±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
    fp4_grid = torch.tensor(
        [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
         0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=blocks_scaled.device, dtype=torch.float32,
    )
    # Round to nearest grid value
    blocks_scaled = blocks_scaled.unsqueeze(-1)
    diff = (blocks_scaled - fp4_grid).abs()
    q_idx = diff.argmin(dim=-1)
    q_fp4_val = fp4_grid[q_idx]
    return q_fp4_val.view(*batch, K).contiguous(), sf_e4m3.contiguous()


def run_test(bench=False):
    device = torch.device("cuda")
    m, n, k = B_H, B_TOPK, D_QK_LATENT
    # Round up problem to MMA tile multiples for the validator (smallest legal tile)
    m_pad = max(m, MMA_TILER_MN[0])
    n_pad = max(n, MMA_TILER_MN[1])

    # Reference inputs
    torch.manual_seed(42)
    Q_bf16 = torch.randn(m_pad, k, dtype=torch.bfloat16, device=device) * 0.5
    K_bf16 = torch.randn(n_pad, k, dtype=torch.bfloat16, device=device) * 0.5

    # Quantize to NVFP4
    Q_fp4_dequant, Q_sf = nvfp4_quantize(Q_bf16, SF_VEC)
    K_fp4_dequant, K_sf = nvfp4_quantize(K_bf16, SF_VEC)
    # The "dequant" values are bf16 representations of the fp4 grid points (for ref math).

    # Reference QK^T (using post-quant values for fair comparison)
    Q_eff = Q_fp4_dequant * Q_sf.repeat_interleave(SF_VEC, dim=-1)
    K_eff = K_fp4_dequant * K_sf.repeat_interleave(SF_VEC, dim=-1)
    ref_acc = (Q_eff @ K_eff.T).float()  # (m_pad, n_pad)

    print(f"Test shape: M={m_pad} N={n_pad} K={k}")
    print(f"Reference acc range: [{ref_acc.min().item():.3f}, {ref_acc.max().item():.3f}]")
    print(f"Reference acc std:   {ref_acc.std().item():.3f}")

    # Pack FP4 values to e2m1 byte storage for the kernel.
    # torch lacks native e2m1, so we manually pack 2 nibbles per byte.
    def pack_fp4(t_fp4_grid):
        # t_fp4_grid is bf16 with values from the fp4 grid
        # Convert to e2m1 nibble indices, pack 2-per-byte
        fp4_grid = torch.tensor(
            [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
             0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            device=t_fp4_grid.device, dtype=torch.float32,
        )
        # Map grid → e2m1 4-bit codes (sign + 3-bit mag).
        # e2m1 encoding: [sign, exp(2), mant(1)] = SEEM, 4 bits
        # Standard NVFP4: 0=+0, 1=+0.5, 2=+1, 3=+1.5, 4=+2, 5=+3, 6=+4, 7=+6
        # negatives: msb=1, so 8=-0, 9=-0.5, ..., 15=-6
        e2m1_codes = torch.tensor(
            [15, 14, 13, 12, 11, 10, 9, 0, 1, 2, 3, 4, 5, 6, 7],
            dtype=torch.uint8, device=t_fp4_grid.device,
        )
        # Find index in fp4_grid for each value
        diff = (t_fp4_grid.float().unsqueeze(-1) - fp4_grid).abs()
        idx = diff.argmin(dim=-1)
        codes = e2m1_codes[idx].view(t_fp4_grid.shape)
        # Pack 2 nibbles per byte (low nibble = even index, high = odd index)
        codes_flat = codes.view(*codes.shape[:-1], -1)
        K = codes_flat.shape[-1]
        assert K % 2 == 0
        low = codes_flat[..., 0::2]
        high = codes_flat[..., 1::2]
        packed = (high << 4) | low
        return packed.contiguous()

    Q_packed = pack_fp4(Q_fp4_dequant)  # (m_pad, k/2) uint8
    K_packed = pack_fp4(K_fp4_dequant)
    print(f"Q_packed shape: {Q_packed.shape}, K_packed shape: {K_packed.shape}")

    # SFA / SFB stored as e4m3 (one byte per block)
    Q_sf_e4m3 = Q_sf.to(torch.float8_e4m3fn)
    K_sf_e4m3 = K_sf.to(torch.float8_e4m3fn)
    print(f"Q_sf shape: {Q_sf_e4m3.shape}, K_sf shape: {K_sf_e4m3.shape}")

    # Output buffer
    C = torch.zeros(m_pad, n_pad, dtype=torch.float32, device=device)

    # Build cute tensors using cutlass.torch helpers
    from cutlass.cute.runtime import from_dlpack
    # NOTE: cutlass-dsl requires (m, k, l) layout for A — add batch dim of 1
    Q_packed = Q_packed.unsqueeze(-1)   # (m_pad, k/2, 1)
    K_packed = K_packed.unsqueeze(-1)
    Q_sf_e4m3 = Q_sf_e4m3.unsqueeze(-1)
    K_sf_e4m3 = K_sf_e4m3.unsqueeze(-1)
    C = C.unsqueeze(-1)

    a_cute = from_dlpack(Q_packed, assumed_align=16)
    b_cute = from_dlpack(K_packed, assumed_align=16)
    sfa_cute = from_dlpack(Q_sf_e4m3, assumed_align=16)
    sfb_cute = from_dlpack(K_sf_e4m3, assumed_align=16)
    c_cute = from_dlpack(C, assumed_align=16)

    print("Compiling kernel...")
    runner = MlaQkValidator()
    err, stream = cuda.cuStreamCreate(0)
    assert err == cuda.CUresult.CUDA_SUCCESS, f"cuStreamCreate failed: {err}"
    compiled = cute.compile(
        runner,
        a_cute.iterator, b_cute.iterator, sfa_cute.iterator, sfb_cute.iterator,
        c_cute.iterator,
        (m_pad, n_pad, k, 1),
        stream,
    )

    print("Launching...")
    compiled(
        a_cute.iterator, b_cute.iterator, sfa_cute.iterator, sfb_cute.iterator,
        c_cute.iterator, (m_pad, n_pad, k, 1), stream,
    )
    torch.cuda.synchronize()

    # Validate
    actual = C.squeeze(-1).float()
    diff = actual - ref_acc
    rms = diff.pow(2).mean().sqrt().item()
    max_abs = diff.abs().max().item()
    rel_rms = rms / ref_acc.std().item()
    print(f"\nResult: max_abs={max_abs:.4f}  rms={rms:.4f}  rel_rms={rel_rms:.4f}")
    print(f"actual range: [{actual.min().item():.3f}, {actual.max().item():.3f}]")
    print(f"Sample diff (top-left 4x4):\n{diff[:4, :4]}")

    if rel_rms < 0.05:
        print("\nPASS — FP4 tensor-core MMA matches reference within 5% RMS")
    else:
        print("\nFAIL — FP4 tensor-core MMA mismatch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    run_test(bench=args.bench)
