"""
LASP Blelloch with Phase 1 Optimization: Stream-Based Overlap

This implements the highest-priority optimization from BLELLOCH_OPTIMIZATION_PLAN.md:
- Run Blelloch scan in separate CUDA stream
- Overlap communication with diagonal kernel computation
- Use events for proper synchronization

Expected improvement: 10-15% faster (150ms → 125-130ms for W=16)
"""

import torch
import torch.distributed as dist
import triton

from .gpu_config import get_config_for_kernel
from .lasp_fuse_parallel import (
    _fwd_diag_kernel,
    _fwd_kv_parallel,
    _fwd_kv_reduce,
    _fwd_none_diag_kernel,
    _bwd_diag_kernel,
    _bwd_dkv_parallel,
    _bwd_dkv_reduce,
    _bwd_none_diag_kernel,
)
from .utils import (
    BlellochScanner,
    get_sequence_parallel_group,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
)


class LaspBlellochOptimized(torch.autograd.Function):
    """
    LASP Blelloch with stream-based communication-computation overlap.

    Key optimization: Run Blelloch tree scan in separate CUDA stream to overlap
    with diagonal kernel computation, reducing overall latency.
    """

    @staticmethod
    def forward(ctx, q, k, v, s, KV, DKV):
        b, h, n, d = q.shape
        e = v.shape[-1]

        # Zero out KV buffer
        KV.zero_()

        # Get distributed context
        group = get_sequence_parallel_group()
        rank = get_sequence_parallel_rank()
        world_size = get_sequence_parallel_world_size()

        # Determine block sizes
        config = get_config_for_kernel('lasp_blelloch', n, d, e, q.device)
        BLOCK = config['BLOCK']
        CBLOCK = config['CBLOCK']

        NUM_BLOCK = n // BLOCK
        NUM_CBLOCK = BLOCK // CBLOCK
        NUM_FBLOCK = 1
        D_FBLOCK = d // NUM_FBLOCK
        E_FBLOCK = e // NUM_FBLOCK

        # Make inputs contiguous
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        s = s.contiguous()

        # Output buffer
        o = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)

        # ===== OPTIMIZATION: Create communication stream and events =====
        comm_stream = torch.cuda.Stream()
        diag_done = torch.cuda.Event()
        local_kv_done = torch.cuda.Event()
        scan_done = torch.cuda.Event()

        # ===== STEP 1: Intra-chunk attention (diagonal) in DEFAULT stream =====
        # This runs independently of communication
        grid = (b * h * NUM_BLOCK, NUM_CBLOCK)
        with torch.cuda.device(q.device.index):
            _fwd_diag_kernel[grid](
                q, k, v, o, s,
                b, h, n, d, e,
                BLOCK=BLOCK,
                NUM_BLOCK=NUM_BLOCK,
                CBLOCK=CBLOCK,
                NUM_CBLOCK=NUM_CBLOCK,
            )
        diag_done.record()  # Signal diagonal kernel completion

        # ===== STEP 2: Compute local KV in DEFAULT stream =====
        kv = torch.empty((b, h, NUM_BLOCK + 1, d, e), dtype=torch.float32, device=q.device)

        with torch.cuda.device(q.device.index):
            grid = (b * h, NUM_BLOCK, NUM_FBLOCK * NUM_FBLOCK)
            _fwd_kv_parallel[grid](
                k, v, s, kv,
                b, h, n, d, e,
                BLOCK=BLOCK,
                NUM_BLOCK=NUM_BLOCK,
                D_FBLOCK=D_FBLOCK,
                E_FBLOCK=E_FBLOCK,
                NUM_FBLOCK=NUM_FBLOCK,
                CBLOCK=CBLOCK,
                NUM_CBLOCK=NUM_CBLOCK,
            )

            grid = (b * h, NUM_FBLOCK, NUM_FBLOCK)
            _fwd_kv_reduce[grid](
                k, v, s, kv,
                b, h, n, d, e,
                BLOCK=BLOCK,
                NUM_BLOCK=NUM_BLOCK,
                D_FBLOCK=D_FBLOCK,
                E_FBLOCK=E_FBLOCK,
                NUM_FBLOCK=NUM_FBLOCK,
                CBLOCK=CBLOCK,
                NUM_CBLOCK=NUM_CBLOCK,
            )

        local_kv = kv[:, :, -1].clone()
        local_kv_done.record()  # Signal local KV computation done

        # ===== STEP 3: Blelloch scan in COMMUNICATION stream =====
        # KEY OPTIMIZATION: This runs in parallel with any remaining default stream work
        if world_size == 1:
            KV_prefix = KV
        else:
            with torch.cuda.stream(comm_stream):
                # Wait for local_kv to be ready
                comm_stream.wait_event(local_kv_done)

                # Run Blelloch scan in communication stream
                lambda_decay = torch.exp(-s.to(torch.float32))
                scanner = BlellochScanner(
                    rank=rank,
                    world_size=world_size,
                    group=group,
                    decay_factor=lambda_decay,
                    chunk_size=n,
                    device=q.device,
                )
                KV_prefix = scanner.scan(local_kv)

                # Signal scan completion
                scan_done.record()

        # ===== STEP 4: Inter-chunk attention =====
        # Wait for both diagonal kernel and scan to complete
        torch.cuda.current_stream().wait_event(diag_done)
        if world_size > 1:
            torch.cuda.current_stream().wait_event(scan_done)

        # Now run inter-chunk kernel with accumulated KV_prefix
        with torch.cuda.device(q.device.index):
            grid = (b * h, NUM_BLOCK * NUM_CBLOCK, NUM_FBLOCK)
            _fwd_none_diag_kernel[grid](
                q, k, v, o, s,
                kv,
                KV_prefix,
                b, h, n, d, e,
                BLOCK=BLOCK,
                NUM_BLOCK=NUM_BLOCK,
                D_FBLOCK=D_FBLOCK,
                E_FBLOCK=E_FBLOCK,
                NUM_FBLOCK=NUM_FBLOCK,
                CBLOCK=CBLOCK,
                NUM_CBLOCK=NUM_CBLOCK,
            )

        # Save for backward
        KV_prefix_saved = KV_prefix.clone()
        ctx.save_for_backward(q, k, v, s, kv, KV_prefix_saved, DKV)
        ctx.group = group
        ctx.rank = rank
        ctx.world_size = world_size
        ctx.BLOCK = BLOCK
        ctx.CBLOCK = CBLOCK
        ctx.NUM_BLOCK = NUM_BLOCK
        ctx.NUM_CBLOCK = NUM_CBLOCK
        ctx.NUM_FBLOCK = NUM_FBLOCK
        ctx.D_FBLOCK = D_FBLOCK
        ctx.E_FBLOCK = E_FBLOCK

        return o

    @staticmethod
    def backward(ctx, do):
        """Backward pass with stream-based overlap."""
        q, k, v, s, kv, KV_prefix, DKV = ctx.saved_tensors
        group = ctx.group
        rank = ctx.rank
        world_size = ctx.world_size
        BLOCK = ctx.BLOCK
        CBLOCK = ctx.CBLOCK
        NUM_BLOCK = ctx.NUM_BLOCK
        NUM_CBLOCK = ctx.NUM_CBLOCK
        NUM_FBLOCK = ctx.NUM_FBLOCK
        D_FBLOCK = ctx.D_FBLOCK
        E_FBLOCK = ctx.E_FBLOCK

        b, h, n, d = q.shape
        e = v.shape[-1]

        DKV.zero_()

        do = do.contiguous()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        # ===== OPTIMIZATION: Create streams and events for backward =====
        comm_stream = torch.cuda.Stream()
        diag_done = torch.cuda.Event()
        local_dkv_done = torch.cuda.Event()
        scan_done = torch.cuda.Event()

        # ===== STEP 1: Backward diagonal in DEFAULT stream =====
        with torch.cuda.device(q.device.index):
            grid = (b * h * NUM_BLOCK, NUM_CBLOCK)
            _bwd_diag_kernel[grid](
                q, k, v, s, do, dq, dk, dv,
                b, h, n, d, e,
                BLOCK=BLOCK,
                NUM_BLOCK=NUM_BLOCK,
                CBLOCK=CBLOCK,
                NUM_CBLOCK=NUM_CBLOCK,
            )
        diag_done.record()

        # ===== STEP 2: Compute local dKV in DEFAULT stream =====
        dkv = torch.empty((b, h, NUM_BLOCK + 1, d, e), dtype=torch.float32, device=q.device)

        with torch.cuda.device(q.device.index):
            grid = (b * h, NUM_BLOCK, NUM_FBLOCK * NUM_FBLOCK)
            _bwd_dkv_parallel[grid](
                q, do, s, dkv,
                b, h, n, d, e,
                BLOCK=BLOCK,
                NUM_BLOCK=NUM_BLOCK,
                D_FBLOCK=D_FBLOCK,
                E_FBLOCK=E_FBLOCK,
                NUM_FBLOCK=NUM_FBLOCK,
                CBLOCK=CBLOCK,
                NUM_CBLOCK=NUM_CBLOCK,
            )

            grid = (b * h, NUM_FBLOCK, NUM_FBLOCK)
            _bwd_dkv_reduce[grid](
                q, do, s, dkv,
                b, h, n, d, e,
                BLOCK=BLOCK,
                NUM_BLOCK=NUM_BLOCK,
                D_FBLOCK=D_FBLOCK,
                E_FBLOCK=E_FBLOCK,
                NUM_FBLOCK=NUM_FBLOCK,
                CBLOCK=CBLOCK,
                NUM_CBLOCK=NUM_CBLOCK,
            )

        local_dkv = dkv[:, :, -1].clone()
        local_dkv_done.record()

        # ===== STEP 3: Reverse Blelloch scan in COMMUNICATION stream =====
        if world_size == 1:
            DKV_suffix = DKV
        else:
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_event(local_dkv_done)

                lambda_decay = torch.exp(-s.to(torch.float32))
                scanner = BlellochScanner(
                    rank=rank,
                    world_size=world_size,
                    group=group,
                    decay_factor=lambda_decay,
                    chunk_size=n,
                    device=do.device,
                    reverse=True,
                )
                DKV_suffix = scanner.scan(local_dkv)

                scan_done.record()

        # ===== STEP 4: Inter-chunk gradients =====
        torch.cuda.current_stream().wait_event(diag_done)
        if world_size > 1:
            torch.cuda.current_stream().wait_event(scan_done)

        with torch.cuda.device(q.device.index):
            grid = (b * h, NUM_BLOCK * NUM_CBLOCK, NUM_FBLOCK)
            _bwd_none_diag_kernel[grid](
                q, k, v, s, do, dq, dk, dv,
                kv,
                dkv,
                KV_prefix,
                DKV_suffix,
                b, h, n, d, e,
                BLOCK=BLOCK,
                NUM_BLOCK=NUM_BLOCK,
                D_FBLOCK=D_FBLOCK,
                E_FBLOCK=E_FBLOCK,
                NUM_FBLOCK=NUM_FBLOCK,
                CBLOCK=CBLOCK,
                NUM_CBLOCK=NUM_CBLOCK,
            )

        return dq, dk, dv, None, None, None


lasp_blelloch_optimized_ = LaspBlellochOptimized.apply


def lasp_blelloch_optimized(q, k, v, ed, KV, DKV):
    """
    Optimized LASP Blelloch with stream-based overlap.

    Usage:
        Same as lasp_blelloch, drop-in replacement.
    """
    b, h, n, d = q.shape
    e = v.shape[-1]

    if d >= 128:
        m = 128
    else:
        m = 64
    arr = [m * i for i in range(d // m + 1)]
    if arr[-1] != d:
        arr.append(d)
    n_splits = len(arr)
    output = 0
    for i in range(n_splits - 1):
        s = arr[i]
        e_idx = arr[i + 1]
        q1 = q[..., s:e_idx]
        k1 = k[..., s:e_idx]
        o = lasp_blelloch_optimized_(
            q1, k1, v, ed, KV[:, :, s:e_idx].contiguous(), DKV[:, :, s:e_idx].contiguous()
        )
        output = output + o

    return output
