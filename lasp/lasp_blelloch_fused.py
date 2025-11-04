"""
LASP Blelloch with Fused Parallel Kernels - The Ultimate Combination

Combines:
- Optimized fused parallel kernels from lasp_fuse_parallel (intra-chunk)
- Blelloch tree communication O(log P) (inter-chunk)

Expected performance at P=128:
- Intra: 0.3ms (fused kernels, -40% vs naive)
- Inter: 4.6ms (Blelloch tree, -83% vs ring)
- Total: 4.9ms vs 28.4ms naive = 5.8× speedup
"""

import torch
import torch.distributed as dist
import triton

# Import optimized kernels from fuse_parallel
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


class LaspBlellochFused(torch.autograd.Function):
    """
    LASP with Blelloch scan + optimized fused parallel kernels.

    Key insight: lasp_fuse_parallel has separate kernels for:
    - Intra-chunk computation (diagonal blocks)
    - Local KV accumulation
    - Inter-chunk computation (using accumulated KV)

    We can use these kernels directly with Blelloch communication!
    """

    @staticmethod
    def forward(ctx, q, k, v, s):
        """
        Forward pass combining fused kernels + Blelloch scan.

        Strategy:
        1. Use _fwd_diag_kernel for intra-chunk attention
        2. Use _fwd_kv_parallel + _fwd_kv_reduce for local KV
        3. Use Blelloch scan to accumulate KV across GPUs
        4. Use _fwd_none_diag_kernel for inter-chunk attention
        """
        b, h, n, d = q.shape
        e = v.shape[-1]

        # Get distributed context
        group = get_sequence_parallel_group()
        rank = get_sequence_parallel_rank()
        world_size = get_sequence_parallel_world_size()

        # Determine block sizes (same logic as lasp_fuse_parallel)
        if n > 128:
            BLOCK = 256
            CBLOCK = 64
        else:
            BLOCK = min(n, 128)
            CBLOCK = min(n, 64)

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

        # ===== STEP 1: Intra-chunk attention (diagonal blocks) =====
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

        # ===== STEP 2: Compute local KV contribution =====
        kv = torch.empty((b, h, NUM_BLOCK + 1, d, e), dtype=torch.float32, device=q.device)

        with torch.cuda.device(q.device.index):
            # Parallel KV accumulation
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

            # Reduce KV across blocks
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

        # Extract local KV contribution (last element of buffer)
        # This is what gets accumulated across GPUs
        local_kv = kv[:, :, -1].clone()  # Shape: (b, h, d, e)

        # ===== STEP 3: Blelloch scan for inter-chunk KV accumulation =====
        if world_size == 1:
            # Single GPU: no inter-chunk communication
            KV_prefix = torch.zeros(b, h, d, e).to(torch.float32).to(q.device)
        else:
            # Multi-GPU: Blelloch tree scan
            block_decay = torch.exp(-s.to(torch.float32) * n)
            lambda_decay = torch.exp(-s.to(torch.float32))

            scanner = BlellochScanner(
                rank=rank,
                world_size=world_size,
                group=group,
                decay_factor=lambda_decay,
                chunk_size=n,
                device=q.device,
            )

            # Blelloch scan: O(log P) tree communication
            # This replaces the O(P) ring in lasp_fuse_parallel
            KV_prefix = scanner.scan(local_kv)

        # ===== STEP 4: Inter-chunk attention using accumulated KV =====
        with torch.cuda.device(q.device.index):
            grid = (b * h, NUM_BLOCK * NUM_CBLOCK, NUM_FBLOCK)
            _fwd_none_diag_kernel[grid](
                q, k, v, o, s,
                kv,  # Local KV buffer
                KV_prefix,  # Accumulated KV from Blelloch (not from ring!)
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
        ctx.save_for_backward(q, k, v, s, kv, KV_prefix)
        ctx.group = group
        ctx.rank = rank
        ctx.world_size = world_size
        ctx.BLOCK = BLOCK
        ctx.CBLOCK = CBLOCK

        return o

    @staticmethod
    def backward(ctx, do):
        """
        Backward pass with fused kernels + reverse Blelloch scan.
        """
        q, k, v, s, kv, KV_prefix = ctx.saved_tensors
        group = ctx.group
        rank = ctx.rank
        world_size = ctx.world_size
        BLOCK = ctx.BLOCK
        CBLOCK = ctx.CBLOCK

        b, h, n, d = q.shape
        e = v.shape[-1]

        NUM_BLOCK = n // BLOCK
        NUM_CBLOCK = BLOCK // CBLOCK
        NUM_FBLOCK = 1
        D_FBLOCK = d // NUM_FBLOCK
        E_FBLOCK = e // NUM_FBLOCK

        # Make inputs contiguous
        do = do.contiguous()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        # ===== STEP 1: Backward diagonal (intra-chunk gradients) =====
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

        # ===== STEP 2: Compute local dKV =====
        dkv = torch.empty((b, h, NUM_BLOCK + 1, d, e), dtype=torch.float32, device=q.device)

        with torch.cuda.device(q.device.index):
            # Parallel dKV computation
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

            # Reduce dKV
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

        # Extract local dKV contribution
        local_dkv = dkv[:, :, -1].clone()

        # ===== STEP 3: Reverse Blelloch scan for gradient accumulation =====
        if world_size == 1:
            # Single GPU: no inter-chunk gradients
            DKV_suffix = torch.zeros(b, h, d, e).to(torch.float32).to(do.device)
        else:
            # Multi-GPU: Reverse Blelloch scan
            lambda_decay = torch.exp(-s.to(torch.float32))

            scanner = BlellochScanner(
                rank=world_size - 1 - rank,  # Reverse for backward
                world_size=world_size,
                group=group,
                decay_factor=lambda_decay,
                chunk_size=n,
                device=do.device,
            )

            # Reverse scan for gradients
            DKV_suffix = scanner.scan(local_dkv)

        # ===== STEP 4: Inter-chunk gradient contribution =====
        with torch.cuda.device(q.device.index):
            grid = (b * h, NUM_BLOCK * NUM_CBLOCK, NUM_FBLOCK)
            _bwd_none_diag_kernel[grid](
                q, k, v, s, do, dq, dk, dv,
                dkv, DKV_suffix, kv, KV_prefix,
                b, h, n, d, e,
                BLOCK=BLOCK,
                NUM_BLOCK=NUM_BLOCK,
                D_FBLOCK=D_FBLOCK,
                E_FBLOCK=E_FBLOCK,
                NUM_FBLOCK=NUM_FBLOCK,
                CBLOCK=CBLOCK,
                NUM_CBLOCK=NUM_CBLOCK,
            )

        return dq, dk, dv, None


lasp_blelloch_fused_ = LaspBlellochFused.apply


def lasp_blelloch_fused(q, k, v, ed):
    """
    Ultimate LASP: Blelloch communication + fused parallel kernels.

    Combines the best optimizations:
    - Intra-chunk: Fused parallel kernels (from lasp_fuse_parallel)
    - Inter-chunk: Blelloch tree O(log P) (from lasp_blelloch)

    Expected speedup at P=128: ~5.8× vs naive (vs 5.5× for basic Blelloch)

    Args:
        q, k, v: Query, key, value tensors
        ed: Exponential decay factors

    Returns:
        Attention output
    """
    d = q.shape[-1]
    ed = ed[:d]
    return lasp_blelloch_fused_(q, k, v, ed)
