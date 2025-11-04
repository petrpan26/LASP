"""
LASP with Blelloch parallel prefix scan using optimized Triton kernels.

Reduces inter-GPU communication from O(P) sequential steps (ring)
to O(log P) parallel steps (tree-based).

Uses fused Triton kernels for both intra-chunk and inter-chunk computation.

For P=128 GPUs: 128 steps → 14 steps (~6-9× speedup)
"""

import torch
import torch.distributed as dist
import triton

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


class LaspBlelloch(torch.autograd.Function):
    """
    LASP attention using Blelloch parallel prefix scan with optimized kernels.

    This class replaces the O(P) ring communication with O(log P) tree-based
    communication while using fused Triton kernels for efficient computation.

    Key improvements:
        - O(log P) communication (Blelloch tree) instead of O(P) (ring)
        - Fused Triton kernels for inter-chunk matmul instead of PyTorch matmul
        - Optimized intra-chunk computation with parallel kernels
    """

    @staticmethod
    def forward(ctx, q, k, v, s):
        """
        Forward pass with Blelloch scan and fused kernels.

        Args:
            q: Query (b, h, n, d)
            k: Key (b, h, n, d)
            v: Value (b, h, n, e)
            s: Decay factor per head (h,)

        Returns:
            o: Output attention (b, h, n, e)
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
        local_kv = kv[:, :, -1].clone()  # Shape: (b, h, d, e)

        # ===== STEP 3: Blelloch scan for inter-chunk KV accumulation =====
        if world_size == 1:
            # Single GPU: no inter-chunk communication
            KV_prefix = torch.zeros(b, h, d, e, dtype=torch.float32, device=q.device)
        else:
            # Multi-GPU: Blelloch tree scan O(log P)
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
            KV_prefix = scanner.scan(local_kv)

        # ===== STEP 4: Inter-chunk attention using fused kernel =====
        # This is the key improvement: use _fwd_none_diag_kernel instead of torch.matmul
        with torch.cuda.device(q.device.index):
            grid = (b * h, NUM_BLOCK * NUM_CBLOCK, NUM_FBLOCK)
            _fwd_none_diag_kernel[grid](
                q, k, v, o, s,
                kv,          # Local KV buffer
                KV_prefix,   # Accumulated KV from Blelloch scan
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
        ctx.NUM_BLOCK = NUM_BLOCK
        ctx.NUM_CBLOCK = NUM_CBLOCK
        ctx.NUM_FBLOCK = NUM_FBLOCK
        ctx.D_FBLOCK = D_FBLOCK
        ctx.E_FBLOCK = E_FBLOCK

        return o

    @staticmethod
    def backward(ctx, do):
        """
        Backward pass with reverse Blelloch scan and fused kernels.
        """
        q, k, v, s, kv, KV_prefix = ctx.saved_tensors
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
            DKV_suffix = torch.zeros(b, h, d, e, dtype=torch.float32, device=do.device)
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

        # ===== STEP 4: Inter-chunk gradient contribution using fused kernel =====
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


lasp_blelloch_ = LaspBlelloch.apply


def lasp_blelloch(q, k, v, ed):
    """
    LASP with Blelloch scan and optimized Triton kernels.

    Combines:
    - Blelloch tree O(log P) communication
    - Fused Triton kernels for computation

    Args:
        q, k, v: Query, key, value tensors
        ed: Exponential decay factors

    Returns:
        Attention output
    """
    d = q.shape[-1]
    ed = ed[:d]
    return lasp_blelloch_(q, k, v, ed)
