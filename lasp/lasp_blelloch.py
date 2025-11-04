"""
LASP with Blelloch parallel prefix scan.

Reduces inter-GPU communication from O(P) sequential steps (ring)
to O(log P) parallel steps (tree-based).

For P=128 GPUs: 128 steps → 14 steps (~6-9× speedup)
"""

import torch
import torch.distributed as dist

from .lasp_naive import lasp_forward, lasp_backward
from .utils import (
    BlellochScanner,
    get_sequence_parallel_group,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
)


class LaspBlelloch(torch.autograd.Function):
    """
    LASP attention using Blelloch parallel prefix scan.

    This class replaces the O(P) ring communication with O(log P) tree-based
    communication while keeping the same interface as LaspNaive.

    Key difference from Ring LASP:
        Ring: GPU i receives KV from GPU i-1, computes, sends to GPU i+1
        Blelloch: GPUs communicate in tree pattern with log(P) levels
    """

    @staticmethod
    def forward(ctx, q, k, v, s):
        """
        Forward pass with Blelloch scan for inter-chunk communication.

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

        # Compute decay factors
        array = torch.arange(n).to(q)
        q_decay = torch.exp(-s[None, :].to(torch.float32) * array.reshape(1, 1, -1, 1))
        k_decay = torch.exp(-s[None, :].to(torch.float32) * (n - array.reshape(-1, 1)))
        block_decay = torch.exp(-s[None, :].to(torch.float32) * n)

        # ===== INTRA-CHUNK: Reuse existing kernel (unchanged from ring) =====
        # This computes causal attention within the local sequence chunk
        kv = torch.empty(b, h, d, e).to(q)
        o_intra = lasp_forward(q, k, v, s, kv).to(torch.float32)

        # ===== INTER-CHUNK: Blelloch scan for KV prefix =====
        # This is the NEW part that replaces ring communication

        if world_size == 1:
            # Single GPU: no inter-chunk communication needed
            KV_prefix = torch.zeros(b, h, d, e).to(torch.float32).to(q.device)
            o = o_intra.to(q.dtype)
        else:
            # Multi-GPU: use Blelloch scan
            # Initialize scanner per head (decay factor is per-head)
            lambda_decay = torch.exp(-s.to(torch.float32))  # Shape: [h]

            scanner = BlellochScanner(
                rank=rank,
                world_size=world_size,
                group=group,
                decay_factor=lambda_decay,
                chunk_size=n,
                device=q.device,
            )

            # Compute local KV contribution: b[rank] = (λ^C Λ^(-1) K)^T V
            # This matches the kv output from lasp_forward, which already computes:
            # kv = Σ(k_decay * K)^T @ V for the local chunk
            # We can reuse it directly!
            local_b = kv.clone()  # Shape: (b, h, d, e)

            # Perform parallel prefix scan
            # Old Ring: O(P) sequential recv-compute-send rounds
            # New Blelloch: O(log P) tree-based parallel communication
            KV_prefix = scanner.scan(local_b)

            # Compute inter-chunk attention: Q @ KV_prefix
            o_inter = torch.matmul(q * q_decay, KV_prefix)

            # Combine intra and inter chunk outputs
            o = (o_intra + o_inter).to(q.dtype)

        # Save for backward
        # Note: For Blelloch backward, we need KV_prefix (not accumulated KV)
        KV_for_backward = KV_prefix if world_size > 1 else torch.zeros(b, h, d, e).to(torch.float32).to(q.device)

        ctx.save_for_backward(q, k, v, s, KV_for_backward, kv)
        ctx.group = group
        ctx.rank = rank
        ctx.world_size = world_size

        return o

    @staticmethod
    def backward(ctx, do):
        """
        Backward pass using reverse Blelloch scan.

        Gradients flow from right to left (opposite of forward).
        """
        q, k, v, s, KV_prefix, kv = ctx.saved_tensors
        group = ctx.group
        rank = ctx.rank
        world_size = ctx.world_size

        b, h, n, d = q.shape
        e = v.shape[-1]

        # Compute decay factors
        array = torch.arange(n).to(do)
        q_decay = torch.exp(-s[None, :].to(torch.float32) * array.reshape(1, 1, -1, 1))
        k_decay = torch.exp(-s[None, :].to(torch.float32) * (n - array.reshape(-1, 1)))
        block_decay = torch.exp(-s[None, :].to(torch.float32) * n)

        # ===== INTRA-CHUNK GRADIENT: Reuse existing kernel =====
        dq_intra, dk_intra, dv_intra, _, dkv = lasp_backward(q, k, v, s, do)

        # ===== INTER-CHUNK GRADIENT =====
        if world_size == 1:
            # Single GPU: no inter-chunk gradients
            dq = dq_intra.to(q.dtype)
            dk = dk_intra.to(q.dtype)
            dv = dv_intra.to(q.dtype)
        else:
            # dL/dKV_prefix from: o_inter = (q * q_decay) @ KV_prefix
            dq_inter = torch.matmul(do.to(KV_prefix.dtype), KV_prefix.transpose(-1, -2)) * q_decay

            # Gradient w.r.t. KV_prefix from inter-chunk attention
            dKV_from_inter = torch.matmul((q * q_decay).transpose(-2, -1), do.to(KV_prefix.dtype))

            # ===== REVERSE BLELLOCH SCAN =====
            # Accumulate gradients from all chunks AFTER this one
            # This is conceptually a "suffix scan" in reverse order

            # For backward, we reverse the direction:
            # Create scanner with reversed rank ordering
            lambda_decay = torch.exp(-s.to(torch.float32))

            # Simple approach: reverse the ranks for backward pass
            # In practice, we can reuse the same Blelloch structure but reverse communication
            scanner = BlellochScanner(
                rank=world_size - 1 - rank,  # Reverse rank!
                world_size=world_size,
                group=group,
                decay_factor=lambda_decay,
                chunk_size=n,
                device=do.device,
            )

            # Scan the gradient in reverse order
            DKV_suffix = scanner.scan(dKV_from_inter)

            # Combine gradients
            # dk, dv need contributions from current chunk and all future chunks
            dk_inter = torch.matmul(v.to(DKV_suffix.dtype), DKV_suffix.transpose(-1, -2)) * k_decay
            dv_inter = torch.matmul((k * k_decay).to(DKV_suffix.dtype), DKV_suffix)

            # Combine intra and inter gradients
            dq = (dq_intra.to(torch.float32) + dq_inter).to(q.dtype)
            dk = (dk_intra.to(torch.float32) + dk_inter).to(q.dtype)
            dv = (dv_intra.to(torch.float32) + dv_inter).to(q.dtype)

        # No gradient for decay parameter s (None)
        return dq, dk, dv, None


lasp_blelloch_ = LaspBlelloch.apply


def lasp_blelloch(q, k, v, ed):
    """
    Convenience function for LASP with Blelloch scan.

    Args:
        q: Query
        k: Key
        v: Value
        ed: Exponential decay factors

    Returns:
        Attention output
    """
    d = q.shape[-1]
    ed = ed[:d]
    return lasp_blelloch_(q, k, v, ed)
