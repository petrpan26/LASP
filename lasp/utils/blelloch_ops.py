"""
Blelloch parallel prefix scan operations for LASP.

This module implements the work-efficient parallel prefix scan algorithm
for computing KV state accumulation in O(log P) time instead of O(P).
"""

import torch
import torch.distributed as dist
import math
from typing import Optional, Tuple


class BlellochScanner:
    """
    Blelloch parallel prefix scan for LASP KV state accumulation.

    Reduces inter-GPU communication from O(P) sequential steps (ring)
    to O(log P) parallel steps (tree-based).

    For P=128 GPUs: 128 steps → 14 steps (9× reduction)

    Algorithm:
        1. Up-sweep: Build tree of partial sums (log P levels)
        2. Down-sweep: Distribute prefix sums to all ranks (log P levels)

    The operation is associative: (A₁, b₁) ⊕ (A₂, b₂) = (A₁·A₂, A₂·b₁ + b₂)
    For LASP: A = λ^C (decay), b = KV state (d×d matrix)
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        group,
        decay_factor: torch.Tensor,  # λ per head (shape: [h])
        chunk_size: int,
        device: torch.device,
        reverse: bool = False,
    ):
        """
        Initialize Blelloch scanner.

        Args:
            rank: Current GPU rank within sequence parallel group (0 to P-1)
            world_size: Size of sequence parallel group (P)
            group: PyTorch distributed group for sequence parallelism
            decay_factor: Decay factor λ per head, shape [h]
            chunk_size: Sequence length per GPU (C)
            device: torch.device for tensors
            reverse: If True, scan in reverse direction (for backward pass)
        """
        self.rank = rank  # Local SP rank
        self.world_size = world_size  # SP world size
        self.group = group
        self.device = device
        self.reverse = reverse

        # Get global ranks for this sequence parallel group
        # This is needed because dist.send/recv with group parameter expects global ranks
        self.global_rank = dist.get_rank()

        # Compute offset to convert local SP rank → global rank
        # For dp_size=2, sp_size=4:
        #   SP group 0: local [0,1,2,3] → global [0,1,2,3], offset=0
        #   SP group 1: local [0,1,2,3] → global [4,5,6,7], offset=4
        self.rank_offset = self.global_rank - self.rank

        # For reverse scan, we reverse the rank order
        if reverse:
            self.scan_rank = world_size - 1 - rank
        else:
            self.scan_rank = rank

        # Compute decay for one chunk: λ^C per head
        self.lambda_C = decay_factor ** chunk_size  # Shape: [h]

        # Pre-compute tree structure
        self.num_levels = math.ceil(math.log2(world_size)) if world_size > 1 else 0
        self.padded_size = 2 ** self.num_levels

        # Check if this rank is active (not a padding rank)
        self.is_active = rank < world_size

    def local_to_global_rank(self, local_rank: int) -> int:
        """Convert local SP rank to global rank."""
        if local_rank == -1:
            return -1
        # For reverse scan, map reversed local rank to actual global rank
        if self.reverse:
            # reversed_local → actual_local → global
            actual_local = self.world_size - 1 - local_rank
            return actual_local + self.rank_offset
        else:
            return local_rank + self.rank_offset

    def get_partner_rank(self, level: int, phase: str) -> int:
        """
        Compute communication partner for this rank at given tree level.

        Args:
            level: Tree level (0 to num_levels-1)
            phase: 'up' for up-sweep, 'down' for down-sweep

        Returns:
            Partner rank (in scan_rank space), or -1 if no communication needed
        """
        stride = 2 ** level

        if phase == 'up':
            # Up-sweep: left sends to right, right receives from left
            if self.scan_rank % (2 * stride) == 0:
                # Left child: send to right sibling
                partner = self.scan_rank + stride
                return partner if partner < self.world_size else -1
            elif self.scan_rank % (2 * stride) == stride:
                # Right child: receive from left sibling
                return self.scan_rank - stride
            else:
                # Inactive at this level
                return -1

        elif phase == 'down':
            # Down-sweep: reversed
            if self.scan_rank % (2 * stride) == stride:
                # Right child: receive from left parent
                return self.scan_rank - stride
            elif self.scan_rank % (2 * stride) == 0:
                # Left child: send to right child
                partner = self.scan_rank + stride
                return partner if partner < self.world_size else -1
            else:
                return -1
        else:
            raise ValueError(f"Unknown phase: {phase}")

    def is_sender(self, level: int, phase: str) -> bool:
        """Check if this rank sends at this level."""
        stride = 2 ** level
        if phase == 'up':
            return self.scan_rank % (2 * stride) == 0
        elif phase == 'down':
            return self.scan_rank % (2 * stride) == 0
        return False

    def is_receiver(self, level: int, phase: str) -> bool:
        """Check if this rank receives at this level."""
        stride = 2 ** level
        if phase == 'up':
            return self.scan_rank % (2 * stride) == stride
        elif phase == 'down':
            return self.scan_rank % (2 * stride) == stride
        return False

    def combine(
        self,
        received: torch.Tensor,
        local: torch.Tensor,
        stride: int,
    ) -> torch.Tensor:
        """
        Combine operation for LASP prefix/suffix scan.

        Forward (prefix): (λ^(stride*C)) * received + local
        Backward (suffix): local + (λ^(stride*C)) * received

        The associative operator remains the same, just the order changes.

        Args:
            received: Tensor from communication partner
            local: Local tensor value
            stride: Tree stride (2^level)

        Returns:
            Combined tensor
        """
        # Compute decay power: λ^(stride * C)
        # Shape: [b, h, ...]
        decay_power = self.lambda_C ** stride  # Broadcast per head

        # Expand decay_power to match tensor dimensions
        # received/local shape: [b, h, d, e]
        # decay_power shape: [h] → [1, h, 1, 1]
        while decay_power.dim() < received.dim():
            decay_power = decay_power.unsqueeze(0)
            if decay_power.dim() < received.dim():
                decay_power = decay_power.unsqueeze(-1)

        # Combine: decay * received + local
        # This works for both prefix and suffix scans with appropriate rank ordering
        return decay_power * received + local

    def scan(self, local_value: torch.Tensor) -> torch.Tensor:
        """
        Perform parallel prefix scan on local KV contribution.

        Args:
            local_value: Local KV state b[rank] (shape: [b, h, d, e])

        Returns:
            prefix_sum: KV[0:rank+1] - prefix sum up to this rank
        """
        if self.world_size == 1:
            # Single GPU: no communication needed
            return local_value

        b, h, d, e = local_value.shape

        # ============ UP-SWEEP PHASE ============
        # Build tree bottom-up, accumulating partial sums

        current_value = local_value.clone()
        tree_values = [current_value]  # Store for down-sweep

        for level in range(self.num_levels):
            partner = self.get_partner_rank(level, 'up')

            if partner == -1:
                # No communication at this level
                continue

            if self.is_sender(level, 'up') and partner < self.world_size:
                # Send to right partner (convert to global rank)
                global_partner = self.local_to_global_rank(partner)
                dist.send(tensor=current_value.contiguous(), dst=global_partner, group=self.group)

            elif self.is_receiver(level, 'up'):
                # Receive from left partner and combine (convert to global rank)
                global_partner = self.local_to_global_rank(partner)
                received = torch.zeros_like(current_value)
                dist.recv(tensor=received, src=global_partner, group=self.group)

                # Combine: (λ^(stride*C)) * received + current
                stride = 2 ** level
                current_value = self.combine(received, current_value, stride)
                tree_values.append(current_value)

        # ============ DOWN-SWEEP PHASE ============
        # Distribute prefix sums top-down

        prefix_sum = None

        for level in range(self.num_levels - 1, -1, -1):
            partner = self.get_partner_rank(level, 'down')

            if partner == -1:
                continue

            if self.is_receiver(level, 'down') and partner >= 0:
                # Receive prefix from left parent (convert to global rank)
                global_partner = self.local_to_global_rank(partner)
                left_prefix = torch.zeros_like(current_value)
                dist.recv(tensor=left_prefix, src=global_partner, group=self.group)

                # Update prefix: combine with left neighbor's prefix
                stride = 2 ** level
                # Use the tree value stored during up-sweep
                tree_idx = min(level, len(tree_values) - 1)
                prefix_sum = self.combine(left_prefix, tree_values[tree_idx], stride)

            elif self.is_sender(level, 'down') and partner < self.world_size:
                # Send to right child (convert to global rank)
                global_partner = self.local_to_global_rank(partner)
                send_value = prefix_sum if prefix_sum is not None else tree_values[min(level, len(tree_values) - 1)]
                dist.send(tensor=send_value.contiguous(), dst=global_partner, group=self.group)

        # Rank 0 has no left prefix, uses its accumulated tree value
        if prefix_sum is None:
            prefix_sum = tree_values[-1] if len(tree_values) > 1 else local_value

        return prefix_sum


def safe_decay_power(base: float, exponent: int, use_log_space: bool = True) -> float:
    """
    Compute base^exponent safely for large exponents.

    For λ^(P*C) where P=128, C=32768: exponent = 4,194,304
    Direct computation causes underflow/overflow.

    Args:
        base: Decay factor λ (typically 0.9-0.999)
        exponent: Power to raise to
        use_log_space: Use log-space arithmetic for stability

    Returns:
        base^exponent computed safely
    """
    if not use_log_space or exponent < 100:
        return base ** exponent

    # Log-space: exp(exponent * log(base))
    log_result = exponent * math.log(base)

    # Clamp to prevent overflow/underflow
    MAX_LOG = 80  # exp(80) ≈ 5e34
    MIN_LOG = -80  # exp(-80) ≈ 2e-35
    log_result = max(MIN_LOG, min(MAX_LOG, log_result))

    return math.exp(log_result)


def is_power_of_two(n: int) -> bool:
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_two(n: int) -> int:
    """Return smallest power of 2 >= n."""
    return 2 ** math.ceil(math.log2(n))
