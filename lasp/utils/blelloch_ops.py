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

    def actual_to_global_rank(self, actual_rank: int) -> int:
        """Convert actual local rank (not scan_rank) to global rank.

        Used for exclusive conversion where we use actual ranks directly.
        """
        if actual_rank == -1:
            return -1
        return actual_rank + self.rank_offset

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
            # Up-sweep: Send from right edge of left subtree to right edge of right subtree
            # This ensures accumulated values flow correctly up the tree
            if level == 0:
                # Level 0: Standard pattern (left edge sends to right edge)
                # rank % 2 == 0 sends to rank % 2 == 1
                if self.scan_rank % 2 == 0:
                    partner = self.scan_rank + 1
                    return partner if partner < self.world_size else -1
                elif self.scan_rank % 2 == 1:
                    return self.scan_rank - 1
                else:
                    return -1
            else:
                # Level >= 1: Right edge of left subtree sends to right edge of right subtree
                # Sender: rank % (2*stride) == stride-1 (right edge of left subtree)
                # Receiver: rank % (2*stride) == 2*stride-1 (right edge of right subtree)
                if self.scan_rank % (2 * stride) == stride - 1:
                    # Right edge of left subtree: send to right edge of right subtree
                    partner = self.scan_rank + stride
                    return partner if partner < self.world_size else -1
                elif self.scan_rank % (2 * stride) == 2 * stride - 1:
                    # Right edge of right subtree: receive from right edge of left subtree
                    return self.scan_rank - stride
                else:
                    # Inactive at this level
                    return -1

        elif phase == 'down':
            # Down-sweep: Distribute accumulated values from right edge of left subtree
            # This mirrors the up-sweep pattern to ensure correct flow
            if level == 0:
                # Level 0: Standard pattern
                if self.scan_rank % 2 == 1:
                    return self.scan_rank - 1
                elif self.scan_rank % 2 == 0:
                    partner = self.scan_rank + 1
                    return partner if partner < self.world_size else -1
                else:
                    return -1
            else:
                # Level >= 1: Send from right edge of left subtree
                if self.scan_rank % (2 * stride) == stride - 1:
                    # Right edge of left subtree: send to middle of right subtree
                    partner = self.scan_rank + 1
                    return partner if partner < self.world_size else -1
                elif self.scan_rank % (2 * stride) == stride:
                    # Middle of right subtree: receive from right edge of left subtree
                    return self.scan_rank - 1
                else:
                    return -1
        else:
            raise ValueError(f"Unknown phase: {phase}")

    def is_sender(self, level: int, phase: str) -> bool:
        """Check if this rank sends at this level."""
        stride = 2 ** level
        if phase == 'up':
            if level == 0:
                # Level 0: rank % 2 == 0 sends
                return self.scan_rank % 2 == 0
            else:
                # Level >= 1: Right edge of left subtree sends (rank % 2*stride == stride-1)
                return self.scan_rank % (2 * stride) == stride - 1
        elif phase == 'down':
            if level == 0:
                # Level 0: rank % 2 == 0 sends
                return self.scan_rank % 2 == 0
            else:
                # Level >= 1: Right edge of left subtree sends
                return self.scan_rank % (2 * stride) == stride - 1
        return False

    def is_receiver(self, level: int, phase: str) -> bool:
        """Check if this rank receives at this level."""
        stride = 2 ** level
        if phase == 'up':
            if level == 0:
                # Level 0: rank % 2 == 1 receives
                return self.scan_rank % 2 == 1
            else:
                # Level >= 1: Right edge of right subtree receives (rank % 2*stride == 2*stride-1)
                return self.scan_rank % (2 * stride) == 2 * stride - 1
        elif phase == 'down':
            if level == 0:
                # Level 0: rank % 2 == 1 receives
                return self.scan_rank % 2 == 1
            else:
                # Level >= 1: Middle of right subtree receives
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
        Perform parallel EXCLUSIVE prefix scan on local KV contribution.

        Args:
            local_value: Local KV state b[rank] (shape: [b, h, d, e])

        Returns:
            exclusive_prefix: KV[0:rank] - prefix sum excluding current rank
                              (rank 0 gets zero, rank i gets sum from ranks 0 to i-1)
        """
        if self.world_size == 1:
            # Single GPU: exclusive prefix is zero (no previous ranks)
            return torch.zeros_like(local_value)

        b, h, d, e = local_value.shape

        # ============ UP-SWEEP PHASE ============
        # Build tree bottom-up, accumulating partial sums (inclusive)

        # Memory optimization: Reuse single buffer for current_value throughout
        # This buffer will be reused for inclusive_prefix and exclusive_prefix later
        working_buffer = local_value.clone()

        # Memory optimization: Only store tree_values when needed for down-sweep
        # List indexed by level: tree_values[i] = state after processing level i-1
        # Use None for levels we don't need (saves ~50% memory)
        tree_values = [working_buffer.clone()]  # tree_values[0] = initial state

        for level in range(self.num_levels):
            partner = self.get_partner_rank(level, 'up')

            if partner == -1:
                # No communication at this level
                tree_values.append(None)  # Don't allocate memory
                continue

            if self.is_sender(level, 'up') and partner < self.world_size:
                # Send to right partner (convert to global rank)
                global_partner = self.local_to_global_rank(partner)
                dist.send(tensor=working_buffer.contiguous(), dst=global_partner, group=self.group)
                # Sender: check if we'll need this value in down-sweep
                # We need it if we're a sender in down-sweep at this level
                if self.is_sender(level, 'down'):
                    # Store current state (will be sent during down-sweep)
                    tree_values.append(working_buffer.clone())
                else:
                    # Don't need this value - save memory
                    tree_values.append(None)

            elif self.is_receiver(level, 'up'):
                # Receive from left partner and combine (convert to global rank)
                global_partner = self.local_to_global_rank(partner)
                received = torch.zeros_like(working_buffer)
                dist.recv(tensor=received, src=global_partner, group=self.group)

                # Combine: (λ^(stride*C)) * received + current
                # Update working_buffer in-place to save memory
                stride = 2 ** level
                working_buffer = self.combine(received, working_buffer, stride)

                # Receiver: always store updated value (needed for down-sweep combine)
                tree_values.append(working_buffer.clone())

        # ============ DOWN-SWEEP PHASE ============
        # Distribute inclusive prefix sums top-down
        # Reuse working_buffer for inclusive_prefix computation

        inclusive_computed = False

        for level in range(self.num_levels - 1, -1, -1):
            partner = self.get_partner_rank(level, 'down')

            if partner == -1:
                continue

            if self.is_receiver(level, 'down') and partner >= 0:
                # Receive prefix from left parent (convert to global rank)
                global_partner = self.local_to_global_rank(partner)
                left_prefix = torch.zeros_like(working_buffer)
                dist.recv(tensor=left_prefix, src=global_partner, group=self.group)

                # Update prefix: combine with left neighbor's prefix
                # Stride is the actual distance between sender and receiver
                distance = abs(self.scan_rank - partner)
                # Use the tree value stored during up-sweep at this level
                tree_idx = min(level, len(tree_values) - 1)
                tree_value = tree_values[tree_idx]
                # If None, find the most recent non-None value
                while tree_value is None and tree_idx > 0:
                    tree_idx -= 1
                    tree_value = tree_values[tree_idx]
                # Reuse working_buffer for inclusive_prefix
                working_buffer = self.combine(left_prefix, tree_value, distance)
                inclusive_computed = True

            elif self.is_sender(level, 'down') and partner < self.world_size:
                # Send to right child (convert to global rank)
                global_partner = self.local_to_global_rank(partner)
                if inclusive_computed:
                    send_value = working_buffer
                else:
                    # Use stored tree value at this level (should always exist for senders)
                    tree_idx = min(level, len(tree_values) - 1)
                    send_value = tree_values[tree_idx]
                    # If None, find the most recent non-None value
                    while send_value is None and tree_idx > 0:
                        tree_idx -= 1
                        send_value = tree_values[tree_idx]
                dist.send(tensor=send_value.contiguous(), dst=global_partner, group=self.group)

        # Compute inclusive prefix for this rank if not already done
        if not inclusive_computed:
            # working_buffer already contains the correct value from up-sweep or initial
            # Find the last non-None tree value
            if len(tree_values) > 1:
                for i in range(len(tree_values) - 1, -1, -1):
                    if tree_values[i] is not None:
                        working_buffer = tree_values[i].clone()
                        break
            else:
                working_buffer = local_value.clone()

        # ============ CONVERT TO EXCLUSIVE ============
        # Shift inclusive prefix to make it exclusive
        # For prefix scan: rank i gets inclusive[i-1] from rank i-1
        # For suffix scan: rank i gets inclusive[i+1] from rank i+1
        #
        # IMPORTANT: Use non-blocking communication to avoid deadlock/serialization

        # Reuse working_buffer for exclusive result (zero it out first)
        # But we need to send inclusive_prefix first, so create result buffer
        result = torch.zeros_like(local_value)

        if not self.reverse:
            # PREFIX SCAN: rank i receives from rank i-1, sends to rank i+1
            recv_req = None
            send_req = None

            if self.rank > 0:
                # Non-blocking receive from left neighbor
                global_left = self.actual_to_global_rank(self.rank - 1)
                recv_req = dist.irecv(tensor=result, src=global_left, group=self.group)

            if self.rank < self.world_size - 1:
                # Non-blocking send to right neighbor
                global_right = self.actual_to_global_rank(self.rank + 1)
                send_req = dist.isend(tensor=working_buffer.contiguous(), dst=global_right, group=self.group)

            # Wait for completion
            if recv_req is not None:
                recv_req.wait()
            if send_req is not None:
                send_req.wait()
        else:
            # SUFFIX SCAN: rank i receives from rank i+1, sends to rank i-1
            recv_req = None
            send_req = None

            if self.rank < self.world_size - 1:
                # Non-blocking receive from right neighbor
                global_right = self.actual_to_global_rank(self.rank + 1)
                recv_req = dist.irecv(tensor=result, src=global_right, group=self.group)

            if self.rank > 0:
                # Non-blocking send to left neighbor
                global_left = self.actual_to_global_rank(self.rank - 1)
                send_req = dist.isend(tensor=working_buffer.contiguous(), dst=global_left, group=self.group)

            # Wait for completion
            if recv_req is not None:
                recv_req.wait()
            if send_req is not None:
                send_req.wait()

        return result


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
