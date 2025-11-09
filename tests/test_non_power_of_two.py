"""
Test Blelloch with non-power-of-2 GPU counts.

Verifies that world_size does NOT need to be 2^k.
"""

import torch
import torch.distributed as dist
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lasp import lasp_naive, lasp_blelloch
from lasp.utils import initialize_lasp


def setup_distributed():
    """Initialize distributed environment."""
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())

    return rank, world_size


def test_non_power_of_two(world_size_expected=None):
    """
    Test Blelloch with non-power-of-2 GPU count.

    How it works:
    - For world_size=7, padded to 8 (next power of 2)
    - Virtual rank 7 doesn't exist
    - Ranks that would communicate with rank 7 skip that communication
    - Creates an unbalanced tree (perfectly fine!)

    Example tree for world_size=7:
        Level 0: 0  1  2  3  4  5  6  [7 virtual]
                 |\ |\ |\ |\  |\ |\  |
        Level 1: | 1  | 3  | 5  |  6 (rank 7 would be here)
                 |  \  |  \ |  \  |
        Level 2: |   3     |   6  (rank 5→7 skipped)
                 |     \   |   /
        Level 3: |      6  (rank 3→7 skipped)
    """
    rank, world_size = setup_distributed()

    if world_size_expected and world_size != world_size_expected:
        if rank == 0:
            print(f"Expected world_size={world_size_expected}, got {world_size}")
            print("Launch with: torchrun --nproc_per_node=N test_non_power_of_two.py")
        return

    # Check if power of 2
    is_power_of_2 = (world_size & (world_size - 1)) == 0 and world_size > 0

    if rank == 0:
        print(f"Testing with world_size={world_size}")
        print(f"Is power of 2: {is_power_of_2}")
        if not is_power_of_2:
            import math
            padded = 2 ** math.ceil(math.log2(world_size))
            print(f"Will be padded to: {padded}")
        print()

    # Initialize
    initialize_lasp(data_parallel_size=1, sequence_parallel_size=world_size)

    device = torch.device(f'cuda:{rank}') if torch.cuda.is_available() else torch.device('cpu')

    # Create test inputs
    torch.manual_seed(42 + rank)
    batch_size, num_heads, seq_len, hidden_dim = 2, 4, 128, 64

    q = torch.randn(batch_size, num_heads, seq_len, hidden_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, hidden_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, hidden_dim, device=device)
    s = torch.rand(num_heads, device=device) * 0.1

    # Test forward pass
    try:
        o_blelloch = lasp_blelloch(q, k, v, s)
        o_ring = lasp_naive(q, k, v, s)

        # Verify they match
        torch.testing.assert_close(o_ring, o_blelloch, rtol=1e-5, atol=1e-6)

        if rank == 0:
            print(f"✓ Forward pass PASSED (world_size={world_size})")
            print(f"  Max difference: {(o_ring - o_blelloch).abs().max().item():.2e}")

    except Exception as e:
        if rank == 0:
            print(f"✗ Forward pass FAILED (world_size={world_size})")
            print(f"  Error: {e}")
        raise

    # Test backward pass
    try:
        q_ring = q.clone().detach().requires_grad_(True)
        k_ring = k.clone().detach().requires_grad_(True)
        v_ring = v.clone().detach().requires_grad_(True)

        q_blelloch = q.clone().detach().requires_grad_(True)
        k_blelloch = k.clone().detach().requires_grad_(True)
        v_blelloch = v.clone().detach().requires_grad_(True)

        o_ring = lasp_naive(q_ring, k_ring, v_ring, s)
        o_blelloch = lasp_blelloch(q_blelloch, k_blelloch, v_blelloch, s)

        grad_out = torch.randn_like(o_ring)
        o_ring.backward(grad_out)
        o_blelloch.backward(grad_out)

        torch.testing.assert_close(q_ring.grad, q_blelloch.grad, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(k_ring.grad, k_blelloch.grad, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(v_ring.grad, v_blelloch.grad, rtol=1e-4, atol=1e-5)

        if rank == 0:
            print(f"✓ Backward pass PASSED (world_size={world_size}")

    except Exception as e:
        if rank == 0:
            print(f"✗ Backward pass FAILED (world_size={world_size})")
            print(f"  Error: {e}")
        raise

    if rank == 0:
        print()
        print("=" * 60)
        print(f"✓ ALL TESTS PASSED for world_size={world_size}")
        if not is_power_of_2:
            print("  (non-power-of-2 handled correctly!)")
        print("=" * 60)


if __name__ == "__main__":
    """
    Test various non-power-of-2 world sizes.

    Usage:
        # Test with 3 GPUs (not power of 2)
        torchrun --nproc_per_node=3 test_non_power_of_two.py

        # Test with 5 GPUs
        torchrun --nproc_per_node=5 test_non_power_of_two.py

        # Test with 7 GPUs
        torchrun --nproc_per_node=7 test_non_power_of_two.py

        # Test with 10 GPUs
        torchrun --nproc_per_node=10 test_non_power_of_two.py

        # Test with 100 GPUs
        torchrun --nproc_per_node=100 test_non_power_of_two.py
    """
    rank, world_size = setup_distributed()

    if rank == 0:
        print("=" * 60)
        print("Testing Blelloch with Non-Power-of-2 World Sizes")
        print("=" * 60)
        print()

    test_non_power_of_two()

    if dist.is_initialized():
        dist.destroy_process_group()
