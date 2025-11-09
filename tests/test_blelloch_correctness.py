"""
Correctness tests for LASP Blelloch implementation.

Verifies that Blelloch outputs match Ring implementation.
"""

import torch
import torch.distributed as dist
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lasp import lasp_naive, lasp_blelloch
from lasp.utils import initialize_lasp


def setup_distributed():
    """Initialize distributed environment for testing."""
    if not dist.is_initialized():
        # For testing, use environment variables
        # Launch with: torchrun --nproc_per_node=N test_blelloch_correctness.py
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())

    return rank, world_size


def test_forward_correctness(
    batch_size=2,
    num_heads=4,
    seq_len_per_gpu=128,
    hidden_dim=64,
    rtol=1e-5,
    atol=1e-6,
):
    """
    Test that Blelloch forward pass matches Ring forward pass.

    Args:
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len_per_gpu: Sequence length per GPU
        hidden_dim: Hidden dimension
        rtol: Relative tolerance
        atol: Absolute tolerance
    """
    rank, world_size = setup_distributed()

    # Initialize LASP
    initialize_lasp(data_parallel_size=1, sequence_parallel_size=world_size)

    device = torch.device(f'cuda:{rank}') if torch.cuda.is_available() else torch.device('cpu')

    # Generate same random inputs on all ranks (for testing)
    torch.manual_seed(42 + rank)  # Different seed per rank for realistic scenario

    # Create inputs
    q = torch.randn(batch_size, num_heads, seq_len_per_gpu, hidden_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len_per_gpu, hidden_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len_per_gpu, hidden_dim, device=device)

    # Decay factors (one per head)
    s = torch.rand(num_heads, device=device) * 0.1  # Small decay for stability

    # Make inputs require grad for backward test
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True

    # ===== Forward: Ring =====
    o_ring = lasp_naive(q.clone().detach().requires_grad_(True),
                        k.clone().detach().requires_grad_(True),
                        v.clone().detach().requires_grad_(True),
                        s)

    # ===== Forward: Blelloch =====
    o_blelloch = lasp_blelloch(q.clone().detach().requires_grad_(True),
                               k.clone().detach().requires_grad_(True),
                               v.clone().detach().requires_grad_(True),
                               s)

    # ===== Verify outputs match =====
    try:
        torch.testing.assert_close(o_ring, o_blelloch, rtol=rtol, atol=atol)
        if rank == 0:
            print(f"✓ Forward pass test PASSED (world_size={world_size})")
            print(f"  Max absolute difference: {(o_ring - o_blelloch).abs().max().item():.2e}")
            print(f"  Mean absolute difference: {(o_ring - o_blelloch).abs().mean().item():.2e}")
        return True
    except AssertionError as e:
        if rank == 0:
            print(f"✗ Forward pass test FAILED (world_size={world_size})")
            print(f"  Error: {e}")
            print(f"  Max difference: {(o_ring - o_blelloch).abs().max().item():.2e}")
        return False


def test_backward_correctness(
    batch_size=2,
    num_heads=4,
    seq_len_per_gpu=128,
    hidden_dim=64,
    rtol=1e-4,
    atol=1e-5,
):
    """
    Test that Blelloch backward pass matches Ring backward pass.
    """
    rank, world_size = setup_distributed()

    # Initialize LASP
    initialize_lasp(data_parallel_size=1, sequence_parallel_size=world_size)

    device = torch.device(f'cuda:{rank}') if torch.cuda.is_available() else torch.device('cpu')

    # Generate inputs
    torch.manual_seed(42 + rank)

    q_ring = torch.randn(batch_size, num_heads, seq_len_per_gpu, hidden_dim, device=device, requires_grad=True)
    k_ring = torch.randn(batch_size, num_heads, seq_len_per_gpu, hidden_dim, device=device, requires_grad=True)
    v_ring = torch.randn(batch_size, num_heads, seq_len_per_gpu, hidden_dim, device=device, requires_grad=True)

    q_blelloch = q_ring.clone().detach().requires_grad_(True)
    k_blelloch = k_ring.clone().detach().requires_grad_(True)
    v_blelloch = v_ring.clone().detach().requires_grad_(True)

    s = torch.rand(num_heads, device=device) * 0.1

    # ===== Forward + Backward: Ring =====
    o_ring = lasp_naive(q_ring, k_ring, v_ring, s)
    grad_out = torch.randn_like(o_ring)  # Random gradient
    o_ring.backward(grad_out)

    dq_ring = q_ring.grad.clone()
    dk_ring = k_ring.grad.clone()
    dv_ring = v_ring.grad.clone()

    # ===== Forward + Backward: Blelloch =====
    o_blelloch = lasp_blelloch(q_blelloch, k_blelloch, v_blelloch, s)
    o_blelloch.backward(grad_out)

    dq_blelloch = q_blelloch.grad
    dk_blelloch = k_blelloch.grad
    dv_blelloch = v_blelloch.grad

    # ===== Verify gradients match =====
    all_passed = True

    try:
        torch.testing.assert_close(dq_ring, dq_blelloch, rtol=rtol, atol=atol)
        if rank == 0:
            print(f"✓ Backward dq test PASSED")
    except AssertionError as e:
        all_passed = False
        if rank == 0:
            print(f"✗ Backward dq test FAILED")
            print(f"  Max difference: {(dq_ring - dq_blelloch).abs().max().item():.2e}")

    try:
        torch.testing.assert_close(dk_ring, dk_blelloch, rtol=rtol, atol=atol)
        if rank == 0:
            print(f"✓ Backward dk test PASSED")
    except AssertionError as e:
        all_passed = False
        if rank == 0:
            print(f"✗ Backward dk test FAILED")
            print(f"  Max difference: {(dk_ring - dk_blelloch).abs().max().item():.2e}")

    try:
        torch.testing.assert_close(dv_ring, dv_blelloch, rtol=rtol, atol=atol)
        if rank == 0:
            print(f"✓ Backward dv test PASSED")
    except AssertionError as e:
        all_passed = False
        if rank == 0:
            print(f"✗ Backward dv test FAILED")
            print(f"  Max difference: {(dv_ring - dv_blelloch).abs().max().item():.2e}")

    return all_passed


def test_single_gpu():
    """Test that Blelloch works correctly with single GPU (no communication)."""
    rank, world_size = setup_distributed()

    if world_size > 1:
        if rank == 0:
            print("Skipping single GPU test (world_size > 1)")
        return True

    device = torch.device(f'cuda:{rank}') if torch.cuda.is_available() else torch.device('cpu')

    # Initialize LASP
    initialize_lasp(data_parallel_size=1, sequence_parallel_size=1)

    # Create inputs
    q = torch.randn(2, 4, 128, 64, device=device, requires_grad=True)
    k = torch.randn(2, 4, 128, 64, device=device, requires_grad=True)
    v = torch.randn(2, 4, 128, 64, device=device, requires_grad=True)
    s = torch.rand(4, device=device) * 0.1

    # Both should give same result with world_size=1
    o_ring = lasp_naive(q.clone().detach().requires_grad_(True),
                        k.clone().detach().requires_grad_(True),
                        v.clone().detach().requires_grad_(True),
                        s)
    o_blelloch = lasp_blelloch(q, k, v, s)

    try:
        torch.testing.assert_close(o_ring, o_blelloch, rtol=1e-5, atol=1e-6)
        print("✓ Single GPU test PASSED")
        return True
    except AssertionError as e:
        print(f"✗ Single GPU test FAILED: {e}")
        return False


if __name__ == "__main__":
    """
    Run tests.

    Usage:
        # Single GPU test
        python test_blelloch_correctness.py

        # Multi-GPU test (4 GPUs)
        torchrun --nproc_per_node=4 test_blelloch_correctness.py

        # Multi-GPU test (8 GPUs)
        torchrun --nproc_per_node=8 test_blelloch_correctness.py
    """
    rank, world_size = setup_distributed()

    if rank == 0:
        print("=" * 80)
        print("LASP Blelloch Correctness Tests")
        print("=" * 80)
        print(f"World size: {world_size}")
        print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        print()

    # Run tests
    passed = []

    if world_size == 1:
        passed.append(test_single_gpu())
    else:
        passed.append(test_forward_correctness())
        passed.append(test_backward_correctness())

    # Summary
    if rank == 0:
        print()
        print("=" * 80)
        if all(passed):
            print("✓ All tests PASSED!")
        else:
            print("✗ Some tests FAILED")
            sys.exit(1)
        print("=" * 80)

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()
