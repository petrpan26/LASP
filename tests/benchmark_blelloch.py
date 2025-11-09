"""
Performance benchmark for LASP Blelloch vs Ring.

Measures communication time, throughput, and speedup.
"""

import argparse
import torch
import torch.distributed as dist
import time
import json
from typing import Dict, List

import sys
import os
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


def benchmark_method(
    method_fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    num_warmup: int = 10,
    num_trials: int = 100,
) -> Dict[str, float]:
    """
    Benchmark a LASP method.

    Args:
        method_fn: Function to benchmark (lasp_naive or lasp_blelloch)
        q, k, v, s: Input tensors
        num_warmup: Number of warmup iterations
        num_trials: Number of benchmark iterations

    Returns:
        Dictionary with timing statistics
    """
    # Warmup
    for _ in range(num_warmup):
        _ = method_fn(q, k, v, s)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark forward pass
    start_time = time.perf_counter()
    for _ in range(num_trials):
        o = method_fn(q, k, v, s)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    forward_time = (time.perf_counter() - start_time) / num_trials

    # Benchmark backward pass
    grad_out = torch.randn_like(o)

    # Clear gradients
    if q.grad is not None:
        q.grad.zero_()
    if k.grad is not None:
        k.grad.zero_()
    if v.grad is not None:
        v.grad.zero_()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(num_trials):
        o = method_fn(q.clone().detach().requires_grad_(True),
                      k.clone().detach().requires_grad_(True),
                      v.clone().detach().requires_grad_(True),
                      s)
        o.backward(grad_out)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    backward_time = (time.perf_counter() - start_time) / num_trials

    total_time = forward_time + backward_time

    return {
        'forward_ms': forward_time * 1000,
        'backward_ms': backward_time * 1000,
        'total_ms': total_time * 1000,
    }


def run_benchmark(
    batch_size: int = 4,
    num_heads: int = 8,
    seq_len_per_gpu: int = 4096,
    hidden_dim: int = 512,
    num_warmup: int = 10,
    num_trials: int = 100,
) -> Dict:
    """
    Run complete benchmark comparing Ring vs Blelloch.

    Returns:
        Dictionary with all benchmark results
    """
    rank, world_size = setup_distributed()

    # Initialize LASP
    initialize_lasp(data_parallel_size=1, sequence_parallel_size=world_size)

    device = torch.device(f'cuda:{rank}') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32

    # Create inputs
    torch.manual_seed(42 + rank)

    q = torch.randn(batch_size, num_heads, seq_len_per_gpu, hidden_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, num_heads, seq_len_per_gpu, hidden_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, num_heads, seq_len_per_gpu, hidden_dim, device=device, dtype=dtype, requires_grad=True)
    s = torch.rand(num_heads, device=device, dtype=torch.float32) * 0.1

    # Benchmark Ring
    if rank == 0:
        print(f"Benchmarking Ring LASP...")
    ring_stats = benchmark_method(lasp_naive, q, k, v, s, num_warmup, num_trials)

    # Benchmark Blelloch
    if rank == 0:
        print(f"Benchmarking Blelloch LASP...")
    blelloch_stats = benchmark_method(lasp_blelloch, q, k, v, s, num_warmup, num_trials)

    # Calculate speedup
    results = {
        'world_size': world_size,
        'batch_size': batch_size,
        'num_heads': num_heads,
        'seq_len_per_gpu': seq_len_per_gpu,
        'hidden_dim': hidden_dim,
        'total_seq_len': seq_len_per_gpu * world_size,
        'ring': ring_stats,
        'blelloch': blelloch_stats,
        'speedup': {
            'forward': ring_stats['forward_ms'] / blelloch_stats['forward_ms'],
            'backward': ring_stats['backward_ms'] / blelloch_stats['backward_ms'],
            'total': ring_stats['total_ms'] / blelloch_stats['total_ms'],
        }
    }

    return results


def print_results(results: Dict):
    """Pretty print benchmark results."""
    print("\n" + "=" * 80)
    print("LASP PERFORMANCE BENCHMARK RESULTS")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  World Size:        {results['world_size']} GPUs")
    print(f"  Batch Size:        {results['batch_size']}")
    print(f"  Num Heads:         {results['num_heads']}")
    print(f"  Seq Len per GPU:   {results['seq_len_per_gpu']}")
    print(f"  Total Seq Len:     {results['total_seq_len']:,}")
    print(f"  Hidden Dim:        {results['hidden_dim']}")

    print(f"\n{'Method':<15} {'Forward (ms)':<15} {'Backward (ms)':<15} {'Total (ms)':<15}")
    print("-" * 60)
    print(f"{'Ring':<15} {results['ring']['forward_ms']:<15.3f} {results['ring']['backward_ms']:<15.3f} {results['ring']['total_ms']:<15.3f}")
    print(f"{'Blelloch':<15} {results['blelloch']['forward_ms']:<15.3f} {results['blelloch']['backward_ms']:<15.3f} {results['blelloch']['total_ms']:<15.3f}")

    print(f"\nSpeedup (Ring / Blelloch):")
    print(f"  Forward:  {results['speedup']['forward']:.2f}×")
    print(f"  Backward: {results['speedup']['backward']:.2f}×")
    print(f"  Total:    {results['speedup']['total']:.2f}×")

    # Calculate theoretical speedup
    import math
    world_size = results['world_size']
    num_levels = math.ceil(math.log2(world_size)) if world_size > 1 else 0
    theoretical_steps_ring = world_size
    theoretical_steps_blelloch = 2 * num_levels
    theoretical_speedup = theoretical_steps_ring / theoretical_steps_blelloch if theoretical_steps_blelloch > 0 else 1.0

    print(f"\nTheoretical Analysis:")
    print(f"  Ring steps:      {theoretical_steps_ring}")
    print(f"  Blelloch steps:  {theoretical_steps_blelloch}")
    print(f"  Theoretical max: {theoretical_speedup:.2f}×")
    print(f"  Efficiency:      {(results['speedup']['total'] / theoretical_speedup * 100):.1f}%")

    print("=" * 80)


def save_results(results: Dict, output_file: str):
    """Save results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


def scaling_benchmark(
    world_sizes: List[int],
    batch_size: int = 4,
    num_heads: int = 8,
    seq_len_per_gpu: int = 4096,
    hidden_dim: int = 512,
):
    """
    Run scaling benchmark across different world sizes.

    Note: This needs to be run separately for each world size.
    """
    rank, world_size = setup_distributed()

    if world_size not in world_sizes:
        if rank == 0:
            print(f"Warning: Current world_size={world_size} not in requested sizes {world_sizes}")
            print("Running benchmark anyway...")

    results = run_benchmark(batch_size, num_heads, seq_len_per_gpu, hidden_dim)

    if rank == 0:
        print_results(results)
        save_results(results, f"benchmark_results_p{world_size}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark LASP Blelloch vs Ring")
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--seq-len', type=int, default=4096, help='Sequence length per GPU')
    parser.add_argument('--hidden-dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--num-warmup', type=int, default=10, help='Number of warmup iterations')
    parser.add_argument('--num-trials', type=int, default=100, help='Number of benchmark trials')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file')

    args = parser.parse_args()

    rank, world_size = setup_distributed()

    if rank == 0:
        print("Starting benchmark...")
        print(f"World size: {world_size}")
        print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        print()

    results = run_benchmark(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        seq_len_per_gpu=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_warmup=args.num_warmup,
        num_trials=args.num_trials,
    )

    if rank == 0:
        print_results(results)

        if args.output:
            save_results(results, args.output)
        else:
            save_results(results, f"benchmark_p{world_size}.json")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()
