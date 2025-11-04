"""
Comprehensive benchmark for all LASP variants.

This script benchmarks all 6 LASP implementations with proper:
- Cache clearing between runs
- Separate forward and backward timing
- Statistical analysis (mean, median, std)
- 100 trials per method
- Warmup iterations
"""

import argparse
import gc
import json
import time
from collections import defaultdict

import torch
import torch.distributed as dist
from einops import rearrange

from lasp import (
    lasp_blelloch,
    lasp_cache,
    lasp_fuse,
    lasp_fuse_parallel,
    lasp_naive,
)
from lasp.utils import (
    build_slope_tensor,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    initialize_lasp,
)


def clear_cache():
    """Clear CUDA cache and run garbage collection."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()


def benchmark_forward(run_fn, num_trials=100, num_warmup=10):
    """Benchmark forward pass only."""
    times = []

    # Warmup
    for _ in range(num_warmup):
        clear_cache()
        _ = run_fn()
        torch.cuda.synchronize()

    # Benchmark
    for _ in range(num_trials):
        clear_cache()

        torch.cuda.synchronize()
        start = time.perf_counter()
        output = run_fn()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000  # ms

        times.append(elapsed)

        # Clean up
        del output

    return times


def benchmark_backward(run_fn, grad_output, num_trials=100, num_warmup=10):
    """Benchmark forward + backward pass."""
    forward_times = []
    backward_times = []
    total_times = []

    # Warmup
    for _ in range(num_warmup):
        clear_cache()
        output = run_fn()
        output.backward(grad_output, retain_graph=False)
        torch.cuda.synchronize()

    # Benchmark
    for _ in range(num_trials):
        clear_cache()

        # Time forward
        torch.cuda.synchronize()
        start_fwd = time.perf_counter()
        output = run_fn()
        torch.cuda.synchronize()
        fwd_time = (time.perf_counter() - start_fwd) * 1000

        # Time backward
        torch.cuda.synchronize()
        start_bwd = time.perf_counter()
        output.backward(grad_output, retain_graph=False)
        torch.cuda.synchronize()
        bwd_time = (time.perf_counter() - start_bwd) * 1000

        forward_times.append(fwd_time)
        backward_times.append(bwd_time)
        total_times.append(fwd_time + bwd_time)

        # Clean up
        del output

    return forward_times, backward_times, total_times


def compute_stats(times):
    """Compute statistics from timing data."""
    import statistics
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min": min(times),
        "max": max(times),
    }


def benchmark_all_methods(
    dp_size,
    num_trials=100,
    num_warmup=10,
    seq_len=2048,
    batch_size_multiplier=2,
    num_heads=12,
    hidden_dim=128,
    value_dim=64,
    output_file=None,
):
    """
    Benchmark all LASP variants.

    Args:
        dp_size: Data parallel size
        num_trials: Number of benchmark iterations per method
        num_warmup: Number of warmup iterations
        seq_len: Total sequence length
        batch_size_multiplier: Batch size = world_size * multiplier
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension
        value_dim: Value dimension
        output_file: Path to save JSON results
    """
    # Initialize distributed
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    sp_size = world_size // dp_size
    initialize_lasp(dp_size, sp_size)

    sp_rank = get_sequence_parallel_rank()

    # Test configuration
    b = world_size * batch_size_multiplier
    n = seq_len
    h = num_heads
    d = hidden_dim
    e = value_dim

    assert n % sp_size == 0, f"Sequence length {n} must be divisible by SP size {sp_size}"

    b_local = b // dp_size
    n_local = n // sp_size

    dtype = torch.bfloat16

    if rank == 0:
        print("="*80)
        print("LASP COMPREHENSIVE BENCHMARK")
        print("="*80)
        print(f"Configuration:")
        print(f"  World size: {world_size}")
        print(f"  Data parallel size: {dp_size}")
        print(f"  Sequence parallel size: {sp_size}")
        print(f"  Batch size: {b} (local: {b_local})")
        print(f"  Sequence length: {n} (local: {n_local})")
        print(f"  Num heads: {h}")
        print(f"  Hidden dim: {d}")
        print(f"  Value dim: {e}")
        print(f"  Dtype: {dtype}")
        print(f"  Num trials: {num_trials}")
        print(f"  Num warmup: {num_warmup}")
        print("="*80)
        print()

    # Create test data (local chunks)
    q = torch.randn(b_local, h, n_local, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(b_local, h, n_local, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(b_local, h, n_local, e, device=device, dtype=dtype, requires_grad=True)
    do_grad = torch.randn(b_local, h, n_local, e, device=device, dtype=dtype)
    s = build_slope_tensor(h).to(device).to(torch.float32)

    # Define all methods
    methods = {
        "naive": {
            "fn": lasp_naive,
            "needs_buffers": False,
        },
        "cache": {
            "fn": lasp_cache,
            "needs_buffers": "cache",  # Special case
        },
        "fuse": {
            "fn": lasp_fuse,
            "needs_buffers": True,
        },
        "fuse_parallel": {
            "fn": lasp_fuse_parallel,
            "needs_buffers": True,
        },
        "blelloch": {
            "fn": lasp_blelloch,
            "needs_buffers": True,
        },
    }

    # Storage for results
    results = {}

    # Benchmark each method
    for method_name, method_info in methods.items():
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"Benchmarking: {method_name}")
            print(f"{'='*80}")

        dist.barrier()
        clear_cache()

        # Prepare inputs based on method interface
        if not method_info["needs_buffers"]:
            # Simple interface: naive, blelloch, blelloch_fused
            def run_forward():
                q.grad = None
                k.grad = None
                v.grad = None
                return method_info["fn"](q, k, v, s)

        elif method_info["needs_buffers"] == "cache":
            # Cache interface
            KV = torch.empty(b_local, h, d, e, dtype=torch.float32, device=device)
            DKV = torch.empty(b_local, h, d, e, dtype=torch.float32, device=device)
            array = torch.arange(n_local, device=device, dtype=dtype)

            def run_forward():
                q.grad = None
                k.grad = None
                v.grad = None
                return method_info["fn"](q, k, v, s, array, KV, DKV)

        else:
            # Fuse interface: fuse, fuse_parallel
            KV = torch.empty(b_local, h, d, e, dtype=torch.float32, device=device)
            DKV = torch.empty(b_local, h, d, e, dtype=torch.float32, device=device)

            def run_forward():
                q.grad = None
                k.grad = None
                v.grad = None
                return method_info["fn"](q, k, v, s, KV, DKV)

        # Benchmark forward + backward
        if rank == 0:
            print(f"  Running {num_trials} trials with {num_warmup} warmup iterations...")

        forward_times, backward_times, total_times = benchmark_backward(
            run_forward, do_grad, num_trials, num_warmup
        )

        # Compute statistics
        forward_stats = compute_stats(forward_times)
        backward_stats = compute_stats(backward_times)
        total_stats = compute_stats(total_times)

        results[method_name] = {
            "forward": forward_stats,
            "backward": backward_stats,
            "total": total_stats,
        }

        if rank == 0:
            print(f"  Forward:  {forward_stats['mean']:.3f} ± {forward_stats['std']:.3f} ms")
            print(f"  Backward: {backward_stats['mean']:.3f} ± {backward_stats['std']:.3f} ms")
            print(f"  Total:    {total_stats['mean']:.3f} ± {total_stats['std']:.3f} ms")

        dist.barrier()
        clear_cache()

    # Print summary table
    if rank == 0:
        print("\n" + "="*80)
        print("SUMMARY RESULTS")
        print("="*80)
        print()

        # Get baseline (naive)
        baseline_fwd = results["naive"]["forward"]["mean"]
        baseline_bwd = results["naive"]["backward"]["mean"]
        baseline_total = results["naive"]["total"]["mean"]

        # Print header
        print(f"{'Method':<20} {'Forward (ms)':<18} {'Backward (ms)':<18} {'Total (ms)':<18} {'Speedup':<10}")
        print("-" * 90)

        # Print each method
        for method_name in methods.keys():
            res = results[method_name]
            fwd_mean = res["forward"]["mean"]
            fwd_std = res["forward"]["std"]
            bwd_mean = res["backward"]["mean"]
            bwd_std = res["backward"]["std"]
            total_mean = res["total"]["mean"]
            total_std = res["total"]["std"]

            speedup = baseline_total / total_mean if total_mean > 0 else 0.0

            print(f"{method_name:<20} {fwd_mean:>7.3f} ± {fwd_std:<5.3f}   {bwd_mean:>7.3f} ± {bwd_std:<5.3f}   {total_mean:>7.3f} ± {total_std:<5.3f}   {speedup:>6.2f}x")

        print("="*80)

        # Detailed statistics
        print("\nDETAILED STATISTICS")
        print("="*80)

        for method_name in methods.keys():
            res = results[method_name]
            print(f"\n{method_name}:")
            print(f"  Forward:  mean={res['forward']['mean']:.3f} ms, "
                  f"median={res['forward']['median']:.3f} ms, "
                  f"std={res['forward']['std']:.3f} ms, "
                  f"min={res['forward']['min']:.3f} ms, "
                  f"max={res['forward']['max']:.3f} ms")
            print(f"  Backward: mean={res['backward']['mean']:.3f} ms, "
                  f"median={res['backward']['median']:.3f} ms, "
                  f"std={res['backward']['std']:.3f} ms, "
                  f"min={res['backward']['min']:.3f} ms, "
                  f"max={res['backward']['max']:.3f} ms")
            print(f"  Total:    mean={res['total']['mean']:.3f} ms, "
                  f"median={res['total']['median']:.3f} ms, "
                  f"std={res['total']['std']:.3f} ms, "
                  f"min={res['total']['min']:.3f} ms, "
                  f"max={res['total']['max']:.3f} ms")

        print("="*80)

        # Save results to JSON
        if output_file:
            output_data = {
                "configuration": {
                    "world_size": world_size,
                    "dp_size": dp_size,
                    "sp_size": sp_size,
                    "batch_size": b,
                    "batch_size_local": b_local,
                    "seq_len": n,
                    "seq_len_local": n_local,
                    "num_heads": h,
                    "hidden_dim": d,
                    "value_dim": e,
                    "dtype": str(dtype),
                    "num_trials": num_trials,
                    "num_warmup": num_warmup,
                },
                "results": results,
            }

            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=2)

            print(f"\nResults saved to: {output_file}")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive benchmark for all LASP variants")
    parser.add_argument("--dp-size", type=int, required=True, help="Data parallel size")
    parser.add_argument("--num-trials", type=int, default=100, help="Number of benchmark trials (default: 100)")
    parser.add_argument("--num-warmup", type=int, default=10, help="Number of warmup iterations (default: 10)")
    parser.add_argument("--seq-len", type=int, default=2048, help="Total sequence length (default: 2048)")
    parser.add_argument("--batch-multiplier", type=int, default=2, help="Batch size multiplier (batch = world_size * multiplier)")
    parser.add_argument("--num-heads", type=int, default=12, help="Number of attention heads (default: 12)")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension (default: 128)")
    parser.add_argument("--value-dim", type=int, default=64, help="Value dimension (default: 64)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for results")

    args = parser.parse_args()

    benchmark_all_methods(
        dp_size=args.dp_size,
        num_trials=args.num_trials,
        num_warmup=args.num_warmup,
        seq_len=args.seq_len,
        batch_size_multiplier=args.batch_multiplier,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        value_dim=args.value_dim,
        output_file=args.output,
    )
