import argparse
import time

import torch
import torch.distributed as dist
from einops import rearrange

from lasp import (
    lasp_blelloch,
    lasp_blelloch_v2,
    lasp_cache,
    lasp_fuse,
    lasp_fuse_parallel,
    lasp_fuse_v2,
    lasp_zeco,
    lasp_naive,
    lightning_attn,
)
from lasp.utils import (
    build_slope_tensor,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    initialize_lasp,
)


def log(msg, a, rank0_only=False):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if rank0_only:
        if rank == 0:
            print(
                f"{msg}: "
                f"mean value: {a.abs().mean().item()}",
                flush=True,
            )
        return

    for i in range(world_size):
        if i == rank:
            if rank == 0:
                print(f"{msg}:")
            print(
                f"[{rank}] "
                f"mean value: {a.abs().mean().item()}",
                flush=True,
            )
        dist.barrier()


def split_data(x):
    # x: b, h, n, d
    sp_size = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()
    dp_size = get_data_parallel_world_size(sp_size > 1)
    dp_rank = get_data_parallel_rank(sp_size > 1)

    # split over batch
    x = rearrange(x, "(b g) ... -> g b ... ", g=dp_size)[dp_rank]
    # split over sequence
    x = rearrange(x, "b h (g n) d -> b h g n d", g=sp_size)[:, :, sp_rank]

    return x.detach().clone()


def test(dp_size, benchmark=False, num_trials=100, num_warmup=10):
    """
    As an example, assume we have 1 node with 8 GPUs and the ranks are {0, 1, 2, 3, 4, 5, 6, 7}. For data parallel size = 2 and sequence parallel size = 4, the DP and SP communication groups will be:

    4 data_parallel groups (with global rank indices):
    (0, 4), (1, 5), (2, 6), (3, 7)

    2 sequence paralell groups (with global rank indices):
    (0, 1, 2, 3), (4, 5, 6, 7)

    In summary, the group maping (with their own rank indices) is as follows:
    Global ranks:             0, 1, 2, 3, 4, 5, 6, 7
    Data parallel ranks:      0, 0, 0, 0, 1, 1, 1, 1
    Sequence parallel ranks:  0, 1, 2, 3, 0, 1, 2, 3

    In the following example, we initialize data loading on global rank 0, then broadcast data chunks to other ranks. Each GPU gets their own data chunk according to the data parallel rank and sequence parallel rank.
    """
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")
    sp_size = world_size // dp_size
    initialize_lasp(dp_size, sp_size)

    name_2_fn_dict = {
        "naive": lasp_naive,
        "cache": lasp_cache,
        "fuse": lasp_fuse,
        "fuse_v2": lasp_fuse_v2,
        "zeco": lasp_zeco,
        "fuse_parallel": lasp_fuse_parallel,
        "blelloch": lasp_blelloch,
        "blelloch_v2": lasp_blelloch_v2,
    }

    # Storage for benchmark results
    benchmark_results = {}

    b, n, h, d, e = world_size * 2, 2048, 12, 128, 64

    assert (
        n % sp_size == 0
    ), f"Sequence length {n} must be devided by sequence prallel size {sp_size}"
    b_local = b // dp_size
    n_local = n // sp_size

    # broadcast data on rank 0, then split along batch and sequence dim
    q = torch.randn(b, h, n, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(b, h, n, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(b, h, n, e, device=device, dtype=dtype, requires_grad=True)
    do = torch.randn(b, h, n, e, device=device, dtype=dtype, requires_grad=True)
    s = build_slope_tensor(h).to(device).to(torch.float32)

    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)
    dist.broadcast(do, src=0)

    get_sequence_parallel_rank()
    qi, ki, vi, doi = map(split_data, [q, k, v, do])

    qi.requires_grad = True
    ki.requires_grad = True
    vi.requires_grad = True

    dist.barrier()

    o = lightning_attn(q, k, v, s)
    o.backward(do)
    dq = q.grad
    dk = k.grad
    dv = v.grad

    oi_ref, dq_ref, dk_ref, dv_ref = map(split_data, [o, dq, dk, dv])

    for name in name_2_fn_dict:
        qi.grad = None
        ki.grad = None
        vi.grad = None

        f = name_2_fn_dict[name]
        if rank == 0:
            print("\n")
            print(
                f"Test lasp_{name} on world size {world_size} with data_parallel_size {dp_size} and sequence_parallel_size {sp_size}:"
            )

        # Determine which interface to use
        if name in ["naive"]:
            # Simple interface
            def run_forward():
                return f(qi, ki, vi, s)
        elif name == "cache":
            # Cache interface with array
            KV = torch.empty(b_local, h, d, e).to(torch.float32).to(q.device)
            DKV = torch.empty(b_local, h, d, e).to(torch.float32).to(q.device)
            array = torch.arange(n_local).to(q)
            def run_forward():
                return f(qi, ki, vi, s, array, KV, DKV)
        elif name == "zeco":
            # ZeCO interface - no KV/DKV buffers needed
            def run_forward():
                return f(qi, ki, vi, s)
        else:
            # Fuse interface with KV, DKV (fuse, fuse_v2, fuse_parallel, blelloch)
            KV = torch.empty(b_local, h, d, e).to(torch.float32).to(q.device)
            DKV = torch.empty(b_local, h, d, e).to(torch.float32).to(q.device)
            def run_forward():
                return f(qi, ki, vi, s, KV, DKV)

        # Benchmarking mode
        if benchmark:
            # Warmup
            for _ in range(num_warmup):
                qi.grad = None
                ki.grad = None
                vi.grad = None
                oi_tmp = run_forward()
                oi_tmp.backward(doi, retain_graph=True)

            dist.barrier()

            # Forward benchmark
            forward_times = []
            for _ in range(num_trials):
                qi.grad = None
                ki.grad = None
                vi.grad = None

                torch.cuda.synchronize()
                start = time.perf_counter()
                oi_tmp = run_forward()
                torch.cuda.synchronize()
                forward_times.append((time.perf_counter() - start) * 1000)

            # Backward benchmark
            backward_times = []
            for _ in range(num_trials):
                qi.grad = None
                ki.grad = None
                vi.grad = None
                oi_tmp = run_forward()

                torch.cuda.synchronize()
                start = time.perf_counter()
                oi_tmp.backward(doi, retain_graph=True)
                torch.cuda.synchronize()
                backward_times.append((time.perf_counter() - start) * 1000)

            # Store results
            avg_forward = sum(forward_times) / len(forward_times)
            avg_backward = sum(backward_times) / len(backward_times)
            benchmark_results[name] = {
                "forward": avg_forward,
                "backward": avg_backward,
                "total": avg_forward + avg_backward,
            }

        # Correctness test
        if rank == 0:
            print("### Forward ###")

        oi = run_forward()
        log("out diff", oi_ref - oi, rank0_only=True)

        dist.barrier()
        if rank == 0:
            print("### Backward ###")

        oi.backward(doi, retain_graph=True)
        dqi = qi.grad.clone()
        dki = ki.grad.clone()
        dvi = vi.grad.clone()

        log("dq diff", dq_ref - dqi, rank0_only=True)
        log("dk diff", dk_ref - dki, rank0_only=True)
        log("dv diff", dv_ref - dvi, rank0_only=True)

    # Print benchmark results
    if benchmark and rank == 0:
        print("\n" + "="*80)
        print("BENCHMARK RESULTS")
        print("="*80)
        print(f"Configuration: world_size={world_size}, dp_size={dp_size}, sp_size={sp_size}")
        print(f"Sequence length per GPU: {n_local}, Total: {n}")
        print(f"Trials: {num_trials}, Warmup: {num_warmup}")
        print("\n")

        # Print table header
        print(f"{'Method':<20} {'Forward (ms)':<15} {'Backward (ms)':<15} {'Total (ms)':<15} {'Speedup':<10}")
        print("-" * 80)

        # Get baseline (naive) for speedup calculation
        baseline_total = benchmark_results.get("naive", {}).get("total", 1.0)

        # Print results for each method
        for name in name_2_fn_dict.keys():
            if name in benchmark_results:
                res = benchmark_results[name]
                speedup = baseline_total / res["total"] if res["total"] > 0 else 0.0
                print(f"{name:<20} {res['forward']:<15.3f} {res['backward']:<15.3f} {res['total']:<15.3f} {speedup:<10.2f}x")

        print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-size", help="data parallel size", type=int, required=True)
    parser.add_argument("--benchmark", help="run performance benchmark", action="store_true")
    parser.add_argument("--num-trials", help="number of benchmark trials", type=int, default=100)
    parser.add_argument("--num-warmup", help="number of warmup iterations", type=int, default=10)
    args = parser.parse_args()

    test(args.dp_size, benchmark=args.benchmark, num_trials=args.num_trials, num_warmup=args.num_warmup)
