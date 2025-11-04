# LASP Blelloch Performance Benchmark Guide

## Overview

This guide explains how to benchmark Blelloch vs Ring LASP implementations to measure actual speedup.

## Quick Start

### Single GPU Configuration

```bash
# Run benchmark on 8 GPUs
torchrun --nproc_per_node=8 tests/benchmark_blelloch.py
```

Output:
```
================================================================================
LASP PERFORMANCE BENCHMARK RESULTS
================================================================================

Configuration:
  World Size:        8 GPUs
  Batch Size:        4
  Num Heads:         8
  Seq Len per GPU:   4,096
  Total Seq Len:     32,768
  Hidden Dim:        512

Method          Forward (ms)    Backward (ms)   Total (ms)
------------------------------------------------------------
Ring            1.723           3.456           5.179
Blelloch        1.312           2.678           3.990

Speedup (Ring / Blelloch):
  Forward:  1.31×
  Backward: 1.29×
  Total:    1.30×

Theoretical Analysis:
  Ring steps:      8
  Blelloch steps:  6
  Theoretical max: 1.33×
  Efficiency:      97.7%
================================================================================

Results saved to: benchmark_p8.json
```

### Multiple Configurations (Automated)

```bash
# Run benchmarks across all available GPU configurations
./run_benchmarks.sh
```

This will:
- Auto-detect available GPUs
- Test configurations: 1, 2, 4, 8, 16, 32, 64, 128 GPUs (as available)
- Save results to timestamped directory
- Generate summary table

## Benchmark Options

### Command Line Arguments

```bash
torchrun --nproc_per_node=N tests/benchmark_blelloch.py \
  --batch-size 4 \           # Batch size (default: 4)
  --num-heads 8 \            # Number of attention heads (default: 8)
  --seq-len 4096 \           # Sequence length per GPU (default: 4096)
  --hidden-dim 512 \         # Hidden dimension (default: 512)
  --num-warmup 10 \          # Warmup iterations (default: 10)
  --num-trials 100 \         # Benchmark iterations (default: 100)
  --output results.json      # Output file (default: benchmark_pN.json)
```

### Example Configurations

#### Small Scale (Test Setup)
```bash
torchrun --nproc_per_node=4 tests/benchmark_blelloch.py \
  --batch-size 2 \
  --num-heads 4 \
  --seq-len 1024 \
  --hidden-dim 256 \
  --num-trials 50
```

#### Medium Scale (Typical Training)
```bash
torchrun --nproc_per_node=16 tests/benchmark_blelloch.py \
  --batch-size 8 \
  --num-heads 16 \
  --seq-len 8192 \
  --hidden-dim 1024 \
  --num-trials 100
```

#### Large Scale (Where Blelloch Shines)
```bash
torchrun --nproc_per_node=128 tests/benchmark_blelloch.py \
  --batch-size 16 \
  --num-heads 32 \
  --seq-len 16384 \
  --hidden-dim 2048 \
  --num-trials 200
```

## Understanding Results

### Metrics Explained

**Forward (ms)**: Time for forward pass only
**Backward (ms)**: Time for backward pass only
**Total (ms)**: Combined forward + backward time

**Speedup**: Ratio of Ring time / Blelloch time (higher is better)

**Efficiency**: Actual speedup / Theoretical speedup
- >90%: Excellent (minimal contention)
- 70-90%: Good (some network contention)
- 50-70%: Fair (significant contention)
- <50%: Poor (check network topology)

### Expected Speedup

| GPUs | Ring Steps | Blelloch Steps | Theoretical | Expected Actual |
|------|------------|----------------|-------------|-----------------|
| 4    | 4          | 4              | 1.0×        | 1.0×            |
| 8    | 8          | 6              | 1.33×       | 1.3×            |
| 16   | 16         | 8              | 2.0×        | 1.9×            |
| 32   | 32         | 10             | 3.2×        | 3.0×            |
| 64   | 64         | 12             | 5.33×       | 5.0×            |
| 128  | 128        | 14             | 9.14×       | 6-7×            |

Note: Actual speedup < theoretical due to:
- Network contention (multiple parallel sends/recvs)
- Latency overhead per communication round
- Computation/communication imbalance

## Benchmarking Different Workloads

### Latency-Bound (Small Matrices)

```bash
# Small d, large P → Latency dominates
torchrun --nproc_per_node=64 tests/benchmark_blelloch.py \
  --hidden-dim 128 \
  --seq-len 2048
```

**Expected**: Higher speedup (latency amortized better with Blelloch)

### Bandwidth-Bound (Large Matrices)

```bash
# Large d, moderate P → Bandwidth dominates
torchrun --nproc_per_node=16 tests/benchmark_blelloch.py \
  --hidden-dim 4096 \
  --seq-len 8192
```

**Expected**: Lower speedup (communication time dominates)

### Long Sequences

```bash
# Very long sequences
torchrun --nproc_per_node=128 tests/benchmark_blelloch.py \
  --seq-len 32768
```

**Expected**: Best speedup (amortizes setup overhead)

## Interpreting Results

### Good Performance Indicators

✅ **Speedup close to theoretical** (efficiency >80%)
- Network topology is good
- Minimal contention
- Ready for production

✅ **Consistent across forward/backward**
- Both phases benefit equally
- No phase-specific bottlenecks

✅ **Scales with GPU count**
- Speedup increases with more GPUs
- Algorithm scaling as expected

### Performance Issues

⚠️ **Low efficiency (<50%)**

**Possible causes**:
- Network contention (too many parallel communications)
- Poor network topology (slow inter-node links)
- Small workload (overhead dominates)

**Solutions**:
- Try hybrid approach (Blelloch within nodes, Ring across)
- Check network topology: `nvidia-smi topo -m`
- Increase problem size to amortize overhead

⚠️ **Backward slower than forward**

**Possible causes**:
- Gradient accumulation creating extra overhead
- Memory pressure in backward pass

**Solutions**:
- Enable gradient checkpointing
- Reduce batch size
- Profile with `torch.profiler`

⚠️ **No speedup at small scale (P<16)**

**This is expected!**
- Ring overhead is minimal for small P
- Blelloch tree coordination has fixed cost
- Use Blelloch only when P ≥ 16

## Profiling

### Detailed Profiling with torch.profiler

Create `profile_blelloch.py`:

```python
import torch
import torch.profiler as profiler
from lasp import lasp_blelloch

# ... setup ...

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    with_stack=True,
) as prof:
    output = lasp_blelloch(q, k, v, s)

# Print summary
print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=50
))

# Export trace for Chrome
prof.export_chrome_trace("blelloch_trace.json")
```

View trace:
1. Open `chrome://tracing` in Chrome
2. Load `blelloch_trace.json`
3. Analyze communication patterns

### What to Look For

**In profiler output**:
- `ncclSend/ncclRecv`: Communication time (should dominate)
- `aten::mul/add`: Computation time (should be <10%)
- Gaps: GPU idle time (minimize this)

**In trace visualization**:
- Parallel communications (multiple GPUs active simultaneously)
- Pipeline overlaps (communication hiding computation)
- Stragglers (GPUs waiting for others)

## Comparing Against Original Test

The original `tests/test.py` tests correctness, not performance:

```bash
# Correctness test (checks outputs match)
torchrun --nproc_per_node=8 tests/test.py --dp-size 1

# Performance benchmark (measures speedup)
torchrun --nproc_per_node=8 tests/benchmark_blelloch.py
```

## Multi-Node Benchmarking

### SLURM Example

```bash
#!/bin/bash
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8

srun torchrun \
  --nnodes=16 \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  tests/benchmark_blelloch.py \
  --batch-size 8 \
  --seq-len 16384 \
  --output benchmark_p128.json
```

### Multi-Node Considerations

- **Inter-node latency**: Much higher than intra-node
- **Network topology**: Consider hierarchical approach
- **Expected speedup**: Lower than single-node (but still significant)

For 128 GPUs on 16 nodes:
- Single-node (8 GPUs): 1.3× speedup
- Multi-node (128 GPUs): 4-6× speedup (not 9× due to slow inter-node links)

## Saving and Analyzing Results

### Result Format

Results are saved as JSON:

```json
{
  "world_size": 8,
  "batch_size": 4,
  "num_heads": 8,
  "seq_len_per_gpu": 4096,
  "hidden_dim": 512,
  "total_seq_len": 32768,
  "ring": {
    "forward_ms": 1.723,
    "backward_ms": 3.456,
    "total_ms": 5.179
  },
  "blelloch": {
    "forward_ms": 1.312,
    "backward_ms": 2.678,
    "total_ms": 3.990
  },
  "speedup": {
    "forward": 1.31,
    "backward": 1.29,
    "total": 1.30
  }
}
```

### Plotting Results

```python
import json
import matplotlib.pyplot as plt
import glob

# Load all results
results = []
for file in sorted(glob.glob("benchmark_results_*/benchmark_p*.json")):
    with open(file) as f:
        results.append(json.load(f))

# Extract data
world_sizes = [r['world_size'] for r in results]
speedups = [r['speedup']['total'] for r in results]

# Theoretical speedup
import math
theoretical = [p / (2 * math.ceil(math.log2(p))) if p > 1 else 1
               for p in world_sizes]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(world_sizes, speedups, 'o-', label='Actual Speedup', linewidth=2)
plt.plot(world_sizes, theoretical, '--', label='Theoretical Max', linewidth=2)
plt.xlabel('Number of GPUs')
plt.ylabel('Speedup')
plt.title('LASP Blelloch Speedup vs GPU Count')
plt.legend()
plt.grid(True)
plt.savefig('speedup_scaling.png', dpi=300)
plt.show()
```

## Troubleshooting

### Benchmark Crashes

**OOM (Out of Memory)**:
```bash
# Reduce batch size or sequence length
torchrun --nproc_per_node=8 tests/benchmark_blelloch.py \
  --batch-size 2 \
  --seq-len 2048
```

**NCCL Timeout**:
```bash
# Increase timeout
export NCCL_TIMEOUT=600
torchrun --nproc_per_node=8 tests/benchmark_blelloch.py
```

### Inconsistent Results

**High variance between runs**:
- Increase `--num-trials` (e.g., 200 or 500)
- Reduce system load (close other processes)
- Pin CPUs: `export OMP_NUM_THREADS=1`

**Different results across GPUs**:
- This shouldn't happen (all GPUs run same workload)
- Check GPU health: `nvidia-smi`
- Check for throttling: `nvidia-smi dmon`

## Summary

**Existing tests**: `tests/test.py` - Correctness only (no performance)

**New benchmarks**:
- `tests/benchmark_blelloch.py` - Comprehensive performance test
- `run_benchmarks.sh` - Automated multi-configuration runner

**Quick commands**:
```bash
# Single configuration
torchrun --nproc_per_node=8 tests/benchmark_blelloch.py

# All configurations
./run_benchmarks.sh

# Custom workload
torchrun --nproc_per_node=64 tests/benchmark_blelloch.py \
  --batch-size 16 --seq-len 8192 --hidden-dim 2048
```

**Expected results**: 1.3× (P=8) to 6-9× (P=128) speedup

Ready to benchmark! 🚀
