# LASP Comprehensive Benchmark Suite

## Overview

Dedicated benchmark script that properly measures all 6 LASP variants with:

✅ **100 trials per method** (configurable)
✅ **Cache clearing between runs** (`torch.cuda.empty_cache()` + garbage collection)
✅ **Separate forward and backward timing**
✅ **Statistical analysis** (mean, median, std, min, max)
✅ **Proper warmup** (10 iterations, configurable)
✅ **JSON output** for result storage

## Quick Start

### Basic Benchmark (8 GPUs)

```bash
./run_benchmark.sh --gpus 8 --dp-size 2
```

This will:
- Run on 8 GPUs with data_parallel_size=2, sequence_parallel_size=4
- Benchmark all 6 LASP variants
- 100 trials each with 10 warmup iterations
- Clear cache between each run

### With Custom Configuration

```bash
./run_benchmark.sh --gpus 8 --dp-size 2 \
  --num-trials 200 \
  --num-warmup 20 \
  --seq-len 4096 \
  --output results_8gpu.json
```

## Usage

### Using the Shell Script (Recommended)

```bash
./run_benchmark.sh [OPTIONS]

Options:
  --gpus N           Number of GPUs to use (required)
  --dp-size N        Data parallel size (default: 1)
  --num-trials N     Number of benchmark trials (default: 100)
  --num-warmup N     Number of warmup iterations (default: 10)
  --seq-len N        Total sequence length (default: 2048)
  --output FILE      Output JSON file for results
  --help             Show help message
```

### Using torchrun Directly

```bash
torchrun --nproc_per_node=8 tests/benchmark_all_methods.py \
  --dp-size 2 \
  --num-trials 100 \
  --num-warmup 10 \
  --seq-len 2048 \
  --output results.json
```

## What Gets Benchmarked

All 6 LASP variants are tested:

1. **naive** - Ring communication + basic kernels (baseline)
2. **cache** - Ring + cached KV buffers
3. **fuse** - Ring + fused kernels
4. **fuse_parallel** - Ring + fused parallel kernels
5. **blelloch** - Tree O(log P) + basic kernels
6. **blelloch_fused** - Tree O(log P) + fused kernels

For each method, we measure:
- **Forward pass time** (separately)
- **Backward pass time** (separately)
- **Total time** (forward + backward)

## Output Format

### Console Output

```
================================================================================
LASP COMPREHENSIVE BENCHMARK
================================================================================
Configuration:
  World size: 8
  Data parallel size: 2
  Sequence parallel size: 4
  Batch size: 16 (local: 8)
  Sequence length: 2048 (local: 512)
  Num heads: 12
  Hidden dim: 128
  Value dim: 64
  Dtype: torch.bfloat16
  Num trials: 100
  Num warmup: 10
================================================================================

================================================================================
Benchmarking: naive
================================================================================
  Running 100 trials with 10 warmup iterations...
  Forward:  1.234 ± 0.045 ms
  Backward: 2.456 ± 0.089 ms
  Total:    3.690 ± 0.112 ms

[... similar for all methods ...]

================================================================================
SUMMARY RESULTS
================================================================================

Method               Forward (ms)       Backward (ms)      Total (ms)         Speedup
------------------------------------------------------------------------------------------
naive                  1.234 ± 0.045      2.456 ± 0.089      3.690 ± 0.112      1.00x
cache                  1.198 ± 0.042      2.412 ± 0.087      3.610 ± 0.108      1.02x
fuse                   0.987 ± 0.038      2.145 ± 0.078      3.132 ± 0.095      1.18x
fuse_parallel          0.876 ± 0.035      1.998 ± 0.072      2.874 ± 0.089      1.28x
blelloch               0.945 ± 0.037      1.876 ± 0.069      2.821 ± 0.093      1.31x
blelloch_fused         0.798 ± 0.032      1.654 ± 0.061      2.452 ± 0.078      1.50x
================================================================================

DETAILED STATISTICS
================================================================================

naive:
  Forward:  mean=1.234 ms, median=1.230 ms, std=0.045 ms, min=1.180 ms, max=1.350 ms
  Backward: mean=2.456 ms, median=2.450 ms, std=0.089 ms, min=2.320 ms, max=2.680 ms
  Total:    mean=3.690 ms, median=3.680 ms, std=0.112 ms, min=3.500 ms, max=4.030 ms

[... similar for all methods ...]
================================================================================

Results saved to: results.json
```

### JSON Output

When using `--output results.json`:

```json
{
  "configuration": {
    "world_size": 8,
    "dp_size": 2,
    "sp_size": 4,
    "batch_size": 16,
    "batch_size_local": 8,
    "seq_len": 2048,
    "seq_len_local": 512,
    "num_heads": 12,
    "hidden_dim": 128,
    "value_dim": 64,
    "dtype": "torch.bfloat16",
    "num_trials": 100,
    "num_warmup": 10
  },
  "results": {
    "naive": {
      "forward": {
        "mean": 1.234,
        "median": 1.230,
        "std": 0.045,
        "min": 1.180,
        "max": 1.350
      },
      "backward": {
        "mean": 2.456,
        "median": 2.450,
        "std": 0.089,
        "min": 2.320,
        "max": 2.680
      },
      "total": {
        "mean": 3.690,
        "median": 3.680,
        "std": 0.112,
        "min": 3.500,
        "max": 4.030
      }
    },
    ...
  }
}
```

## Key Features

### 1. Cache Clearing

Between each trial, the script:
```python
torch.cuda.empty_cache()  # Clear CUDA cache
gc.collect()              # Python garbage collection
torch.cuda.synchronize()  # Ensure completion
```

This ensures:
- No memory fragmentation affects results
- Each trial starts from clean state
- More consistent timing measurements

### 2. Separate Forward/Backward Timing

```python
# Forward timing (separately)
torch.cuda.synchronize()
start = time.perf_counter()
output = run_forward()
torch.cuda.synchronize()
forward_time = (time.perf_counter() - start) * 1000

# Backward timing (separately)
torch.cuda.synchronize()
start = time.perf_counter()
output.backward(grad_output)
torch.cuda.synchronize()
backward_time = (time.perf_counter() - start) * 1000
```

You can see which methods optimize forward vs backward.

### 3. Statistical Analysis

For each measurement, we compute:
- **Mean**: Average time across all trials
- **Median**: Middle value (robust to outliers)
- **Std**: Standard deviation (variability)
- **Min**: Best case performance
- **Max**: Worst case performance

### 4. Proper Warmup

The first 10 iterations (configurable) are discarded to:
- Let GPU clocks stabilize
- Warm up kernel caches
- Eliminate JIT compilation overhead
- Reach steady-state performance

## Examples

### 1. Quick Benchmark (8 GPUs)

```bash
./run_benchmark.sh --gpus 8 --dp-size 2
```

### 2. High-Precision Benchmark (200 trials)

```bash
./run_benchmark.sh --gpus 8 --dp-size 2 --num-trials 200 --num-warmup 20
```

### 3. Large Sequence Length

```bash
./run_benchmark.sh --gpus 8 --dp-size 2 --seq-len 8192
```

### 4. Save Results to JSON

```bash
./run_benchmark.sh --gpus 8 --dp-size 2 --output benchmark_8gpu.json
```

### 5. Multi-Scale Benchmarking

Benchmark across multiple GPU counts:

```bash
# 4 GPUs
./run_benchmark.sh --gpus 4 --dp-size 1 --output results_4gpu.json

# 8 GPUs
./run_benchmark.sh --gpus 8 --dp-size 2 --output results_8gpu.json

# 16 GPUs (if available)
./run_benchmark.sh --gpus 16 --dp-size 2 --output results_16gpu.json

# 32 GPUs (if available)
./run_benchmark.sh --gpus 32 --dp-size 4 --output results_32gpu.json
```

### 6. Direct torchrun (More Control)

```bash
torchrun --nproc_per_node=8 tests/benchmark_all_methods.py \
  --dp-size 2 \
  --num-trials 100 \
  --num-warmup 10 \
  --seq-len 2048 \
  --batch-multiplier 2 \
  --num-heads 12 \
  --hidden-dim 128 \
  --value-dim 64 \
  --output results.json
```

## Understanding Results

### Speedup Calculation

Speedup is relative to **naive** method:
```
Speedup = naive_total_time / method_total_time
```

- **1.0x**: Same speed as baseline
- **1.5x**: 50% faster than baseline
- **2.0x**: 2× faster (half the time)
- **6.0x**: 6× faster (6x less time)

### Expected Speedups at Different Scales

| Method | P=4 | P=8 | P=16 | P=32 | P=64 | P=128 |
|--------|-----|-----|------|------|------|-------|
| naive | 1.0× | 1.0× | 1.0× | 1.0× | 1.0× | 1.0× |
| cache | 1.05× | 1.05× | 1.05× | 1.05× | 1.05× | 1.05× |
| fuse | 1.2× | 1.2× | 1.2× | 1.2× | 1.2× | 1.2× |
| fuse_parallel | 1.4× | 1.4× | 1.4× | 1.4× | 1.4× | 1.4× |
| blelloch | 1.0× | 1.3× | 1.9× | 3.0× | 5.0× | **6-9×** |
| blelloch_fused | 1.0× | 1.4× | 2.0× | 3.2× | 5.3× | **7-10×** |

**Key Insight**: Blelloch variants scale better with more GPUs due to O(log P) communication.

### Interpreting Standard Deviation

- **Low std (< 5% of mean)**: Consistent performance, reliable results
- **Medium std (5-10% of mean)**: Some variance, acceptable
- **High std (> 10% of mean)**: Unstable, may need more trials or investigation

### Median vs Mean

- **Median ≈ Mean**: Symmetrical distribution, no outliers
- **Median < Mean**: Some slow outliers pulling mean up
- **Median > Mean**: Some fast outliers pulling mean down

Use median for more robust comparison if you see outliers.

## Troubleshooting

### CUDA Out of Memory

Reduce sequence length:
```bash
./run_benchmark.sh --gpus 8 --dp-size 2 --seq-len 1024
```

### High Variance in Results

Increase warmup and trials:
```bash
./run_benchmark.sh --gpus 8 --dp-size 2 --num-warmup 20 --num-trials 200
```

### NCCL Errors

Enable debug output:
```bash
export NCCL_DEBUG=INFO
./run_benchmark.sh --gpus 8 --dp-size 2
```

### Slow Benchmark

Reduce trials for quick testing:
```bash
./run_benchmark.sh --gpus 8 --dp-size 2 --num-trials 20
```

## Advanced Usage

### Custom Problem Size

```bash
torchrun --nproc_per_node=8 tests/benchmark_all_methods.py \
  --dp-size 2 \
  --seq-len 4096 \
  --batch-multiplier 4 \
  --num-heads 16 \
  --hidden-dim 256 \
  --value-dim 128
```

### Benchmark Specific Methods Only

Edit `tests/benchmark_all_methods.py` and comment out methods you don't want:

```python
methods = {
    # "naive": {...},     # Skip naive
    # "cache": {...},     # Skip cache
    # "fuse": {...},      # Skip fuse
    # "fuse_parallel": {...},  # Skip fuse_parallel
    "blelloch": {...},        # Only test blelloch
    "blelloch_fused": {...},  # Only test blelloch_fused
}
```

### Multi-Node Benchmarking

```bash
# Node 0
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
  --master_addr=<IP> --master_port=29500 \
  tests/benchmark_all_methods.py --dp-size 2 --output results.json

# Node 1
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
  --master_addr=<IP> --master_port=29500 \
  tests/benchmark_all_methods.py --dp-size 2 --output results.json
```

## Performance Tips

### 1. GPU Topology

Check GPU interconnect:
```bash
nvidia-smi topo -m
```

NVLink/NVSwitch gives better performance than PCIe.

### 2. GPU Clocks

Ensure GPUs are not throttled:
```bash
nvidia-smi -q -d CLOCK
```

Set persistence mode:
```bash
sudo nvidia-smi -pm 1
```

### 3. Reduce Interference

- Close other GPU applications
- Use exclusive GPU mode if possible
- Run during low system load

### 4. Consistent Environment

- Same GPU driver version
- Same CUDA version
- Same PyTorch version
- Same NCCL version

## Comparison with test.py

| Feature | test.py --benchmark | benchmark_all_methods.py |
|---------|-------------------|--------------------------|
| **Cache clearing** | ❌ No | ✅ Yes (between each trial) |
| **Trials** | ✅ Configurable | ✅ Default 100 |
| **Statistics** | ❌ Mean only | ✅ Mean, median, std, min, max |
| **JSON output** | ❌ No | ✅ Yes |
| **Dedicated script** | ❌ Integrated | ✅ Standalone |
| **Correctness testing** | ✅ Yes | ❌ No (benchmark only) |

**Recommendation**:
- Use `test.py --benchmark` for quick correctness + performance check
- Use `benchmark_all_methods.py` for thorough performance analysis

## Files

- **`tests/benchmark_all_methods.py`** - Main benchmark script
- **`run_benchmark.sh`** - Convenient runner script
- **`BENCHMARK_README.md`** - This file

## Summary

This benchmark suite provides:

✅ **Accurate timing** - Cache clearing, proper synchronization
✅ **Statistical rigor** - 100 trials, multiple metrics
✅ **Complete coverage** - All 6 LASP variants
✅ **Easy to use** - Simple shell script interface
✅ **Detailed output** - Console table + JSON export
✅ **Flexible configuration** - All parameters configurable

**Typical workflow**:

1. **Quick correctness check**: `torchrun --nproc_per_node=8 tests/test.py --dp-size 2`
2. **Thorough benchmark**: `./run_benchmark.sh --gpus 8 --dp-size 2 --output results.json`
3. **Analyze results**: Review console output and JSON file
4. **Compare scales**: Run at P=4, 8, 16, 32, 64 to see scaling behavior

Enjoy comprehensive, accurate LASP benchmarking! 🚀
