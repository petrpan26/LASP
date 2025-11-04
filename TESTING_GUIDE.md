# LASP Testing Guide

## Overview

The updated `tests/test.py` now includes all 6 LASP variants with integrated benchmarking capabilities.

## Supported Variants

All LASP implementations are now tested:

1. **`lasp_naive`** - Ring communication with basic kernels (baseline)
2. **`lasp_cache`** - Ring with cached KV buffers
3. **`lasp_fuse`** - Ring with fused kernels
4. **`lasp_fuse_parallel`** - Ring with fused parallel kernels
5. **`lasp_blelloch`** ⭐ - Blelloch tree O(log P) with basic kernels
6. **`lasp_blelloch_fused`** ⭐ - Blelloch tree O(log P) with fused kernels

## Quick Start

### Correctness Testing Only

Test all variants for correctness (compares against `lightning_attn` baseline):

```bash
# 8 GPUs with data_parallel_size=2, sequence_parallel_size=4
torchrun --nproc_per_node=8 tests/test.py --dp-size 2

# 4 GPUs with data_parallel_size=1, sequence_parallel_size=4
torchrun --nproc_per_node=4 tests/test.py --dp-size 1
```

### Correctness + Benchmarking

Add `--benchmark` flag to measure performance:

```bash
torchrun --nproc_per_node=8 tests/test.py --dp-size 2 --benchmark
```

This will:
- ✅ Run correctness tests for all variants
- ✅ Measure forward and backward pass timing
- ✅ Calculate speedup relative to naive baseline
- ✅ Print formatted results table

### Custom Benchmark Parameters

Control the number of trials and warmup iterations:

```bash
torchrun --nproc_per_node=8 tests/test.py --dp-size 2 --benchmark \
  --num-trials 200 --num-warmup 20
```

## Command Line Options

| Flag | Description | Default | Required |
|------|-------------|---------|----------|
| `--dp-size` | Data parallel size | - | ✅ Yes |
| `--benchmark` | Enable performance benchmarking | False | No |
| `--num-trials` | Number of benchmark iterations | 100 | No |
| `--num-warmup` | Number of warmup iterations | 10 | No |

## Output Examples

### Correctness Test Output

```
Test lasp_naive on world size 8 with data_parallel_size 2 and sequence_parallel_size 4:
### Forward ###
out diff: mean value: 1.234e-06
### Backward ###
dq diff: mean value: 2.345e-06
dk diff: mean value: 1.876e-06
dv diff: mean value: 1.543e-06
```

### Benchmark Output

```
================================================================================
BENCHMARK RESULTS
================================================================================
Configuration: world_size=8, dp_size=2, sp_size=4
Sequence length per GPU: 512, Total: 2048
Trials: 100, Warmup: 10

Method               Forward (ms)    Backward (ms)   Total (ms)      Speedup
--------------------------------------------------------------------------------
naive                1.234           2.456           3.690           1.00x
cache                1.198           2.412           3.610           1.02x
fuse                 0.987           2.145           3.132           1.18x
fuse_parallel        0.876           1.998           2.874           1.28x
blelloch             0.945           1.876           2.821           1.31x
blelloch_fused       0.798           1.654           2.452           1.50x
================================================================================
```

## Understanding Results

### Correctness Metrics

The test compares each LASP variant against `lightning_attn` (non-distributed reference):

- **out diff**: Forward pass output difference
- **dq/dk/dv diff**: Backward pass gradient differences

✅ **Expected**: Differences should be < 1e-4 (numerical precision)

### Performance Metrics

- **Forward (ms)**: Average time for forward pass over `num_trials` iterations
- **Backward (ms)**: Average time for backward pass over `num_trials` iterations
- **Total (ms)**: Forward + Backward time
- **Speedup**: Ratio of `naive_total / method_total`

### Expected Speedups

At different scales (sequence_parallel_size):

| Method | P=4 | P=8 | P=16 | P=32 | P=64 | P=128 |
|--------|-----|-----|------|------|------|-------|
| `naive` | 1.0× | 1.0× | 1.0× | 1.0× | 1.0× | 1.0× |
| `cache` | ~1.05× | ~1.05× | ~1.05× | ~1.05× | ~1.05× | ~1.05× |
| `fuse` | ~1.2× | ~1.2× | ~1.2× | ~1.2× | ~1.2× | ~1.2× |
| `fuse_parallel` | ~1.4× | ~1.4× | ~1.4× | ~1.4× | ~1.4× | ~1.4× |
| `blelloch` | ~1.0× | ~1.3× | ~1.9× | ~3.0× | ~5.0× | **~6-9×** |
| `blelloch_fused` | ~1.0× | ~1.4× | ~2.0× | ~3.2× | ~5.3× | **~7-10×** |

**Key Insight**: Blelloch variants shine at large scale (P ≥ 16) due to O(log P) communication.

## Test Configuration

The test uses these parameters (defined in `test.py`):

```python
b = world_size * 2      # Batch size (scales with GPUs)
n = 2048                # Total sequence length
h = 12                  # Number of heads
d = 128                 # Hidden dimension
e = 64                  # Value dimension
```

- **Sequence length per GPU**: `n_local = n // sequence_parallel_size`
- **Batch size per GPU**: `b_local = b // data_parallel_size`

## Parallelism Explanation

With `world_size=8` and `--dp-size 2`:

- **Data parallel groups** (2 groups, 4 GPUs each):
  - Group 0: GPUs {0, 1, 2, 3}
  - Group 1: GPUs {4, 5, 6, 7}

- **Sequence parallel groups** (4 groups, 2 GPUs each):
  - Group 0: GPUs {0, 4}
  - Group 1: GPUs {1, 5}
  - Group 2: GPUs {2, 6}
  - Group 3: GPUs {3, 7}

Each GPU processes:
- Batch chunk: `b_local = b // 2` (2 data parallel groups)
- Sequence chunk: `n_local = n // 4` (4 sequence parallel groups)

## Troubleshooting

### NCCL Errors

If you see communication errors:

```bash
# Check GPU availability
nvidia-smi

# Enable NCCL debug output
export NCCL_DEBUG=INFO
torchrun --nproc_per_node=8 tests/test.py --dp-size 2
```

### Numerical Differences

If correctness tests fail with large differences:

1. **Check data type**: Test uses `torch.bfloat16` by default
2. **Check decay factors**: Very small/large decays can cause numerical issues
3. **Try float32**: Modify `dtype = torch.float32` in test.py for testing

### Performance Lower Than Expected

If speedup is less than expected:

1. **Check GPU topology**: `nvidia-smi topo -m`
2. **Increase problem size**: Larger sequences show better speedup
3. **Profile communication**: Use `torch.profiler` to identify bottlenecks
4. **Check world_size**: Blelloch needs P ≥ 16 for significant gains

## Advanced Usage

### Custom Test Configuration

Modify these parameters in `test.py` for your use case:

```python
# Line 103: Adjust problem size
b, n, h, d, e = world_size * 2, 2048, 12, 128, 64

# Increase sequence length for large-scale testing
b, n, h, d, e = world_size * 2, 8192, 12, 128, 64

# Line 83: Change data type
dtype = torch.float32  # or torch.float16
```

### Testing Specific Methods

Comment out methods you don't want to test in `name_2_fn_dict`:

```python
name_2_fn_dict = {
    "naive": lasp_naive,
    # "cache": lasp_cache,  # Skip cache
    # "fuse": lasp_fuse,    # Skip fuse
    "blelloch": lasp_blelloch,
    "blelloch_fused": lasp_blelloch_fused,
}
```

### Multi-Node Testing

For multi-node testing with multiple machines:

```bash
# Node 0 (master)
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
  --master_addr=<master_ip> --master_port=29500 \
  tests/test.py --dp-size 2 --benchmark

# Node 1
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
  --master_addr=<master_ip> --master_port=29500 \
  tests/test.py --dp-size 2 --benchmark
```

## Integration with CI/CD

Add to your continuous integration pipeline:

```yaml
# .github/workflows/test.yml
- name: Test LASP variants
  run: |
    torchrun --nproc_per_node=4 tests/test.py --dp-size 1

- name: Benchmark LASP
  run: |
    torchrun --nproc_per_node=8 tests/test.py --dp-size 2 \
      --benchmark --num-trials 50 --num-warmup 5
```

## Summary

The updated `tests/test.py` provides:

✅ **Complete coverage** - All 6 LASP variants tested
✅ **Correctness validation** - Compares against reference implementation
✅ **Performance benchmarking** - Integrated timing with proper warmup
✅ **Flexible configuration** - Customizable trials, warmup, parallelism
✅ **Clear output** - Formatted tables with speedup calculations

**Recommended workflow**:
1. Run correctness tests first to validate implementation
2. Add `--benchmark` to measure performance
3. Test at multiple scales (P=4, 8, 16, 32, 64) to see Blelloch benefits
4. Use results to choose optimal variant for your use case

Enjoy comprehensive LASP testing! 🚀
