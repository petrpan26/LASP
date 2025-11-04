#!/bin/bash
# Run LASP Blelloch benchmarks across different GPU counts

set -e

echo "=========================================="
echo "LASP Blelloch Performance Benchmarks"
echo "=========================================="
echo ""

# Configuration
BATCH_SIZE=4
NUM_HEADS=8
SEQ_LEN=4096
HIDDEN_DIM=512
NUM_TRIALS=100

# Create results directory
RESULTS_DIR="benchmark_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Function to run benchmark for a given GPU count
run_benchmark() {
    local num_gpus=$1
    echo "----------------------------------------"
    echo "Benchmarking with $num_gpus GPUs..."
    echo "----------------------------------------"

    if ! torchrun --nproc_per_node=$num_gpus tests/benchmark_blelloch.py \
        --batch-size $BATCH_SIZE \
        --num-heads $NUM_HEADS \
        --seq-len $SEQ_LEN \
        --hidden-dim $HIDDEN_DIM \
        --num-trials $NUM_TRIALS \
        --output "$RESULTS_DIR/benchmark_p${num_gpus}.json"; then
        echo "Warning: Benchmark failed for $num_gpus GPUs (may not have enough GPUs)"
        return 1
    fi

    echo ""
    return 0
}

# Auto-detect available GPUs
if command -v nvidia-smi &> /dev/null; then
    MAX_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Detected $MAX_GPUS GPUs"
    echo ""
else
    echo "Warning: nvidia-smi not found, assuming 8 GPUs"
    MAX_GPUS=8
fi

# GPU configurations to test (adjust based on your hardware)
GPU_CONFIGS=(1 2 4 8 16 32 64 128)

echo "Testing configurations: ${GPU_CONFIGS[@]}"
echo ""

# Run benchmarks
for num_gpus in "${GPU_CONFIGS[@]}"; do
    if [ $num_gpus -le $MAX_GPUS ]; then
        run_benchmark $num_gpus
    else
        echo "Skipping $num_gpus GPUs (only $MAX_GPUS available)"
    fi
done

echo "=========================================="
echo "Benchmarks Complete!"
echo "=========================================="
echo ""
echo "Results saved in: $RESULTS_DIR"
echo ""

# Generate summary if Python is available
if command -v python3 &> /dev/null; then
    echo "Generating summary..."
    python3 - <<EOF
import json
import os
import glob

results_dir = "$RESULTS_DIR"
files = sorted(glob.glob(os.path.join(results_dir, "benchmark_p*.json")))

if not files:
    print("No results found")
    exit(0)

print("\n" + "=" * 80)
print("BENCHMARK SUMMARY")
print("=" * 80)
print(f"\n{'GPUs':<10} {'Ring (ms)':<12} {'Blelloch (ms)':<15} {'Speedup':<10} {'Efficiency':<12}")
print("-" * 80)

import math

for file in files:
    with open(file) as f:
        data = json.load(f)

    world_size = data['world_size']
    ring_time = data['ring']['total_ms']
    blelloch_time = data['blelloch']['total_ms']
    speedup = data['speedup']['total']

    # Calculate efficiency
    num_levels = math.ceil(math.log2(world_size)) if world_size > 1 else 0
    theoretical = world_size / (2 * num_levels) if num_levels > 0 else 1.0
    efficiency = (speedup / theoretical * 100) if theoretical > 0 else 0

    print(f"{world_size:<10} {ring_time:<12.2f} {blelloch_time:<15.2f} {speedup:<10.2f}× {efficiency:<12.1f}%")

print("=" * 80)
EOF
fi

echo ""
echo "Done! 🚀"
