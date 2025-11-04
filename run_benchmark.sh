#!/bin/bash

# Comprehensive benchmark runner for LASP variants
# Runs dedicated benchmark with cache clearing and 100 trials

set -e

# Default configuration
NUM_TRIALS=100
NUM_WARMUP=10
SEQ_LEN=2048
DP_SIZE=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --dp-size)
            DP_SIZE="$2"
            shift 2
            ;;
        --num-trials)
            NUM_TRIALS="$2"
            shift 2
            ;;
        --num-warmup)
            NUM_WARMUP="$2"
            shift 2
            ;;
        --seq-len)
            SEQ_LEN="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpus N           Number of GPUs to use (required)"
            echo "  --dp-size N        Data parallel size (default: 1)"
            echo "  --num-trials N     Number of benchmark trials (default: 100)"
            echo "  --num-warmup N     Number of warmup iterations (default: 10)"
            echo "  --seq-len N        Total sequence length (default: 2048)"
            echo "  --output FILE      Output JSON file for results"
            echo "  --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --gpus 8 --dp-size 2"
            echo "  $0 --gpus 8 --dp-size 2 --num-trials 200 --output results.json"
            echo "  $0 --gpus 16 --dp-size 2 --seq-len 4096"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$GPUS" ]; then
    echo "Error: --gpus is required"
    echo "Use --help for usage information"
    exit 1
fi

# Calculate SP size
SP_SIZE=$((GPUS / DP_SIZE))

# Check if valid
if [ $((DP_SIZE * SP_SIZE)) -ne $GPUS ]; then
    echo "Error: dp_size ($DP_SIZE) must divide evenly into number of GPUs ($GPUS)"
    exit 1
fi

# Build command
CMD="torchrun --nproc_per_node=$GPUS tests/benchmark_all_methods.py"
CMD="$CMD --dp-size $DP_SIZE"
CMD="$CMD --num-trials $NUM_TRIALS"
CMD="$CMD --num-warmup $NUM_WARMUP"
CMD="$CMD --seq-len $SEQ_LEN"

if [ ! -z "$OUTPUT" ]; then
    CMD="$CMD --output $OUTPUT"
fi

echo "="
echo "LASP Comprehensive Benchmark"
echo "="
echo "Configuration:"
echo "  GPUs: $GPUS"
echo "  Data Parallel Size: $DP_SIZE"
echo "  Sequence Parallel Size: $SP_SIZE"
echo "  Sequence Length: $SEQ_LEN"
echo "  Trials: $NUM_TRIALS"
echo "  Warmup: $NUM_WARMUP"
if [ ! -z "$OUTPUT" ]; then
    echo "  Output: $OUTPUT"
fi
echo "="
echo ""
echo "Running: $CMD"
echo ""

# Run benchmark
$CMD
