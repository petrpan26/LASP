"""
GPU configuration utility for tuning block sizes based on architecture-specific shared memory limits.
"""
import torch


# Shared memory limits per thread block (in bytes) by compute capability
# Based on NVIDIA documentation:
# - Compute Capability 6.x (Pascal): 48 KB per thread block
# - Compute Capability 7.0 (Volta): 48 KB per thread block
# - Compute Capability 7.5 (Turing): 48 KB per thread block
# - Compute Capability 8.x (Ampere): 163 KB per thread block (static: 48 KB, dynamic: up to 163 KB)
# - Compute Capability 8.9 (Ada Lovelace/RTX 4090): ~99 KB per thread block (varies by model)
# - Compute Capability 9.0 (Hopper): 227 KB per thread block (static: 48 KB, dynamic: up to 227 KB)

SMEM_LIMITS = {
    # Compute capability 6.x (Pascal)
    6.0: 48 * 1024,
    6.1: 48 * 1024,
    6.2: 48 * 1024,
    # Compute capability 7.0 (Volta)
    7.0: 48 * 1024,
    # Compute capability 7.5 (Turing)
    7.5: 48 * 1024,
    # Compute capability 8.0 (Ampere A100)
    8.0: 163 * 1024,
    # Compute capability 8.6 (Ampere consumer, RTX 3090, etc.)
    8.6: 163 * 1024,
    # Compute capability 8.9 (Ada Lovelace, RTX 4090)
    # Note: RTX 4090 typically has ~99 KB limit per thread block
    8.9: 99 * 1024,
    # Compute capability 9.0 (Hopper)
    9.0: 227 * 1024,
}

# Default to conservative 48 KB if architecture not found
DEFAULT_SMEM_LIMIT = 48 * 1024


def get_compute_capability(device=None):
    """Get the compute capability of the current or specified GPU."""
    if device is None:
        device = torch.cuda.current_device()

    props = torch.cuda.get_device_properties(device)
    major = props.major
    minor = props.minor
    compute_cap = float(f"{major}.{minor}")

    return compute_cap


def get_shared_memory_limit(device=None):
    """Get the shared memory limit per thread block for the current GPU."""
    compute_cap = get_compute_capability(device)

    # Try exact match first
    if compute_cap in SMEM_LIMITS:
        return SMEM_LIMITS[compute_cap]

    # Try matching by major version
    major_version = int(compute_cap)
    for cap, limit in SMEM_LIMITS.items():
        if int(cap) == major_version:
            return limit

    # Fall back to default
    return DEFAULT_SMEM_LIMIT


def get_optimal_block_sizes(n, d, e, device=None):
    """
    Calculate optimal BLOCK and BLOCK_MODEL sizes based on shared memory constraints.

    Args:
        n: Sequence length
        d: Query/key dimension
        e: Value dimension
        device: CUDA device (optional)

    Returns:
        tuple: (BLOCK, BLOCK_MODEL) sizes
    """
    smem_limit = get_shared_memory_limit(device)

    # Estimate shared memory usage per block
    # For forward kernel:
    # - q: BLOCK * d * 4 bytes (float32)
    # - k_trans: BLOCK * d * 4 bytes
    # - v: BLOCK * BLOCK_MODEL * 4 bytes
    # - kv: d * BLOCK_MODEL * 4 bytes
    # - Various temporary arrays: ~BLOCK^2 * 4 bytes for diag_decay
    # Total approximation: ~(2 * BLOCK * d + BLOCK * BLOCK_MODEL + d * BLOCK_MODEL + BLOCK^2) * 4

    # Start with conservative values
    BLOCK = 32
    BLOCK_MODEL = 16

    # Try to increase block sizes while staying within limit
    for block_size in [64, 128, 256]:
        for block_model in [16, 32, 64]:
            if block_model > e:
                continue

            # Rough estimate of shared memory usage
            qk_mem = 2 * block_size * d * 4  # q and k_trans
            v_mem = block_size * block_model * 4
            kv_mem = d * block_model * 4
            diag_mem = block_size * block_size * 4  # diag_decay matrix
            temp_mem = block_size * block_model * 4  # o_intra, o_inter

            total_mem = qk_mem + v_mem + kv_mem + diag_mem + temp_mem

            # Add 20% overhead for safety
            if total_mem * 1.2 <= smem_limit:
                BLOCK = block_size
                BLOCK_MODEL = block_model
            else:
                break
        if total_mem * 1.2 > smem_limit:
            break

    # Ensure BLOCK_MODEL doesn't exceed e and is power of 2
    try:
        import triton
        BLOCK_MODEL = min(BLOCK_MODEL, triton.next_power_of_2(e), 64)
    except ImportError:
        # Fallback: round down to nearest power of 2
        import math
        max_pow2 = 2 ** int(math.log2(min(BLOCK_MODEL, e, 64)))
        BLOCK_MODEL = max_pow2

    # Cap BLOCK at reasonable values
    BLOCK = min(BLOCK, 128)

    return BLOCK, BLOCK_MODEL


def get_optimal_cblock_size(BLOCK, device=None):
    """
    Calculate optimal CBLOCK size for backward kernels.

    Args:
        BLOCK: Main block size
        device: CUDA device (optional)

    Returns:
        int: CBLOCK size
    """
    smem_limit = get_shared_memory_limit(device)

    # CBLOCK is typically BLOCK // 2 or BLOCK // 4
    # For backward kernels, shared memory usage is similar to forward
    # but with CBLOCK instead of BLOCK for some operations

    # Start conservative
    CBLOCK = 16

    # Try increasing CBLOCK
    for cblock_size in [32, 64]:
        if cblock_size <= BLOCK and BLOCK % cblock_size == 0:
            # Estimate shared memory (conservative)
            # Similar to forward but with CBLOCK
            estimated_mem = 4 * cblock_size * cblock_size * 4  # Rough estimate
            if estimated_mem * 1.2 <= smem_limit:
                CBLOCK = cblock_size
            else:
                break

    return min(CBLOCK, BLOCK // 2)


# Fixed configurations pre-computed for common GPU architectures and dimensions
# Format: (compute_capability, kernel_type, n_range, d_range, e_range): {BLOCK, BLOCK_MODEL, CBLOCK}
# Ranges are (min, max) inclusive
FIXED_CONFIGS = {
    # RTX 4090 / Ada Lovelace (8.9) - 99KB shared memory limit
    (8.9, 'lightning', (512, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'BLOCK_MODEL': 16, 'CBLOCK': 16},
    (8.9, 'lasp_naive', (512, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'BLOCK_MODEL': 16, 'CBLOCK': 16},
    (8.9, 'lasp_cache', (512, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'BLOCK_MODEL': 16, 'CBLOCK': 16},
    (8.9, 'lasp_fuse', (128, 2048), (64, 256), (32, 128)): {'BLOCK': 32, 'CBLOCK': 16},
    (8.9, 'lasp_fuse', (2049, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'CBLOCK': 16},
    (8.9, 'lasp_fuse_parallel', (128, 2048), (64, 256), (32, 128)): {'BLOCK': 32, 'CBLOCK': 16},
    (8.9, 'lasp_fuse_parallel', (2049, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'CBLOCK': 16},
    (8.9, 'lasp_blelloch', (128, 2048), (64, 256), (32, 128)): {'BLOCK': 32, 'CBLOCK': 16},
    (8.9, 'lasp_blelloch', (2049, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'CBLOCK': 16},

    # Ampere A100 / RTX 3090 (8.0, 8.6) - 163KB shared memory limit
    (8.0, 'lightning', (512, 8192), (64, 256), (32, 128)): {'BLOCK': 64, 'BLOCK_MODEL': 32, 'CBLOCK': 32},
    (8.6, 'lightning', (512, 8192), (64, 256), (32, 128)): {'BLOCK': 64, 'BLOCK_MODEL': 32, 'CBLOCK': 32},
    (8.0, 'lasp_naive', (512, 8192), (64, 256), (32, 128)): {'BLOCK': 64, 'BLOCK_MODEL': 32, 'CBLOCK': 32},
    (8.6, 'lasp_naive', (512, 8192), (64, 256), (32, 128)): {'BLOCK': 64, 'BLOCK_MODEL': 32, 'CBLOCK': 32},
    (8.0, 'lasp_cache', (512, 8192), (64, 256), (32, 128)): {'BLOCK': 64, 'BLOCK_MODEL': 32, 'CBLOCK': 32},
    (8.6, 'lasp_cache', (512, 8192), (64, 256), (32, 128)): {'BLOCK': 64, 'BLOCK_MODEL': 32, 'CBLOCK': 32},
    (8.0, 'lasp_fuse', (128, 2048), (64, 256), (32, 128)): {'BLOCK': 64, 'CBLOCK': 32},
    (8.6, 'lasp_fuse', (128, 2048), (64, 256), (32, 128)): {'BLOCK': 64, 'CBLOCK': 32},
    (8.0, 'lasp_fuse', (2049, 8192), (64, 256), (32, 128)): {'BLOCK': 128, 'CBLOCK': 32},
    (8.6, 'lasp_fuse', (2049, 8192), (64, 256), (32, 128)): {'BLOCK': 128, 'CBLOCK': 32},
    (8.0, 'lasp_fuse_parallel', (128, 2048), (64, 256), (32, 128)): {'BLOCK': 64, 'CBLOCK': 32},
    (8.6, 'lasp_fuse_parallel', (128, 2048), (64, 256), (32, 128)): {'BLOCK': 64, 'CBLOCK': 32},
    (8.0, 'lasp_fuse_parallel', (2049, 8192), (64, 256), (32, 128)): {'BLOCK': 128, 'CBLOCK': 32},
    (8.6, 'lasp_fuse_parallel', (2049, 8192), (64, 256), (32, 128)): {'BLOCK': 128, 'CBLOCK': 32},
    (8.0, 'lasp_blelloch', (128, 2048), (64, 256), (32, 128)): {'BLOCK': 64, 'CBLOCK': 32},
    (8.6, 'lasp_blelloch', (128, 2048), (64, 256), (32, 128)): {'BLOCK': 64, 'CBLOCK': 32},
    (8.0, 'lasp_blelloch', (2049, 8192), (64, 256), (32, 128)): {'BLOCK': 128, 'CBLOCK': 32},
    (8.6, 'lasp_blelloch', (2049, 8192), (64, 256), (32, 128)): {'BLOCK': 128, 'CBLOCK': 32},

    # Hopper H100 (9.0) - 227KB shared memory limit
    (9.0, 'lightning', (512, 8192), (64, 256), (32, 128)): {'BLOCK': 64, 'BLOCK_MODEL': 32, 'CBLOCK': 32},
    (9.0, 'lasp_naive', (512, 8192), (64, 256), (32, 128)): {'BLOCK': 64, 'BLOCK_MODEL': 32, 'CBLOCK': 32},
    (9.0, 'lasp_cache', (512, 8192), (64, 256), (32, 128)): {'BLOCK': 64, 'BLOCK_MODEL': 32, 'CBLOCK': 32},
    (9.0, 'lasp_fuse', (128, 2048), (64, 256), (32, 128)): {'BLOCK': 64, 'CBLOCK': 32},
    (9.0, 'lasp_fuse', (2049, 8192), (64, 256), (32, 128)): {'BLOCK': 128, 'CBLOCK': 32},
    (9.0, 'lasp_fuse_parallel', (128, 2048), (64, 256), (32, 128)): {'BLOCK': 64, 'CBLOCK': 32},
    (9.0, 'lasp_fuse_parallel', (2049, 8192), (64, 256), (32, 128)): {'BLOCK': 128, 'CBLOCK': 32},
    (9.0, 'lasp_blelloch', (128, 2048), (64, 256), (32, 128)): {'BLOCK': 64, 'CBLOCK': 32},
    (9.0, 'lasp_blelloch', (2049, 8192), (64, 256), (32, 128)): {'BLOCK': 128, 'CBLOCK': 32},

    # Pascal/Turing/Volta (6.x, 7.0, 7.5) - 48KB shared memory limit (conservative)
    (6.0, 'lightning', (512, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'BLOCK_MODEL': 16, 'CBLOCK': 16},
    (6.1, 'lightning', (512, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'BLOCK_MODEL': 16, 'CBLOCK': 16},
    (6.2, 'lightning', (512, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'BLOCK_MODEL': 16, 'CBLOCK': 16},
    (7.0, 'lightning', (512, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'BLOCK_MODEL': 16, 'CBLOCK': 16},
    (7.5, 'lightning', (512, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'BLOCK_MODEL': 16, 'CBLOCK': 16},
    (6.0, 'lasp_naive', (512, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'BLOCK_MODEL': 16, 'CBLOCK': 16},
    (6.1, 'lasp_naive', (512, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'BLOCK_MODEL': 16, 'CBLOCK': 16},
    (6.2, 'lasp_naive', (512, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'BLOCK_MODEL': 16, 'CBLOCK': 16},
    (7.0, 'lasp_naive', (512, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'BLOCK_MODEL': 16, 'CBLOCK': 16},
    (7.5, 'lasp_naive', (512, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'BLOCK_MODEL': 16, 'CBLOCK': 16},
    (6.0, 'lasp_cache', (512, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'BLOCK_MODEL': 16, 'CBLOCK': 16},
    (6.1, 'lasp_cache', (512, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'BLOCK_MODEL': 16, 'CBLOCK': 16},
    (6.2, 'lasp_cache', (512, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'BLOCK_MODEL': 16, 'CBLOCK': 16},
    (7.0, 'lasp_cache', (512, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'BLOCK_MODEL': 16, 'CBLOCK': 16},
    (7.5, 'lasp_cache', (512, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'BLOCK_MODEL': 16, 'CBLOCK': 16},
    (6.0, 'lasp_fuse', (128, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'CBLOCK': 16},
    (6.1, 'lasp_fuse', (128, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'CBLOCK': 16},
    (6.2, 'lasp_fuse', (128, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'CBLOCK': 16},
    (7.0, 'lasp_fuse', (128, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'CBLOCK': 16},
    (7.5, 'lasp_fuse', (128, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'CBLOCK': 16},
    (6.0, 'lasp_fuse_parallel', (128, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'CBLOCK': 16},
    (6.1, 'lasp_fuse_parallel', (128, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'CBLOCK': 16},
    (6.2, 'lasp_fuse_parallel', (128, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'CBLOCK': 16},
    (7.0, 'lasp_fuse_parallel', (128, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'CBLOCK': 16},
    (7.5, 'lasp_fuse_parallel', (128, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'CBLOCK': 16},
    (6.0, 'lasp_blelloch', (128, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'CBLOCK': 16},
    (6.1, 'lasp_blelloch', (128, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'CBLOCK': 16},
    (6.2, 'lasp_blelloch', (128, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'CBLOCK': 16},
    (7.0, 'lasp_blelloch', (128, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'CBLOCK': 16},
    (7.5, 'lasp_blelloch', (128, 8192), (64, 256), (32, 128)): {'BLOCK': 32, 'CBLOCK': 16},
}


def _match_fixed_config(compute_cap, kernel_type, n, d, e):
    """Match dimensions to fixed configuration ranges."""
    # Try exact compute capability match first
    for (cap, ktype, (n_min, n_max), (d_min, d_max), (e_min, e_max)), config in FIXED_CONFIGS.items():
        if cap == compute_cap and ktype == kernel_type:
            if n_min <= n <= n_max and d_min <= d <= d_max and e_min <= e <= e_max:
                return config

    # Try matching by major version
    major_version = int(compute_cap)
    for (cap, ktype, (n_min, n_max), (d_min, d_max), (e_min, e_max)), config in FIXED_CONFIGS.items():
        if int(cap) == major_version and ktype == kernel_type:
            if n_min <= n <= n_max and d_min <= d <= d_max and e_min <= e <= e_max:
                return config

    return None


# Cache for performance (only caches lookup results, not computation)
_smem_cache = {}


def get_config_for_kernel(kernel_type, n, d, e, device=None):
    """
    Get configuration for a specific kernel type using fixed lookup table.
    Falls back to dynamic computation if no match found.

    Args:
        kernel_type: 'lightning', 'lasp_naive', 'lasp_cache', 'lasp_fuse', etc.
        n: Sequence length
        d: Query/key dimension
        e: Value dimension
        device: CUDA device (optional)

    Returns:
        dict: Configuration with BLOCK, BLOCK_MODEL, CBLOCK, etc.
    """
    if device is None:
        device = torch.cuda.current_device()

    cache_key = (kernel_type, device, n, d, e)
    if cache_key in _smem_cache:
        return _smem_cache[cache_key]

    compute_cap = get_compute_capability(device)

    # Try fixed configuration first (fast lookup)
    config = _match_fixed_config(compute_cap, kernel_type, n, d, e)

    if config is not None:
        _smem_cache[cache_key] = config
        return config

    # Fall back to dynamic computation for edge cases
    smem_limit = get_shared_memory_limit(device)

    if kernel_type == 'lightning':
        BLOCK, BLOCK_MODEL = get_optimal_block_sizes(n, d, e, device)
        config = {
            'BLOCK': BLOCK,
            'BLOCK_MODEL': BLOCK_MODEL,
            'CBLOCK': get_optimal_cblock_size(BLOCK, device),
        }
    elif kernel_type in ['lasp_naive', 'lasp_cache']:
        BLOCK, BLOCK_MODEL = get_optimal_block_sizes(n, d, e, device)
        config = {
            'BLOCK': BLOCK,
            'BLOCK_MODEL': BLOCK_MODEL,
            'CBLOCK': get_optimal_cblock_size(BLOCK, device),
        }
    elif kernel_type in ['lasp_fuse', 'lasp_fuse_parallel', 'lasp_blelloch']:
        if n > 128:
            if smem_limit <= 99 * 1024:
                BLOCK = 32
                CBLOCK = 16
            else:
                BLOCK = 128
                CBLOCK = 32
        else:
            BLOCK = min(n, 32)
            CBLOCK = min(n, 16)
        config = {
            'BLOCK': BLOCK,
            'CBLOCK': CBLOCK,
        }

    _smem_cache[cache_key] = config
    return config

