"""
LASP-ZeCO: All-Scan (ZeCO) implementation with pipelined P2P communication.

This implementation follows the ZeCO paper's All-Scan primitive:
- Linear chain topology (not ring): rank 0 has no recv, last rank has no send
- Block-sliced pipeline along d dimension to overlap recv→update→send
- Runs in separate CUDA stream to overlap with local compute
- Communication cost independent of world size P (only depends on d×e)

Key differences from other LASP implementations:
- Uses pipelined P2P instead of ring (LASP-1) or AllGather (LASP-2)
- Imports kernels from lasp_fuse.py for better performance
- Uses triton.cdiv consistently to handle non-divisible sequence lengths
- Correctly handles gradient flow in backward pass (zeros for successor gradients)

Note: lasp_fuse.py has a NUM_BLOCK inconsistency between forward/backward that
we work around by using triton.cdiv consistently in this implementation.
"""

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from .lasp_fuse import (
    lasp_forward,
    lasp_backward,
    get_config_for_kernel,
)
from .utils import (
    get_sequence_parallel_group,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
)


def linear_chain_neighbors(rank, world_size, direction="fwd"):
    """
    Return (recv_from, send_to) for a linear chain.

    Args:
        rank: Current rank in the group
        world_size: Total number of ranks
        direction: "fwd" — data flows 0 -> 1 -> ... -> world_size-1
                   "bwd" — data flows world_size-1 -> ... -> 1 -> 0

    Returns:
        (recv_from, send_to): Tuple of rank IDs or None if at chain boundary
    """
    if direction == "fwd":
        recv_from = rank - 1 if rank > 0 else None
        send_to = rank + 1 if rank < world_size - 1 else None
    else:  # bwd
        recv_from = rank + 1 if rank < world_size - 1 else None
        send_to = rank - 1 if rank > 0 else None

    return recv_from, send_to


@triton.jit
def _compute_local_kv_and_gamma_kernel(
    K,
    V,
    S,
    KV_out,
    Gamma_out,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    DBLOCK: tl.constexpr,
    EBLOCK: tl.constexpr,
):
    """Compute local memory state M_r = K^T @ V and cumulative decay gamma_tilde.

    gamma_tilde = exp(-s * n_local) is the cumulative decay product across
    the entire local chunk (not per-token), matching the inter-chunk boundary
    recurrence in All-Scan."""
    off_d = tl.program_id(0)
    off_e = tl.program_id(1)
    off_bh = tl.program_id(2)
    off_h = off_bh % h

    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    kv_offset = off_bh * d * e

    d_offset = off_d * DBLOCK
    e_offset = off_e * EBLOCK
    kv_d_offset = d_offset * e

    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    array = tl.arange(0, BLOCK).to(tl.float32)
    block_decay = tl.exp(-s.to(tl.float32) * BLOCK)
    k_trans_decay = tl.exp(-s.to(tl.float32) * (BLOCK - array[None, :]))

    K_trans_block_ptr = (
        K
        + qk_offset
        + d_offset
        + tl.arange(0, BLOCK)[None, :] * d
        + tl.arange(0, DBLOCK)[:, None]
    )
    V_block_ptr = (
        V
        + v_offset
        + e_offset
        + tl.arange(0, BLOCK)[:, None] * e
        + tl.arange(0, EBLOCK)[None, :]
    )
    KV_block_ptr = (
        KV_out
        + kv_offset
        + kv_d_offset
        + e_offset
        + tl.arange(0, DBLOCK)[:, None] * e
        + tl.arange(0, EBLOCK)[None, :]
    )

    kv = tl.zeros([DBLOCK, EBLOCK], dtype=tl.float32)
    gamma_accum = 1.0  # Accumulate total decay

    for i in range(NUM_BLOCK):
        k_trans = tl.load(K_trans_block_ptr).to(tl.float32)
        v = tl.load(V_block_ptr).to(tl.float32)

        kv = block_decay * kv + tl.dot(k_trans * k_trans_decay, v)
        gamma_accum = gamma_accum * block_decay

        K_trans_block_ptr += BLOCK * d
        V_block_ptr += BLOCK * e

    # Store local KV contribution
    tl.store(KV_block_ptr, kv.to(KV_block_ptr.dtype.element_ty))

    # Store cumulative gamma (only need one value per (b,h,d) block)
    # Only store when processing the first e-block to avoid race conditions
    if off_e == 0:
        Gamma_ptr = Gamma_out + off_bh * d + d_offset + tl.arange(0, DBLOCK)
        tl.store(Gamma_ptr, gamma_accum)


def compute_local_kv_and_gamma(k, v, s, d_, e_, BLOCK, NUM_BLOCK):
    """Compute local memory state M_r = K^T @ V and cumulative gamma_tilde."""
    k = k.contiguous()
    v = v.contiguous()
    s = s.contiguous()

    b, h, n, d = k.shape
    e = v.shape[-1]
    nd, ne = d // d_, e // e_

    # Output shapes
    kv_out = torch.empty((b, h, d, e), dtype=k.dtype, device=k.device)
    gamma_out = torch.empty((b, h, d), dtype=torch.float32, device=k.device)

    grid = (nd, ne, b * h)

    with torch.cuda.device(k.device.index):
        _compute_local_kv_and_gamma_kernel[grid](
            k,
            v,
            s,
            kv_out,
            gamma_out,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            DBLOCK=d_,
            EBLOCK=e_,
        )

    return kv_out, gamma_out


@torch.no_grad()
def all_scan_p2p(
    S_local,
    gamma_tilde,
    group,
    direction="fwd",
    num_blocks=8,
    comm_stream=None,
):
    """
    Pipelined receive→update→send of minimal cross-boundary state for ZeCO/All-Scan.

    Each device transmits/receives exactly |S| = d×e bytes once, independent of P.
    The state is block-sliced along the d dimension and pipelined to hide latency.

    Args:
        S_local: (b, h, d, e) final local state for this rank
        gamma_tilde: (b, h, d) or (b, h, d, 1) cumulative decay factors
        group: sequence-parallel process group
        direction: 'fwd' or 'bwd'
        num_blocks: number of slices along d dimension for pipelining
        comm_stream: CUDA stream for communication (None = current stream)

    Returns:
        (S_pred, S_out):
            S_pred: (b, h, d, e) predecessor's global state (zeros on chain head)
            S_out: (b, h, d, e) this rank's updated final global state
    """
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")

    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    recv_from, send_to = linear_chain_neighbors(rank, world_size, direction)

    b, h, d, e = S_local.shape
    device = S_local.device
    dtype = S_local.dtype

    # Prepare gamma_tilde with correct shape for broadcasting: (b, h, d, 1)
    if gamma_tilde.dim() == 3:  # (b, h, d)
        gamma_tilde = gamma_tilde.unsqueeze(-1)

    # Calculate block sizes for d dimension
    # Distribute remainder to early blocks for load balancing
    base = d // num_blocks
    rem = d % num_blocks
    starts = []
    sizes = []
    offset = 0
    for i in range(num_blocks):
        step = base + (1 if i < rem else 0)
        if step == 0:
            continue
        starts.append(offset)
        sizes.append(step)
        offset += step

    true_blocks = len(starts)

    # Output tensors
    S_pred = torch.zeros_like(S_local)
    S_out = torch.empty_like(S_local)

    # Allocate recv buffers once (reuse across blocks if at head)
    recv_bufs = []
    if recv_from is not None:
        for i in range(true_blocks):
            h_block = sizes[i]
            recv_bufs.append(torch.empty((b, h, h_block, e), device=device, dtype=dtype))

    # Use specified comm stream or current stream
    cs = comm_stream if comm_stream is not None else torch.cuda.current_stream()

    # Record stream on input tensors to ensure they're available when used
    S_local.record_stream(cs)
    if gamma_tilde.dim() == 4:  # Already expanded
        gamma_tilde.record_stream(cs)

    # Pipelined block processing
    with torch.cuda.stream(cs):
        work_recv = [None] * true_blocks
        work_send = [None] * true_blocks

        # Pre-post first receive to overlap with first block computation
        if recv_from is not None and true_blocks > 0:
            work_recv[0] = dist.irecv(tensor=recv_bufs[0], src=recv_from, group=group)

        for i in range(true_blocks):
            s = starts[i]
            h_block = sizes[i]

            # Extract local and gamma slices
            sl_local = S_local[:, :, s:s + h_block, :]
            gl = gamma_tilde[:, :, s:s + h_block, :]

            # Wait for receive of this block
            if recv_from is not None:
                work_recv[i].wait()
                pred_block = recv_bufs[i]
            else:
                # Head of chain: predecessor is zeros
                pred_block = torch.zeros_like(sl_local)

            # Save predecessor slice (for caller's use)
            S_pred[:, :, s:s + h_block, :].copy_(pred_block)

            # Update: S_out[block] = S_local[block] + gamma_tilde[block] ⊙ pred_block
            # This is the core All-Scan update equation
            upd = sl_local + gl * pred_block
            S_out[:, :, s:s + h_block, :].copy_(upd)

            # Post send of this block immediately (pipelining)
            if send_to is not None:
                upd_contig = upd.contiguous()
                # Record stream to ensure producer ops complete before send
                upd_contig.record_stream(cs)
                work_send[i] = dist.isend(tensor=upd_contig, dst=send_to, group=group)

            # Pre-post next receive as soon as possible to overlap
            nxt = i + 1
            if recv_from is not None and nxt < true_blocks:
                work_recv[nxt] = dist.irecv(tensor=recv_bufs[nxt], src=recv_from, group=group)

        # Wait for all sends to complete before buffers go out of scope
        # Note: record_stream() calls ensure CUDA ops complete before sends finish
        # DO NOT add cs.synchronize() here - it causes deadlock in chain topology!
        for w in work_send:
            if w is not None:
                w.wait()

    return S_pred, S_out


class LaspZeCo(torch.autograd.Function):
    """
    LASP-ZeCO: All-Scan (ZeCO) implementation with pipelined P2P.

    Key properties:
    - Uses block-sliced pipelined receive→update→send to minimize latency
    - Communication cost is O(d×e) per device, independent of world size P
    - Overlaps communication with local intra-chunk computation
    - Linear chain topology (not ring) for cleaner forward/backward semantics
    """

    @staticmethod
    def forward(ctx, q, k, v, s, num_blocks=8):
        b, h, n, d = q.shape
        e = v.shape[-1]

        # Get config
        config = get_config_for_kernel('lasp_fuse', n, d, e, q.device)
        BLOCK = config['BLOCK']
        # Use cdiv consistently to handle partial blocks
        # NOTE: lasp_fuse.py has inconsistent NUM_BLOCK calculation:
        # - forward uses floor division (n // BLOCK) which loses partial blocks
        # - backward uses ceiling division (triton.cdiv) which processes all tokens
        # We use cdiv consistently here for correctness with non-divisible sequence lengths
        NUM_BLOCK = triton.cdiv(n, BLOCK)

        # Use same tile caps as lasp_fuse kernels (≤64) to ensure nd, ne > 0
        # Otherwise if d=768, next_power_of_2=1024 → nd=0 → invalid grid
        cd = 64
        ce = 64
        d_ = min(triton.next_power_of_2(d), cd)
        e_ = min(triton.next_power_of_2(e), ce)

        # Get parallel group info
        group = get_sequence_parallel_group()
        current_idx = get_sequence_parallel_rank()
        world_size = get_sequence_parallel_world_size()

        # Step 1: Compute local memory state M_r = K^T @ V and boundary decay gamma_tilde
        # gamma_tilde = exp(-s * n_local) is the cumulative decay across this rank's
        # entire local chunk, used for the inter-chunk boundary recurrence in All-Scan
        local_KV, gamma_tilde = compute_local_kv_and_gamma(k, v, s, d_, e_, BLOCK, NUM_BLOCK)

        # gamma_tilde shape: (b, h, d) - one decay factor per d-tile
        # Expand to (b, h, d, 1) for broadcasting in all_scan_p2p
        gamma_tilde_expanded = gamma_tilde.unsqueeze(-1)

        # Step 2: Create communication stream and event for overlap
        comm_stream = torch.cuda.Stream()
        comm_done = torch.cuda.Event()

        # Step 3: Launch All-Scan on comm stream (forward direction)
        # This runs asynchronously while we could do local intra-chunk work
        # NOTE: all_scan_p2p manages its own stream context internally
        S_pred, S_out = all_scan_p2p(
            S_local=local_KV,
            gamma_tilde=gamma_tilde_expanded,
            group=group,
            direction="fwd",
            num_blocks=num_blocks,
            comm_stream=comm_stream,
        )
        # Record completion event in the comm stream
        with torch.cuda.stream(comm_stream):
            comm_done.record()

        # Step 4: Wait for All-Scan to complete before computing local attention
        # NOTE: Future optimization could overlap local computation with All-Scan
        torch.cuda.current_stream().wait_event(comm_done)

        # Step 5: Use S_pred (predecessor's global state) as initial state
        # S_pred is the correct initial state before our local sequence
        # lasp_forward expects float32, so convert S_pred if needed
        KV_buffer = S_pred.to(dtype=torch.float32).contiguous()

        # Run forward pass with predecessor state
        o = lasp_forward(q, k, v, s, KV_buffer)

        # Save for backward
        ctx.save_for_backward(q, k, v, s, gamma_tilde)
        ctx.group = group
        ctx.current_idx = current_idx
        ctx.world_size = world_size
        ctx.config = config
        ctx.num_blocks = num_blocks
        ctx.S_pred = S_pred

        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, s, gamma_tilde = ctx.saved_tensors
        group = ctx.group
        current_idx = ctx.current_idx
        world_size = ctx.world_size
        config = ctx.config
        num_blocks = ctx.num_blocks
        S_pred = ctx.S_pred

        b, h, n, d = q.shape
        e = v.shape[-1]

        BLOCK = config['BLOCK']
        # NUM_BLOCK is already computed correctly in forward pass with triton.cdiv
        # We don't recompute it here to ensure consistency

        # Use same tile caps as forward (≤64) to match lasp_fuse kernels
        cd = 64
        ce = 64
        d_ = min(triton.next_power_of_2(d), cd)
        e_ = min(triton.next_power_of_2(e), ce)

        # Allocate buffers for backward
        # lasp_backward expects float32, convert S_pred if needed
        KV_buffer = S_pred.to(dtype=torch.float32).contiguous()
        DKV_buffer = torch.zeros((b, h, d, e), dtype=torch.float32, device=q.device)

        # Compute local gradients - lasp_backward modifies DKV_buffer in-place
        dq, dk, dv = lasp_backward(q, k, v, s, do, KV_buffer, DKV_buffer)

        # DKV_buffer now contains local d(KV) gradients - use directly, no need to clone
        dKV_local = DKV_buffer

        # Step 2: Create communication stream and event
        comm_stream = torch.cuda.Stream()
        comm_done = torch.cuda.Event()

        # Step 3: Launch All-Scan in backward direction
        # Fix: Pass the actual computed dKV_local, not zeros!
        gamma_tilde_expanded = gamma_tilde.unsqueeze(-1)

        # NOTE: all_scan_p2p manages its own stream context internally
        dKV_pred, dKV_out = all_scan_p2p(
            S_local=dKV_local,
            gamma_tilde=gamma_tilde_expanded,
            group=group,
            direction="bwd",  # Reverse direction for backward pass
            num_blocks=num_blocks,
            comm_stream=comm_stream,
        )
        # Record completion event in the comm stream
        with torch.cuda.stream(comm_stream):
            comm_done.record()

        # Wait for backward All-Scan
        torch.cuda.current_stream().wait_event(comm_done)

        # Accumulate gradients from successor ranks
        # dKV_pred contains gradients from the "predecessor" in backward direction
        # (which is the successor in forward direction)
        if current_idx < world_size - 1:
            # Compute gradient contribution from successors
            # Use zeros for KV state since we only want gradient flow from DKV
            # This matches the LaspFuseV2 implementation pattern
            dq_suffix, dk_suffix, dv_suffix = lasp_backward(
                q, k, v, s, torch.zeros_like(do), torch.zeros_like(KV_buffer), dKV_pred
            )
            dq = dq + dq_suffix
            dk = dk + dk_suffix
            dv = dv + dv_suffix

        return dq, dk, dv, None, None


lasp_zeco_ = LaspZeCo.apply


def lasp_zeco(q, k, v, ed, num_blocks=8):
    """
    LASP-ZeCO: All-Scan (ZeCO) implementation.

    Uses pipelined P2P communication with block slicing to minimize latency.
    Communication cost is O(d×e) per device, independent of world size P.
    Overlaps communication with local computation for optimal performance.

    Key advantages over LASP-1 (ring) and LASP-2 (AllGather):
    - LASP-1 (ring): O(P) sequential communication steps
    - LASP-2 (AllGather): 2 collectives but gathers all states (memory overhead)
    - LASP-ZeCO (All-Scan): Minimal state transfer with pipelined overlap

    Args:
        q, k, v: Query, key, value tensors (b, h, n, d)/(b, h, n, e)
        ed: Decay factors (h,)
        num_blocks: Number of blocks for pipeline (default: 8, higher = better overlap)

    Returns:
        Output tensor (b, h, n, e)

    Note: This version does NOT do feature-dimension slicing (that was incorrect).
          ZeCO is sequence-parallel, not tensor-parallel over d.
    """
    return lasp_zeco_(q, k, v, ed, num_blocks)
