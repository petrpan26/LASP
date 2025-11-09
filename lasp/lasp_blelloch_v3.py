"""
LASP Blelloch V3: Pipelined Tree-Scan (ZeCO-inspired) with O(log P) communication.

Goals:
- Retain Blelloch O(log P) communication topology for large-scale efficiency.
- Borrow ZeCO's practical wins: d-sliced, non-blocking P2P on a dedicated CUDA stream.
- Robust block math (use triton.cdiv for NUM_BLOCK).

Key ideas:
- Compute local KV using fused kernels (same as V1/V2).
- Per tree level, exchange KV in d-slices using irecv/isend on a comm stream.
- Apply per-level decay powers: lambda^(stride * n_local) when combining.
- Mirror the pipeline for backward (reverse scan over ranks).
"""

import math
import torch
import torch.distributed as dist
import triton

from .gpu_config import get_config_for_kernel
from .lasp_fuse_parallel import (
    _fwd_diag_kernel,
    _fwd_kv_parallel,
    _fwd_kv_reduce,
    _fwd_none_diag_kernel,
    _bwd_diag_kernel,
    _bwd_dkv_parallel,
    _bwd_dkv_reduce,
    _bwd_none_diag_kernel,
)
from .utils import (
    get_sequence_parallel_group,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
)


def _compute_d_slices(d: int, num_blocks: int):
    """Compute balanced slices along d dimension."""
    base = d // num_blocks
    rem = d % num_blocks
    starts, sizes = [], []
    off = 0
    for i in range(num_blocks):
        step = base + (1 if i < rem else 0)
        if step > 0:
            starts.append(off)
            sizes.append(step)
            off += step
    return starts, sizes


def _expand_decay(decay_vec: torch.Tensor, target_ndim: int) -> torch.Tensor:
    """
    Expand [h] to [1, h, 1, 1] (or appropriate) to match [b, h, d, e].
    """
    decay = decay_vec
    while decay.dim() < target_ndim:
        # Add singleton dims at front then back
        decay = decay.unsqueeze(0)
        if decay.dim() < target_ndim:
            decay = decay.unsqueeze(-1)
    return decay


class _PipelinedTreeScanner:
    """
    ZeCO-inspired, d-sliced, non-blocking P2P exchange per Blelloch tree level.

    Produces EXCLUSIVE prefix (forward) or EXCLUSIVE suffix (backward when reverse=True).
    """

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        group,
        decay_factor: torch.Tensor,  # λ per head [h]
        chunk_size: int,  # local n
        device: torch.device,
        reverse: bool = False,
        num_slices: int = 8,
        comm_stream: torch.cuda.Stream | None = None,
    ):
        self.rank = rank
        self.world_size = world_size
        self.group = group
        self.device = device
        self.reverse = reverse
        self.num_slices = max(1, int(num_slices))
        self.cs = comm_stream if comm_stream is not None else torch.cuda.current_stream()

        # Map local SP ranks to global ranks for P2P as in ZeCO
        self.local_rank = dist.get_rank(group)
        self.global_rank = dist.get_rank()
        self.rank_offset = self.global_rank - self.local_rank

        # Reverse scan rank space if suffix is requested
        self.scan_rank = (world_size - 1 - self.local_rank) if reverse else self.local_rank

        # Precompute lambda^C (C=n_local per rank)
        # Keep in float32 for stability (as in other implementations)
        self.lambda_C = (decay_factor.to(torch.float32)) ** chunk_size  # [h]

        # Number of Blelloch levels
        self.num_levels = math.ceil(math.log2(world_size)) if world_size > 1 else 0

    # ---- Partner selection helpers (match Blelloch semantics) ----
    @staticmethod
    def _stride(level: int) -> int:
        return 2 ** level

    def _partner_up(self, level: int) -> int:
        stride = self._stride(level)
        if self.scan_rank % (2 * stride) == stride - 1:
            partner = self.scan_rank + stride
            return partner if partner < self.world_size else -1
        if self.scan_rank % (2 * stride) == 2 * stride - 1:
            return self.scan_rank - stride
        return -1

    def _is_sender_up(self, level: int) -> bool:
        stride = self._stride(level)
        return self.scan_rank % (2 * stride) == stride - 1

    def _is_receiver_up(self, level: int) -> bool:
        stride = self._stride(level)
        return self.scan_rank % (2 * stride) == 2 * stride - 1

    def _partner_down(self, level: int) -> int:
        stride = self._stride(level)
        if self.scan_rank % (2 * stride) == stride - 1:
            partner = self.scan_rank + 1  # right subtree middle
            return partner if partner < self.world_size else -1
        if self.scan_rank % (2 * stride) == stride:
            return self.scan_rank - 1
        return -1

    def _is_sender_down(self, level: int) -> bool:
        stride = self._stride(level)
        return self.scan_rank % (2 * stride) == stride - 1

    def _is_receiver_down(self, level: int) -> bool:
        stride = self._stride(level)
        return self.scan_rank % (2 * stride) == stride

    def _scan_to_global(self, scan_rank: int) -> int:
        if scan_rank < 0:
            return -1
        if self.reverse:
            actual_local = self.world_size - 1 - scan_rank
            return actual_local + self.rank_offset
        return scan_rank + self.rank_offset

    def _actual_to_global(self, actual_local: int) -> int:
        if actual_local < 0:
            return -1
        return actual_local + self.rank_offset

    def _combine(self, recv: torch.Tensor, local: torch.Tensor, level: int) -> torch.Tensor:
        """
        Combine with per-level decay: (lambda_C^(2^level)) * recv + local
        """
        stride = self._stride(level)
        decay = (self.lambda_C ** stride)  # [h]
        decay = _expand_decay(decay, target_ndim=local.dim()).to(local.device)
        return decay * recv + local

    def scan(self, local_value: torch.Tensor) -> torch.Tensor:
        """
        Perform pipelined EXCLUSIVE scan (prefix for forward, suffix for reverse).
        local_value: [b, h, d, e] float32
        """
        if self.world_size == 1:
            return torch.zeros_like(local_value)

        b, h, d, e = local_value.shape
        starts, sizes = _compute_d_slices(d, self.num_slices)
        # Working buffer (inclusive rolling aggregate during up-sweep)
        working = local_value.clone()

        # Store selected tree values for down-sweep (opt-in to save memory)
        tree_values = [working.clone()]

        # ========== Up-sweep (bottom-up) ==========
        for level in range(self.num_levels):
            partner_scan = self._partner_up(level)
            if partner_scan == -1:
                tree_values.append(None)
                continue

            partner_global = self._scan_to_global(partner_scan)

            # Use comm stream for P2P ops
            with torch.cuda.stream(self.cs):
                if self._is_sender_up(level) and partner_scan < self.world_size:
                    # Send our current aggregate in d-slices
                    send_reqs = []
                    for i, (s, w) in enumerate(zip(starts, sizes)):
                        slice_to_send = working[:, :, s:s + w, :].contiguous()
                        slice_to_send.record_stream(self.cs)
                        send_reqs.append(
                            dist.isend(tensor=slice_to_send, dst=partner_global, group=self.group)
                        )
                    for req in send_reqs:
                        req.wait()
                    # Decide whether to store current value for down-sweep
                    if self._is_sender_down(level):
                        tree_values.append(working.clone())
                    else:
                        tree_values.append(None)

                elif self._is_receiver_up(level):
                    # Receive partner slices and combine on the fly
                    for i, (s, w) in enumerate(zip(starts, sizes)):
                        recv_buf = torch.empty((b, h, w, e), dtype=working.dtype, device=working.device)
                        req = dist.irecv(tensor=recv_buf, src=partner_global, group=self.group)
                        req.wait()
                        combined = self._combine(recv_buf, working[:, :, s:s + w, :], level)
                        working[:, :, s:s + w, :].copy_(combined)
                    # Always store updated value for down-sweep needs
                    tree_values.append(working.clone())

        # ========== Down-sweep (top-down) ==========
        inclusive_ready = False
        for level in range(self.num_levels - 1, -1, -1):
            partner_scan = self._partner_down(level)
            if partner_scan == -1:
                continue
            partner_global = self._scan_to_global(partner_scan)

            with torch.cuda.stream(self.cs):
                if self._is_receiver_down(level) and partner_scan >= 0:
                    # Receive left prefix in d-slices and combine with stored tree value
                    # Use the most recent non-None tree value up to this level
                    tree_idx = min(level, len(tree_values) - 1)
                    tree_val = tree_values[tree_idx]
                    while tree_val is None and tree_idx > 0:
                        tree_idx -= 1
                        tree_val = tree_values[tree_idx]
                    # Combine per slice
                    for i, (s, w) in enumerate(zip(starts, sizes)):
                        left_slice = torch.empty((b, h, w, e), dtype=working.dtype, device=working.device)
                        req = dist.irecv(tensor=left_slice, src=partner_global, group=self.group)
                        req.wait()
                        base_slice = tree_val[:, :, s:s + w, :] if tree_val is not None else working[:, :, s:s + w, :]
                        # Distance equals actual separation in scan order
                        # Use level as stride proxy (2^level)
                        combined = self._combine(left_slice, base_slice, level)
                        working[:, :, s:s + w, :].copy_(combined)
                    inclusive_ready = True

                elif self._is_sender_down(level) and partner_scan < self.world_size:
                    # Send either current inclusive (if ready) or stored tree value slice-by-slice
                    if inclusive_ready:
                        send_source = working
                    else:
                        tree_idx = min(level, len(tree_values) - 1)
                        send_source = tree_values[tree_idx]
                        while send_source is None and tree_idx > 0:
                            tree_idx -= 1
                            send_source = tree_values[tree_idx]
                        if send_source is None:
                            send_source = working
                    send_reqs = []
                    for i, (s, w) in enumerate(zip(starts, sizes)):
                        src_slice = send_source[:, :, s:s + w, :].contiguous()
                        src_slice.record_stream(self.cs)
                        send_reqs.append(
                            dist.isend(tensor=src_slice, dst=partner_global, group=self.group)
                        )
                    for req in send_reqs:
                        req.wait()

        # If inclusive not set during down-sweep, keep working as-is
        # ========== Convert inclusive → exclusive via neighbor exchange ==========
        exclusive = torch.zeros_like(working)
        with torch.cuda.stream(self.cs):
            if not self.reverse:
                # Prefix: recv from left neighbor (rank-1), send to right neighbor (rank+1)
                recv_req = None
                send_req = None
                if self.local_rank > 0:
                    left_global = self._actual_to_global(self.local_rank - 1)
                    # Receive d-slices
                    for s, w in zip(starts, sizes):
                        buf = exclusive[:, :, s:s + w, :]
                        req = dist.irecv(tensor=buf, src=left_global, group=self.group)
                        req.wait()
                if self.local_rank < self.world_size - 1:
                    right_global = self._actual_to_global(self.local_rank + 1)
                    # Send our inclusive in d-slices
                    send_reqs = []
                    for s, w in zip(starts, sizes):
                        src_slice = working[:, :, s:s + w, :].contiguous()
                        src_slice.record_stream(self.cs)
                        send_reqs.append(
                            dist.isend(tensor=src_slice, dst=right_global, group=self.group)
                        )
                    for req in send_reqs:
                        req.wait()
            else:
                # Suffix: recv from right neighbor (rank+1), send to left neighbor (rank-1)
                if self.local_rank < self.world_size - 1:
                    right_global = self._actual_to_global(self.local_rank + 1)
                    for s, w in zip(starts, sizes):
                        buf = exclusive[:, :, s:s + w, :]
                        req = dist.irecv(tensor=buf, src=right_global, group=self.group)
                        req.wait()
                if self.local_rank > 0:
                    left_global = self._actual_to_global(self.local_rank - 1)
                    send_reqs = []
                    for s, w in zip(starts, sizes):
                        src_slice = working[:, :, s:s + w, :].contiguous()
                        src_slice.record_stream(self.cs)
                        send_reqs.append(
                            dist.isend(tensor=src_slice, dst=left_global, group=self.group)
                        )
                    for req in send_reqs:
                        req.wait()

        return exclusive


class LaspBlellochV3(torch.autograd.Function):
    """
    LASP Blelloch V3 with pipelined, d-sliced tree-scan on a dedicated comm stream.
    """

    @staticmethod
    def forward(ctx, q, k, v, s, KV, DKV, num_pipeline_blocks=8):
        b, h, n, d = q.shape
        e = v.shape[-1]

        # Reuse KV buffer
        KV.zero_()

        # Distributed context
        group = get_sequence_parallel_group()
        rank = get_sequence_parallel_rank()
        world_size = get_sequence_parallel_world_size()

        # Kernel config
        config = get_config_for_kernel('lasp_blelloch', n, d, e, q.device)
        BLOCK = config['BLOCK']
        CBLOCK = config['CBLOCK']

        # Use cdiv for robustness on tail blocks
        NUM_BLOCK = triton.cdiv(n, BLOCK)
        NUM_CBLOCK = BLOCK // CBLOCK
        NUM_FBLOCK = 1
        D_FBLOCK = d // NUM_FBLOCK
        E_FBLOCK = e // NUM_FBLOCK

        # Contiguity
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        s = s.contiguous()

        # Output
        o = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)

        # Streams and events
        comm_stream = torch.cuda.Stream()
        diag_done = torch.cuda.Event()
        local_kv_done = torch.cuda.Event()
        scan_done = torch.cuda.Event()

        # Step 1: Diagonal kernel (intra-chunk attention)
        grid = (b * h * NUM_BLOCK, NUM_CBLOCK)
        with torch.cuda.device(q.device.index):
            _fwd_diag_kernel[grid](
                q, k, v, o, s,
                b, h, n, d, e,
                BLOCK=BLOCK,
                NUM_BLOCK=NUM_BLOCK,
                CBLOCK=CBLOCK,
                NUM_CBLOCK=NUM_CBLOCK,
            )
        diag_done.record()

        # Step 2: Local KV contribution
        kv = torch.empty((b, h, NUM_BLOCK + 1, d, e), dtype=torch.float32, device=q.device)
        with torch.cuda.device(q.device.index):
            grid = (b * h, NUM_BLOCK, NUM_FBLOCK * NUM_FBLOCK)
            _fwd_kv_parallel[grid](
                k, v, s, kv,
                b, h, n, d, e,
                BLOCK=BLOCK,
                NUM_BLOCK=NUM_BLOCK,
                D_FBLOCK=D_FBLOCK,
                E_FBLOCK=E_FBLOCK,
                NUM_FBLOCK=NUM_FBLOCK,
                CBLOCK=CBLOCK,
                NUM_CBLOCK=NUM_CBLOCK,
            )
            grid = (b * h, NUM_FBLOCK, NUM_FBLOCK)
            _fwd_kv_reduce[grid](
                k, v, s, kv,
                b, h, n, d, e,
                BLOCK=BLOCK,
                NUM_BLOCK=NUM_BLOCK,
                D_FBLOCK=D_FBLOCK,
                E_FBLOCK=E_FBLOCK,
                NUM_FBLOCK=NUM_FBLOCK,
                CBLOCK=CBLOCK,
                NUM_CBLOCK=NUM_CBLOCK,
            )
        local_kv = kv[:, :, -1].clone()  # [b, h, d, e], float32
        local_kv_done.record()

        # Step 3: Pipelined tree-scan to get EXCLUSIVE prefix
        if world_size == 1:
            KV_prefix = KV
        else:
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_event(local_kv_done)
                lambda_decay = torch.exp(-s.to(torch.float32))
                scanner = _PipelinedTreeScanner(
                    rank=rank,
                    world_size=world_size,
                    group=group,
                    decay_factor=lambda_decay,
                    chunk_size=n,
                    device=q.device,
                    reverse=False,
                    num_slices=num_pipeline_blocks,
                    comm_stream=comm_stream,
                )
                KV_prefix = scanner.scan(local_kv)
                scan_done.record()

        # Step 4: Inter-chunk kernel using KV_prefix
        torch.cuda.current_stream().wait_event(diag_done)
        if world_size > 1:
            torch.cuda.current_stream().wait_event(scan_done)
        with torch.cuda.device(q.device.index):
            grid = (b * h, NUM_BLOCK * NUM_CBLOCK, NUM_FBLOCK)
            _fwd_none_diag_kernel[grid](
                q, k, v, o, s,
                kv,
                KV_prefix,
                b, h, n, d, e,
                BLOCK=BLOCK,
                NUM_BLOCK=NUM_BLOCK,
                D_FBLOCK=D_FBLOCK,
                E_FBLOCK=E_FBLOCK,
                NUM_FBLOCK=NUM_FBLOCK,
                CBLOCK=CBLOCK,
                NUM_CBLOCK=NUM_CBLOCK,
            )

        # Save for backward
        KV_prefix_saved = KV_prefix.clone()
        ctx.save_for_backward(q, k, v, s, kv, KV_prefix_saved, DKV)
        ctx.group = group
        ctx.rank = rank
        ctx.world_size = world_size
        ctx.BLOCK = BLOCK
        ctx.CBLOCK = CBLOCK
        ctx.NUM_BLOCK = NUM_BLOCK
        ctx.NUM_CBLOCK = NUM_CBLOCK
        ctx.NUM_FBLOCK = NUM_FBLOCK
        ctx.D_FBLOCK = D_FBLOCK
        ctx.E_FBLOCK = E_FBLOCK
        ctx.num_pipeline_blocks = num_pipeline_blocks

        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, s, kv, KV_prefix, DKV = ctx.saved_tensors
        group = ctx.group
        rank = ctx.rank
        world_size = ctx.world_size
        BLOCK = ctx.BLOCK
        CBLOCK = ctx.CBLOCK
        NUM_BLOCK = ctx.NUM_BLOCK
        NUM_CBLOCK = ctx.NUM_CBLOCK
        NUM_FBLOCK = ctx.NUM_FBLOCK
        D_FBLOCK = ctx.D_FBLOCK
        E_FBLOCK = ctx.E_FBLOCK
        num_pipeline_blocks = ctx.num_pipeline_blocks

        b, h, n, d = q.shape
        e = v.shape[-1]

        DKV.zero_()

        do = do.contiguous()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        # Streams and events
        comm_stream = torch.cuda.Stream()
        diag_done = torch.cuda.Event()
        local_dkv_done = torch.cuda.Event()
        scan_done = torch.cuda.Event()

        # Step 1: Backward diagonal (intra-chunk)
        with torch.cuda.device(q.device.index):
            grid = (b * h * NUM_BLOCK, NUM_CBLOCK)
            _bwd_diag_kernel[grid](
                q, k, v, s, do, dq, dk, dv,
                b, h, n, d, e,
                BLOCK=BLOCK,
                NUM_BLOCK=NUM_BLOCK,
                CBLOCK=CBLOCK,
                NUM_CBLOCK=NUM_CBLOCK,
            )
        diag_done.record()

        # Step 2: Local dKV
        dkv = torch.empty((b, h, NUM_BLOCK + 1, d, e), dtype=torch.float32, device=q.device)
        with torch.cuda.device(q.device.index):
            grid = (b * h, NUM_BLOCK, NUM_FBLOCK * NUM_FBLOCK)
            _bwd_dkv_parallel[grid](
                q, do, s, dkv,
                b, h, n, d, e,
                BLOCK=BLOCK,
                NUM_BLOCK=NUM_BLOCK,
                D_FBLOCK=D_FBLOCK,
                E_FBLOCK=E_FBLOCK,
                NUM_FBLOCK=NUM_FBLOCK,
                CBLOCK=CBLOCK,
                NUM_CBLOCK=NUM_CBLOCK,
            )
            grid = (b * h, NUM_FBLOCK, NUM_FBLOCK)
            _bwd_dkv_reduce[grid](
                q, do, s, dkv,
                b, h, n, d, e,
                BLOCK=BLOCK,
                NUM_BLOCK=NUM_BLOCK,
                D_FBLOCK=D_FBLOCK,
                E_FBLOCK=E_FBLOCK,
                NUM_FBLOCK=NUM_FBLOCK,
                CBLOCK=CBLOCK,
                NUM_CBLOCK=NUM_CBLOCK,
            )
        local_dkv = dkv[:, :, -1].clone()
        local_dkv_done.record()

        # Step 3: Reverse pipelined tree-scan to get EXCLUSIVE suffix of dKV
        if world_size == 1:
            DKV_suffix = DKV
        else:
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_event(local_dkv_done)
                lambda_decay = torch.exp(-s.to(torch.float32))
                scanner = _PipelinedTreeScanner(
                    rank=rank,
                    world_size=world_size,
                    group=group,
                    decay_factor=lambda_decay,
                    chunk_size=n,
                    device=do.device,
                    reverse=True,
                    num_slices=num_pipeline_blocks,
                    comm_stream=comm_stream,
                )
                DKV_suffix = scanner.scan(local_dkv)
                scan_done.record()

        # Step 4: Inter-chunk gradient kernel
        torch.cuda.current_stream().wait_event(diag_done)
        if world_size > 1:
            torch.cuda.current_stream().wait_event(scan_done)
        with torch.cuda.device(q.device.index):
            grid = (b * h, NUM_BLOCK * NUM_CBLOCK, NUM_FBLOCK)
            _bwd_none_diag_kernel[grid](
                q, k, v, s, do, dq, dk, dv,
                kv,
                dkv,
                KV_prefix,
                DKV_suffix,
                b, h, n, d, e,
                BLOCK=BLOCK,
                NUM_BLOCK=NUM_BLOCK,
                D_FBLOCK=D_FBLOCK,
                E_FBLOCK=E_FBLOCK,
                NUM_FBLOCK=NUM_FBLOCK,
                CBLOCK=CBLOCK,
                NUM_CBLOCK=NUM_CBLOCK,
            )

        return dq, dk, dv, None, None, None, None


lasp_blelloch_v3_ = LaspBlellochV3.apply


def lasp_blelloch_v3(q, k, v, ed, KV, DKV, num_pipeline_blocks=8):
    """
    LASP Blelloch V3: Pipelined tree-scan across ranks with d-sliced P2P.

    Args:
        q, k, v: Input tensors
        ed: Decay factors per head (h,)
        KV: KV buffer (b, h, d, e)
        DKV: DKV buffer (b, h, d, e)
        num_pipeline_blocks: Number of d-slices for pipelining (default: 8)
    """
    b, h, n, d = q.shape
    e = v.shape[-1]

    # Split across d to keep kernel tiling stable (same pattern as V1/V2)
    if d >= 128:
        m = 128
    else:
        m = 64
    arr = [m * i for i in range(d // m + 1)]
    if arr[-1] != d:
        arr.append(d)
    n_splits = len(arr)
    output = 0
    for i in range(n_splits - 1):
        s_idx = arr[i]
        e_idx = arr[i + 1]
        q1 = q[..., s_idx:e_idx]
        k1 = k[..., s_idx:e_idx]
        o = lasp_blelloch_v3_(
            q1, k1, v, ed,
            KV[:, :, s_idx:e_idx].contiguous(),
            DKV[:, :, s_idx:e_idx].contiguous(),
            num_pipeline_blocks,
        )
        output = output + o
    return output


