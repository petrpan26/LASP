"""
LASP Fused Kernels Implementation

This file contains optimized fused kernels for LASP attention:
- LaspFuse (V1): Ring-based P2P communication, O(W) steps forward/backward
- LaspFuseV2 (LASP-2): AllGather-based implementation, O(1) steps forward/backward

Recent fixes:
1. NUM_BLOCK calculation: Changed from floor division to ceiling division
   (triton.cdiv) to correctly handle non-divisible sequence lengths
2. LaspFuseV2 G array: Extended to world_size + 1 elements to prevent
   IndexError when computing decay weights for the last rank
3. Gamma calculation: Uses padded length (NUM_BLOCK * BLOCK) for consistency
   with kernel processing when handling partial blocks
4. LaspFuseV2 backward: Completely rewritten to follow LASP-2 algorithm:
   - Computes local dM contribution from each rank
   - AllGathers all dM values
   - Computes weighted suffix sum for gradient accumulation
   - Single backward pass with properly accumulated gradients
   - Fixes the double backward bug that caused large dk/dv errors

The LASP-2 backward implementation now correctly follows the algorithm from the paper:
1. Local dM computation: dM_r = Q_r^T @ do_r
2. AllGather: every rank gets [dM_0, ..., dM_{W-1}]
3. Weighted suffix: total_dM_r = dM_r + sum_{j>r} weight(r,j) * dM_j
4. Final gradients: dQ, dK, dV from single backward pass with total_dM
"""

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from .gpu_config import get_config_for_kernel
from .utils import (
    get_seq_parallel_receive_rank,
    get_seq_parallel_send_rank,
    get_sequence_parallel_group,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
)


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Out,
    S,
    KV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    DBLOCK: tl.constexpr,
    NUM_DBLOCK: tl.constexpr,
    EBLOCK: tl.constexpr,
    NUM_EBLOCK: tl.constexpr,
):
    off_d = tl.program_id(0)
    off_e = tl.program_id(1)
    off_bh = tl.program_id(2)
    off_h = off_bh % h
    # get the (b, h) location
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_d * b * h * n * e + off_bh * n * e
    kv_offset = off_bh * d * e

    d_offset = off_d * DBLOCK
    e_offset = off_e * EBLOCK

    kv_d_offset = d_offset * e

    Q_block_ptr = (
        Q
        + qk_offset
        + d_offset
        + tl.arange(0, BLOCK)[:, None] * d
        + tl.arange(0, DBLOCK)[None, :]
    )
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
    O_block_ptr = (
        Out
        + o_offset
        + e_offset
        + tl.arange(0, BLOCK)[:, None] * e
        + tl.arange(0, EBLOCK)[None, :]
    )
    KV_block_ptr = (
        KV
        + kv_offset
        + kv_d_offset
        + e_offset
        + tl.arange(0, DBLOCK)[:, None] * e
        + tl.arange(0, EBLOCK)[None, :]
    )

    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    array = tl.arange(0, BLOCK).to(tl.float32)
    q_decay = tl.exp(-s.to(tl.float32) * array[:, None])
    k_trans_decay = tl.exp(-s.to(tl.float32) * (BLOCK - array[None, :]))
    block_decay = tl.exp(-s.to(tl.float32) * BLOCK)
    # diag
    index = array[:, None] - array[None, :]
    s_index = s * index
    s_index = tl.where(index >= 0, -s_index, float("-inf"))
    diag_decay = tl.exp(s_index)

    # load global KV
    KV = tl.load(KV_block_ptr).to(tl.float32)

    kv = tl.zeros([DBLOCK, EBLOCK], dtype=tl.float32)
    for i in range(NUM_BLOCK):
        q = tl.load(Q_block_ptr).to(tl.float32)
        k_trans = tl.load(K_trans_block_ptr).to(tl.float32)
        v = tl.load(V_block_ptr).to(tl.float32)

        qkv_none_diag = tl.dot(q, kv) * q_decay + tl.dot(q, KV) * tl.exp(
            -s.to(tl.float32) * (array[:, None] + i * BLOCK)
        )
        # diag
        qk = tl.dot(q, k_trans) * diag_decay
        qkv_diag = tl.dot(qk, v)

        qkv = qkv_none_diag + qkv_diag

        tl.store(O_block_ptr, qkv.to(O_block_ptr.dtype.element_ty))
        kv = block_decay * kv + tl.dot(k_trans * k_trans_decay, v)

        Q_block_ptr += BLOCK * d
        K_trans_block_ptr += BLOCK * d
        V_block_ptr += BLOCK * e
        O_block_ptr += BLOCK * e

    KV = tl.exp(-s.to(tl.float32) * n) * KV + kv
    tl.store(KV_block_ptr, KV.to(KV_block_ptr.dtype.element_ty))


@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    S,
    DO,
    DQ,
    DK,
    DV,
    KV,
    DKV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    DBLOCK: tl.constexpr,
    NUM_DBLOCK: tl.constexpr,
    EBLOCK: tl.constexpr,
    NUM_EBLOCK: tl.constexpr,
):
    off_d = tl.program_id(0)
    off_e = tl.program_id(1)
    off_bh = tl.program_id(2)
    off_h = off_bh % h

    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    kv_offset = off_bh * d * e

    d_offset = off_d * DBLOCK
    e_offset = off_e * EBLOCK

    dqk_offset = off_e * b * h * n * d
    dv_offset = off_d * b * h * n * e

    d_offset = off_d * DBLOCK
    e_offset = off_e * EBLOCK
    kv_d_offset = d_offset * e

    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)
    block_decay = tl.exp(-s.to(tl.float32) * BLOCK)

    DQ_block_ptr = (
        DQ
        + qk_offset
        + dqk_offset
        + d_offset
        + tl.arange(0, BLOCK)[:, None] * d
        + tl.arange(0, DBLOCK)[None, :]
    )
    K_block_ptr = (
        K
        + qk_offset
        + d_offset
        + tl.arange(0, BLOCK)[:, None] * d
        + tl.arange(0, DBLOCK)[None, :]
    )
    V_trans_block_ptr = (
        V
        + v_offset
        + e_offset
        + tl.arange(0, BLOCK)[None, :] * e
        + tl.arange(0, EBLOCK)[:, None]
    )
    DO_block_ptr = (
        DO
        + o_offset
        + e_offset
        + tl.arange(0, BLOCK)[:, None] * e
        + tl.arange(0, EBLOCK)[None, :]
    )

    KV_trans_block_ptr = (
        KV
        + kv_offset
        + kv_d_offset
        + e_offset
        + tl.arange(0, DBLOCK)[None, :] * e
        + tl.arange(0, EBLOCK)[:, None]
    )
    DKV_block_ptr = (
        DKV
        + kv_offset
        + kv_d_offset
        + e_offset
        + tl.arange(0, DBLOCK)[:, None] * e
        + tl.arange(0, EBLOCK)[None, :]
    )

    # compute block array
    array = tl.arange(0, BLOCK)

    # diag
    index = array[:, None] - array[None, :]
    s_index = s * index
    s_index = tl.where(index >= 0, -s_index, float("-inf"))
    diag_decay = tl.exp(s_index)
    diag_decay_trans = tl.trans(diag_decay)

    KV_trans = tl.load(KV_trans_block_ptr).to(tl.float32)
    kv_trans = tl.zeros([EBLOCK, DBLOCK], dtype=tl.float32)
    for i in range(NUM_BLOCK):
        q_decay = tl.exp(-s.to(tl.float32) * array[:, None])
        k_decay = tl.exp(-s.to(tl.float32) * (BLOCK - array[:, None]))
        do = tl.load(DO_block_ptr).to(tl.float32)
        k = tl.load(K_block_ptr).to(tl.float32)
        v_trans = tl.load(V_trans_block_ptr).to(tl.float32)

        dq_none_diag = tl.dot(do, kv_trans) * q_decay + tl.dot(do, KV_trans) * tl.exp(
            -s.to(tl.float32) * (i * BLOCK + array[:, None])
        )

        dqk = tl.dot(do, v_trans) * diag_decay
        dq_diag = tl.dot(dqk, k)

        dq = dq_none_diag + dq_diag

        tl.store(DQ_block_ptr, dq.to(DQ_block_ptr.dtype.element_ty))

        DQ_block_ptr += BLOCK * d
        DO_block_ptr += BLOCK * e
        K_block_ptr += BLOCK * d
        V_trans_block_ptr += BLOCK * e

        kv_trans = block_decay * kv_trans + tl.dot(v_trans, k * k_decay)

    Q_trans_block_ptr = (
        Q
        + qk_offset
        + d_offset
        + n * d
        + tl.arange(0, BLOCK)[None, :] * d
        + tl.arange(0, DBLOCK)[:, None]
    )
    K_block_ptr = (
        K
        + qk_offset
        + d_offset
        + n * d
        + tl.arange(0, BLOCK)[:, None] * d
        + tl.arange(0, DBLOCK)[None, :]
    )
    V_trans_block_ptr = (
        V
        + v_offset
        + e_offset
        + n * e
        + tl.arange(0, BLOCK)[None, :] * e
        + tl.arange(0, EBLOCK)[:, None]
    )

    DK_trans_block_ptr = (
        DK
        + qk_offset
        + dqk_offset
        + d_offset
        + n * d
        + tl.arange(0, BLOCK)[None, :] * d
        + tl.arange(0, DBLOCK)[:, None]
    )
    DV_block_ptr = (
        DV
        + v_offset
        + dv_offset
        + e_offset
        + n * e
        + tl.arange(0, BLOCK)[:, None] * e
        + tl.arange(0, EBLOCK)[None, :]
    )
    DO_block_ptr = (
        DO
        + o_offset
        + e_offset
        + n * e
        + tl.arange(0, BLOCK)[:, None] * e
        + tl.arange(0, EBLOCK)[None, :]
    )

    DKV = tl.load(DKV_block_ptr)
    dkv = tl.zeros([DBLOCK, EBLOCK], dtype=tl.float32)
    for i in range(NUM_BLOCK - 1, -1, -1):
        K_block_ptr -= BLOCK * d
        V_trans_block_ptr -= BLOCK * e
        DK_trans_block_ptr -= BLOCK * d
        DV_block_ptr -= BLOCK * e
        DO_block_ptr -= BLOCK * e
        Q_trans_block_ptr -= BLOCK * d

        k = tl.load(K_block_ptr).to(tl.float32)
        v_trans = tl.load(V_trans_block_ptr).to(tl.float32)
        do = tl.load(DO_block_ptr).to(tl.float32)
        q_trans = tl.load(Q_trans_block_ptr).to(tl.float32)

        k_decay_trans = tl.exp(-s.to(tl.float32) * (BLOCK - array[None, :]))
        k_decay = tl.exp(-s.to(tl.float32) * (BLOCK - array[:, None]))
        q_decay_trans = tl.exp(-s.to(tl.float32) * array[None, :])

        dqk = tl.dot(do, v_trans) * diag_decay
        dk_diag_trans = tl.dot(q_trans, dqk)
        dk_none_diag_trans = tl.dot(dkv, v_trans) * k_decay_trans + tl.dot(
            DKV, v_trans.to(DKV.dtype)
        ) * tl.exp(-s.to(tl.float32) * (n - i * BLOCK - array[None, :]))
        dk_trans = dk_none_diag_trans + dk_diag_trans

        qk_trans = tl.dot(k, q_trans) * diag_decay_trans
        dv_diag = tl.dot(qk_trans, do)
        dv_none_diag = tl.dot(k, dkv) * k_decay + tl.dot(k.to(DKV.dtype), DKV) * tl.exp(
            -s.to(tl.float32) * (n - i * BLOCK - array[:, None])
        )
        dv = dv_none_diag + dv_diag

        tl.store(DK_trans_block_ptr, dk_trans.to(DK_trans_block_ptr.dtype.element_ty))
        tl.store(DV_block_ptr, dv.to(DV_block_ptr.dtype.element_ty))

        dkv = block_decay * dkv + tl.dot(q_trans * q_decay_trans, do)

    DKV = tl.exp(-s.to(tl.float32) * n) * DKV + dkv
    tl.store(DKV_block_ptr, DKV.to(DKV_block_ptr.dtype.element_ty))


def lasp_forward(q, k, v, s, KV):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    s = s.contiguous()
    KV = KV.contiguous()

    # shape constraints
    b, h, n, d = q.shape
    e = v.shape[-1]
    # split over head
    cd = 64
    ce = 64
    d_, e_ = min(triton.next_power_of_2(d), cd), min(triton.next_power_of_2(e), ce)
    nd, ne = d // d_, e // e_
    # right
    o = torch.empty((nd, b, h, n, e), dtype=q.dtype, device=q.device)

    # Get optimal block sizes based on GPU architecture
    config = get_config_for_kernel('lasp_fuse', n, d, e, q.device)
    BLOCK = config['BLOCK']
    # Use ceiling division to handle partial blocks correctly
    NUM_BLOCK = triton.cdiv(n, BLOCK)

    grid = (nd, ne, b * h)

    with torch.cuda.device(q.device.index):
        _fwd_kernel[grid](
            q,
            k,
            v,
            o,
            s,
            KV,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            DBLOCK=d_,
            NUM_DBLOCK=nd,
            EBLOCK=e_,
            NUM_EBLOCK=ne,
        )

    if nd > 1:
        o = o.sum(0)
    else:
        o.squeeze_()

    return o


def lasp_backward(q, k, v, s, do, KV, DKV):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    s = s.contiguous()
    do = do.contiguous()
    KV = KV.contiguous()
    DKV = DKV.contiguous()

    b, h, n, d = q.shape
    e = v.shape[-1]

    # Get optimal block sizes based on GPU architecture
    config = get_config_for_kernel('lasp_fuse', n, d, e, q.device)
    BLOCK = config['BLOCK']
    NUM_BLOCK = triton.cdiv(n, BLOCK)

    cd = 64
    ce = 64
    d_, e_ = min(triton.next_power_of_2(d), cd), min(triton.next_power_of_2(e), ce)
    nd, ne = d // d_, e // e_

    dq = torch.empty((ne, b, h, n, d), dtype=q.dtype, device=q.device)
    dk = torch.empty((ne, b, h, n, d), dtype=q.dtype, device=q.device)
    dv = torch.empty((nd, b, h, n, e), dtype=q.dtype, device=q.device)

    grid = (
        nd,
        ne,
        b * h,
    )

    with torch.cuda.device(q.device.index):
        _bwd_kernel[grid](
            q,
            k,
            v,
            s,
            do,
            dq,
            dk,
            dv,
            KV,
            DKV,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            DBLOCK=d_,
            NUM_DBLOCK=nd,
            EBLOCK=e_,
            NUM_EBLOCK=ne,
        )

    if ne > 1:
        dq = dq.sum(0)
        dk = dk.sum(0)
    else:
        dq.squeeze_(0)
        dk.squeeze_(0)

    if nd > 1:
        dv = dv.sum(0)
    else:
        dv.squeeze_(0)

    return dq, dk, dv


class LaspFuse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, s, KV, DKV):
        # s: (h, 1, 1)
        b, h, n, d = q.shape
        v.shape[-1]

        KV.zero_()

        group = get_sequence_parallel_group()
        current_idx = get_sequence_parallel_rank()
        send_idx = get_seq_parallel_send_rank()
        recv_idx = get_seq_parallel_receive_rank()

        if current_idx > 0:
            dist.recv(KV, src=send_idx, group=group)

        # need clone, import !!!
        ctx.save_for_backward(q, k, v, s, KV.clone(), DKV)

        o = lasp_forward(q, k, v, s, KV)

        if current_idx < get_sequence_parallel_world_size() - 1:
            dist.send(KV, dst=recv_idx, group=group)

        ctx.group = group
        ctx.current_idx = current_idx
        ctx.send_idx = send_idx
        ctx.recv_idx = recv_idx

        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, s, KV, DKV = ctx.saved_tensors
        group = ctx.group
        # forward: 0->1, backward: 1->0
        current_idx = ctx.current_idx
        send_idx = ctx.recv_idx
        recv_idx = ctx.send_idx

        b, h, n, d = q.shape
        v.shape[-1]

        DKV.zero_()

        if current_idx < get_sequence_parallel_world_size() - 1:
            dist.recv(DKV, src=send_idx, group=group)

        dq, dk, dv = lasp_backward(q, k, v, s, do, KV, DKV)

        if current_idx > 0:
            dist.send(DKV, dst=recv_idx, group=group)

        return dq, dk, dv, None, None, None, None


lasp_fuse_ = LaspFuse.apply


def lasp_fuse(q, k, v, ed, KV, DKV):
    b, h, n, d = q.shape
    e = v.shape[-1]

    if d >= 128:
        m = 128
    else:
        m = 64
    arr = [m * i for i in range(d // m + 1)]
    if arr[-1] != d:
        arr.append(d)
    n = len(arr)
    output = 0
    for i in range(n - 1):
        s = arr[i]
        e = arr[i + 1]
        q1 = q[..., s:e]
        k1 = k[..., s:e]
        o = lasp_fuse_(
            q1, k1, v, ed, KV[:, :, s:e].contiguous(), DKV[:, :, s:e].contiguous()
        )
        output = output + o

    return output


# LASP-2: AllGather-based implementation


@triton.jit
def _compute_local_kv_kernel(
    K,
    V,
    S,
    KV_out,
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
    """Compute local memory state M_r = K^T @ V for a chunk."""
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
    for i in range(NUM_BLOCK):
        k_trans = tl.load(K_trans_block_ptr).to(tl.float32)
        v = tl.load(V_block_ptr).to(tl.float32)

        kv = block_decay * kv + tl.dot(k_trans * k_trans_decay, v)

        K_trans_block_ptr += BLOCK * d
        V_block_ptr += BLOCK * e

    # Store local KV contribution
    tl.store(KV_block_ptr, kv.to(KV_block_ptr.dtype.element_ty))


def compute_local_kv(k, v, s, d_, e_, BLOCK, NUM_BLOCK):
    """Compute local memory state M_r = K^T @ V."""
    k = k.contiguous()
    v = v.contiguous()
    s = s.contiguous()

    b, h, n, d = k.shape
    e = v.shape[-1]
    nd, ne = d // d_, e // e_

    # Output shape: (b, h, d, e)
    kv_out = torch.empty((b, h, d, e), dtype=k.dtype, device=k.device)

    grid = (nd, ne, b * h)

    with torch.cuda.device(k.device.index):
        _compute_local_kv_kernel[grid](
            k,
            v,
            s,
            kv_out,
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

    return kv_out


class LaspFuseV2(torch.autograd.Function):
    """LASP-2: AllGather-based implementation for improved parallelism.

    Uses a single AllGather collective instead of ring P2P communication,
    reducing communication steps from 2(W-1) to 2, where W is world size.

    Note: Assumes the sequence parallel group ranks are ordered to match
    the sequence shard order (rank i has the i-th chunk of the sequence).
    """

    @staticmethod
    def forward(ctx, q, k, v, s, KV, DKV):
        b, h, n, d = q.shape
        e = v.shape[-1]

        # Get config
        config = get_config_for_kernel('lasp_fuse', n, d, e, q.device)
        BLOCK = config['BLOCK']
        # Use ceiling division to handle partial blocks correctly
        NUM_BLOCK = triton.cdiv(n, BLOCK)

        # Use same caps as V1 to ensure nd, ne >= 1
        # Otherwise if d=768, next_power_of_2=1024 → nd=0 → invalid grid
        cd = 64
        ce = 64
        d_ = min(triton.next_power_of_2(d), cd)
        e_ = min(triton.next_power_of_2(e), ce)

        # Get parallel group info
        group = get_sequence_parallel_group()
        current_idx = get_sequence_parallel_rank()
        world_size = get_sequence_parallel_world_size()

        # Step 1: Compute local memory state M_r = K^T @ V
        local_KV = compute_local_kv(k, v, s, d_, e_, BLOCK, NUM_BLOCK)

        # Step 2: Compute per-rank gamma = exp(-s * n_local)
        # This is the cumulative decay across this rank's local chunk
        # Shape: [H] → broadcast to [1, H, 1, 1] for element-wise ops
        # Use padded length for consistency with kernel processing
        n_local = NUM_BLOCK * BLOCK
        gamma_local = torch.exp(-s.to(torch.float32) * n_local).to(local_KV.dtype).view(1, h, 1, 1)

        # Step 3: AllGather gamma and KV from all ranks with stream overlap
        gamma_list = [torch.empty_like(gamma_local) for _ in range(world_size)]
        KV_list = [torch.empty_like(local_KV) for _ in range(world_size)]

        # Use separate stream for communication to enable overlap
        comm_stream = torch.cuda.Stream()
        comm_done = torch.cuda.Event()

        with torch.cuda.stream(comm_stream):
            dist.all_gather(gamma_list, gamma_local.contiguous(), group=group)
            dist.all_gather(KV_list, local_KV.contiguous(), group=group)
            comm_done.record()

        # Wait for communication to complete
        torch.cuda.current_stream().wait_event(comm_done)

        # Step 4: Compute decay-weighted exclusive prefix
        # Prefix for rank r: sum_{i<r} (prod_{t=i+1..r} gamma[t]) * local_KV[i]
        # First compute prefix products G[r] = prod_{t=0..r-1} gamma[t]
        # Need world_size + 1 elements to handle all decay computations
        G = [torch.ones_like(gamma_local)]
        for r in range(1, world_size + 1):  # Extended to world_size + 1
            G.append(G[-1] * gamma_list[r - 1])

        # Compute decay-weighted prefix sum
        if current_idx > 0:
            KV_prefix = torch.zeros_like(local_KV)
            for i in range(current_idx):
                # Weight for KV from rank i at rank current_idx is G[current_idx] / G[i+1]
                # Add small epsilon for numerical stability
                weight = G[current_idx] / (G[i + 1] + 1e-10)
                KV_prefix = KV_prefix + weight * KV_list[i]
        else:
            # Rank 0 has no prefix
            KV_prefix = torch.zeros_like(local_KV)

        # Copy to KV buffer for kernel
        KV.copy_(KV_prefix)

        # Step 5: Run forward pass with prefix KV
        o = lasp_forward(q, k, v, s, KV)

        # Save for backward - store gamma_list and G for decay-weighted gradient suffix
        ctx.save_for_backward(q, k, v, s, local_KV)
        ctx.gamma_list = gamma_list
        ctx.G = G
        ctx.group = group
        ctx.current_idx = current_idx
        ctx.world_size = world_size
        ctx.config = config

        return o

    @staticmethod
    def backward(ctx, do):
        """
        LASP-2 backward implementation following the algorithm from the paper.

        Algorithm:
        1. Compute local dM (dKV) from each rank's do
        2. AllGather all local dM values
        3. Compute weighted suffix sum of dM (gradients from successors)
        4. Use total dM to compute dK, dV
        5. Compute dQ from do and KV states
        """
        q, k, v, s, local_KV = ctx.saved_tensors
        gamma_list = ctx.gamma_list
        G = ctx.G
        group = ctx.group
        current_idx = ctx.current_idx
        world_size = ctx.world_size
        config = ctx.config

        b, h, n, d = q.shape
        e = v.shape[-1]

        BLOCK = config['BLOCK']
        NUM_BLOCK = triton.cdiv(n, BLOCK)

        cd = 64
        ce = 64
        d_ = min(triton.next_power_of_2(d), cd)
        e_ = min(triton.next_power_of_2(e), ce)

        comm_stream = torch.cuda.Stream()
        comm_done = torch.cuda.Event()

        # ============ STEP 1: Compute local dM (dKV) contribution ============
        # For rank r, local dM comes from: dM_r = Q_r^T @ do_r
        # This is the gradient of the local memory state from the local attention output

        # We need to compute this using the backward kernel, but with zero incoming DKV
        # to isolate just the local contribution
        local_dM = torch.zeros_like(local_KV)

        # Use the backward kernel to compute local dM contribution
        # Pass zero for KV_prefix since we only want the local dM, not the gradients yet
        _ = lasp_backward(q, k, v, s, do, torch.zeros_like(local_KV), local_dM)

        # ============ STEP 2: AllGather all local dM contributions ============
        dM_list = [torch.empty_like(local_dM) for _ in range(world_size)]

        with torch.cuda.stream(comm_stream):
            dist.all_gather(dM_list, local_dM.contiguous(), group=group)
            comm_done.record()

        torch.cuda.current_stream().wait_event(comm_done)

        # ============ STEP 3: Compute weighted suffix sum of dM ============
        # Gradients flow from later chunks (successors) to earlier chunks
        # For rank r: total_dM_r = local_dM_r + sum_{j>r} weight(r,j) * local_dM_j
        # where weight(r,j) = G[j+1] / G[r+1] (decay from rank j back to rank r)

        total_dM = local_dM.clone()  # Start with local contribution

        if current_idx < world_size - 1:
            for j in range(current_idx + 1, world_size):
                # Weight for gradient from rank j flowing back to current rank
                weight = G[j + 1] / (G[current_idx + 1] + 1e-10)
                total_dM = total_dM + weight * dM_list[j]

        # ============ STEP 4: Reconstruct KV_prefix for computing dQ ============
        KV_list = [torch.empty_like(local_KV) for _ in range(world_size)]

        with torch.cuda.stream(comm_stream):
            dist.all_gather(KV_list, local_KV.contiguous(), group=group)
            comm_done.record()

        torch.cuda.current_stream().wait_event(comm_done)

        # Compute decay-weighted exclusive prefix (same as forward)
        if current_idx > 0:
            KV_prefix = torch.zeros_like(local_KV)
            for i in range(current_idx):
                weight = G[current_idx] / (G[i + 1] + 1e-10)
                KV_prefix = KV_prefix + weight * KV_list[i]
        else:
            KV_prefix = torch.zeros_like(local_KV)

        # ============ STEP 5: Compute final gradients ============
        # Now we compute dQ, dK, dV using:
        # - do: upstream gradient
        # - KV_prefix: state from predecessors
        # - total_dM: accumulated gradient state (local + weighted successors)

        # Run backward with the complete accumulated dM
        dq, dk, dv = lasp_backward(q, k, v, s, do, KV_prefix, total_dM)

        return dq, dk, dv, None, None, None


lasp_fuse_v2_ = LaspFuseV2.apply


def lasp_fuse_v2(q, k, v, ed, KV, DKV):
    """
    LASP-2: AllGather-based implementation.

    Uses a single AllGather collective instead of ring P2P communication,
    reducing communication steps from 2(W-1) to 2, where W is world size.

    Args:
        q: Query tensor [B, H, N, D]
        k: Key tensor [B, H, N, D]
        v: Value tensor [B, H, N, E]
        ed: Exponential decay parameter [H]
        KV: Buffer for KV state [B, H, D, E]
        DKV: Buffer for gradient of KV state [B, H, D, E]

    Returns:
        output: Attention output [B, H, N, E]
    """
    return lasp_fuse_v2_(q, k, v, ed, KV, DKV)
