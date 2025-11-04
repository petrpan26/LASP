# Optimizing LASP with Blelloch Parallel Prefix Scan

## Problem Statement

### Current LASP Bottleneck

LASP (Linear Attention Sequence Parallelism) achieves **sequence-length independent communication** by passing compressed KV states (size `d×d`) between GPUs instead of full K,V matrices. However, it suffers from a critical **sequential dependency**:

```
GPU 0 → GPU 1 → GPU 2 → GPU 3 → ... → GPU P-1
  ↓      ↓        ↓        ↓              ↓
 KV[0]  KV[1]   KV[2]   KV[3]         KV[P-1]
```

**Time Complexity**: `O(P)` sequential communication steps where P is the number of GPUs.

For **P = 128 GPUs**, this means **128 sequential communication rounds**, each with latency overhead.

### Why This Matters

**Communication Time** = `P × (α + d²/β)`

Where:
- `α` = network latency (typically 1-10 μs)
- `β` = bandwidth (GB/s)
- `d²` = KV state size (e.g., 4096² = 16M elements)

For large clusters:
- P=128, α=5μs, d=4096, β=300GB/s → **640μs latency + 8.7ms transfer ≈ 1.2 seconds** just for communication!
- This grows linearly with P, limiting scalability

---

## Solution: Blelloch Parallel Prefix Scan

### Key Insight

The KV state update is a **linear recurrence**:

```
KV[t] = λ^C · KV[t-1] + b[t]
```

Where `b[t] = (λ^C Λ^(-1) K[t])^T V[t]` is computed locally on GPU t.

This recurrence can be parallelized using **prefix scan** because the operation is **associative**.

### Associative Operator

Define the operator `⊕` on pairs `(A, b)`:

```
(A₁, b₁) ⊕ (A₂, b₂) = (A₁ · A₂,  A₂ · b₁ + b₂)
```

For LASP:
- `A[t] = λ^C` (scalar, same for all t)
- `b[t]` is the local KV contribution (d×d matrix)

**Verification of Associativity**:
```
((A₁, b₁) ⊕ (A₂, b₂)) ⊕ (A₃, b₃) 
= (A₁·A₂, A₂·b₁ + b₂) ⊕ (A₃, b₃)
= (A₁·A₂·A₃, A₃·(A₂·b₁ + b₂) + b₃)
= (A₁·A₂·A₃, A₃·A₂·b₁ + A₃·b₂ + b₃)

(A₁, b₁) ⊕ ((A₂, b₂) ⊕ (A₃, b₃))
= (A₁, b₁) ⊕ (A₂·A₃, A₃·b₂ + b₃)
= (A₁·A₂·A₃, A₂·A₃·b₁ + A₃·b₂ + b₃)  ✓
```

### Complexity Improvement

| Method | Sequential Steps | Parallel Complexity | Total Communication |
|--------|------------------|---------------------|---------------------|
| **LASP (Ring)** | P | O(P) | P × d² |
| **LASP + Blelloch** | 2 log₂(P) | O(log P) | 2 log₂(P) × d² |

For **P = 128 GPUs**: 128 steps → **14 steps** (9× reduction!)

---

## Blelloch Algorithm Explained

### Two-Phase Approach

1. **Up-sweep (Reduce)**: Build partial sums in a tree structure
2. **Down-sweep (Distribute)**: Propagate partial sums to compute all prefixes

### Concrete Example: 8 GPUs

#### Initial State

Each GPU has its local contribution:
```
GPU 0: b[0] = K[0]^T V[0]
GPU 1: b[1] = K[1]^T V[1]
GPU 2: b[2] = K[2]^T V[2]
...
GPU 7: b[7] = K[7]^T V[7]
```

#### Up-Sweep Phase

**Step 1** (4 parallel operations, stride=1):
```
GPU 0 → GPU 1:  s[1] = λ^C · b[0] + b[1]
GPU 2 → GPU 3:  s[3] = λ^C · b[2] + b[3]
GPU 4 → GPU 5:  s[5] = λ^C · b[4] + b[5]
GPU 6 → GPU 7:  s[7] = λ^C · b[6] + b[7]
```

**Step 2** (2 parallel operations, stride=2):
```
GPU 1 → GPU 3:  s[3] = λ^(2C) · s[1] + s[3]  (covers GPUs 0-3)
GPU 5 → GPU 7:  s[7] = λ^(2C) · s[5] + s[7]  (covers GPUs 4-7)
```

**Step 3** (1 operation, stride=4):
```
GPU 3 → GPU 7:  s[7] = λ^(4C) · s[3] + s[7]  (covers GPUs 0-7)
```

Now GPU 7 has the total sum representing KV[0:7].

#### Down-Sweep Phase

**Step 4** (1 operation):
```
GPU 7 → GPU 3:  Send left_sum = s[7] - λ^(4C)·s[3]
GPU 3 stores this as prefix sum for GPU 3
```

**Step 5** (2 parallel operations):
```
GPU 7 → GPU 5:  Send appropriate prefix
GPU 3 → GPU 1:  Send appropriate prefix
```

**Step 6** (4 parallel operations):
```
GPU 7 → GPU 6:  Final prefix for GPU 6
GPU 5 → GPU 4:  Final prefix for GPU 4
GPU 3 → GPU 2:  Final prefix for GPU 2
GPU 1 → GPU 0:  Final prefix for GPU 0
```

After down-sweep, each GPU t has `KV[0:t]` (the prefix sum up to position t).

### Visual Tree Structure

```
Up-sweep (reduce):
        Level 0:  b₀   b₁   b₂   b₃   b₄   b₅   b₆   b₇
                  │╲   │╲   │╲   │╲   │╲   │╲   │╲   │╲
        Level 1:  │ s₁  │ s₃  │ s₅  │ s₇ 
                  │  ╲  │  ╲  │  ╲  │  ╲
        Level 2:  │   s₃     │   s₇
                  │     ╲    │    ╱
        Level 3:  │      s₇

Down-sweep (distribute):
        Level 3:         s₇
                        ╱  ╲
        Level 2:      s₃    s₇
                     ╱ ╲   ╱ ╲
        Level 1:    s₁ s₃ s₅ s₇
                   ╱│ ╱│ ╱│ ╱│
        Level 0:  KV₀ KV₁ KV₂ KV₃ KV₄ KV₅ KV₆ KV₇
```

---

## Implementation

### Pseudocode

```python
def blelloch_scan_lasp(local_K, local_V, lambda_decay, chunk_size, rank, world_size):
    """
    Parallel prefix scan for LASP using Blelloch algorithm.
    
    Args:
        local_K: Key matrix for local chunk (C × d)
        local_V: Value matrix for local chunk (C × d)
        lambda_decay: Decay factor λ
        chunk_size: Size of each chunk C
        rank: GPU rank (0 to P-1)
        world_size: Total number of GPUs (P)
    
    Returns:
        KV_prefix: Prefix KV state up to this rank (d × d)
    """
    d = local_K.shape[1]
    
    # Compute local contribution: b[rank] = (λ^C Λ^(-1) K)^T V
    Lambda_inv = torch.diag(1 / lambda_decay ** torch.arange(1, chunk_size + 1))
    decay_factor = lambda_decay ** chunk_size
    local_b = (decay_factor * Lambda_inv @ local_K).T @ local_V  # (d × d)
    
    # Store intermediate values
    tree_values = [local_b]
    
    # ============ UP-SWEEP PHASE ============
    num_levels = int(np.log2(world_size))
    
    for level in range(num_levels):
        stride = 2 ** level
        partner_rank = rank + stride
        
        if (rank % (2 * stride) == 0) and (partner_rank < world_size):
            # This GPU sends to partner
            send_value = tree_values[-1]
            dist.send(tensor=send_value, dst=partner_rank)
            
        elif (rank % (2 * stride) == stride):
            # This GPU receives and combines
            received = torch.zeros(d, d)
            dist.recv(tensor=received, src=rank - stride)
            
            # Combine: (λ^(stride*C) · received) + local
            combined = (decay_factor ** stride) * received + tree_values[-1]
            tree_values.append(combined)
    
    # ============ DOWN-SWEEP PHASE ============
    prefix_sum = None
    
    for level in range(num_levels - 1, -1, -1):
        stride = 2 ** level
        partner_rank = rank - stride
        
        if (rank % (2 * stride) == stride) and (partner_rank >= 0):
            # This GPU receives prefix from left partner
            if prefix_sum is None:
                prefix_sum = tree_values[-1]
            
            left_prefix = torch.zeros(d, d)
            dist.recv(tensor=left_prefix, src=partner_rank)
            
            # Update prefix: combine with left neighbor's prefix
            prefix_sum = (decay_factor ** stride) * left_prefix + tree_values[level]
            
        elif (rank % (2 * stride) == 0) and (partner_rank >= 0):
            # This GPU sends to right partner
            send_value = prefix_sum if prefix_sum is not None else tree_values[level]
            dist.send(tensor=send_value, dst=rank + stride)
    
    if prefix_sum is None:
        prefix_sum = tree_values[-1] if tree_values else local_b
    
    return prefix_sum  # KV[0:rank]
```

### Integration with LASP Forward Pass

```python
def lasp_forward_with_blelloch(X, W_Q, W_K, W_V, lambda_decay, rank, world_size):
    """
    LASP forward pass using Blelloch scan for inter-chunk computation.
    """
    # Split sequence across GPUs
    local_X = split_sequence(X, rank, world_size)
    C = local_X.shape[0]  # chunk size
    d = local_X.shape[1]  # hidden dim
    
    # Compute Q, K, V for local chunk
    Q_local = local_X @ W_Q
    K_local = local_X @ W_K
    V_local = local_X @ W_V
    
    # ======== INTRA-CHUNK COMPUTATION (Parallel) ========
    # Standard causal attention within chunk
    causal_mask = torch.tril(torch.ones(C, C))
    attn_scores = Q_local @ K_local.T
    attn_masked = attn_scores * causal_mask
    O_intra = attn_masked @ V_local
    
    # ======== INTER-CHUNK COMPUTATION (Blelloch Scan) ========
    # Use Blelloch to get KV_prefix in O(log P) steps
    KV_prefix = blelloch_scan_lasp(
        K_local, V_local, lambda_decay, C, rank, world_size
    )
    
    # Compute inter-chunk attention
    Lambda = torch.diag(lambda_decay ** torch.arange(1, C + 1))
    O_inter = Lambda @ Q_local @ KV_prefix
    
    # Combine
    O_local = O_intra + O_inter
    
    return O_local
```

### Backward Pass

The backward pass follows similar logic but in reverse:

```python
def blelloch_scan_backward(dO_local, Q_local, lambda_decay, chunk_size, 
                           rank, world_size):
    """
    Backward pass for LASP using reverse Blelloch scan.
    Gradients flow from right to left (opposite of forward).
    """
    # Compute local gradient contribution
    Lambda = torch.diag(lambda_decay ** torch.arange(1, chunk_size + 1))
    local_dKV = (Lambda @ Q_local).T @ dO_local
    
    # Blelloch scan in reverse direction (right to left)
    # Implementation similar to forward but with communication reversed
    ...
    
    return dKV_suffix  # Gradient from all chunks after this one
```

---

## Performance Analysis

### Theoretical Speedup

**Communication Time Comparison**:

```
T_ring = P × (α + d²/β)
T_blelloch = 2 log₂(P) × (α + d²/β + contention)
```

**Speedup** = `T_ring / T_blelloch ≈ P / (2 log₂ P)` when contention is low.

| GPUs (P) | Ring Steps | Blelloch Steps | Theoretical Speedup |
|----------|------------|----------------|---------------------|
| 8 | 8 | 6 | 1.3× |
| 16 | 16 | 8 | 2.0× |
| 32 | 32 | 10 | 3.2× |
| 64 | 64 | 12 | 5.3× |
| 128 | 128 | 14 | **9.1×** |
| 256 | 256 | 16 | **16×** |

### When Blelloch Wins

**Advantages**:
1. **Large clusters** (P ≥ 64): Log growth dominates
2. **High latency networks**: Latency α is amortized over fewer rounds
3. **Good interconnect topology**: NVSwitch, fat-tree networks
4. **Small d²**: Latency-bound regime

**Disadvantages**:
1. **Network contention**: Multiple parallel communications compete for bandwidth
2. **Complex topology mapping**: Tree pattern may not match physical network
3. **Implementation complexity**: More communication patterns to handle

### Hybrid Strategy

For realistic clusters with hierarchical topology:

```python
# Use Blelloch within nodes (fast NVLink/NVSwitch)
# Use Ring across nodes (slower InfiniBand)

def hybrid_scan(local_K, local_V, rank, world_size, gpus_per_node=8):
    node_id = rank // gpus_per_node
    local_rank = rank % gpus_per_node
    num_nodes = world_size // gpus_per_node
    
    # Step 1: Blelloch scan within node (O(log 8) = 3 steps)
    node_KV = blelloch_scan_lasp(
        local_K, local_V, 
        rank=local_rank, world_size=gpus_per_node
    )
    
    # Step 2: Ring scan across nodes (O(num_nodes) steps)
    if local_rank == gpus_per_node - 1:  # Last GPU in node
        inter_node_KV = ring_scan_lasp(
            node_KV,
            rank=node_id, world_size=num_nodes
        )
    
    # Broadcast inter-node result within node
    ...
```

**Complexity**: `O(log(gpus_per_node) + num_nodes)`
- For 128 GPUs on 16 nodes: `O(log 8 + 16) = O(3 + 16) = 19 steps`
- vs pure ring: 128 steps (6.7× faster)
- vs pure Blelloch: 14 steps (but with less contention)

---

## Implementation Considerations

### 1. Communication Pattern

**Ring LASP**: Simple point-to-point
```python
if rank < world_size - 1:
    dist.send(KV, dst=rank + 1)
if rank > 0:
    dist.recv(KV_prev, src=rank - 1)
```

**Blelloch LASP**: Tree-based, requires careful orchestration
```python
# Different communication partners at each level
# Need to compute: partner_rank = rank ± 2^level
```

### 2. Numerical Stability

With Blelloch, you're composing many `λ^C` multiplications:

```
KV[127] = λ^(127C) · b[0] + λ^(126C) · b[1] + ... + b[127]
```

For large P, `λ^(PC)` can cause numerical issues:
- **Underflow** if λ < 1
- **Overflow** if λ > 1

**Solution**: Use log-space arithmetic or block-wise normalization.

### 3. Memory

Each GPU needs to store:
- `O(log P)` intermediate tree values during up-sweep
- Each intermediate is `d × d`
- Total extra memory: `O(d² log P)`

This is still much smaller than storing full K,V for the sequence!

### 4. Load Balancing

Blelloch requires power-of-2 GPUs for perfect tree structure. For arbitrary P:
- Pad to next power of 2 with identity operations
- Or use a work-efficient variant with unbalanced trees

---

## Experimental Validation

### Expected Results

For a 1B parameter model with d=4096, P=128 GPUs, sequence length 4M:

**Ring LASP**:
- Communication rounds: 128
- Per-round latency: 5 μs
- Per-round transfer: 16M floats × 4 bytes / 300 GB/s ≈ 213 μs
- **Total: 128 × 218 μs = 27.9 ms**

**Blelloch LASP**:
- Communication rounds: 14
- Same per-round cost but with potential 1.5× contention
- **Total: 14 × 327 μs = 4.6 ms**

**Speedup: 6× faster** (accounting for contention)

### Profiling Plan

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, 
                profiler.ProfilerActivity.CUDA],
    with_stack=True
) as prof:
    output = lasp_forward_with_blelloch(...)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

Key metrics:
- Communication time per level
- Computation/communication overlap
- Network bandwidth utilization
- Idle time during tree operations

---

## Summary

### Key Takeaways

1. **Blelloch scan reduces sequential steps from O(P) to O(log P)** for LASP
2. **No increase in memory footprint** - still O(d²) per GPU
3. **Requires associative operator** - linear recurrence fits perfectly
4. **Best for large clusters** (P ≥ 64) with good interconnects
5. **Hybrid approach** combines benefits of both methods

### Next Steps

1. **Implement basic Blelloch version** for LASP forward pass
2. **Profile on real hardware** to measure contention and speedup
3. **Add backward pass** with reverse scan
4. **Optimize for specific topology** (DGX, cloud, etc.)
5. **Compare with hybrid strategies** for realistic deployments

### Code Repository Structure

```
lasp-blelloch/
├── core/
│   ├── ring_scan.py          # Original LASP ring implementation
│   ├── blelloch_scan.py       # New Blelloch scan
│   └── hybrid_scan.py         # Hybrid strategy
├── kernels/
│   ├── combine_op.py          # Optimized ⊕ operator
│   └── fused_attention.py     # Kernel fusion
├── tests/
│   ├── test_correctness.py    # Verify output matches ring
│   ├── test_performance.py    # Benchmark speedup
│   └── test_scaling.py        # Weak/strong scaling
└── experiments/
    └── cluster_configs/        # DGX, AWS, etc.
```

---

## References

1. Blelloch, G. E. (1990). "Prefix sums and their applications"
2. Sun et al. (2024). "Linear Attention Sequence Parallelism" (arXiv:2404.02882)
3. Martin & Cundy (2018). "Parallelizing Linear Recurrent Neural Nets Over Sequence Length"
4. Gu & Dao (2024). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

---

## Appendix: Mathematical Derivation

### Why the Operator is Associative

The recurrence `KV[t] = λ^C · KV[t-1] + b[t]` can be viewed as a linear transformation:

```
[KV[t]]   [λ^C  b[t]] [KV[t-1]]
[  1  ] = [ 0    1  ] [   1   ]
```

Matrix multiplication is associative, therefore our operator is associative.

### Generalization to Other Linear Sequence Models

Any model with recurrence:
```
h[t] = A[t] · h[t-1] + B[t] · x[t]
```

Can use Blelloch scan with operator:
```
(A₁, B₁) ⊕ (A₂, B₂) = (A₂·A₁, A₂·B₁ + B₂)
```

This includes:
- **S4/S5** (state space models)
- **Mamba** (selective SSM)
- **RWKV** (linear RNN)
- **RetNet** (retention networks)
- **All models in Table 5 of LASP paper**