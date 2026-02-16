# Understanding MicroGPT: A Structured Learning Guide

This document breaks down the concepts in Karpathy's microgpt into digestible pieces, progressing from fundamentals to the complete system.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Automatic Differentiation (Autograd)](#2-automatic-differentiation-autograd)
3. [Neural Network Building Blocks](#3-neural-network-building-blocks)
4. [The Transformer Architecture](#4-the-transformer-architecture)
5. [Training: Optimization & Learning](#5-training-optimization--learning)
6. [Inference: Generating Text](#6-inference-generating-text)
7. [Go-Specific Considerations](#7-go-specific-considerations)

---

## 1. The Big Picture

### What is MicroGPT?

MicroGPT is a **complete implementation** of a GPT (Generative Pre-trained Transformer) language model in ~200 lines. It demonstrates that the core algorithm is surprisingly simple—everything else in production systems is "just efficiency."

```
┌──────────────────────────────────────────────────────────────────┐
│                    MicroGPT Pipeline                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Raw Text ──▶ Tokenizer ──▶ Token IDs ──▶ Model ──▶ Predictions │
│     │                                       │                    │
│     │         "hello" → [BOS,7,4,11,11,14]  │                    │
│     │                         ↓              │                    │
│     │              [Embeddings + Transformer]│                    │
│     │                         ↓              │                    │
│     │              Probability Distribution  │                    │
│     │              over next character       │                    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### The Learning Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│    ┌───────────┐     ┌───────────┐     ┌───────────────┐        │
│    │  Forward  │────▶│   Loss    │────▶│   Backward    │        │
│    │   Pass    │     │ (how bad?)│     │(who's guilty?)│        │
│    └───────────┘     └───────────┘     └───────────────┘        │
│          ▲                                     │                │
│          │                                     ▼                │
│    ┌───────────┐                       ┌───────────────┐        │
│    │  Update   │◀──────────────────────│   Gradients   │        │
│    │  Weights  │                       │(how to fix it)│        │
│    └───────────┘                       └───────────────┘        │
│                                                                 │
│            REPEAT 1000+ times until model is good               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Automatic Differentiation (Autograd)

### Why Do We Need Derivatives?

Neural networks learn by adjusting their parameters (weights) to minimize error. To know *how* to adjust them, we need derivatives—they tell us the direction to move each parameter.

**Analogy**: Imagine you're blindfolded on a hill trying to find the lowest point. The derivative (gradient) is like feeling the slope under your feet—it tells you which way is downhill.

### The Chain Rule

The fundamental insight: complex functions are composed of simple operations. The derivative of the whole is computed by multiplying derivatives along the path.

```
        f(g(x))
           │
    ┌──────┴──────┐
    │             │
  g(x)     ×    f'(g(x))
    │
    ├─────────────┐
    │             │
   x        ×   g'(x)
    
    
df/dx = df/dg × dg/dx    (Chain Rule)
```

### The Value Type

In microgpt, `Value` wraps a number and tracks:
1. **data**: The actual computed value
2. **grad**: The derivative of the final loss with respect to this value
3. **children**: What values were used to compute this one
4. **local_grads**: The derivative of this value with respect to each child

```
         c = a × b
         
    a ─────────┐
               │
          ┌────┴────┐
          │    ×    │ ──── c
          │ dc/da=b │
          │ dc/db=a │
          └────┬────┘
               │
    b ─────────┘
```

### Forward vs Backward Pass

**Forward Pass**: Compute the output, building a graph of operations
```
a=2, b=3 ──▶ c=a×b=6 ──▶ d=c+1=7 ──▶ loss=d²=49
```

**Backward Pass**: Start from loss, propagate gradients back using chain rule
```
d(loss)/d(loss) = 1
d(loss)/d(d) = 2×d = 14
d(loss)/d(c) = d(loss)/d(d) × d(d)/d(c) = 14 × 1 = 14
d(loss)/d(a) = d(loss)/d(c) × d(c)/d(a) = 14 × b = 14 × 3 = 42
```

### Topological Sort

Before backpropagating, we must process nodes in reverse dependency order (children before parents in backward direction). This is a topological sort:

```go
func buildTopo(v *Value) []*Value {
    var topo []*Value
    visited := make(map[*Value]struct{})
    
    var dfs func(*Value)
    dfs = func(node *Value) {
        if _, seen := visited[node]; seen {
            return
        }
        visited[node] = struct{}{}
        for _, child := range node.children {
            dfs(child)
        }
        topo = append(topo, node)  // Add AFTER processing children
    }
    
    dfs(v)
    return topo  // Process in reverse order for backward pass
}
```

---

## 3. Neural Network Building Blocks

### Embeddings: From Discrete to Continuous

Tokens are discrete IDs (integers). Neural networks work with continuous vectors. **Embeddings** are learned lookup tables that map each token to a dense vector.

```
Token ID: 5
                     ┌─────────────────┐
Embedding Table:     │ 0: [0.1, 0.3..] │
[vocab_size × dim]   │ 1: [0.5, 0.2..] │
                     │ ...             │
                     │ 5: [0.8, 0.1..] │ ◀── Lookup row 5
                     │ ...             │
                     └─────────────────┘
                            ↓
                     [0.8, 0.1, 0.4, ...]  ← 16-dimensional vector
```

**Position Embeddings**: Same idea, but encode *where* in the sequence a token appears. Added to token embeddings.

### Linear Layer (Matrix Multiplication)

The workhorse of neural networks: multiply input by learned weights.

```
Input x: [3 elements]      Weights W: [4×3]           Output: [4 elements]

    ┌───┐                ┌───────────────┐               ┌───┐
    │ x₀│                │ w₀₀ w₀₁ w₀₂  │               │ y₀│
    │ x₁│    ×           │ w₁₀ w₁₁ w₁₂  │     =         │ y₁│
    │ x₂│                │ w₂₀ w₂₁ w₂₂  │               │ y₂│
    └───┘                │ w₃₀ w₃₁ w₃₂  │               │ y₃│
                         └───────────────┘               └───┘

y₀ = w₀₀×x₀ + w₀₁×x₁ + w₀₂×x₂
y₁ = w₁₀×x₀ + w₁₁×x₁ + w₁₂×x₂
...
```

### RMSNorm (Root Mean Square Normalization)

Keeps activations at a reasonable scale to prevent exploding/vanishing gradients.

```
     Input x: [2.0, 4.0, 6.0]
              │
     Mean of squares: (4 + 16 + 36) / 3 = 18.67
              │
     RMS = √18.67 ≈ 4.32
              │
     Output: [2/4.32, 4/4.32, 6/4.32]
           = [0.46, 0.93, 1.39]  ← Normalized to unit variance
```

### Softmax: Probabilities from Scores

Converts arbitrary scores (logits) into a probability distribution (sums to 1).

```
Logits: [2.0, 1.0, 0.5]
           │
        exp each: [7.39, 2.72, 1.65]
           │
        sum: 11.76
           │
Probabilities: [0.63, 0.23, 0.14]  ← Sums to 1.0
```

**Numerical stability trick**: Subtract max before exp to prevent overflow:
```
[2.0, 1.0, 0.5] - 2.0 = [0, -1.0, -1.5]
exp: [1.0, 0.37, 0.22]  ← No overflow risk
```

### ReLU: Non-linearity

Without non-linear functions, stacking linear layers would just be one big linear layer!

```
ReLU(x) = max(0, x)

     x: [-2, -1, 0, 1, 2]
          │
ReLU(x): [ 0,  0, 0, 1, 2]

Simple, fast, and works well in practice.
```

---

## 4. The Transformer Architecture

### The Core Idea: Attention

**Problem**: How should the model combine information from different positions in a sequence?

**Solution**: Let the model *learn* which positions are relevant to each other.

```
Sequence: "The cat sat on the mat"
           ↑                  ↑
           └──────────────────┘
           "mat" attends to "cat" (both are important for context)
```

### Query, Key, Value (QKV)

Three learned projections of the input:
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What information should I pass along?"

```
    Input x
       │
       ├────▶ Q = x × W_Q  (query projection)
       │
       ├────▶ K = x × W_K  (key projection)
       │
       └────▶ V = x × W_V  (value projection)
```

### Computing Attention

1. **Dot product Q and K**: Find similarity between query and all keys
2. **Scale**: Divide by √d to keep values stable
3. **Softmax**: Convert to probabilities
4. **Weighted sum of V**: Combine values based on attention weights

```
For position t looking at all positions ≤ t:

    Attention(Q, K, V) = softmax(Q × Kᵀ / √d) × V

    Q_t · K_0, Q_t · K_1, ... Q_t · K_t    (dot products)
           │
    Scale by 1/√d
           │
    Softmax → [0.1, 0.2, 0.7]  (attention weights)
           │
    0.1×V_0 + 0.2×V_1 + 0.7×V_t = weighted output
```

### Multi-Head Attention

Run multiple attention "heads" in parallel, each learning different relationships.

```
         ┌─────────────────────────────────────────┐
         │              Input (d=16)               │
         └─────────────────────────────────────────┘
                            │
         ┌──────┬───────────┼───────────┬──────────┐
         ▼      ▼           ▼           ▼          ▼
     ┌──────┐┌──────┐  ┌──────┐    ┌──────┐   ┌──────┐
     │Head 1││Head 2│  │Head 3│    │Head 4│   │Head n│
     │(d/n) ││(d/n) │  │(d/n) │    │(d/n) │   │(d/n) │
     └──────┘└──────┘  └──────┘    └──────┘   └──────┘
         │      │           │           │          │
         └──────┴───────────┼───────────┴──────────┘
                            │
                      Concatenate
                            │
                      Output (d=16)
```

### The Full Transformer Block

```
      Input x
         │
    ┌────┴────┐
    │         │
    │    ┌────▼────┐
    │    │ RMSNorm │
    │    └────┬────┘
    │         │
    │    ┌────▼────┐
    │    │  Attn   │
    │    └────┬────┘
    │         │
    └────►───Add◄───┘   ← Residual Connection
              │
         ┌────┴────┐
         │         │
         │    ┌────▼────┐
         │    │ RMSNorm │
         │    └────┬────┘
         │         │
         │    ┌────▼────┐
         │    │   MLP   │ (linear → ReLU → linear)
         │    └────┬────┘
         │         │
         └────►───Add◄───┘   ← Residual Connection
                   │
                Output
```

**Residual connections**: Add input to output of each sub-block. Helps gradients flow during training.

### KV-Cache for Efficiency

During generation, we process one token at a time. Instead of recomputing K and V for all previous tokens, we **cache** them:

```
Position 0: Compute K₀, V₀ → store in cache
Position 1: Compute K₁, V₁ → store, use K₀, V₀ from cache
Position 2: Compute K₂, V₂ → store, use K₀, K₁, V₀, V₁ from cache
...

Saves O(T) computation at each step!
```

---

## 5. Training: Optimization & Learning

### The Loss Function

**Cross-Entropy Loss**: How surprised is the model by the correct answer?

```
Model predicts probabilities: [0.1, 0.2, 0.7]  (for tokens A, B, C)
Correct answer: B (index 1)

Loss = -log(probability of correct answer)
     = -log(0.2)
     = 1.61

Lower loss = higher confidence in correct answer = better model
```

### Gradient Descent

Move parameters in the direction that decreases loss:

```
new_param = old_param - learning_rate × gradient

If gradient is positive: increasing param increases loss → decrease param
If gradient is negative: increasing param decreases loss → increase param
```

### Adam Optimizer

Plain gradient descent has issues:
- Too sensitive to learning rate
- Gets stuck in flat regions
- Oscillates in steep regions

**Adam** fixes this with:
1. **Momentum (m)**: Running average of gradients → smooth updates
2. **RMSProp (v)**: Running average of squared gradients → adaptive learning rate per parameter

```
For each parameter p with gradient g:

    m = β₁ × m + (1-β₁) × g        # Momentum
    v = β₂ × v + (1-β₂) × g²       # RMSProp
    
    m̂ = m / (1 - β₁ᵗ)              # Bias correction
    v̂ = v / (1 - β₂ᵗ)
    
    p = p - lr × m̂ / (√v̂ + ε)     # Update
```

### Learning Rate Schedule

Start with a higher learning rate, then decay over time:

```
lr_t = lr_base × (1 - step/total_steps)

Step   0: lr = 0.01 × 1.0 = 0.01
Step 500: lr = 0.01 × 0.5 = 0.005
Step 999: lr = 0.01 × 0.001 = 0.00001

Big steps early to find good region, small steps later to fine-tune.
```

---

## 6. Inference: Generating Text

### Autoregressive Generation

Generate one token at a time, feeding each output back as input:

```
Start: [BOS]
         │
Model predicts: p("a")=0.3, p("b")=0.5, p("c")=0.2
         │
Sample "b" → [BOS, "b"]
                  │
Model predicts: p("a")=0.1, p("b")=0.1, p("c")=0.8
                  │
Sample "c" → [BOS, "b", "c"]
                       │
... continue until BOS (end) or max length
```

### Temperature Sampling

**Temperature** controls randomness in generation:

```
logits = [2.0, 1.0, 0.5]

Temperature = 1.0 (normal):
    probs = softmax([2.0, 1.0, 0.5]) = [0.63, 0.23, 0.14]

Temperature = 0.5 (more deterministic):
    probs = softmax([4.0, 2.0, 1.0]) = [0.84, 0.11, 0.05]  ← More peaked

Temperature = 2.0 (more random):
    probs = softmax([1.0, 0.5, 0.25]) = [0.44, 0.30, 0.26]  ← Flatter
```

Lower temperature → more confident/repetitive  
Higher temperature → more creative/chaotic

---

## 7. Go-Specific Considerations

### Memory Layout Matters

**Bad (Python default)**: Slice of slices
```go
// [][]Value - each inner slice is separate allocation
// Cache misses when iterating!
weights := make([][]Value, rows)
for i := range weights {
    weights[i] = make([]Value, cols)
}
```

**Good**: Flat slice with manual indexing
```go
// []Value - contiguous memory, cache-friendly
// 2-3x faster for matrix operations
type Matrix struct {
    data []Value
    rows, cols int
}

func (m *Matrix) At(i, j int) *Value {
    return &m.data[i*m.cols + j]
}
```

### Avoid Allocations in Hot Paths

**Bad**: New slice every iteration
```go
for step := 0; step < 1000; step++ {
    grads := make([]float64, numParams)  // 1000 allocations!
}
```

**Good**: Pre-allocate and reuse
```go
grads := make([]float64, numParams)  // 1 allocation
for step := 0; step < 1000; step++ {
    // Zero using fast pattern
    for i := range grads {
        grads[i] = 0
    }
}
```

### Operator Overloading Alternative

Go doesn't have operator overloading. Instead, use method chaining:

```go
// Python: a + b * c
// Go:    a.Add(b.Mul(c))

result := a.Add(b.Mul(c)).Sub(d.Div(e))
```

### Reproducible Randomness

Always create your own RNG with a fixed seed:

```go
// Bad: Uses global state, non-reproducible
x := rand.Float64()

// Good: Reproducible across runs
rng := rand.New(rand.NewPCG(42, 42))
x := rng.Float64()
```

---

## Summary: The Complete Flow

```
┌────────────────────────────────────────────────────────────────────────┐
│                            TRAINING                                    │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  1. Load data: "alice", "bob", "carl", ...                            │
│                      │                                                 │
│  2. Tokenize: [BOS,0,11,8,2,4,BOS]                                    │
│                      │                                                 │
│  3. For each position:                                                 │
│     ├─ Embed token + position                                          │
│     ├─ Pass through transformer blocks                                 │
│     ├─ Compute logits (scores for each vocab token)                   │
│     └─ Softmax → probability of next token                            │
│                      │                                                 │
│  4. Cross-entropy loss: -log(P(correct_next_token))                   │
│                      │                                                 │
│  5. Backward pass: Compute gradients via chain rule                   │
│                      │                                                 │
│  6. Adam update: Adjust weights to reduce loss                        │
│                      │                                                 │
│  7. Repeat 1000 times                                                  │
│                                                                        │
├────────────────────────────────────────────────────────────────────────┤
│                            INFERENCE                                   │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  1. Start with [BOS]                                                   │
│                      │                                                 │
│  2. Forward pass → logits                                              │
│                      │                                                 │
│  3. Temperature scaling + softmax → probabilities                     │
│                      │                                                 │
│  4. Sample next token from distribution                               │
│                      │                                                 │
│  5. If token == BOS (end), stop                                       │
│     Else append token, go to step 2                                   │
│                      │                                                 │
│  6. Decode tokens → "harley" (new generated name!)                    │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

1. **Autograd is just recursion + chain rule**: Track operations, replay them backward
2. **Attention is learned similarity**: Q·K tells which tokens to look at
3. **Transformers are just attention + MLP + residuals**: Simple building blocks
4. **Adam is smart gradient descent**: Momentum + adaptive learning rates
5. **Temperature controls creativity**: Lower = deterministic, higher = random
6. **Go performance: contiguous memory wins**: Use flat slices, avoid allocations

The entire algorithm fits in 200 lines because it's fundamentally simple. The complexity in production systems comes from:
- Efficiency optimizations (batching, parallelism, GPU kernels)
- Scale (billions of parameters, trillions of tokens)
- Infrastructure (distributed training, checkpointing, serving)

But the *algorithm* is exactly what you see in microgpt. Everything else is just efficiency.
