# MicroGPT-Go

A Go port of [Andrej Karpathy's microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) - a complete GPT (Generative Pre-trained Transformer) implementation with **zero external dependencies**.

Based on the concepts from Karpathy's blog post: [*"The most atomic GPT"*](https://karpathy.github.io/2026/02/12/microgpt/)

> "This file is the complete algorithm. Everything else is just efficiency."  
> — @karpathy

## Features

- ✅ **Pure Go stdlib** - No external dependencies
- ✅ **Complete autograd engine** - Automatic differentiation from scratch
- ✅ **GPT transformer** - Multi-head attention, residual connections, RMSNorm
- ✅ **Adam optimizer** - With learning rate decay and bias correction
- ✅ **Character-level tokenizer** - Simple and educational
- ✅ **~1400 lines** - Clean, readable, idiomatic Go

## Quick Start

```bash
# Clone the repository
git clone https://github.com/joelsearcy/microgpt-go
cd microgpt-go

# Run training and inference
go run ./cmd/microgpt
```

The model will:
1. Download the names dataset (32k unique names)
2. Train a 1-layer transformer for 1000 steps (~4 seconds)
3. Generate 20 new, hallucinated names

## Example Output

```
num docs: 32033
vocab size: 27
num params: 4192
step 1000 / 1000 | loss 2.4672
--- inference (new, hallucinated names) ---
sample  1: jaren
sample  2: chenily
sample  3: zarin
sample  4: elan
sample  5: ranidh
sample  6: jona
sample  7: shan
sample  8: caldi
sample  9: manana
sample 10: gavy
```

## Architecture

```
┌─────────────────────────────────────────────┐
│              MicroGPT-Go                    │
├─────────────────────────────────────────────┤
│  pkg/autograd/    Automatic differentiation │
│  pkg/model/       Transformer architecture  │
│  pkg/optim/       Adam optimizer            │
│  pkg/tokenizer/   Character-level tokenizer │
│  pkg/data/        Dataset loading           │
│  cmd/microgpt/    Training loop & inference │
└─────────────────────────────────────────────┘
```

### Model Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_layer` | 1 | Number of transformer layers |
| `n_embd` | 16 | Embedding dimension |
| `block_size` | 16 | Maximum sequence length |
| `n_head` | 4 | Number of attention heads |
| **Total params** | **4,192** | Tiny but functional! |

## How It Works

### 1. Autograd Engine

Custom automatic differentiation system that builds a computation graph and applies backpropagation via the chain rule:

```go
// Example: gradient flows backward through operations
loss := a.Mul(b).Add(c).Exp()
loss.Backward()  // Computes gradients for a, b, c
```

### 2. Transformer Architecture

Each layer contains:
- **Multi-head attention** - Learn relationships between tokens
- **RMSNorm** - Normalize activations for training stability  
- **MLP (2-layer)** - Non-linear transformations with ReLU
- **Residual connections** - Enable gradient flow

### 3. Training Loop

```go
for step := 0; step < 1000; step++ {
    // Forward: predict next character
    logits := model.Forward(token, position, cache)
    
    // Loss: cross-entropy
    loss := -log(P(correct_token))
    
    // Backward: compute gradients
    loss.Backward()
    
    // Update: Adam optimizer with LR decay
    optimizer.Step(params, lrDecay)
}
```

### 4. Inference

Autoregressive generation with temperature sampling:

```go
for pos := 0; pos < maxLength; pos++ {
    logits := model.Forward(token, pos, cache)
    probs := Softmax(logits / temperature)
    token = Sample(probs)  // Stochastic sampling
}
```

## Performance Optimizations

This implementation achieves a **7x speedup** over a naive port through careful optimization:

| Metric | Naive | Optimized | Improvement |
|--------|-------|-----------|-------------|
| Training time | ~30s | **~4.3s** | **85% faster** |
| Memory/step | 13.7 MB | 2.2 MB | 84% reduction |
| Allocs/step | 191,790 | 19,346 | 90% reduction |

### Key Optimizations

| Technique | Impact | Description |
|-----------|--------|-------------|
| **FlatMatrix layout** | 2-3x faster | Contiguous memory for cache efficiency |
| **Struct vs map** | 61x faster | Direct field access vs map lookup |
| **Fused DotProduct** | 97% fewer allocs | Single graph node for vector dot products |
| **Fused Softmax** | 90% fewer allocs | Single operation instead of N intermediate Values |
| **Pre-allocated buffers** | Eliminates reallocs | Capacity hints for KV cache and backward pass |
| **Sequential execution** | 3.5x faster | No goroutines for small operations |

See [PLAN.md](PLAN.md) for detailed performance analysis and [pkg/model/benchmark_test.go](pkg/model/benchmark_test.go) for benchmarks.

## Testing & Benchmarking

### Run Tests

```bash
# Run all tests
go test ./... -v

# Run with coverage
go test ./... -cover
```

Tests include:
- Gradient correctness (analytical vs numerical)
- Basic operations (add, mul, exp, log, relu)
- Complex expression backpropagation
- Fused operations (DotProduct, FusedSoftmax)
- Neuron simulation

### Run Benchmarks

```bash
# Benchmark core operations
go test -bench=. -benchmem ./pkg/model/

# Profile CPU usage
go run ./cmd/profile
```

Benchmarks measure:
- Linear layer performance
- Softmax computation
- GPT forward pass
- Full training step with backward pass
- Multi-head attention

## Project Structure

```
microgpt-go/
├── README.md                   # This file
├── PLAN.md                     # Implementation strategy
├── CONCEPTS.md                 # Educational guide
├── go.mod                      # Go module definition
├── cmd/
│   ├── microgpt/
│   │   └── main.go             # Training & inference entry point
│   └── profile/
│       └── main.go             # CPU/memory profiling harness
└── pkg/
    ├── autograd/
    │   ├── value.go            # Value type with operations
    │   └── value_test.go       # Gradient tests (23 tests)
    ├── model/
    │   ├── params.go           # Weight matrices (FlatMatrix)
    │   ├── layers.go           # Linear, Softmax, RMSNorm
    │   ├── gpt.go              # Transformer forward pass
    │   └── benchmark_test.go   # Performance benchmarks
    ├── optim/
    │   └── adam.go             # Adam optimizer
    ├── tokenizer/
    │   └── char.go             # Character tokenizer
    └── data/
        └── dataset.go          # File loading & shuffling
```

## Educational Resources

- **[CONCEPTS.md](CONCEPTS.md)** - Learn the fundamentals: autograd, attention, transformers, optimization
- **[PLAN.md](PLAN.md)** - Implementation details and design decisions

## Credits

This is a direct port of [Andrej Karpathy's microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95), adapted to idiomatic Go with performance optimizations.

**Related resources:**
- [Blog post: "The most atomic GPT"](https://karpathy.github.io/2026/02/12/microgpt/) - Karpathy's explanation of the philosophy and design
- [Original Python gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) - The reference implementation

Original work by [@karpathy](https://github.com/karpathy) - thank you for making deep learning accessible!

## License

MIT (same as the original microgpt.py)

---

*"The algorithm is simple. Everything else is just efficiency." -- Andrej Karpathy*
