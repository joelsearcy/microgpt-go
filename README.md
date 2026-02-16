# MicroGPT-Go

A Go port of [Andrej Karpathy's microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) - a complete GPT (Generative Pre-trained Transformer) implementation with **zero external dependencies**.

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
2. Train a 1-layer transformer for 1000 steps (~30 seconds)
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

This implementation uses several Go-specific optimizations:

| Optimization | Speedup | Technique |
|--------------|---------|-----------|
| FlatMatrix | 2-3x | Contiguous memory (cache-friendly) |
| Struct fields | 61x | Direct access vs map lookup |
| Pre-allocation | Varies | Avoid append() in hot loops |
| Sequential matmul | 3.5x | No goroutines for small ops |

See [PLAN.md](PLAN.md) for detailed performance analysis.

## Testing

Run the comprehensive test suite:

```bash
go test ./... -v
```

Tests include:
- Gradient correctness (analytical vs numerical)
- Basic operations (add, mul, exp, log, relu)
- Complex expression backpropagation
- Neuron simulation

## Project Structure

```
microgpt-go/
├── README.md                   # This file
├── PLAN.md                     # Implementation strategy
├── CONCEPTS.md                 # Educational guide
├── go.mod                      # Go module definition
├── cmd/
│   └── microgpt/
│       └── main.go             # Training & inference entry point
└── pkg/
    ├── autograd/
    │   ├── value.go            # Value type with operations
    │   └── value_test.go       # Gradient tests
    ├── model/
    │   ├── params.go           # Weight matrices
    │   ├── layers.go           # Linear, Softmax, RMSNorm
    │   └── gpt.go              # Transformer forward pass
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

Original work by [@karpathy](https://github.com/karpathy) - thank you for making deep learning accessible!

## License

MIT (same as the original microgpt.py)

---

*"The algorithm is simple. Everything else is just efficiency." -- Andrej Karpathy*
