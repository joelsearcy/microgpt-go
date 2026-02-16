# MicroGPT-Go Implementation Plan

A Go port of [Karpathy's microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) - a complete GPT implementation in ~200 lines of pure Python with zero dependencies.

## Project Philosophy

> "This file is the complete algorithm. Everything else is just efficiency."  
> — @karpathy

The goal is to produce an idiomatic Go implementation that:
- Has **zero external dependencies** (pure stdlib Go)
- Maintains **educational clarity** while being performant
- Demonstrates Go patterns for numerical computing
- Avoids common performance footguns

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         MicroGPT-Go                              │
├─────────────────────────────────────────────────────────────────┤
│  cmd/microgpt/main.go          Entry point & training loop     │
├─────────────────────────────────────────────────────────────────┤
│  pkg/autograd/                                                   │
│    ├── value.go                Value type + operators           │
│    └── backward.go             Topological sort + chain rule    │
├─────────────────────────────────────────────────────────────────┤
│  pkg/model/                                                      │
│    ├── params.go               Weight matrices & state_dict     │
│    ├── layers.go               linear, rmsnorm, softmax         │
│    └── gpt.go                  Transformer forward pass         │
├─────────────────────────────────────────────────────────────────┤
│  pkg/optim/                                                      │
│    └── adam.go                 Adam optimizer                   │
├─────────────────────────────────────────────────────────────────┤
│  pkg/tokenizer/                                                  │
│    └── char.go                 Character-level tokenizer        │
├─────────────────────────────────────────────────────────────────┤
│  pkg/data/                                                       │
│    └── dataset.go              File loading & batching          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Modules

### Module 1: Autograd Engine (`pkg/autograd/`)

**Python Reference (lines 29-72):**
```python
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
    def __add__(self, other): ...
    def backward(self): # topological sort + chain rule
```

**Go Design Decisions:**

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Type | Concrete struct, not interface | Zero overhead, simpler code |
| Operations | Method chaining: `v.Add(other).Mul(c)` | Idiomatic Go, readable |
| Graph storage | `[]*Value` children, `[]float64` localGrads | Cache-friendly |
| Topo sort | Recursive DFS with `map[*Value]struct{}` | Simple, efficient for scale |

**Go Implementation:**
```go
type Value struct {
    Data       float64
    Grad       float64
    children   []*Value   // computation graph edges
    localGrads []float64  // ∂self/∂child for each child
}

func (v *Value) Add(other *Value) *Value {
    return &Value{
        Data:       v.Data + other.Data,
        children:   []*Value{v, other},
        localGrads: []float64{1, 1},
    }
}

func (v *Value) Mul(other *Value) *Value {
    return &Value{
        Data:       v.Data * other.Data,
        children:   []*Value{v, other},
        localGrads: []float64{other.Data, v.Data}, // d(a*b)/da = b
    }
}

func (v *Value) Backward() {
    // Build topological order via DFS
    var topo []*Value
    visited := make(map[*Value]struct{})
    var buildTopo func(*Value)
    buildTopo = func(node *Value) {
        if _, seen := visited[node]; seen {
            return
        }
        visited[node] = struct{}{}
        for _, child := range node.children {
            buildTopo(child)
        }
        topo = append(topo, node)
    }
    buildTopo(v)
    
    // Backpropagate
    v.Grad = 1.0
    for i := len(topo) - 1; i >= 0; i-- {
        node := topo[i]
        for j, child := range node.children {
            child.Grad += node.localGrads[j] * node.Grad
        }
    }
}
```

**Operations to Implement:**
- `Add`, `Sub`, `Mul`, `Div`, `Neg`
- `Pow(exp float64)`
- `Log()`, `Exp()`
- `ReLU()`

**Performance Notes:**
- ⚠️ **Footgun**: Avoid creating Value wrappers for constants in hot loops. Use `AddScalar(float64)` variants.
- ⚠️ **Memory**: Each Value creates heap allocations. Consider sync.Pool for large graphs if profiling shows GC pressure.

---

### Module 2: Model Parameters (`pkg/model/params.go`)

**Python Reference (lines 74-90):**
```python
matrix = lambda nout, nin, std=0.08: [[Value(...) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), ...}
```

**Go Design Decisions:**

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Matrix type | FlatMatrix (1D slice) | **2-3x faster** than `[][]Value` due to cache locality |
| Storage | Struct with named fields | **61x faster** than map[string] for access |
| Param iteration | Pre-computed `allParams []*Value` slice | Efficient optimizer iteration |

**Go Implementation:**
```go
// FlatMatrix stores 2D matrix as contiguous 1D slice (row-major)
type FlatMatrix struct {
    Data         []*Value
    Rows, Cols   int
}

func NewMatrix(rows, cols int, std float64, rng *rand.Rand) *FlatMatrix {
    data := make([]*Value, rows*cols)
    for i := range data {
        data[i] = &Value{Data: rng.NormFloat64() * std}
    }
    return &FlatMatrix{Data: data, Rows: rows, Cols: cols}
}

func (m *FlatMatrix) At(row, col int) *Value {
    return m.Data[row*m.Cols+col]
}

func (m *FlatMatrix) Row(row int) []*Value {
    start := row * m.Cols
    return m.Data[start : start+m.Cols]
}

// TransformerBlock holds weights for one layer
type TransformerBlock struct {
    AttnWQ, AttnWK, AttnWV, AttnWO *FlatMatrix
    MlpFC1, MlpFC2                  *FlatMatrix
}

// ModelParams holds all trainable parameters
type ModelParams struct {
    Wte    *FlatMatrix // token embeddings [vocab_size, n_embd]
    Wpe    *FlatMatrix // position embeddings [block_size, n_embd]
    LmHead *FlatMatrix // output projection [vocab_size, n_embd]
    Blocks []TransformerBlock
    
    // Cached flat list for optimizer
    allParams []*Value
}

func (p *ModelParams) AllParams() []*Value {
    if p.allParams == nil {
        // Build once, cache for training loop
        p.allParams = append(p.allParams, p.Wte.Data...)
        p.allParams = append(p.allParams, p.Wpe.Data...)
        p.allParams = append(p.allParams, p.LmHead.Data...)
        for _, b := range p.Blocks {
            p.allParams = append(p.allParams, b.AttnWQ.Data...)
            // ... etc
        }
    }
    return p.allParams
}
```

---

### Module 3: Neural Network Layers (`pkg/model/layers.go`)

**Python Reference (lines 94-106):**
```python
def linear(x, w): return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
def softmax(logits): ...
def rmsnorm(x): ...
```

**Go Implementation:**
```go
// Linear performs matrix-vector multiplication: output = W @ x
// w is [out_dim, in_dim], x is [in_dim], returns [out_dim]
func Linear(x []*Value, w *FlatMatrix) []*Value {
    out := make([]*Value, w.Rows)
    for i := 0; i < w.Rows; i++ {
        sum := &Value{Data: 0}
        for j := 0; j < w.Cols; j++ {
            wij := w.At(i, j)
            sum = sum.Add(wij.Mul(x[j]))
        }
        out[i] = sum
    }
    return out
}

// Softmax with numerical stability (subtract max)
func Softmax(logits []*Value) []*Value {
    // Find max (data only, no grad needed)
    maxVal := logits[0].Data
    for _, v := range logits[1:] {
        if v.Data > maxVal {
            maxVal = v.Data
        }
    }
    
    // exp(logit - max)
    exps := make([]*Value, len(logits))
    for i, v := range logits {
        shifted := v.Sub(&Value{Data: maxVal})
        exps[i] = shifted.Exp()
    }
    
    // Sum and normalize
    total := exps[0]
    for _, e := range exps[1:] {
        total = total.Add(e)
    }
    
    probs := make([]*Value, len(logits))
    for i, e := range exps {
        probs[i] = e.Div(total)
    }
    return probs
}

// RMSNorm: x / sqrt(mean(x²) + eps)
func RMSNorm(x []*Value) []*Value {
    n := float64(len(x))
    
    // Compute mean of squares
    ms := &Value{Data: 0}
    for _, xi := range x {
        ms = ms.Add(xi.Mul(xi))
    }
    ms = ms.Div(&Value{Data: n})
    
    // scale = (ms + eps)^(-0.5)
    scale := ms.Add(&Value{Data: 1e-5}).Pow(-0.5)
    
    out := make([]*Value, len(x))
    for i, xi := range x {
        out[i] = xi.Mul(scale)
    }
    return out
}
```

**Performance Notes:**
- ⚠️ **Footgun**: Creating `&Value{Data: constant}` in loops allocates. Pre-allocate constants.
- ⚠️ **Loop order**: For large matrices, i-k-j order is 2.6x faster due to cache locality.

---

### Module 4: GPT Forward Pass (`pkg/model/gpt.go`)

**Python Reference (lines 108-144):**
```python
def gpt(token_id, pos_id, keys, values):
    # Embeddings + position
    # For each layer: attention + MLP with residual connections
    # Return logits
```

**Go Implementation:**
```go
type GPT struct {
    Params    *ModelParams
    NLayer    int
    NEmbd     int
    NHead     int
    HeadDim   int
    BlockSize int
}

// KVCache stores key-value pairs for autoregressive generation
type KVCache struct {
    Keys   [][]*Value // [layer][pos][embd]
    Values [][]*Value
}

func NewKVCache(nLayers int) *KVCache {
    return &KVCache{
        Keys:   make([][]*Value, nLayers),
        Values: make([][]*Value, nLayers),
    }
}

func (g *GPT) Forward(tokenID, posID int, cache *KVCache) []*Value {
    // Get embeddings
    tokEmb := g.Params.Wte.Row(tokenID)
    posEmb := g.Params.Wpe.Row(posID)
    
    // x = tok_emb + pos_emb
    x := make([]*Value, g.NEmbd)
    for i := 0; i < g.NEmbd; i++ {
        x[i] = tokEmb[i].Add(posEmb[i])
    }
    x = RMSNorm(x)
    
    // Transformer blocks
    for li := 0; li < g.NLayer; li++ {
        block := &g.Params.Blocks[li]
        xResidual := x
        
        // Self-attention
        x = RMSNorm(x)
        q := Linear(x, block.AttnWQ)
        k := Linear(x, block.AttnWK)
        v := Linear(x, block.AttnWV)
        
        cache.Keys[li] = append(cache.Keys[li], k)
        cache.Values[li] = append(cache.Values[li], v)
        
        // Multi-head attention
        xAttn := g.multiHeadAttention(q, cache.Keys[li], cache.Values[li])
        x = Linear(xAttn, block.AttnWO)
        
        // Residual connection
        for i := range x {
            x[i] = x[i].Add(xResidual[i])
        }
        
        // MLP block
        xResidual = x
        x = RMSNorm(x)
        x = Linear(x, block.MlpFC1)
        for i := range x {
            x[i] = x[i].ReLU()
        }
        x = Linear(x, block.MlpFC2)
        for i := range x {
            x[i] = x[i].Add(xResidual[i])
        }
    }
    
    return Linear(x, g.Params.LmHead)
}

func (g *GPT) multiHeadAttention(q []*Value, keys, vals [][]*Value) []*Value {
    xAttn := make([]*Value, g.NEmbd)
    
    for h := 0; h < g.NHead; h++ {
        hs := h * g.HeadDim
        
        // Extract head slices
        qH := q[hs : hs+g.HeadDim]
        
        // Compute attention scores
        attnLogits := make([]*Value, len(keys))
        scale := math.Sqrt(float64(g.HeadDim))
        for t, kt := range keys {
            kH := kt[hs : hs+g.HeadDim]
            dot := &Value{Data: 0}
            for j := 0; j < g.HeadDim; j++ {
                dot = dot.Add(qH[j].Mul(kH[j]))
            }
            attnLogits[t] = dot.Div(&Value{Data: scale})
        }
        
        // Softmax over attention scores
        attnWeights := Softmax(attnLogits)
        
        // Weighted sum of values
        for j := 0; j < g.HeadDim; j++ {
            sum := &Value{Data: 0}
            for t, vt := range vals {
                vH := vt[hs : hs+g.HeadDim]
                sum = sum.Add(attnWeights[t].Mul(vH[j]))
            }
            xAttn[hs+j] = sum
        }
    }
    return xAttn
}
```

---

### Module 5: Adam Optimizer (`pkg/optim/adam.go`)

**Python Reference (lines 146-182):**
```python
m[i] = beta1 * m[i] + (1 - beta1) * p.grad
v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
```

**Go Implementation:**
```go
type AdamOptimizer struct {
    LR      float64
    Beta1   float64
    Beta2   float64
    Epsilon float64
    
    m, v []float64 // moment buffers
    t    int       // step counter
}

func NewAdam(numParams int, lr, beta1, beta2, eps float64) *AdamOptimizer {
    return &AdamOptimizer{
        LR:      lr,
        Beta1:   beta1,
        Beta2:   beta2,
        Epsilon: eps,
        m:       make([]float64, numParams),
        v:       make([]float64, numParams),
    }
}

func (opt *AdamOptimizer) Step(params []*Value, lrDecay float64) {
    opt.t++
    bc1 := 1.0 - math.Pow(opt.Beta1, float64(opt.t))
    bc2 := 1.0 - math.Pow(opt.Beta2, float64(opt.t))
    lr := opt.LR * lrDecay
    
    for i, p := range params {
        g := p.Grad
        opt.m[i] = opt.Beta1*opt.m[i] + (1-opt.Beta1)*g
        opt.v[i] = opt.Beta2*opt.v[i] + (1-opt.Beta2)*g*g
        
        mHat := opt.m[i] / bc1
        vHat := opt.v[i] / bc2
        
        p.Data -= lr * mHat / (math.Sqrt(vHat) + opt.Epsilon)
        p.Grad = 0 // Zero grad for next iteration
    }
}
```

---

### Module 6: Tokenizer (`pkg/tokenizer/char.go`)

**Go Implementation:**
```go
type CharTokenizer struct {
    runeToID map[rune]int
    idToRune []rune
    BOS      int
}

func NewCharTokenizer(docs []string) *CharTokenizer {
    seen := make(map[rune]struct{})
    for _, doc := range docs {
        for _, r := range doc {
            seen[r] = struct{}{}
        }
    }
    
    runes := make([]rune, 0, len(seen))
    for r := range seen {
        runes = append(runes, r)
    }
    slices.Sort(runes)
    
    runeToID := make(map[rune]int, len(runes))
    for i, r := range runes {
        runeToID[r] = i
    }
    
    return &CharTokenizer{
        runeToID: runeToID,
        idToRune: runes,
        BOS:      len(runes),
    }
}

func (t *CharTokenizer) VocabSize() int { return len(t.idToRune) + 1 }

func (t *CharTokenizer) Encode(s string) []int {
    tokens := make([]int, 0, len(s)+2)
    tokens = append(tokens, t.BOS)
    for _, r := range s {
        tokens = append(tokens, t.runeToID[r])
    }
    return append(tokens, t.BOS)
}

func (t *CharTokenizer) Decode(tokens []int) string {
    var sb strings.Builder
    sb.Grow(len(tokens))
    for _, id := range tokens {
        if id != t.BOS && id < len(t.idToRune) {
            sb.WriteRune(t.idToRune[id])
        }
    }
    return sb.String()
}
```

---

### Module 7: Dataset Loading (`pkg/data/dataset.go`)

```go
func LoadDataset(path string) ([]string, error) {
    f, err := os.Open(path)
    if err != nil {
        return nil, err
    }
    defer f.Close()
    
    var docs []string
    scanner := bufio.NewScanner(f)
    for scanner.Scan() {
        if line := strings.TrimSpace(scanner.Text()); line != "" {
            docs = append(docs, line)
        }
    }
    return docs, scanner.Err()
}

func Shuffle(docs []string, seed int64) {
    rng := rand.New(rand.NewPCG(uint64(seed), 0))
    rng.Shuffle(len(docs), func(i, j int) {
        docs[i], docs[j] = docs[j], docs[i]
    })
}
```

---

## Implementation Phases

### Phase 1: Core Foundation
1. `pkg/autograd/value.go` - Value type and operations
2. `pkg/autograd/backward.go` - Backpropagation
3. Unit tests for gradient correctness

### Phase 2: Model Architecture  
4. `pkg/model/params.go` - FlatMatrix and ModelParams
5. `pkg/model/layers.go` - linear, softmax, rmsnorm
6. `pkg/model/gpt.go` - Transformer forward pass

### Phase 3: Training Infrastructure
7. `pkg/optim/adam.go` - Adam optimizer
8. `pkg/tokenizer/char.go` - Character tokenizer
9. `pkg/data/dataset.go` - File loading

### Phase 4: Integration
10. `cmd/microgpt/main.go` - Training loop
11. Inference with temperature sampling
12. End-to-end testing

---

## Performance Footguns to Avoid

| Footgun | Problem | Solution |
|---------|---------|----------|
| `[][]Value` matrices | Poor cache locality | Use FlatMatrix (1D slice) |
| `map[string]` for params | 61x slower lookup | Use struct with named fields |
| Creating Value{} for constants | Heap allocation per op | Pre-allocate constant pool |
| Parallelizing small matmuls | Goroutine overhead > gain | Keep sequential; parallelize at batch level |
| `append()` in hot loops | Unpredictable reallocs | Pre-allocate with `make([]T, 0, cap)` |
| Global rand | Not reproducible | Use `rand.New(rand.NewPCG(seed, 0))` |

---

## Hyperparameters (matching Python)

```go
const (
    NLayer    = 1
    NEmbd     = 16
    BlockSize = 16
    NHead     = 4
    
    LearningRate = 0.01
    Beta1        = 0.85
    Beta2        = 0.99
    EpsAdam      = 1e-8
    
    NumSteps    = 1000
    Temperature = 0.5
)
```

---

## Testing Strategy

1. **Gradient checking**: Numerical vs analytical gradients
2. **Layer tests**: Known input → expected output
3. **Integration test**: Train on small corpus, ensure loss decreases
4. **Determinism test**: Same seed → identical outputs

---

## File Structure

```
microgpt-go/
├── PLAN.md                 # This document
├── go.mod
├── cmd/
│   └── microgpt/
│       └── main.go         # Entry point
├── pkg/
│   ├── autograd/
│   │   ├── value.go
│   │   ├── value_test.go
│   │   └── backward.go
│   ├── model/
│   │   ├── params.go
│   │   ├── layers.go
│   │   ├── layers_test.go
│   │   └── gpt.go
│   ├── optim/
│   │   └── adam.go
│   ├── tokenizer/
│   │   └── char.go
│   └── data/
│       └── dataset.go
└── input.txt               # Training data (auto-downloaded)
```
