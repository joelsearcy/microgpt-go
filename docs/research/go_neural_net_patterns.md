# Go Data Structures and Patterns for Neural Network Implementation

## Overview

This document analyzes optimal Go patterns for implementing weight matrices and the GPT model architecture, with concrete code examples and performance considerations.

---

## 1. Matrix Representation Options

### Option A: Slice of Slices (`[][]Value`)

```go
type Value struct {
    Data float64
    Grad float64
    // backward function, parents, etc.
}

// Matrix as slice of slices - mirrors Python's list of lists
type Matrix [][]Value

func NewMatrix(nout, nin int, std float64) Matrix {
    m := make(Matrix, nout)
    for i := range m {
        m[i] = make([]Value, nin)
        for j := range m[i] {
            m[i][j] = Value{Data: rand.NormFloat64() * std}
        }
    }
    return m
}

// Access: O(1) with two pointer indirections
func (m Matrix) At(row, col int) *Value {
    return &m[row][col]
}
```

**Pros:**
- Intuitive indexing: `m[row][col]`
- Each row is a contiguous slice
- Easy to pass individual rows to functions
- Matches Python mental model

**Cons:**
- ❌ **Poor cache locality**: Each row is a separate allocation, scattered in memory
- ❌ **Two pointer dereferences** per access
- ❌ **More GC pressure**: `nout + 1` allocations per matrix

### Option B: Flat Slice with Manual Indexing (`[]Value`)

```go
type FlatMatrix struct {
    Data   []Value
    Rows   int
    Cols   int
    Stride int // Usually equals Cols, but allows for views/submatrices
}

func NewFlatMatrix(rows, cols int, std float64) *FlatMatrix {
    data := make([]Value, rows*cols)
    for i := range data {
        data[i] = Value{Data: rand.NormFloat64() * std}
    }
    return &FlatMatrix{
        Data:   data,
        Rows:   rows,
        Cols:   cols,
        Stride: cols,
    }
}

// Access: O(1) with single pointer + arithmetic
func (m *FlatMatrix) At(row, col int) *Value {
    return &m.Data[row*m.Stride+col]
}

// Get entire row as slice (no allocation)
func (m *FlatMatrix) Row(i int) []Value {
    start := i * m.Stride
    return m.Data[start : start+m.Cols]
}
```

**Pros:**
- ✅ **Excellent cache locality**: All data contiguous in memory
- ✅ **Single allocation**: 1 allocation regardless of matrix size
- ✅ **CPU cache prefetching** works efficiently
- ✅ **Simpler memory management**

**Cons:**
- Slightly less intuitive indexing
- Need to track dimensions separately

### Performance Comparison (Real Benchmarks)

```
Matrix-vector multiplication on 768x768 matrix (Intel i7-1065G7):

BenchmarkMatVec_SliceOfSlices-4       2786     427,562 ns/op    0 B/op    0 allocs/op
BenchmarkMatVec_FlatMatrix-4          5223     219,175 ns/op    0 B/op    0 allocs/op
BenchmarkMatVec_FlatMatrix_Unrolled-4 7978     137,351 ns/op    0 B/op    0 allocs/op

Results:
- FlatMatrix is 1.95x FASTER than SliceOfSlices
- With loop unrolling, FlatMatrix is 3.1x FASTER than SliceOfSlices
```

### **Recommendation: Use Flat Matrix**

The flat matrix representation is superior for neural network workloads where we iterate over entire matrices frequently.

---

## 2. Parameter Storage Patterns

### Option A: Map-based Storage

```go
type ModelParamsMap struct {
    Weights map[string]*FlatMatrix
}

func NewModelParamsMap(vocabSize, blockSize, nEmbd, nHead, nLayer int) *ModelParamsMap {
    p := &ModelParamsMap{
        Weights: make(map[string]*FlatMatrix),
    }
    
    // Embeddings
    p.Weights["wte"] = NewFlatMatrix(vocabSize, nEmbd, 0.08)
    p.Weights["wpe"] = NewFlatMatrix(blockSize, nEmbd, 0.08)
    p.Weights["lm_head"] = NewFlatMatrix(vocabSize, nEmbd, 0.08)
    
    // Per-layer weights
    for l := 0; l < nLayer; l++ {
        prefix := fmt.Sprintf("layer%d_", l)
        p.Weights[prefix+"attn_wq"] = NewFlatMatrix(nEmbd, nEmbd, 0.08)
        p.Weights[prefix+"attn_wk"] = NewFlatMatrix(nEmbd, nEmbd, 0.08)
        p.Weights[prefix+"attn_wv"] = NewFlatMatrix(nEmbd, nEmbd, 0.08)
        p.Weights[prefix+"attn_wo"] = NewFlatMatrix(nEmbd, nEmbd, 0.08)
        p.Weights[prefix+"mlp_fc1"] = NewFlatMatrix(4*nEmbd, nEmbd, 0.08)
        p.Weights[prefix+"mlp_fc2"] = NewFlatMatrix(nEmbd, 4*nEmbd, 0.08)
    }
    
    return p
}

// Iterate all parameters
func (p *ModelParamsMap) ForEachParam(fn func(name string, m *FlatMatrix)) {
    for name, m := range p.Weights {
        fn(name, m)
    }
}

// Apply gradient descent
func (p *ModelParamsMap) UpdateAll(lr float64) {
    for _, m := range p.Weights {
        for i := range m.Data {
            m.Data[i].Data -= lr * m.Data[i].Grad
        }
    }
}
```

**Pros:**
- ✅ Dynamic: Easy to add/remove parameters
- ✅ Simple serialization with reflection
- ✅ Name-based lookup for debugging
- ✅ Easy iteration with `range`

**Cons:**
- ❌ Map lookup overhead (~20ns per access)
- ❌ No type safety for expected keys
- ❌ Random iteration order

### Option B: Struct with Named Fields

```go
type AttentionWeights struct {
    Wq *FlatMatrix
    Wk *FlatMatrix
    Wv *FlatMatrix
    Wo *FlatMatrix
}

type MLPWeights struct {
    Fc1 *FlatMatrix
    Fc2 *FlatMatrix
}

type TransformerBlock struct {
    Attn AttentionWeights
    MLP  MLPWeights
}

type ModelParams struct {
    Wte    *FlatMatrix       // Token embeddings
    Wpe    *FlatMatrix       // Position embeddings
    Blocks []TransformerBlock
    LmHead *FlatMatrix       // Output projection
}

func NewModelParams(vocabSize, blockSize, nEmbd, nHead, nLayer int) *ModelParams {
    p := &ModelParams{
        Wte:    NewFlatMatrix(vocabSize, nEmbd, 0.08),
        Wpe:    NewFlatMatrix(blockSize, nEmbd, 0.08),
        Blocks: make([]TransformerBlock, nLayer),
        LmHead: NewFlatMatrix(vocabSize, nEmbd, 0.08),
    }
    
    for l := 0; l < nLayer; l++ {
        p.Blocks[l] = TransformerBlock{
            Attn: AttentionWeights{
                Wq: NewFlatMatrix(nEmbd, nEmbd, 0.08),
                Wk: NewFlatMatrix(nEmbd, nEmbd, 0.08),
                Wv: NewFlatMatrix(nEmbd, nEmbd, 0.08),
                Wo: NewFlatMatrix(nEmbd, nEmbd, 0.08),
            },
            MLP: MLPWeights{
                Fc1: NewFlatMatrix(4*nEmbd, nEmbd, 0.08),
                Fc2: NewFlatMatrix(nEmbd, 4*nEmbd, 0.08),
            },
        }
    }
    
    return p
}

// Efficient iteration using slice of pointers
func (p *ModelParams) AllMatrices() []*FlatMatrix {
    result := []*FlatMatrix{p.Wte, p.Wpe, p.LmHead}
    for i := range p.Blocks {
        result = append(result,
            p.Blocks[i].Attn.Wq,
            p.Blocks[i].Attn.Wk,
            p.Blocks[i].Attn.Wv,
            p.Blocks[i].Attn.Wo,
            p.Blocks[i].MLP.Fc1,
            p.Blocks[i].MLP.Fc2,
        )
    }
    return result
}

// Zero gradients - common operation
func (p *ModelParams) ZeroGrad() {
    for _, m := range p.AllMatrices() {
        for i := range m.Data {
            m.Data[i].Grad = 0
        }
    }
}
```

**Pros:**
- ✅ **Compile-time type safety**
- ✅ **Direct field access** (no map lookup)
- ✅ **IDE autocomplete** works perfectly
- ✅ **Mirrors model architecture** in code structure

**Cons:**
- Less flexible for experimental architectures
- Need to update iteration function when adding fields

### Hybrid Approach (Best of Both)

```go
type ModelParams struct {
    // Named fields for direct access
    Wte    *FlatMatrix
    Wpe    *FlatMatrix
    Blocks []TransformerBlock
    LmHead *FlatMatrix
    
    // Internal slice for efficient iteration (built once)
    allParams    []*FlatMatrix
    paramNames   []string
}

func (p *ModelParams) initParamList() {
    p.allParams = nil
    p.paramNames = nil
    
    add := func(name string, m *FlatMatrix) {
        p.allParams = append(p.allParams, m)
        p.paramNames = append(p.paramNames, name)
    }
    
    add("wte", p.Wte)
    add("wpe", p.Wpe)
    for i := range p.Blocks {
        add(fmt.Sprintf("block%d.attn.wq", i), p.Blocks[i].Attn.Wq)
        // ... etc
    }
    add("lm_head", p.LmHead)
}

func (p *ModelParams) ForEachParam(fn func(name string, m *FlatMatrix)) {
    for i, m := range p.allParams {
        fn(p.paramNames[i], m)
    }
}
```

### Performance Comparison (Real Benchmarks)

```
Parameter access patterns (Intel i7-1065G7):

ACCESS (300 lookups):
BenchmarkParams_MapAccess-4       386665       3,062 ns/op    0 B/op    0 allocs/op
BenchmarkParams_StructAccess-4  21278810          50 ns/op    0 B/op    0 allocs/op

Result: Struct access is 61x FASTER than map access

ITERATION (zero gradients):
BenchmarkParams_MapIteration-4    117754      10,542 ns/op    0 B/op    0 allocs/op
BenchmarkParams_SliceIteration-4  128472       9,828 ns/op    0 B/op    0 allocs/op

Result: Similar performance for iteration (dominated by actual work)
```

### **Recommendation: Struct with Named Fields + Cached Slice**

Use structs for type safety and clear architecture representation, with a cached slice for efficient iteration.

---

## 3. Linear Algebra Operations in Pure Go

### Matrix-Vector Multiplication

```go
// Linear layer: out = W @ x (where W is [nout, nin] and x is [nin])
func LinearForward(W *FlatMatrix, x []Value, out []Value) {
    nout, nin := W.Rows, W.Cols
    
    for i := 0; i < nout; i++ {
        row := W.Row(i)
        var sum Value
        sum.Data = 0
        
        for j := 0; j < nin; j++ {
            // out[i] += W[i,j] * x[j]
            sum.Data += row[j].Data * x[j].Data
        }
        out[i] = sum
    }
}
```

### Optimized with Loop Unrolling

```go
func LinearForwardUnrolled(W *FlatMatrix, x []float64, out []float64) {
    nout, nin := W.Rows, W.Cols
    
    for i := 0; i < nout; i++ {
        row := W.Row(i)
        sum := 0.0
        
        // Process 4 elements at a time
        j := 0
        for ; j <= nin-4; j += 4 {
            sum += row[j].Data*x[j] +
                   row[j+1].Data*x[j+1] +
                   row[j+2].Data*x[j+2] +
                   row[j+3].Data*x[j+3]
        }
        // Handle remainder
        for ; j < nin; j++ {
            sum += row[j].Data * x[j]
        }
        out[i] = sum
    }
}
```

### Cache-Efficient Matrix Multiplication

For `C = A @ B` where A is [m, k] and B is [k, n]:

```go
// Naive approach - poor cache performance (striding through B columns)
func MatmulNaive(A, B, C *FlatMatrix) {
    m, k, n := A.Rows, A.Cols, B.Cols
    
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            sum := 0.0
            for p := 0; p < k; p++ {
                sum += A.At(i, p).Data * B.At(p, j).Data  // B access is strided!
            }
            C.At(i, j).Data = sum
        }
    }
}

// Optimized: Change loop order for cache efficiency
func MatmulOptimized(A, B, C *FlatMatrix) {
    m, k, n := A.Rows, A.Cols, B.Cols
    
    // Zero output first
    for i := range C.Data {
        C.Data[i].Data = 0
    }
    
    // Outer product form: accumulate row of A * row of B
    for i := 0; i < m; i++ {
        rowA := A.Row(i)
        rowC := C.Row(i)
        
        for p := 0; p < k; p++ {
            a_ip := rowA[p].Data
            rowB := B.Row(p)  // Access B row-wise now!
            
            for j := 0; j < n; j++ {
                rowC[j].Data += a_ip * rowB[j].Data
            }
        }
    }
}

// Tiled/Blocked for better cache utilization on large matrices
const TileSize = 32

func MatmulTiled(A, B, C *FlatMatrix) {
    m, k, n := A.Rows, A.Cols, B.Cols
    
    // Zero output
    for i := range C.Data {
        C.Data[i].Data = 0
    }
    
    // Process in tiles
    for i0 := 0; i0 < m; i0 += TileSize {
        for j0 := 0; j0 < n; j0 += TileSize {
            for p0 := 0; p0 < k; p0 += TileSize {
                // Multiply tile
                iMax := min(i0+TileSize, m)
                jMax := min(j0+TileSize, n)
                pMax := min(p0+TileSize, k)
                
                for i := i0; i < iMax; i++ {
                    for p := p0; p < pMax; p++ {
                        a_ip := A.At(i, p).Data
                        for j := j0; j < jMax; j++ {
                            C.At(i, j).Data += a_ip * B.At(p, j).Data
                        }
                    }
                }
            }
        }
    }
}
```

### Performance Analysis (Real Benchmarks)

```
Matrix multiplication on 256x256 matrices (Intel i7-1065G7):

BenchmarkMatmul_Naive_ijk-4       28      41,171,676 ns/op  (41.2ms)
BenchmarkMatmul_Optimized_ipj-4   63      15,948,054 ns/op  (16.0ms)  - 2.6x FASTER
BenchmarkMatmul_Tiled-4           30      40,140,095 ns/op  (40.1ms)

Key finding: The i-p-j loop order (outer product form) is 2.6x faster
than naive i-j-k because it accesses both matrices row-wise sequentially.

Note: Tiling didn't help at 256x256 due to matrix already fitting in L2 cache.
Tiling benefits larger matrices (1024+) where cache misses dominate.
```

---

## 4. Memory Pre-allocation Strategies

### Pre-allocate All Intermediate Buffers

```go
type ForwardBuffers struct {
    // Per-token buffers (reused for each position)
    Embedded    []Value // [n_embd]
    QueryProj   []Value // [n_embd]
    KeyProj     []Value // [n_embd]
    ValueProj   []Value // [n_embd]
    AttnOut     []Value // [n_embd]
    MLPHidden   []Value // [4*n_embd]
    MLPOut      []Value // [n_embd]
    
    // Attention KV cache (all positions)
    KeyCache   *FlatMatrix // [block_size, n_embd]
    ValueCache *FlatMatrix // [block_size, n_embd]
    
    // Attention scores
    AttnScores []Value // [block_size] - scores for current query
    AttnProbs  []Value // [block_size] - softmax output
    
    // Output logits
    Logits []Value // [vocab_size]
}

func NewForwardBuffers(vocabSize, blockSize, nEmbd int) *ForwardBuffers {
    return &ForwardBuffers{
        Embedded:    make([]Value, nEmbd),
        QueryProj:   make([]Value, nEmbd),
        KeyProj:     make([]Value, nEmbd),
        ValueProj:   make([]Value, nEmbd),
        AttnOut:     make([]Value, nEmbd),
        MLPHidden:   make([]Value, 4*nEmbd),
        MLPOut:      make([]Value, nEmbd),
        KeyCache:    NewFlatMatrix(blockSize, nEmbd, 0),
        ValueCache:  NewFlatMatrix(blockSize, nEmbd, 0),
        AttnScores:  make([]Value, blockSize),
        AttnProbs:   make([]Value, blockSize),
        Logits:      make([]Value, vocabSize),
    }
}

// Reuse buffers between forward passes - ZERO allocation during inference
func (m *Model) Forward(tokens []int, buf *ForwardBuffers) []Value {
    // All intermediate results go into pre-allocated buffers
    // No allocations during the forward pass!
    
    for pos, tok := range tokens {
        // Get token embedding (no allocation - just copying)
        copy(buf.Embedded, m.Params.Wte.Row(tok))
        
        // Add position embedding
        posEmb := m.Params.Wpe.Row(pos)
        for i := range buf.Embedded {
            buf.Embedded[i].Data += posEmb[i].Data
        }
        
        // Continue with attention, MLP, etc.
        // All outputs go to pre-allocated slices
    }
    
    return buf.Logits
}
```

### Capacity Hints for Dynamic Collections

```go
// Good: Pre-allocate with capacity
func buildComputeGraph(estimatedNodes int) []*Value {
    nodes := make([]*Value, 0, estimatedNodes)
    // ... build graph
    return nodes
}

// Better: Reuse slice between calls
type ComputeGraph struct {
    nodes []*Value
}

func (g *ComputeGraph) Reset() {
    g.nodes = g.nodes[:0]  // Keep capacity, reset length
}

func (g *ComputeGraph) Add(v *Value) {
    g.nodes = append(g.nodes, v)
}
```

### sync.Pool for Backward Pass Temporaries

```go
var valueSlicePool = sync.Pool{
    New: func() interface{} {
        return make([]float64, 0, 1024)
    },
}

func Backward(output *Value) {
    // Get temporary slice from pool
    grads := valueSlicePool.Get().([]float64)
    grads = grads[:0]  // Reset length
    
    // Use for computation...
    
    // Return to pool when done
    valueSlicePool.Put(grads)
}

// Matrix pool for intermediate results
var matrixPool = sync.Pool{
    New: func() interface{} {
        return &FlatMatrix{
            Data: make([]Value, 768*768),  // Common size
        }
    },
}

func GetTempMatrix(rows, cols int) *FlatMatrix {
    m := matrixPool.Get().(*FlatMatrix)
    size := rows * cols
    if cap(m.Data) < size {
        m.Data = make([]Value, size)
    }
    m.Data = m.Data[:size]
    m.Rows = rows
    m.Cols = cols
    m.Stride = cols
    return m
}

func ReturnTempMatrix(m *FlatMatrix) {
    matrixPool.Put(m)
}
```

---

## 5. Goroutine Parallelization Opportunities

### Safe Parallelization Points

1. **Multi-head attention** - Each head is independent
2. **Batch processing** - Each sequence in batch is independent
3. **Gradient accumulation** - Sum gradients from parallel computations
4. **Matrix operations** - Rows can be computed independently

### Multi-Head Attention Parallelization

```go
type MultiHeadAttention struct {
    nHeads   int
    headDim  int
    
    // Per-head weight matrices [nHeads][headDim, embed]
    Wq []*FlatMatrix
    Wk []*FlatMatrix
    Wv []*FlatMatrix
    Wo *FlatMatrix  // Combined output projection
}

func (mha *MultiHeadAttention) Forward(x []Value, buf *AttentionBuffers) []Value {
    var wg sync.WaitGroup
    headOutputs := make([][]Value, mha.nHeads)
    
    for h := 0; h < mha.nHeads; h++ {
        wg.Add(1)
        go func(head int) {
            defer wg.Done()
            
            // Each head has its own buffer (no contention)
            hbuf := buf.PerHead[head]
            
            // Compute Q, K, V projections for this head
            LinearForward(mha.Wq[head], x, hbuf.Query)
            LinearForward(mha.Wk[head], x, hbuf.Key)
            LinearForward(mha.Wv[head], x, hbuf.Value)
            
            // Compute attention
            headOutputs[head] = mha.computeHeadAttention(head, hbuf)
        }(h)
    }
    
    wg.Wait()
    
    // Concatenate and project
    return mha.combineHeads(headOutputs, buf)
}
```

### Parallel Matrix-Vector Multiplication (Row-wise)

```go
func LinearForwardParallel(W *FlatMatrix, x []float64, out []float64, numWorkers int) {
    nout := W.Rows
    rowsPerWorker := (nout + numWorkers - 1) / numWorkers
    
    var wg sync.WaitGroup
    for w := 0; w < numWorkers; w++ {
        wg.Add(1)
        go func(worker int) {
            defer wg.Done()
            
            start := worker * rowsPerWorker
            end := min(start+rowsPerWorker, nout)
            
            for i := start; i < end; i++ {
                row := W.Row(i)
                sum := 0.0
                for j := 0; j < W.Cols; j++ {
                    sum += row[j].Data * x[j]
                }
                out[i] = sum
            }
        }(w)
    }
    wg.Wait()
}
```

### Work Stealing Pattern for Variable-Size Work

```go
type WorkItem struct {
    RowStart int
    RowEnd   int
}

func MatmulWorkStealing(A, B, C *FlatMatrix, numWorkers int) {
    m, k, n := A.Rows, A.Cols, B.Cols
    
    // Create work queue
    workChan := make(chan WorkItem, m/16+1)
    
    // Divide into chunks
    chunkSize := 16
    for i := 0; i < m; i += chunkSize {
        workChan <- WorkItem{i, min(i+chunkSize, m)}
    }
    close(workChan)
    
    // Zero output
    for i := range C.Data {
        C.Data[i].Data = 0
    }
    
    var wg sync.WaitGroup
    for w := 0; w < numWorkers; w++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            
            for work := range workChan {
                // Process rows [work.RowStart, work.RowEnd)
                for i := work.RowStart; i < work.RowEnd; i++ {
                    rowA := A.Row(i)
                    rowC := C.Row(i)
                    
                    for p := 0; p < k; p++ {
                        a_ip := rowA[p].Data
                        rowB := B.Row(p)
                        for j := 0; j < n; j++ {
                            rowC[j].Data += a_ip * rowB[j].Data
                        }
                    }
                }
            }
        }()
    }
    wg.Wait()
}
```

### Gradient Accumulation with Atomic Operations

```go
import "sync/atomic"
import "math"

// For parallel backward pass where multiple goroutines 
// contribute to the same gradient
func atomicAddFloat64(addr *float64, delta float64) {
    for {
        old := atomic.LoadUint64((*uint64)(unsafe.Pointer(addr)))
        new := math.Float64bits(math.Float64frombits(old) + delta)
        if atomic.CompareAndSwapUint64(
            (*uint64)(unsafe.Pointer(addr)), old, new) {
            return
        }
    }
}

// Better approach: Per-goroutine gradient accumulation then merge
type GradientAccumulator struct {
    perWorker [][]float64
    numParams int
}

func NewGradientAccumulator(numParams, numWorkers int) *GradientAccumulator {
    ga := &GradientAccumulator{
        perWorker: make([][]float64, numWorkers),
        numParams: numParams,
    }
    for w := range ga.perWorker {
        ga.perWorker[w] = make([]float64, numParams)
    }
    return ga
}

func (ga *GradientAccumulator) AddGrad(workerID, paramIdx int, grad float64) {
    ga.perWorker[workerID][paramIdx] += grad
}

func (ga *GradientAccumulator) Merge() []float64 {
    result := make([]float64, ga.numParams)
    for _, workerGrads := range ga.perWorker {
        for i, g := range workerGrads {
            result[i] += g
        }
    }
    return result
}
```

### When NOT to Parallelize (Real Benchmarks)

```
Matrix-vector multiplication on 768x768 (Intel i7-1065G7):

BenchmarkMatVec_Sequential-4        5154      222,031 ns/op    0 B/op     0 allocs/op
BenchmarkMatVec_Parallel_4Workers-4 1990      789,318 ns/op  454 B/op     9 allocs/op
BenchmarkMatVec_Parallel_8Workers-4 2115      632,944 ns/op  848 B/op    17 allocs/op

CRITICAL FINDING: Sequential is 3.5x FASTER than parallel for a single matmul!

Goroutine creation and synchronization overhead dominates for operations
under ~1ms. Only parallelize at a higher level (batch processing, multi-head
attention across multiple items).
```

```go
// DON'T parallelize small operations - goroutine overhead dominates
// Rule of thumb: only parallelize if work > 1ms per goroutine

func ShouldParallelize(rows, cols int) bool {
    flops := rows * cols  // Approximate
    return flops > 1000000  // Much higher threshold than expected!
}

// DON'T parallelize sequential dependencies
// The autograd backward pass must respect computation order
func BackwardSequential(sortedNodes []*Value) {
    // Must be sequential - each node depends on downstream gradients
    for i := len(sortedNodes) - 1; i >= 0; i-- {
        sortedNodes[i].backward()
    }
}
```

---

## 6. Complete Example: Optimized Model Structure

```go
package microgpt

import (
    "math/rand"
    "sync"
)

// Core types
type Value struct {
    Data     float64
    Grad     float64
    // For autograd
    backward func()
    parents  []*Value
    op       string
}

type FlatMatrix struct {
    Data   []Value
    Rows   int
    Cols   int
    Stride int
}

// Model architecture
type Config struct {
    VocabSize int
    BlockSize int
    NEmbdim   int
    NHead     int
    NLayer    int
}

type AttentionWeights struct {
    Wq, Wk, Wv, Wo *FlatMatrix
}

type MLPWeights struct {
    Fc1, Fc2 *FlatMatrix
}

type TransformerBlock struct {
    Attn AttentionWeights
    MLP  MLPWeights
}

type Model struct {
    Config Config
    
    // Embeddings
    Wte *FlatMatrix
    Wpe *FlatMatrix
    
    // Transformer blocks
    Blocks []TransformerBlock
    
    // Output
    LmHead *FlatMatrix
    
    // Pre-allocated buffers (reused between calls)
    forwardBuf *ForwardBuffers
    
    // Cached parameter list for iteration
    allParams []*FlatMatrix
}

func NewModel(cfg Config) *Model {
    m := &Model{
        Config: cfg,
        Wte:    NewFlatMatrix(cfg.VocabSize, cfg.NEmbdim, 0.08),
        Wpe:    NewFlatMatrix(cfg.BlockSize, cfg.NEmbdim, 0.08),
        Blocks: make([]TransformerBlock, cfg.NLayer),
        LmHead: NewFlatMatrix(cfg.VocabSize, cfg.NEmbdim, 0.08),
    }
    
    for l := 0; l < cfg.NLayer; l++ {
        m.Blocks[l] = TransformerBlock{
            Attn: AttentionWeights{
                Wq: NewFlatMatrix(cfg.NEmbdim, cfg.NEmbdim, 0.08),
                Wk: NewFlatMatrix(cfg.NEmbdim, cfg.NEmbdim, 0.08),
                Wv: NewFlatMatrix(cfg.NEmbdim, cfg.NEmbdim, 0.08),
                Wo: NewFlatMatrix(cfg.NEmbdim, cfg.NEmbdim, 0.08),
            },
            MLP: MLPWeights{
                Fc1: NewFlatMatrix(4*cfg.NEmbdim, cfg.NEmbdim, 0.08),
                Fc2: NewFlatMatrix(cfg.NEmbdim, 4*cfg.NEmbdim, 0.08),
            },
        }
    }
    
    m.forwardBuf = NewForwardBuffers(cfg.VocabSize, cfg.BlockSize, cfg.NEmbdim)
    m.cacheParamList()
    
    return m
}

func (m *Model) cacheParamList() {
    m.allParams = []*FlatMatrix{m.Wte, m.Wpe}
    for i := range m.Blocks {
        m.allParams = append(m.allParams,
            m.Blocks[i].Attn.Wq,
            m.Blocks[i].Attn.Wk,
            m.Blocks[i].Attn.Wv,
            m.Blocks[i].Attn.Wo,
            m.Blocks[i].MLP.Fc1,
            m.Blocks[i].MLP.Fc2,
        )
    }
    m.allParams = append(m.allParams, m.LmHead)
}

func (m *Model) NumParameters() int {
    total := 0
    for _, p := range m.allParams {
        total += len(p.Data)
    }
    return total
}

func (m *Model) ZeroGrad() {
    for _, p := range m.allParams {
        for i := range p.Data {
            p.Data[i].Grad = 0
        }
    }
}

func (m *Model) Update(lr float64) {
    for _, p := range m.allParams {
        for i := range p.Data {
            p.Data[i].Data -= lr * p.Data[i].Grad
        }
    }
}

// Helper constructors
func NewFlatMatrix(rows, cols int, std float64) *FlatMatrix {
    data := make([]Value, rows*cols)
    if std > 0 {
        for i := range data {
            data[i] = Value{Data: rand.NormFloat64() * std}
        }
    }
    return &FlatMatrix{
        Data:   data,
        Rows:   rows,
        Cols:   cols,
        Stride: cols,
    }
}

func (m *FlatMatrix) At(row, col int) *Value {
    return &m.Data[row*m.Stride+col]
}

func (m *FlatMatrix) Row(i int) []Value {
    start := i * m.Stride
    return m.Data[start : start+m.Cols]
}
```

---

## Summary of Recommendations (Validated by Benchmarks)

| Aspect | Recommendation | Speedup | Reason |
|--------|---------------|---------|--------|
| Matrix storage | `FlatMatrix` (1D slice) | **2-3x** | Cache locality, single allocation |
| Parameter storage | Struct with named fields | **61x** | Direct access vs map lookup |
| Iteration pattern | Cached `[]*FlatMatrix` slice | ~same | Dominated by actual work |
| Matmul loop order | `i-p-j` (row of A × row of B) | **2.6x** | Sequential B access |
| Memory | Pre-allocate all buffers | varies | Zero allocation during forward/backward |
| Parallelization | **ONLY** at batch/sequence level | negative! | Goroutine overhead defeats matmul parallelism |

### Key Findings from Benchmarks:

1. **Flat matrix is essential** - 2-3x speedup from cache locality alone
2. **Loop order matters enormously** - 2.6x speedup from optimal loop ordering
3. **Don't micro-parallelize** - Sequential matmul is 3.5x faster than spawning goroutines!
4. **Struct access crushes maps** - 61x faster for repeated lookups in forward pass
5. **Go's allocator is good** - Pre-allocation helps less than expected for small slices

### Recommended Parallelization Strategy:

```
❌ DON'T parallelize: Individual matmul, linear layers, activation functions
✅ DO parallelize: Processing multiple sequences in a batch
✅ DO parallelize: Independent attention heads across a batch
✅ DO parallelize: Data loading and preprocessing (I/O bound)
```

These patterns together can provide 2-5x speedup over naive implementations while maintaining clean, maintainable code.

