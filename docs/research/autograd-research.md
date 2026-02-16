# Autograd Implementation Research for microgpt-go

This document provides comprehensive research on implementing an automatic differentiation (autograd) system in Go, based on Karpathy's micrograd Python implementation and informed by patterns from Gorgonia and Go best practices.

## Table of Contents

1. [Go Struct Design for Value Type](#1-go-struct-design-for-value-type)
2. [Operator Overloading Alternatives](#2-operator-overloading-alternatives)
3. [Efficient Topological Sort for Backward Pass](#3-efficient-topological-sort-for-backward-pass)
4. [Memory Management](#4-memory-management)
5. [Interface vs Concrete Types](#5-interface-vs-concrete-types)
6. [Final Recommendation](#6-final-recommendation)

---

## 1. Go Struct Design for Value Type

### Performance-Oriented Memory Layout

Go struct fields should be ordered to minimize padding. Fields are aligned based on their size, and the compiler adds padding between fields to maintain proper alignment.

**Key alignment rules:**
- `bool`, `int8`, `uint8`: 1-byte aligned
- `int16`, `uint16`: 2-byte aligned
- `int32`, `uint32`, `float32`: 4-byte aligned
- `int64`, `uint64`, `float64`: 8-byte aligned
- Pointers: 8-byte aligned on 64-bit systems

### Recommended Value Struct Design

```go
package autograd

// Value represents a scalar value with automatic differentiation support.
// Fields are ordered largest-to-smallest for optimal memory layout.
type Value struct {
    // 8-byte aligned fields first
    data     float64         // 8 bytes - the actual scalar value
    grad     float64         // 8 bytes - gradient (accumulated during backward)
    backward func()          // 8 bytes (pointer) - closure for gradient computation
    children []*Value        // 24 bytes (slice header: ptr + len + cap)
    op       string          // 16 bytes (string header: ptr + len)
    
    // Internal tracking
    id       uint64          // 8 bytes - unique ID for graph operations
}

// Total: ~72 bytes on 64-bit systems (no padding wasted)
```

### Alternative: Minimal Struct with Separate Graph

For better cache locality and reduced memory per node:

```go
// ValueCompact is a minimal representation focusing on hot data.
type ValueCompact struct {
    data float64  // 8 bytes - hot path
    grad float64  // 8 bytes - backward pass
    id   uint32   // 4 bytes - index into graph metadata
    _    uint32   // 4 bytes - padding for alignment
}

// Graph stores metadata separately for better cache efficiency.
type Graph struct {
    values    []ValueCompact
    children  [][]uint32       // children[id] = list of child IDs
    backwards []func()         // backward functions indexed by ID
    ops       []string         // operation names for debugging
}
```

### Comparison with Python micrograd

| Python micrograd          | Go Equivalent                        |
|---------------------------|--------------------------------------|
| `self.data` (float)       | `data float64`                       |
| `self.grad` (float)       | `grad float64`                       |
| `self._prev` (set)        | `children []*Value`                  |
| `self._backward` (lambda) | `backward func()`                    |
| `self._op` (str)          | `op string`                          |

---

## 2. Operator Overloading Alternatives

Go doesn't support operator overloading, so we need alternative patterns. Here are three approaches:

### Approach A: Method Chaining

```go
// Method chaining style - fluent interface
type Value struct {
    data     float64
    grad     float64
    children []*Value
    backward func()
}

func NewValue(data float64) *Value {
    return &Value{data: data, backward: func() {}}
}

func (v *Value) Add(other *Value) *Value {
    out := &Value{
        data:     v.data + other.data,
        children: []*Value{v, other},
    }
    out.backward = func() {
        v.grad += out.grad
        other.grad += out.grad
    }
    return out
}

func (v *Value) Mul(other *Value) *Value {
    out := &Value{
        data:     v.data * other.data,
        children: []*Value{v, other},
    }
    out.backward = func() {
        v.grad += other.data * out.grad
        other.grad += v.data * out.grad
    }
    return out
}

func (v *Value) Pow(n float64) *Value {
    out := &Value{
        data:     math.Pow(v.data, n),
        children: []*Value{v},
    }
    out.backward = func() {
        v.grad += n * math.Pow(v.data, n-1) * out.grad
    }
    return out
}

func (v *Value) ReLU() *Value {
    data := v.data
    if data < 0 {
        data = 0
    }
    out := &Value{
        data:     data,
        children: []*Value{v},
    }
    out.backward = func() {
        if out.data > 0 {
            v.grad += out.grad
        }
    }
    return out
}

func (v *Value) Neg() *Value {
    return v.Mul(NewValue(-1))
}

func (v *Value) Sub(other *Value) *Value {
    return v.Add(other.Neg())
}

func (v *Value) Div(other *Value) *Value {
    return v.Mul(other.Pow(-1))
}

// Usage:
// result := a.Add(b).Mul(c).ReLU()
```

**Pros:**
- Familiar OOP style
- Easy to read for simple expressions
- Natural for Go developers

**Cons:**
- Long chains can be hard to read
- Each method call allocates a new Value

### Approach B: Functional Style (Gorgonia Pattern)

```go
// Package-level functions (Gorgonia style)
func Add(a, b *Value) *Value {
    out := &Value{
        data:     a.data + b.data,
        children: []*Value{a, b},
    }
    out.backward = func() {
        a.grad += out.grad
        b.grad += out.grad
    }
    return out
}

func Mul(a, b *Value) *Value {
    out := &Value{
        data:     a.data * b.data,
        children: []*Value{a, b},
    }
    out.backward = func() {
        a.grad += b.data * out.grad
        b.grad += a.data * out.grad
    }
    return out
}

func Pow(v *Value, n float64) *Value {
    out := &Value{
        data:     math.Pow(v.data, n),
        children: []*Value{v},
    }
    out.backward = func() {
        v.grad += n * math.Pow(v.data, n-1) * out.grad
    }
    return out
}

func ReLU(v *Value) *Value {
    data := v.data
    if data < 0 {
        data = 0
    }
    out := &Value{
        data:     data,
        children: []*Value{v},
    }
    out.backward = func() {
        if out.data > 0 {
            v.grad += out.grad
        }
    }
    return out
}

func Neg(v *Value) *Value {
    return Mul(v, NewValue(-1))
}

func Sub(a, b *Value) *Value {
    return Add(a, Neg(b))
}

func Div(a, b *Value) *Value {
    return Mul(a, Pow(b, -1))
}

func Exp(v *Value) *Value {
    x := v.data
    out := &Value{
        data:     math.Exp(x),
        children: []*Value{v},
    }
    out.backward = func() {
        v.grad += out.data * out.grad // derivative of exp(x) is exp(x)
    }
    return out
}

func Tanh(v *Value) *Value {
    x := v.data
    t := math.Tanh(x)
    out := &Value{
        data:     t,
        children: []*Value{v},
    }
    out.backward = func() {
        v.grad += (1 - t*t) * out.grad
    }
    return out
}

// Usage:
// result := ReLU(Mul(Add(a, b), c))
```

**Pros:**
- Matches Gorgonia's established patterns
- Clearer for complex nested expressions
- Easier to compose with higher-order functions

**Cons:**
- Deeply nested calls can be hard to read
- Less discoverable via IDE autocomplete

### Approach C: DSL/Builder Pattern

```go
// Expression builder for complex computations
type Expr struct {
    value *Value
}

func Val(data float64) *Expr {
    return &Expr{value: NewValue(data)}
}

func From(v *Value) *Expr {
    return &Expr{value: v}
}

func (e *Expr) Add(other *Expr) *Expr {
    return &Expr{value: Add(e.value, other.value)}
}

func (e *Expr) Mul(other *Expr) *Expr {
    return &Expr{value: Mul(e.value, other.value)}
}

func (e *Expr) Sub(other *Expr) *Expr {
    return &Expr{value: Sub(e.value, other.value)}
}

func (e *Expr) Div(other *Expr) *Expr {
    return &Expr{value: Div(e.value, other.value)}
}

func (e *Expr) Pow(n float64) *Expr {
    return &Expr{value: Pow(e.value, n)}
}

func (e *Expr) ReLU() *Expr {
    return &Expr{value: ReLU(e.value)}
}

func (e *Expr) Tanh() *Expr {
    return &Expr{value: Tanh(e.value)}
}

func (e *Expr) Value() *Value {
    return e.value
}

// Usage:
// result := From(a).Add(From(b)).Mul(From(c)).ReLU().Value()
```

**Pros:**
- Clean, readable expressions
- IDE-friendly with autocomplete
- Separates expression building from execution

**Cons:**
- Extra wrapper allocation
- Indirection overhead

### Recommendation: Hybrid Approach

Combine method chaining for common operations with functional style for complex compositions:

```go
// Value type with basic method chaining
func (v *Value) Add(other *Value) *Value { ... }
func (v *Value) Mul(other *Value) *Value { ... }

// Package functions for operations that don't chain naturally
func Sum(values ...*Value) *Value { ... }
func Dot(a, b []*Value) *Value { ... }
func Softmax(values []*Value) []*Value { ... }
```

---

## 3. Efficient Topological Sort for Backward Pass

The backward pass requires visiting nodes in reverse topological order – from output to inputs.

### Go Implementation of Topological Sort

```go
// Backward performs backpropagation from this value through the graph.
func (v *Value) Backward() {
    // Build topological order using DFS
    topo := make([]*Value, 0)
    visited := make(map[*Value]bool)
    
    var buildTopo func(node *Value)
    buildTopo = func(node *Value) {
        if visited[node] {
            return
        }
        visited[node] = true
        for _, child := range node.children {
            buildTopo(child)
        }
        topo = append(topo, node)
    }
    
    buildTopo(v)
    
    // Set gradient of output to 1
    v.grad = 1.0
    
    // Traverse in reverse order, applying backward functions
    for i := len(topo) - 1; i >= 0; i-- {
        if topo[i].backward != nil {
            topo[i].backward()
        }
    }
}
```

### Optimized Version: Pre-allocated Slices

```go
// BackwardOptimized uses pre-allocated memory to reduce allocations.
func (v *Value) BackwardOptimized() {
    // Count nodes first to pre-allocate
    nodeCount := v.countNodes()
    
    topo := make([]*Value, 0, nodeCount)
    visited := make(map[*Value]struct{}, nodeCount) // struct{} is zero-size
    
    var buildTopo func(node *Value)
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
    v.grad = 1.0
    
    for i := len(topo) - 1; i >= 0; i-- {
        if topo[i].backward != nil {
            topo[i].backward()
        }
    }
}

func (v *Value) countNodes() int {
    visited := make(map[*Value]struct{})
    var count func(*Value) int
    count = func(node *Value) int {
        if _, seen := visited[node]; seen {
            return 0
        }
        visited[node] = struct{}{}
        n := 1
        for _, child := range node.children {
            n += count(child)
        }
        return n
    }
    return count(v)
}
```

### Iterative (Non-Recursive) Topological Sort

For very deep graphs, avoid stack overflow with an iterative approach:

```go
// BackwardIterative uses iterative DFS to avoid stack overflow.
func (v *Value) BackwardIterative() {
    // Use Kahn's algorithm (iterative BFS-based topological sort)
    // First, compute in-degrees
    inDegree := make(map[*Value]int)
    allNodes := make(map[*Value]struct{})
    
    var collectNodes func(*Value)
    collectNodes = func(node *Value) {
        if _, seen := allNodes[node]; seen {
            return
        }
        allNodes[node] = struct{}{}
        for _, child := range node.children {
            collectNodes(child)
        }
    }
    collectNodes(v)
    
    // Build parent map (reverse of children)
    parents := make(map[*Value][]*Value)
    for node := range allNodes {
        inDegree[node] = 0
    }
    for node := range allNodes {
        for _, child := range node.children {
            parents[child] = append(parents[child], node)
            inDegree[node]++
        }
    }
    
    // Find all nodes with in-degree 0 (leaves)
    queue := make([]*Value, 0)
    for node, deg := range inDegree {
        if deg == 0 {
            queue = append(queue, node)
        }
    }
    
    // Process in topological order (forward), store for reverse
    topo := make([]*Value, 0, len(allNodes))
    for len(queue) > 0 {
        node := queue[0]
        queue = queue[1:]
        topo = append(topo, node)
        
        for _, parent := range parents[node] {
            inDegree[parent]--
            if inDegree[parent] == 0 {
                queue = append(queue, parent)
            }
        }
    }
    
    // Backward pass in reverse topological order
    v.grad = 1.0
    for i := len(topo) - 1; i >= 0; i-- {
        if topo[i].backward != nil {
            topo[i].backward()
        }
    }
}
```

### Comparison with Gorgonia

Gorgonia uses `Sort()` and `UnstableSort()` functions:

```go
// From Gorgonia: Sort performs a topological sort on the graph
func Sort(g *ExprGraph) (Nodes, error)

// UnstableSort is faster but doesn't guarantee deterministic order
func UnstableSort(g *ExprGraph) (Nodes, error)
```

For micrograd-go, the recursive DFS approach is sufficient and idiomatic.

---

## 4. Memory Management

### Object Pooling with sync.Pool

```go
var valuePool = sync.Pool{
    New: func() any {
        return &Value{
            children: make([]*Value, 0, 4), // Pre-allocate common capacity
        }
    },
}

// NewValuePooled gets a Value from the pool.
func NewValuePooled(data float64) *Value {
    v := valuePool.Get().(*Value)
    v.data = data
    v.grad = 0
    v.children = v.children[:0] // Reset slice, keep capacity
    v.backward = nil
    return v
}

// Release returns a Value to the pool.
func (v *Value) Release() {
    v.data = 0
    v.grad = 0
    v.children = v.children[:0]
    v.backward = nil
    valuePool.Put(v)
}
```

### Batch Operations to Reduce Allocations

```go
// Graph manages all values in a computation graph.
type Graph struct {
    values []*Value
    pool   sync.Pool
}

func NewGraph() *Graph {
    g := &Graph{
        values: make([]*Value, 0, 1024),
    }
    g.pool.New = func() any {
        return &Value{children: make([]*Value, 0, 4)}
    }
    return g
}

func (g *Graph) NewValue(data float64) *Value {
    v := g.pool.Get().(*Value)
    v.data = data
    v.grad = 0
    v.children = v.children[:0]
    v.backward = nil
    g.values = append(g.values, v)
    return v
}

// ZeroGrad resets all gradients in the graph.
func (g *Graph) ZeroGrad() {
    for _, v := range g.values {
        v.grad = 0
    }
}

// Reset clears the graph and returns all values to the pool.
func (g *Graph) Reset() {
    for _, v := range g.values {
        v.data = 0
        v.grad = 0
        v.children = v.children[:0]
        v.backward = nil
        g.pool.Put(v)
    }
    g.values = g.values[:0]
}
```

### Avoiding Closure Allocations

Closures in Go can cause allocations. For hot paths, consider alternatives:

```go
// Instead of closures, store gradient computation info directly
type Value struct {
    data      float64
    grad      float64
    children  []*Value
    opType    OpType
    opParam   float64 // For Pow, stores exponent
}

type OpType uint8

const (
    OpNone OpType = iota
    OpAdd
    OpMul
    OpPow
    OpReLU
    OpTanh
    OpExp
)

// ComputeBackward computes gradients based on op type
func (v *Value) ComputeBackward() {
    switch v.opType {
    case OpAdd:
        v.children[0].grad += v.grad
        v.children[1].grad += v.grad
    case OpMul:
        v.children[0].grad += v.children[1].data * v.grad
        v.children[1].grad += v.children[0].data * v.grad
    case OpPow:
        child := v.children[0]
        child.grad += v.opParam * math.Pow(child.data, v.opParam-1) * v.grad
    case OpReLU:
        if v.data > 0 {
            v.children[0].grad += v.grad
        }
    case OpTanh:
        v.children[0].grad += (1 - v.data*v.data) * v.grad
    case OpExp:
        v.children[0].grad += v.data * v.grad
    }
}
```

### Memory Layout Optimization

```go
// Cache-line aware struct (128-byte aligned for false sharing prevention)
type ValueAligned struct {
    // Hot data - accessed in forward pass (first cache line)
    data float64
    grad float64
    id   uint32
    _    uint32
    
    // Cold data - accessed less frequently
    children []*ValueAligned
    opType   OpType
    opParam  float64
    
    // Padding to prevent false sharing in concurrent scenarios
    _ [128 - 64]byte // Adjust based on actual struct size
}
```

---

## 5. Interface vs Concrete Types

### Approach A: Concrete Types Only (Recommended for Micrograd)

```go
// Concrete type - simple, fast, sufficient for scalar autograd
type Value struct {
    data     float64
    grad     float64
    children []*Value
    backward func()
}
```

**Pros:**
- Zero interface overhead
- Direct memory layout, better cache performance
- Simpler code, easier to reason about
- No reflection needed

**Cons:**
- Less extensible
- Can't swap implementations

### Approach B: Interface-Based (Gorgonia Pattern)

```go
// Value interface for different scalar types
type Value interface {
    Data() float64
    SetData(float64)
    Grad() float64
    SetGrad(float64)
    Children() []Value
    Backward()
}

// Float64Value implements Value for float64
type Float64Value struct {
    data     float64
    grad     float64
    children []Value
    backward func()
}

func (v *Float64Value) Data() float64      { return v.data }
func (v *Float64Value) SetData(d float64)  { v.data = d }
func (v *Float64Value) Grad() float64      { return v.grad }
func (v *Float64Value) SetGrad(g float64)  { v.grad = g }
func (v *Float64Value) Children() []Value  { return v.children }
func (v *Float64Value) Backward()          { v.backward() }
```

**Pros:**
- Extensible to different numeric types (float32, complex128, etc.)
- Can mock for testing
- Matches Gorgonia's Value interface pattern

**Cons:**
- Interface indirection (~1-2ns per call)
- No inline optimization
- More complex code

### Approach C: Generics (Go 1.18+)

```go
// Generic Value type
type Value[T constraints.Float] struct {
    data     T
    grad     T
    children []*Value[T]
    backward func()
}

func NewValue[T constraints.Float](data T) *Value[T] {
    return &Value[T]{data: data, backward: func() {}}
}

func Add[T constraints.Float](a, b *Value[T]) *Value[T] {
    out := &Value[T]{
        data:     a.data + b.data,
        children: []*Value[T]{a, b},
    }
    out.backward = func() {
        a.grad += out.grad
        b.grad += out.grad
    }
    return out
}

// Usage:
// v := NewValue[float64](3.0)
// f32 := NewValue[float32](3.0)
```

**Pros:**
- Type safety with compile-time checking
- Zero runtime overhead (monomorphization)
- Flexible for different float types

**Cons:**
- Slightly more complex API
- Code duplication in compiled binary

### Performance Comparison

| Approach     | Method Call Overhead | Memory Overhead | Flexibility |
|--------------|----------------------|-----------------|-------------|
| Concrete     | ~0ns                 | Minimal         | Low         |
| Interface    | ~1-2ns               | +16 bytes/value | High        |
| Generics     | ~0ns                 | Minimal         | Medium      |

### Recommendation

For a micrograd port, use **concrete types** with optional generics:

```go
// Default to float64 for simplicity
type Value = ValueT[float64]

// Generic implementation available if needed
type ValueT[T constraints.Float] struct {
    data     T
    grad     T
    children []*ValueT[T]
    backward func()
}
```

---

## 6. Final Recommendation

For the most idiomatic Go solution that balances readability with performance:

### Complete Implementation

```go
package autograd

import (
    "math"
    "sync"
)

// Value represents a differentiable scalar value.
type Value struct {
    data     float64
    grad     float64
    children []*Value
    backward func()
    op       string // For debugging/visualization
}

// ============================================================
// CONSTRUCTORS
// ============================================================

var valuePool = sync.Pool{
    New: func() any {
        return &Value{children: make([]*Value, 0, 2)}
    },
}

// NewValue creates a new Value with the given data.
func NewValue(data float64) *Value {
    return &Value{
        data:     data,
        backward: func() {},
    }
}

// Data returns the value's data.
func (v *Value) Data() float64 { return v.data }

// Grad returns the value's gradient.
func (v *Value) Grad() float64 { return v.grad }

// ============================================================
// OPERATIONS (Method Chaining Style)
// ============================================================

// Add returns a new Value that is the sum of v and other.
func (v *Value) Add(other *Value) *Value {
    out := &Value{
        data:     v.data + other.data,
        children: []*Value{v, other},
        op:       "+",
    }
    out.backward = func() {
        v.grad += out.grad
        other.grad += out.grad
    }
    return out
}

// Mul returns a new Value that is the product of v and other.
func (v *Value) Mul(other *Value) *Value {
    out := &Value{
        data:     v.data * other.data,
        children: []*Value{v, other},
        op:       "*",
    }
    out.backward = func() {
        v.grad += other.data * out.grad
        other.grad += v.data * out.grad
    }
    return out
}

// Pow returns a new Value that is v raised to the power n.
func (v *Value) Pow(n float64) *Value {
    out := &Value{
        data:     math.Pow(v.data, n),
        children: []*Value{v},
        op:       "**",
    }
    out.backward = func() {
        v.grad += n * math.Pow(v.data, n-1) * out.grad
    }
    return out
}

// ReLU returns the ReLU activation of v.
func (v *Value) ReLU() *Value {
    data := v.data
    if data < 0 {
        data = 0
    }
    out := &Value{
        data:     data,
        children: []*Value{v},
        op:       "ReLU",
    }
    out.backward = func() {
        if out.data > 0 {
            v.grad += out.grad
        }
    }
    return out
}

// Tanh returns the hyperbolic tangent of v.
func (v *Value) Tanh() *Value {
    t := math.Tanh(v.data)
    out := &Value{
        data:     t,
        children: []*Value{v},
        op:       "tanh",
    }
    out.backward = func() {
        v.grad += (1 - t*t) * out.grad
    }
    return out
}

// Exp returns e^v.
func (v *Value) Exp() *Value {
    out := &Value{
        data:     math.Exp(v.data),
        children: []*Value{v},
        op:       "exp",
    }
    out.backward = func() {
        v.grad += out.data * out.grad
    }
    return out
}

// Neg returns -v.
func (v *Value) Neg() *Value {
    return v.Mul(NewValue(-1))
}

// Sub returns v - other.
func (v *Value) Sub(other *Value) *Value {
    return v.Add(other.Neg())
}

// Div returns v / other.
func (v *Value) Div(other *Value) *Value {
    return v.Mul(other.Pow(-1))
}

// ============================================================
// BACKPROPAGATION
// ============================================================

// Backward performs backpropagation from this value.
func (v *Value) Backward() {
    // Topological sort using DFS
    topo := make([]*Value, 0)
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
    
    // Set output gradient to 1
    v.grad = 1.0
    
    // Backward pass in reverse topological order
    for i := len(topo) - 1; i >= 0; i-- {
        if topo[i].backward != nil {
            topo[i].backward()
        }
    }
}

// ZeroGrad resets the gradient of this value and all its children.
func (v *Value) ZeroGrad() {
    visited := make(map[*Value]struct{})
    var zero func(*Value)
    zero = func(node *Value) {
        if _, seen := visited[node]; seen {
            return
        }
        visited[node] = struct{}{}
        node.grad = 0
        for _, child := range node.children {
            zero(child)
        }
    }
    zero(v)
}

// ============================================================
// CONVENIENCE FUNCTIONS
// ============================================================

// AddScalar adds a scalar to a Value.
func (v *Value) AddScalar(s float64) *Value {
    return v.Add(NewValue(s))
}

// MulScalar multiplies a Value by a scalar.
func (v *Value) MulScalar(s float64) *Value {
    return v.Mul(NewValue(s))
}
```

### Usage Example

```go
package main

import (
    "fmt"
    "microgpt-go/autograd"
)

func main() {
    // Create inputs
    x := autograd.NewValue(2.0)
    y := autograd.NewValue(3.0)
    
    // Forward pass: f(x, y) = (x + y) * x
    z := x.Add(y).Mul(x)
    
    // Backward pass
    z.Backward()
    
    fmt.Printf("z = %f\n", z.Data())    // z = 10.0
    fmt.Printf("dz/dx = %f\n", x.Grad()) // dz/dx = 7.0 (2x + y = 2*2 + 3)
    fmt.Printf("dz/dy = %f\n", y.Grad()) // dz/dy = 2.0 (x)
}
```

### Key Design Decisions Summary

| Aspect              | Decision                              | Rationale                           |
|---------------------|---------------------------------------|-------------------------------------|
| Value Type          | Concrete struct, not interface        | Performance, simplicity             |
| Operations          | Method chaining with value receivers  | Readable, chainable                 |
| Memory Layout       | Largest fields first                  | Minimize padding                    |
| Topo Sort           | Recursive DFS with map               | Simple, efficient for small graphs  |
| Gradient Closures   | Inline func() closures               | Flexibility, readable code          |
| Pooling             | Optional sync.Pool                   | Add if profiling shows need         |

This implementation closely mirrors Karpathy's micrograd while following idiomatic Go patterns. It prioritizes readability and correctness while maintaining good performance characteristics for typical neural network training workloads.
