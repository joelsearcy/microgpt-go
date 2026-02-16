package autograd

import "math"

// Value represents a scalar value with automatic differentiation support.
// It tracks the computation graph for backpropagation.
type Value struct {
	Data       float64   // the actual scalar value
	Grad       float64   // gradient accumulated during backward pass
	children   []*Value  // computation graph edges (inputs to this operation)
	localGrads []float64 // local derivatives for chain rule (∂self/∂child for each child)
}

// NewValue creates a leaf Value node with the given data.
func NewValue(data float64) *Value {
	return &Value{
		Data:       data,
		Grad:       0,
		children:   nil,
		localGrads: nil,
	}
}

// Scalar is an alias for NewValue, typically used for constants.
func Scalar(data float64) *Value {
	return NewValue(data)
}

// Add returns a new Value representing self + other.
// Local gradients: ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1
func (v *Value) Add(other *Value) *Value {
	return &Value{
		Data:       v.Data + other.Data,
		Grad:       0,
		children:   []*Value{v, other},
		localGrads: []float64{1, 1},
	}
}

// Mul returns a new Value representing self * other.
// Local gradients: ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
func (v *Value) Mul(other *Value) *Value {
	return &Value{
		Data:       v.Data * other.Data,
		Grad:       0,
		children:   []*Value{v, other},
		localGrads: []float64{other.Data, v.Data},
	}
}

// Neg returns a new Value representing -self.
// Implemented as self * (-1)
func (v *Value) Neg() *Value {
	return v.Mul(Scalar(-1))
}

// Sub returns a new Value representing self - other.
// Implemented as self + (-other)
func (v *Value) Sub(other *Value) *Value {
	return v.Add(other.Neg())
}

// Pow returns a new Value representing self^exp.
// Local gradient: ∂(x^n)/∂x = n * x^(n-1)
func (v *Value) Pow(exp float64) *Value {
	return &Value{
		Data:       math.Pow(v.Data, exp),
		Grad:       0,
		children:   []*Value{v},
		localGrads: []float64{exp * math.Pow(v.Data, exp-1)},
	}
}

// Div returns a new Value representing self / other.
// Implemented as self * other^(-1)
func (v *Value) Div(other *Value) *Value {
	return v.Mul(other.Pow(-1))
}

// Exp returns a new Value representing e^self.
// Local gradient: ∂(e^x)/∂x = e^x
func (v *Value) Exp() *Value {
	result := math.Exp(v.Data)
	return &Value{
		Data:       result,
		Grad:       0,
		children:   []*Value{v},
		localGrads: []float64{result},
	}
}

// Log returns a new Value representing ln(self).
// Local gradient: ∂(ln(x))/∂x = 1/x
func (v *Value) Log() *Value {
	return &Value{
		Data:       math.Log(v.Data),
		Grad:       0,
		children:   []*Value{v},
		localGrads: []float64{1.0 / v.Data},
	}
}

// ReLU returns a new Value representing max(0, self).
// Local gradient: 1 if x > 0, else 0
func (v *Value) ReLU() *Value {
	var data float64
	var localGrad float64
	if v.Data > 0 {
		data = v.Data
		localGrad = 1
	} else {
		data = 0
		localGrad = 0
	}
	return &Value{
		Data:       data,
		Grad:       0,
		children:   []*Value{v},
		localGrads: []float64{localGrad},
	}
}

// Backward performs backpropagation starting from this Value.
// It builds a topological ordering via DFS, then propagates gradients backward.
// The gradient of this Value is set to 1 (assumed to be the loss).
func (v *Value) Backward() {
	// Build topological order using iterative DFS
	var topo []*Value
	visited := make(map[*Value]bool)

	// Iterative DFS using explicit stack
	type stackItem struct {
		node    *Value
		visited bool
	}
	stack := []stackItem{{v, false}}

	for len(stack) > 0 {
		item := stack[len(stack)-1]
		stack = stack[:len(stack)-1]

		if item.visited {
			topo = append(topo, item.node)
			continue
		}

		if visited[item.node] {
			continue
		}
		visited[item.node] = true

		// Push the node again to be added to topo after children
		stack = append(stack, stackItem{item.node, true})

		// Push children
		for _, child := range item.node.children {
			if !visited[child] {
				stack = append(stack, stackItem{child, false})
			}
		}
	}

	// Reverse to get topological order from output to inputs
	// Actually, the order is already from leaves to root, so we need to reverse
	for i, j := 0, len(topo)-1; i < j; i, j = i+1, j-1 {
		topo[i], topo[j] = topo[j], topo[i]
	}

	// Initialize the gradient of the output to 1
	v.Grad = 1

	// Backpropagate gradients
	for _, node := range topo {
		for i, child := range node.children {
			// Chain rule: child.grad += node.grad * local_grad
			child.Grad += node.Grad * node.localGrads[i]
		}
	}
}

// ZeroGrad resets the gradient of this Value to 0.
func (v *Value) ZeroGrad() {
	v.Grad = 0
}
