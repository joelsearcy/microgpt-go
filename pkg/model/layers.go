package model

import (
	"math"

	"github.com/joelsearcy/microgpt-go/pkg/autograd"
)

// Linear performs matrix-vector multiplication: W @ x
// w is [out_dim, in_dim], x is [in_dim], returns [out_dim]
func Linear(x []*autograd.Value, w *FlatMatrix) []*autograd.Value {
	out := make([]*autograd.Value, w.Rows)
	for i := 0; i < w.Rows; i++ {
		row := w.Row(i)
		out[i] = autograd.DotProduct(row, x)
	}
	return out
}

// Softmax computes softmax with numerical stability using a fused operation
func Softmax(logits []*autograd.Value) []*autograd.Value {
	return autograd.FusedSoftmax(logits)
}

// RMSNorm: x / sqrt(mean(x²) + eps)
func RMSNorm(x []*autograd.Value) []*autograd.Value {
	const eps = 1e-5
	n := float64(len(x))

	// Compute mean of squares
	sumSq := autograd.NewValue(0)
	for _, v := range x {
		sumSq = sumSq.Add(v.Mul(v))
	}
	meanSq := sumSq.Div(autograd.NewValue(n))

	// Compute 1/sqrt(mean(x²) + eps)
	// We need to compute this as a scalar operation
	rmsVal := math.Sqrt(meanSq.Data + eps)
	rmsInv := autograd.NewValue(1.0 / rmsVal)

	// Scale each element
	out := make([]*autograd.Value, len(x))
	for i, v := range x {
		out[i] = v.Mul(rmsInv)
	}

	return out
}
