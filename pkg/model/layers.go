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

// Softmax with numerical stability (subtract max before exp)
func Softmax(logits []*autograd.Value) []*autograd.Value {
	// Find max for numerical stability
	maxVal := logits[0].Data
	for _, v := range logits[1:] {
		if v.Data > maxVal {
			maxVal = v.Data
		}
	}

	// Compute exp(x - max)
	exps := make([]*autograd.Value, len(logits))
	for i, v := range logits {
		shifted := v.Sub(autograd.NewValue(maxVal))
		exps[i] = shifted.Exp()
	}

	// Sum of exps
	sumExp := autograd.NewValue(0)
	for _, e := range exps {
		sumExp = sumExp.Add(e)
	}

	// Normalize
	out := make([]*autograd.Value, len(logits))
	for i, e := range exps {
		out[i] = e.Div(sumExp)
	}

	return out
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
