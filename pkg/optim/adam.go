package optim

import (
	"math"

	"github.com/joelsearcy/microgpt-go/pkg/autograd"
)

// AdamOptimizer implements the Adam optimization algorithm
type AdamOptimizer struct {
	LR      float64 // base learning rate
	Beta1   float64 // exponential decay rate for first moment
	Beta2   float64 // exponential decay rate for second moment
	Epsilon float64 // small constant for numerical stability

	m []float64 // first moment estimates
	v []float64 // second moment estimates
	t int       // timestep counter
}

// NewAdam creates a new Adam optimizer
// Default hyperparameters from microgpt: lr=0.01, beta1=0.85, beta2=0.99, eps=1e-8
func NewAdam(numParams int, lr, beta1, beta2, eps float64) *AdamOptimizer {
	return &AdamOptimizer{
		LR:      lr,
		Beta1:   beta1,
		Beta2:   beta2,
		Epsilon: eps,
		m:       make([]float64, numParams),
		v:       make([]float64, numParams),
		t:       0,
	}
}

// Step performs one optimization step
// lrDecay is multiplied with base LR (for learning rate scheduling)
func (opt *AdamOptimizer) Step(params []*autograd.Value, lrDecay float64) {
	// 1. Increment timestep t
	opt.t++

	// 2. Compute bias correction terms: bc1 = 1 - beta1^t, bc2 = 1 - beta2^t
	bc1 := 1 - math.Pow(opt.Beta1, float64(opt.t))
	bc2 := 1 - math.Pow(opt.Beta2, float64(opt.t))

	// 3. For each parameter p with gradient g:
	for i, p := range params {
		g := p.Grad

		// m[i] = beta1 * m[i] + (1 - beta1) * g
		opt.m[i] = opt.Beta1*opt.m[i] + (1-opt.Beta1)*g

		// v[i] = beta2 * v[i] + (1 - beta2) * g * g
		opt.v[i] = opt.Beta2*opt.v[i] + (1-opt.Beta2)*g*g

		// mHat = m[i] / bc1
		mHat := opt.m[i] / bc1

		// vHat = v[i] / bc2
		vHat := opt.v[i] / bc2

		// p.Data -= lr * lrDecay * mHat / (sqrt(vHat) + epsilon)
		p.Data -= opt.LR * lrDecay * mHat / (math.Sqrt(vHat) + opt.Epsilon)

		// p.Grad = 0 (zero gradient for next iteration)
		p.Grad = 0
	}
}

// Reset resets the optimizer state for a new training run
func (opt *AdamOptimizer) Reset() {
	for i := range opt.m {
		opt.m[i] = 0
	}
	for i := range opt.v {
		opt.v[i] = 0
	}
	opt.t = 0
}
