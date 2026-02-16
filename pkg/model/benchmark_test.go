package model

import (
	"math/rand/v2"
	"testing"

	"github.com/joelsearcy/microgpt-go/pkg/autograd"
)

// BenchmarkLinear benchmarks the Linear layer with typical dimensions (16x16 matrix, 16-dim vector)
func BenchmarkLinear(b *testing.B) {
	rng := rand.New(rand.NewPCG(42, 42))

	// Create a 16x16 weight matrix
	w := NewMatrix(16, 16, 0.02, rng)

	// Create a 16-dim input vector
	x := make([]*autograd.Value, 16)
	for i := range x {
		x[i] = autograd.NewValue(rng.NormFloat64())
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Linear(x, w)
	}
}

// BenchmarkSoftmax benchmarks Softmax on 27-element logits (vocab size)
func BenchmarkSoftmax(b *testing.B) {
	rng := rand.New(rand.NewPCG(42, 42))

	// Create 27-element logits (typical vocab size for chars)
	logits := make([]*autograd.Value, 27)
	for i := range logits {
		logits[i] = autograd.NewValue(rng.NormFloat64())
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Softmax(logits)
	}
}

// BenchmarkRMSNorm benchmarks RMSNorm on a 16-dim vector
func BenchmarkRMSNorm(b *testing.B) {
	rng := rand.New(rand.NewPCG(42, 42))

	// Create a 16-dim input vector
	x := make([]*autograd.Value, 16)
	for i := range x {
		x[i] = autograd.NewValue(rng.NormFloat64())
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = RMSNorm(x)
	}
}

// BenchmarkGPTForward benchmarks the full GPT forward pass
func BenchmarkGPTForward(b *testing.B) {
	rng := rand.New(rand.NewPCG(42, 42))

	// Create a small GPT model matching training config
	vocabSize := 27
	blockSize := 16
	nEmbd := 16
	nLayer := 1
	nHead := 4

	gpt := NewGPT(vocabSize, blockSize, nEmbd, nLayer, nHead, rng)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache := NewKVCache(nLayer, blockSize)
		// Process a single token at position 0
		_ = gpt.Forward(0, 0, cache)
	}
}

// BenchmarkGPTForwardSequence benchmarks GPT forward pass for a full sequence
func BenchmarkGPTForwardSequence(b *testing.B) {
	rng := rand.New(rand.NewPCG(42, 42))

	vocabSize := 27
	blockSize := 16
	nEmbd := 16
	nLayer := 1
	nHead := 4
	seqLen := 8 // Half of block size

	gpt := NewGPT(vocabSize, blockSize, nEmbd, nLayer, nHead, rng)

	// Create a random token sequence
	tokens := make([]int, seqLen)
	for i := range tokens {
		tokens[i] = rng.IntN(vocabSize)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache := NewKVCache(nLayer, blockSize)
		for posID := 0; posID < seqLen; posID++ {
			_ = gpt.Forward(tokens[posID], posID, cache)
		}
	}
}

// BenchmarkBackward benchmarks the backward pass on a typical loss computation
func BenchmarkBackward(b *testing.B) {
	rng := rand.New(rand.NewPCG(42, 42))

	vocabSize := 27
	blockSize := 16
	nEmbd := 16
	nLayer := 1
	nHead := 4

	gpt := NewGPT(vocabSize, blockSize, nEmbd, nLayer, nHead, rng)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()

		// Forward pass
		cache := NewKVCache(nLayer, blockSize)
		tokenID := 0
		targetID := 1

		logits := gpt.Forward(tokenID, 0, cache)
		probs := Softmax(logits)
		loss := probs[targetID].Log().Neg()

		// Zero gradients
		gpt.Params.ZeroGrads()

		b.StartTimer()

		// Benchmark backward pass
		loss.Backward()
	}
}

// BenchmarkFullTrainingStep benchmarks a complete training step (forward + backward)
func BenchmarkFullTrainingStep(b *testing.B) {
	rng := rand.New(rand.NewPCG(42, 42))

	vocabSize := 27
	blockSize := 16
	nEmbd := 16
	nLayer := 1
	nHead := 4
	seqLen := 8

	gpt := NewGPT(vocabSize, blockSize, nEmbd, nLayer, nHead, rng)

	// Create a random token sequence
	tokens := make([]int, seqLen+1)
	for i := range tokens {
		tokens[i] = rng.IntN(vocabSize)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Forward pass for sequence
		cache := NewKVCache(nLayer, blockSize)
		var losses []*autograd.Value

		for posID := 0; posID < seqLen; posID++ {
			tokenID := tokens[posID]
			targetID := tokens[posID+1]

			logits := gpt.Forward(tokenID, posID, cache)
			probs := Softmax(logits)
			lossT := probs[targetID].Log().Neg()
			losses = append(losses, lossT)
		}

		// Average loss
		loss := autograd.NewValue(0)
		for _, l := range losses {
			loss = loss.Add(l)
		}
		loss = loss.Mul(autograd.NewValue(1.0 / float64(seqLen)))

		// Backward pass
		loss.Backward()

		// Zero gradients for next iteration
		gpt.Params.ZeroGrads()
	}
}

// BenchmarkAttention benchmarks the multi-head attention computation
func BenchmarkAttention(b *testing.B) {
	rng := rand.New(rand.NewPCG(42, 42))

	vocabSize := 27
	blockSize := 16
	nEmbd := 16
	nLayer := 1
	nHead := 4

	gpt := NewGPT(vocabSize, blockSize, nEmbd, nLayer, nHead, rng)

	// Pre-populate cache with a few positions
	cache := NewKVCache(nLayer, blockSize)
	for posID := 0; posID < 4; posID++ {
		gpt.Forward(rng.IntN(vocabSize), posID, cache)
	}

	// Create query vector
	q := make([]*autograd.Value, nEmbd)
	for i := range q {
		q[i] = autograd.NewValue(rng.NormFloat64())
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = gpt.multiHeadAttention(q, cache.Keys[0], cache.Values[0])
	}
}
