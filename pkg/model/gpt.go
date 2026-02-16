package model

import (
	"math"
	"math/rand/v2"

	"github.com/joelsearcy/microgpt-go/pkg/autograd"
)

// GPT holds the model configuration and parameters
type GPT struct {
	Params    *ModelParams
	NLayer    int
	NEmbd     int
	NHead     int
	HeadDim   int
	BlockSize int
}

// NewGPT creates a new GPT model with initialized parameters
func NewGPT(vocabSize, blockSize, nEmbd, nLayer, nHead int, rng *rand.Rand) *GPT {
	return &GPT{
		Params:    NewModelParams(vocabSize, blockSize, nEmbd, nLayer, rng),
		NLayer:    nLayer,
		NEmbd:     nEmbd,
		NHead:     nHead,
		HeadDim:   nEmbd / nHead,
		BlockSize: blockSize,
	}
}

// KVCache stores key-value pairs for autoregressive generation
type KVCache struct {
	Keys   [][][]*autograd.Value // [layer][position][embedding]
	Values [][][]*autograd.Value
}

// NewKVCache creates an empty KV cache for nLayers
func NewKVCache(nLayers int) *KVCache {
	return &KVCache{
		Keys:   make([][][]*autograd.Value, nLayers),
		Values: make([][][]*autograd.Value, nLayers),
	}
}

// Reset clears the cache for a new sequence
func (c *KVCache) Reset() {
	for i := range c.Keys {
		c.Keys[i] = nil
		c.Values[i] = nil
	}
}

// Forward performs one forward pass for a single token
// tokenID: the token index, posID: position in sequence
// Returns logits over vocabulary
func (g *GPT) Forward(tokenID, posID int, cache *KVCache) []*autograd.Value {
	// 1. Get token and position embeddings
	tokEmb := g.Params.Wte.Row(tokenID)
	posEmb := g.Params.Wpe.Row(posID)

	// 2. x = tok_emb + pos_emb
	x := make([]*autograd.Value, g.NEmbd)
	for i := 0; i < g.NEmbd; i++ {
		x[i] = tokEmb[i].Add(posEmb[i])
	}
	x = RMSNorm(x)

	// 3. For each transformer layer
	for li := 0; li < g.NLayer; li++ {
		block := &g.Params.Blocks[li]
		xResidual := x

		// Self-attention block
		x = RMSNorm(x)
		q := Linear(x, block.AttnWQ)
		k := Linear(x, block.AttnWK)
		v := Linear(x, block.AttnWV)

		// Append to KV cache
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

	// 4. Output projection
	return Linear(x, g.Params.LmHead)
}

// multiHeadAttention computes multi-head attention
func (g *GPT) multiHeadAttention(q []*autograd.Value, keys, vals [][]*autograd.Value) []*autograd.Value {
	xAttn := make([]*autograd.Value, g.NEmbd)

	for h := 0; h < g.NHead; h++ {
		hs := h * g.HeadDim

		// Extract query head slice
		qH := q[hs : hs+g.HeadDim]

		// Compute attention scores: q · k / sqrt(d)
		attnLogits := make([]*autograd.Value, len(keys))
		scale := math.Sqrt(float64(g.HeadDim))

		for t, kt := range keys {
			kH := kt[hs : hs+g.HeadDim]
			// Dot product
			dot := autograd.NewValue(0)
			for j := 0; j < g.HeadDim; j++ {
				dot = dot.Add(qH[j].Mul(kH[j]))
			}
			// Scale
			attnLogits[t] = dot.Mul(autograd.NewValue(1.0 / scale))
		}

		// Softmax over attention scores
		attnWeights := Softmax(attnLogits)

		// Weighted sum of values
		for j := 0; j < g.HeadDim; j++ {
			sum := autograd.NewValue(0)
			for t, vt := range vals {
				vH := vt[hs : hs+g.HeadDim]
				sum = sum.Add(attnWeights[t].Mul(vH[j]))
			}
			xAttn[hs+j] = sum
		}
	}

	return xAttn
}
