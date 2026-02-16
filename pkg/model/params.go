package model

import (
	"math/rand/v2"

	"github.com/joelsearcy/microgpt-go/pkg/autograd"
)

// FlatMatrix stores 2D matrix as contiguous 1D slice (row-major)
type FlatMatrix struct {
	Data       []*autograd.Value
	Rows, Cols int
}

// NewMatrix creates a matrix with Gaussian-initialized values
func NewMatrix(rows, cols int, std float64, rng *rand.Rand) *FlatMatrix {
	data := make([]*autograd.Value, rows*cols)
	for i := range data {
		data[i] = autograd.NewValue(rng.NormFloat64() * std)
	}
	return &FlatMatrix{
		Data: data,
		Rows: rows,
		Cols: cols,
	}
}

// At returns pointer to element at (row, col)
func (m *FlatMatrix) At(row, col int) *autograd.Value {
	return m.Data[row*m.Cols+col]
}

// Row returns a slice view of row
func (m *FlatMatrix) Row(row int) []*autograd.Value {
	start := row * m.Cols
	return m.Data[start : start+m.Cols]
}

// TransformerBlock holds weights for one transformer layer
type TransformerBlock struct {
	AttnWQ, AttnWK, AttnWV, AttnWO *FlatMatrix
	MlpFC1, MlpFC2                 *FlatMatrix
}

// ModelParams holds all trainable parameters
type ModelParams struct {
	Wte    *FlatMatrix // token embeddings [vocab_size, n_embd]
	Wpe    *FlatMatrix // position embeddings [block_size, n_embd]
	LmHead *FlatMatrix // output projection [vocab_size, n_embd]
	Blocks []TransformerBlock

	allParams []*autograd.Value // cached flat list
}

// NewModelParams creates and initializes all model parameters
func NewModelParams(vocabSize, blockSize, nEmbd, nLayer int, rng *rand.Rand) *ModelParams {
	std := 0.02

	p := &ModelParams{
		Wte:    NewMatrix(vocabSize, nEmbd, std, rng),
		Wpe:    NewMatrix(blockSize, nEmbd, std, rng),
		LmHead: NewMatrix(vocabSize, nEmbd, std, rng),
		Blocks: make([]TransformerBlock, nLayer),
	}

	// Initialize transformer blocks
	for i := 0; i < nLayer; i++ {
		p.Blocks[i] = TransformerBlock{
			AttnWQ: NewMatrix(nEmbd, nEmbd, std, rng),
			AttnWK: NewMatrix(nEmbd, nEmbd, std, rng),
			AttnWV: NewMatrix(nEmbd, nEmbd, std, rng),
			AttnWO: NewMatrix(nEmbd, nEmbd, std, rng),
			MlpFC1: NewMatrix(4*nEmbd, nEmbd, std, rng), // 4x expansion
			MlpFC2: NewMatrix(nEmbd, 4*nEmbd, std, rng),
		}
	}

	// Cache all parameters
	p.cacheAllParams()

	return p
}

// cacheAllParams builds the flat list of all parameters
func (p *ModelParams) cacheAllParams() {
	var params []*autograd.Value

	// Add embedding weights
	params = append(params, p.Wte.Data...)
	params = append(params, p.Wpe.Data...)
	params = append(params, p.LmHead.Data...)

	// Add block weights
	for _, block := range p.Blocks {
		params = append(params, block.AttnWQ.Data...)
		params = append(params, block.AttnWK.Data...)
		params = append(params, block.AttnWV.Data...)
		params = append(params, block.AttnWO.Data...)
		params = append(params, block.MlpFC1.Data...)
		params = append(params, block.MlpFC2.Data...)
	}

	p.allParams = params
}

// AllParams returns flattened list of all parameters (cached)
func (p *ModelParams) AllParams() []*autograd.Value {
	return p.allParams
}

// ZeroGrads resets all parameter gradients to 0
func (p *ModelParams) ZeroGrads() {
	for _, param := range p.allParams {
		param.Grad = 0
	}
}
