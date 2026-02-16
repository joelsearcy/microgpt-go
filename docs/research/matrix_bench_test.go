package research

import (
	"math/rand"
	"sync"
	"testing"
)

// ============================================================================
// Matrix Representations
// ============================================================================

type Value struct {
	Data float64
	Grad float64
}

// Option A: Slice of slices
type SliceMatrix [][]Value

func NewSliceMatrix(rows, cols int, std float64) SliceMatrix {
	m := make(SliceMatrix, rows)
	for i := range m {
		m[i] = make([]Value, cols)
		for j := range m[i] {
			m[i][j] = Value{Data: rand.NormFloat64() * std}
		}
	}
	return m
}

// Option B: Flat matrix
type FlatMatrix struct {
	Data   []Value
	Rows   int
	Cols   int
	Stride int
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

func (m *FlatMatrix) At(row, col int) *Value {
	return &m.Data[row*m.Stride+col]
}

func (m *FlatMatrix) Row(i int) []Value {
	start := i * m.Stride
	return m.Data[start : start+m.Cols]
}

// ============================================================================
// Benchmark: Matrix Representation - MatVec Multiply
// ============================================================================

func BenchmarkMatVec_SliceOfSlices(b *testing.B) {
	const size = 768
	m := NewSliceMatrix(size, size, 0.08)
	v := make([]float64, size)
	out := make([]float64, size)
	for i := range v {
		v[i] = rand.Float64()
	}

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		for i := 0; i < size; i++ {
			sum := 0.0
			for j := 0; j < size; j++ {
				sum += m[i][j].Data * v[j]
			}
			out[i] = sum
		}
	}
}

func BenchmarkMatVec_FlatMatrix(b *testing.B) {
	const size = 768
	m := NewFlatMatrix(size, size, 0.08)
	v := make([]float64, size)
	out := make([]float64, size)
	for i := range v {
		v[i] = rand.Float64()
	}

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		for i := 0; i < size; i++ {
			row := m.Row(i)
			sum := 0.0
			for j := 0; j < size; j++ {
				sum += row[j].Data * v[j]
			}
			out[i] = sum
		}
	}
}

func BenchmarkMatVec_FlatMatrix_Unrolled(b *testing.B) {
	const size = 768
	m := NewFlatMatrix(size, size, 0.08)
	v := make([]float64, size)
	out := make([]float64, size)
	for i := range v {
		v[i] = rand.Float64()
	}

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		for i := 0; i < size; i++ {
			row := m.Row(i)
			sum := 0.0
			j := 0
			// Unroll by 4
			for ; j <= size-4; j += 4 {
				sum += row[j].Data*v[j] +
					row[j+1].Data*v[j+1] +
					row[j+2].Data*v[j+2] +
					row[j+3].Data*v[j+3]
			}
			for ; j < size; j++ {
				sum += row[j].Data * v[j]
			}
			out[i] = sum
		}
	}
}

// ============================================================================
// Benchmark: Matrix Multiplication Loop Order
// ============================================================================

func BenchmarkMatmul_Naive_ijk(b *testing.B) {
	const size = 256 // Smaller for faster benchmarks
	A := NewFlatMatrix(size, size, 0.08)
	B := NewFlatMatrix(size, size, 0.08)
	C := NewFlatMatrix(size, size, 0)

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		// Naive i-j-k order
		for i := 0; i < size; i++ {
			for j := 0; j < size; j++ {
				sum := 0.0
				for k := 0; k < size; k++ {
					sum += A.At(i, k).Data * B.At(k, j).Data // Strided B access!
				}
				C.At(i, j).Data = sum
			}
		}
	}
}

func BenchmarkMatmul_Optimized_ipj(b *testing.B) {
	const size = 256
	A := NewFlatMatrix(size, size, 0.08)
	B := NewFlatMatrix(size, size, 0.08)
	C := NewFlatMatrix(size, size, 0)

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		// Zero output
		for i := range C.Data {
			C.Data[i].Data = 0
		}

		// Optimized i-p-j order (outer product)
		for i := 0; i < size; i++ {
			rowA := A.Row(i)
			rowC := C.Row(i)

			for p := 0; p < size; p++ {
				a_ip := rowA[p].Data
				rowB := B.Row(p) // Sequential B access!

				for j := 0; j < size; j++ {
					rowC[j].Data += a_ip * rowB[j].Data
				}
			}
		}
	}
}

func BenchmarkMatmul_Tiled(b *testing.B) {
	const size = 256
	const tileSize = 32
	A := NewFlatMatrix(size, size, 0.08)
	B := NewFlatMatrix(size, size, 0.08)
	C := NewFlatMatrix(size, size, 0)

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		// Zero output
		for i := range C.Data {
			C.Data[i].Data = 0
		}

		// Tiled multiplication
		for i0 := 0; i0 < size; i0 += tileSize {
			for j0 := 0; j0 < size; j0 += tileSize {
				for p0 := 0; p0 < size; p0 += tileSize {
					// Process tile
					iMax := min(i0+tileSize, size)
					jMax := min(j0+tileSize, size)
					pMax := min(p0+tileSize, size)

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
}

// ============================================================================
// Benchmark: Parameter Storage - Map vs Struct
// ============================================================================

type ParamsMap struct {
	Weights map[string]*FlatMatrix
}

type ParamsStruct struct {
	Wte    *FlatMatrix
	Wpe    *FlatMatrix
	LmHead *FlatMatrix
	// Cached slice for iteration
	all []*FlatMatrix
}

func (p *ParamsStruct) initAll() {
	p.all = []*FlatMatrix{p.Wte, p.Wpe, p.LmHead}
}

func BenchmarkParams_MapAccess(b *testing.B) {
	p := &ParamsMap{
		Weights: map[string]*FlatMatrix{
			"wte":     NewFlatMatrix(1000, 768, 0.08),
			"wpe":     NewFlatMatrix(128, 768, 0.08),
			"lm_head": NewFlatMatrix(1000, 768, 0.08),
		},
	}

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		// Access each weight 100 times (simulating forward pass)
		for i := 0; i < 100; i++ {
			_ = p.Weights["wte"]
			_ = p.Weights["wpe"]
			_ = p.Weights["lm_head"]
		}
	}
}

func BenchmarkParams_StructAccess(b *testing.B) {
	p := &ParamsStruct{
		Wte:    NewFlatMatrix(1000, 768, 0.08),
		Wpe:    NewFlatMatrix(128, 768, 0.08),
		LmHead: NewFlatMatrix(1000, 768, 0.08),
	}

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		// Access each weight 100 times
		for i := 0; i < 100; i++ {
			_ = p.Wte
			_ = p.Wpe
			_ = p.LmHead
		}
	}
}

func BenchmarkParams_MapIteration(b *testing.B) {
	p := &ParamsMap{
		Weights: map[string]*FlatMatrix{
			"wte":     NewFlatMatrix(100, 64, 0.08),
			"wpe":     NewFlatMatrix(32, 64, 0.08),
			"lm_head": NewFlatMatrix(100, 64, 0.08),
		},
	}

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		// Zero gradients - common operation
		for _, m := range p.Weights {
			for i := range m.Data {
				m.Data[i].Grad = 0
			}
		}
	}
}

func BenchmarkParams_SliceIteration(b *testing.B) {
	p := &ParamsStruct{
		Wte:    NewFlatMatrix(100, 64, 0.08),
		Wpe:    NewFlatMatrix(32, 64, 0.08),
		LmHead: NewFlatMatrix(100, 64, 0.08),
	}
	p.initAll()

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		// Zero gradients using cached slice
		for _, m := range p.all {
			for i := range m.Data {
				m.Data[i].Grad = 0
			}
		}
	}
}

// ============================================================================
// Benchmark: Parallel vs Sequential MatVec
// ============================================================================

func BenchmarkMatVec_Sequential(b *testing.B) {
	const size = 768
	m := NewFlatMatrix(size, size, 0.08)
	v := make([]float64, size)
	out := make([]float64, size)
	for i := range v {
		v[i] = rand.Float64()
	}

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		for i := 0; i < size; i++ {
			row := m.Row(i)
			sum := 0.0
			for j := 0; j < size; j++ {
				sum += row[j].Data * v[j]
			}
			out[i] = sum
		}
	}
}

func BenchmarkMatVec_Parallel_4Workers(b *testing.B) {
	const size = 768
	const numWorkers = 4
	m := NewFlatMatrix(size, size, 0.08)
	v := make([]float64, size)
	out := make([]float64, size)
	for i := range v {
		v[i] = rand.Float64()
	}

	rowsPerWorker := (size + numWorkers - 1) / numWorkers

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		var wg sync.WaitGroup
		for w := 0; w < numWorkers; w++ {
			wg.Add(1)
			go func(worker int) {
				defer wg.Done()
				start := worker * rowsPerWorker
				end := start + rowsPerWorker
				if end > size {
					end = size
				}

				for i := start; i < end; i++ {
					row := m.Row(i)
					sum := 0.0
					for j := 0; j < size; j++ {
						sum += row[j].Data * v[j]
					}
					out[i] = sum
				}
			}(w)
		}
		wg.Wait()
	}
}

func BenchmarkMatVec_Parallel_8Workers(b *testing.B) {
	const size = 768
	const numWorkers = 8
	m := NewFlatMatrix(size, size, 0.08)
	v := make([]float64, size)
	out := make([]float64, size)
	for i := range v {
		v[i] = rand.Float64()
	}

	rowsPerWorker := (size + numWorkers - 1) / numWorkers

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		var wg sync.WaitGroup
		for w := 0; w < numWorkers; w++ {
			wg.Add(1)
			go func(worker int) {
				defer wg.Done()
				start := worker * rowsPerWorker
				end := start + rowsPerWorker
				if end > size {
					end = size
				}

				for i := start; i < end; i++ {
					row := m.Row(i)
					sum := 0.0
					for j := 0; j < size; j++ {
						sum += row[j].Data * v[j]
					}
					out[i] = sum
				}
			}(w)
		}
		wg.Wait()
	}
}

// ============================================================================
// Benchmark: Memory Allocation
// ============================================================================

func BenchmarkAlloc_NewEveryTime(b *testing.B) {
	const size = 768

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		// Allocate new slice each iteration
		buf := make([]float64, size)
		for i := range buf {
			buf[i] = float64(i)
		}
	}
}

func BenchmarkAlloc_PreAllocated(b *testing.B) {
	const size = 768
	buf := make([]float64, size)

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		// Reuse pre-allocated slice
		for i := range buf {
			buf[i] = float64(i)
		}
	}
}

var bufPool = sync.Pool{
	New: func() interface{} {
		return make([]float64, 768)
	},
}

func BenchmarkAlloc_SyncPool(b *testing.B) {
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		buf := bufPool.Get().([]float64)
		for i := range buf {
			buf[i] = float64(i)
		}
		bufPool.Put(buf)
	}
}

// ============================================================================
// Helper
// ============================================================================

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
