package main

import (
	"fmt"
	"math/rand/v2"
	"os"
	"runtime/pprof"

	"github.com/joelsearcy/microgpt-go/pkg/autograd"
	"github.com/joelsearcy/microgpt-go/pkg/data"
	"github.com/joelsearcy/microgpt-go/pkg/model"
	"github.com/joelsearcy/microgpt-go/pkg/optim"
	"github.com/joelsearcy/microgpt-go/pkg/tokenizer"
)

const (
	// Model hyperparameters
	NLayer    = 1
	NEmbd     = 16
	BlockSize = 16
	NHead     = 4

	// Training hyperparameters
	LearningRate = 0.01
	Beta1        = 0.85
	Beta2        = 0.99
	EpsAdam      = 1e-8
	NumSteps     = 100 // Reduced from 1000 for faster profiling

	// Data
	Seed     = 42
	DataURL  = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
	DataPath = "input.txt"
)

func main() {
	// 1. Start CPU profiling
	cpuFile, err := os.Create("cpu.prof")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating CPU profile: %v\n", err)
		os.Exit(1)
	}
	defer cpuFile.Close()

	if err := pprof.StartCPUProfile(cpuFile); err != nil {
		fmt.Fprintf(os.Stderr, "Error starting CPU profile: %v\n", err)
		os.Exit(1)
	}
	defer pprof.StopCPUProfile()

	fmt.Println("CPU profiling enabled - writing to cpu.prof")

	// 2. Load dataset
	if err := data.DownloadIfNotExists(DataURL, DataPath); err != nil {
		fmt.Fprintf(os.Stderr, "Error downloading data: %v\n", err)
		os.Exit(1)
	}

	docs, err := data.LoadDataset(DataPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading data: %v\n", err)
		os.Exit(1)
	}

	// Shuffle dataset
	data.Shuffle(docs, Seed)
	fmt.Printf("num docs: %d\n", len(docs))

	// 3. Create tokenizer
	tok := tokenizer.NewCharTokenizer(docs)
	fmt.Printf("vocab size: %d\n", tok.VocabSize())

	// 4. Initialize model
	rng := rand.New(rand.NewPCG(Seed, Seed))
	gpt := model.NewGPT(tok.VocabSize(), BlockSize, NEmbd, NLayer, NHead, rng)

	params := gpt.Params.AllParams()
	fmt.Printf("num params: %d\n", len(params))

	// 5. Initialize optimizer
	optimizer := optim.NewAdam(len(params), LearningRate, Beta1, Beta2, EpsAdam)

	// 6. Training loop
	fmt.Printf("Running %d training steps for profiling...\n", NumSteps)
	for step := 0; step < NumSteps; step++ {
		// Get document for this step
		doc := docs[step%len(docs)]
		tokens := tok.Encode(doc)
		n := min(BlockSize, len(tokens)-1)

		// Forward pass
		cache := model.NewKVCache(NLayer, BlockSize)
		var losses []*autograd.Value

		for posID := 0; posID < n; posID++ {
			tokenID := tokens[posID]
			targetID := tokens[posID+1]

			logits := gpt.Forward(tokenID, posID, cache)
			probs := model.Softmax(logits)

			// Cross-entropy loss: -log(prob[target])
			lossT := probs[targetID].Log().Neg()
			losses = append(losses, lossT)
		}

		// Average loss
		loss := autograd.NewValue(0)
		for _, l := range losses {
			loss = loss.Add(l)
		}
		loss = loss.Mul(autograd.NewValue(1.0 / float64(n)))

		// Backward pass
		loss.Backward()

		// Optimizer step with linear LR decay
		lrDecay := 1.0 - float64(step)/float64(NumSteps)
		optimizer.Step(params, lrDecay)

		// Zero gradients
		gpt.Params.ZeroGrads()

		// Print progress
		if step%10 == 0 || step == NumSteps-1 {
			fmt.Printf("\rstep %4d / %4d | loss %.4f", step+1, NumSteps, loss.Data)
		}
	}
	fmt.Println()

	// 7. Write memory profile
	memFile, err := os.Create("mem.prof")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating memory profile: %v\n", err)
		os.Exit(1)
	}
	defer memFile.Close()

	if err := pprof.WriteHeapProfile(memFile); err != nil {
		fmt.Fprintf(os.Stderr, "Error writing memory profile: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("Memory profiling complete - written to mem.prof")
	fmt.Println("\nTo analyze profiles:")
	fmt.Println("  go tool pprof cpu.prof")
	fmt.Println("  go tool pprof mem.prof")
	fmt.Println("\nOr view in browser:")
	fmt.Println("  go tool pprof -http=:8080 cpu.prof")
}
