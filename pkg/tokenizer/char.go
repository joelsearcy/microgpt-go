package tokenizer

import (
	"slices"
	"strings"
)

// CharTokenizer handles character-level tokenization
type CharTokenizer struct {
	runeToID map[rune]int
	idToRune []rune
	BOS      int // Beginning/End of Sequence token ID
}

// NewCharTokenizer builds vocabulary from documents
func NewCharTokenizer(docs []string) *CharTokenizer {
	// Collect all unique runes from docs
	runeSet := make(map[rune]struct{})
	for _, doc := range docs {
		for _, r := range doc {
			runeSet[r] = struct{}{}
		}
	}

	// Convert to slice and sort for deterministic ordering
	runes := make([]rune, 0, len(runeSet))
	for r := range runeSet {
		runes = append(runes, r)
	}
	slices.Sort(runes)

	// Build bidirectional mapping
	runeToID := make(map[rune]int, len(runes))
	for i, r := range runes {
		runeToID[r] = i
	}

	return &CharTokenizer{
		runeToID: runeToID,
		idToRune: runes,
		BOS:      len(runes),
	}
}

// VocabSize returns total vocabulary size including BOS
func (t *CharTokenizer) VocabSize() int {
	return len(t.idToRune) + 1
}

// Encode converts string to token IDs wrapped with BOS
// Returns [BOS, char_ids..., BOS]
func (t *CharTokenizer) Encode(s string) []int {
	runes := []rune(s)
	tokens := make([]int, len(runes)+2)
	tokens[0] = t.BOS
	for i, r := range runes {
		tokens[i+1] = t.runeToID[r]
	}
	tokens[len(tokens)-1] = t.BOS
	return tokens
}

// Decode converts token IDs back to string (skips BOS tokens)
func (t *CharTokenizer) Decode(tokens []int) string {
	var builder strings.Builder
	builder.Grow(len(tokens))
	for _, id := range tokens {
		if id == t.BOS {
			continue
		}
		if id >= 0 && id < len(t.idToRune) {
			builder.WriteRune(t.idToRune[id])
		}
	}
	return builder.String()
}
