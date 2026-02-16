package data

import (
	"bufio"
	"io"
	"math/rand/v2"
	"net/http"
	"os"
	"strings"
)

// LoadDataset reads lines from a file, returning non-empty trimmed lines
func LoadDataset(path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var lines []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			lines = append(lines, line)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return lines, nil
}

// DownloadIfNotExists downloads a URL to path if path doesn't exist
func DownloadIfNotExists(url, path string) error {
	if _, err := os.Stat(path); err == nil {
		return nil // File already exists
	}

	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	_, err = io.Copy(file, resp.Body)
	return err
}

// Shuffle randomly shuffles docs in place with a seed
func Shuffle(docs []string, seed uint64) {
	rng := rand.New(rand.NewPCG(seed, 0))
	rng.Shuffle(len(docs), func(i, j int) {
		docs[i], docs[j] = docs[j], docs[i]
	})
}
