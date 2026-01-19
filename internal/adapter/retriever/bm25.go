package retriever

import (
	"math"
	"sort"

	"rag/internal/adapter/analyzer"
	"rag/internal/adapter/store"
	"rag/internal/domain"
)

// BM25Retriever implements BM25 scoring for retrieval.
type BM25Retriever struct {
	store     *store.BoltStore
	tokenizer *analyzer.Tokenizer
	k1        float64
	b         float64
}

// NewBM25Retriever creates a new BM25 retriever.
func NewBM25Retriever(store *store.BoltStore, tokenizer *analyzer.Tokenizer, k1, b float64) *BM25Retriever {
	return &BM25Retriever{
		store:     store,
		tokenizer: tokenizer,
		k1:        k1,
		b:         b,
	}
}

// Search finds chunks matching the query using BM25 scoring.
func (r *BM25Retriever) Search(query string, k int) ([]domain.ScoredChunk, error) {
	queryTokens := r.tokenizer.Tokenize(query)
	if len(queryTokens) == 0 {
		return nil, nil
	}

	stats, err := r.store.GetStats()
	if err != nil {
		return nil, err
	}

	if stats.TotalChunks == 0 {
		return nil, nil
	}

	// Collect all matching chunks and their scores
	chunkScores := make(map[string]float64)
	chunkLengths := make(map[string]int)

	for _, term := range queryTokens {
		postings, err := r.store.GetPostings(term)
		if err != nil {
			continue
		}

		// Calculate IDF for this term
		n := float64(len(postings))
		N := float64(stats.TotalChunks)
		idf := math.Log((N-n+0.5)/(n+0.5) + 1)

		for _, posting := range postings {
			// Get chunk length if not cached
			if _, exists := chunkLengths[posting.ChunkID]; !exists {
				chunk, err := r.store.GetChunk(posting.ChunkID)
				if err != nil {
					continue
				}
				chunkLengths[posting.ChunkID] = len(chunk.Tokens)
			}

			// BM25 term score
			dl := float64(chunkLengths[posting.ChunkID])
			avgDl := stats.AvgChunkLen
			tf := float64(posting.TF)

			// BM25 formula: IDF * (tf * (k1+1)) / (tf + k1 * (1-b + b*dl/avgDl))
			score := idf * (tf * (r.k1 + 1)) / (tf + r.k1*(1-r.b+r.b*dl/avgDl))
			chunkScores[posting.ChunkID] += score
		}
	}

	// Convert to scored chunks and sort
	results := make([]domain.ScoredChunk, 0, len(chunkScores))
	for chunkID, score := range chunkScores {
		chunk, err := r.store.GetChunk(chunkID)
		if err != nil {
			continue
		}
		results = append(results, domain.ScoredChunk{
			Chunk: chunk,
			Score: score,
		})
	}

	// Sort by score descending
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Limit to k results
	if len(results) > k {
		results = results[:k]
	}

	return results, nil
}

// ComputeBM25Score computes BM25 score for a single chunk against query tokens.
func ComputeBM25Score(queryTokens []string, chunk domain.Chunk, stats domain.Stats, k1, b float64) float64 {
	// Build term frequency map for the chunk
	chunkTF := make(map[string]int)
	for _, token := range chunk.Tokens {
		chunkTF[token]++
	}

	score := 0.0
	dl := float64(len(chunk.Tokens))
	avgDl := stats.AvgChunkLen
	N := float64(stats.TotalChunks)

	for _, term := range queryTokens {
		tf, exists := chunkTF[term]
		if !exists {
			continue
		}

		// For IDF, we'd need the document frequency, but for a simplified version
		// we'll assume uniform distribution. In production, you'd look this up.
		n := 1.0 // Simplified: assume term appears in at least 1 doc
		idf := math.Log((N-n+0.5)/(n+0.5) + 1)

		tfFloat := float64(tf)
		termScore := idf * (tfFloat * (k1 + 1)) / (tfFloat + k1*(1-b+b*dl/avgDl))
		score += termScore
	}

	return score
}
