package retriever

import (
	"sort"

	"rag/internal/adapter/store"
	"rag/internal/domain"
	"rag/internal/port"
)

// HybridRetriever combines BM25 lexical search with vector similarity search.
type HybridRetriever struct {
	bm25        *BM25Retriever
	vectorStore port.VectorStore
	embedder    port.Embedder
	chunkStore  *store.BoltStore
	rrfK        int     // RRF constant (typically 60)
	bm25Weight  float64 // Weight for BM25 results (0-1)
}

// NewHybridRetriever creates a new hybrid retriever.
func NewHybridRetriever(
	bm25 *BM25Retriever,
	vectorStore port.VectorStore,
	embedder port.Embedder,
	chunkStore *store.BoltStore,
	rrfK int,
	bm25Weight float64,
) *HybridRetriever {
	if rrfK <= 0 {
		rrfK = 60 // Standard default
	}
	if bm25Weight < 0 || bm25Weight > 1 {
		bm25Weight = 0.5 // Equal weighting
	}

	return &HybridRetriever{
		bm25:        bm25,
		vectorStore: vectorStore,
		embedder:    embedder,
		chunkStore:  chunkStore,
		rrfK:        rrfK,
		bm25Weight:  bm25Weight,
	}
}

// Search performs hybrid search combining BM25 and vector similarity.
func (r *HybridRetriever) Search(query string, k int) ([]domain.ScoredChunk, error) {
	// If no vector store or embedder, fall back to BM25 only
	if r.vectorStore == nil || r.embedder == nil {
		return r.bm25.Search(query, k)
	}

	// Get expanded candidate pool from both retrievers
	candidateK := k * 3
	if candidateK < 20 {
		candidateK = 20
	}

	// BM25 search
	bm25Results, err := r.bm25.Search(query, candidateK)
	if err != nil {
		// Fall back to vector-only if BM25 fails
		return r.vectorOnlySearch(query, k)
	}

	// Vector search
	vectorResults, err := r.vectorSearch(query, candidateK)
	if err != nil {
		// Fall back to BM25-only if vector search fails
		return bm25Results[:min(k, len(bm25Results))], nil
	}

	// Fuse results using Reciprocal Rank Fusion
	fused := r.rrfFuse(bm25Results, vectorResults)

	// Limit to k results
	if len(fused) > k {
		fused = fused[:k]
	}

	return fused, nil
}

// vectorSearch performs vector similarity search.
func (r *HybridRetriever) vectorSearch(query string, k int) ([]domain.ScoredChunk, error) {
	// Embed the query
	embeddings, err := r.embedder.Embed([]string{query})
	if err != nil {
		return nil, err
	}
	if len(embeddings) == 0 {
		return nil, nil
	}

	// Search vector store
	results, err := r.vectorStore.Search(embeddings[0], k)
	if err != nil {
		return nil, err
	}

	// Convert to ScoredChunks
	chunks := make([]domain.ScoredChunk, 0, len(results))
	for _, result := range results {
		chunk, err := r.chunkStore.GetChunk(result.ID)
		if err != nil {
			continue
		}
		chunks = append(chunks, domain.ScoredChunk{
			Chunk: chunk,
			Score: result.Score,
		})
	}

	return chunks, nil
}

// vectorOnlySearch performs search using only vectors.
func (r *HybridRetriever) vectorOnlySearch(query string, k int) ([]domain.ScoredChunk, error) {
	return r.vectorSearch(query, k)
}

// rrfFuse combines results using Reciprocal Rank Fusion.
// RRF score = Σ 1/(k + rank) for each result list where the document appears.
func (r *HybridRetriever) rrfFuse(bm25Results, vectorResults []domain.ScoredChunk) []domain.ScoredChunk {
	rrfScores := make(map[string]float64)
	chunkMap := make(map[string]domain.Chunk)

	// Score BM25 results
	for rank, result := range bm25Results {
		rrfScores[result.Chunk.ID] += r.bm25Weight / float64(r.rrfK+rank+1)
		chunkMap[result.Chunk.ID] = result.Chunk
	}

	// Score vector results
	vectorWeight := 1.0 - r.bm25Weight
	for rank, result := range vectorResults {
		rrfScores[result.Chunk.ID] += vectorWeight / float64(r.rrfK+rank+1)
		if _, exists := chunkMap[result.Chunk.ID]; !exists {
			chunkMap[result.Chunk.ID] = result.Chunk
		}
	}

	// Convert to scored chunks
	fused := make([]domain.ScoredChunk, 0, len(rrfScores))
	for id, score := range rrfScores {
		fused = append(fused, domain.ScoredChunk{
			Chunk: chunkMap[id],
			Score: score,
		})
	}

	// Sort by RRF score descending
	sort.Slice(fused, func(i, j int) bool {
		return fused[i].Score > fused[j].Score
	})

	return fused
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
