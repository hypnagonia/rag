package retriever

import (
	"sort"

	"rag/internal/domain"
	"rag/internal/port"
)

type HybridRetriever struct {
	bm25        *BM25Retriever
	vectorStore port.VectorStore
	embedder    port.Embedder
	chunkStore  port.IndexStore
	rrfK        int
	bm25Weight  float64
}

func NewHybridRetriever(
	bm25 *BM25Retriever,
	vectorStore port.VectorStore,
	embedder port.Embedder,
	chunkStore port.IndexStore,
	rrfK int,
	bm25Weight float64,
) *HybridRetriever {
	if rrfK <= 0 {
		rrfK = 60
	}
	if bm25Weight < 0 || bm25Weight > 1 {
		bm25Weight = 0.5
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

func (r *HybridRetriever) Search(query string, k int) ([]domain.ScoredChunk, error) {
	if r.vectorStore == nil || r.embedder == nil {
		return r.bm25.Search(query, k)
	}

	candidateK := k * 10
	if candidateK < 50 {
		candidateK = 50
	}

	bm25Results, err := r.bm25.Search(query, candidateK)
	if err != nil || len(bm25Results) == 0 {
		return r.vectorOnlySearch(query, k)
	}

	queryEmbedding, err := r.embedder.Embed([]string{query})
	if err != nil || len(queryEmbedding) == 0 {
		return bm25Results[:min(k, len(bm25Results))], nil
	}

	chunkIDs := make([]string, len(bm25Results))
	for i, result := range bm25Results {
		chunkIDs[i] = result.Chunk.ID
	}

	vectorScores, err := r.vectorStore.SearchSubset(queryEmbedding[0], chunkIDs)
	if err != nil {
		return bm25Results[:min(k, len(bm25Results))], nil
	}

	vectorScoreMap := make(map[string]float64)
	for _, vs := range vectorScores {
		vectorScoreMap[vs.ID] = vs.Score
	}

	reranked := r.combineScores(bm25Results, vectorScoreMap)

	if len(reranked) > k {
		reranked = reranked[:k]
	}

	return reranked, nil
}

func (r *HybridRetriever) vectorSearch(query string, k int) ([]domain.ScoredChunk, error) {

	embeddings, err := r.embedder.Embed([]string{query})
	if err != nil {
		return nil, err
	}
	if len(embeddings) == 0 {
		return nil, nil
	}

	results, err := r.vectorStore.Search(embeddings[0], k)
	if err != nil {
		return nil, err
	}

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

func (r *HybridRetriever) vectorOnlySearch(query string, k int) ([]domain.ScoredChunk, error) {
	return r.vectorSearch(query, k)
}

func (r *HybridRetriever) combineScores(bm25Results []domain.ScoredChunk, vectorScores map[string]float64) []domain.ScoredChunk {
	if len(bm25Results) == 0 {
		return nil
	}

	maxBM25 := bm25Results[0].Score
	minBM25 := bm25Results[len(bm25Results)-1].Score
	bm25Range := maxBM25 - minBM25
	if bm25Range == 0 {
		bm25Range = 1
	}

	vectorWeight := 1.0 - r.bm25Weight
	results := make([]domain.ScoredChunk, 0, len(bm25Results))

	for _, result := range bm25Results {
		normalizedBM25 := (result.Score - minBM25) / bm25Range

		vectorScore := vectorScores[result.Chunk.ID]

		combinedScore := r.bm25Weight*normalizedBM25 + vectorWeight*vectorScore

		results = append(results, domain.ScoredChunk{
			Chunk: result.Chunk,
			Score: combinedScore,
		})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
