package retriever

import (
	"sort"

	"rag/internal/adapter/store"
	"rag/internal/domain"
	"rag/internal/port"
)

type HybridRetriever struct {
	bm25        *BM25Retriever
	vectorStore port.VectorStore
	embedder    port.Embedder
	chunkStore  *store.BoltStore
	rrfK        int
	bm25Weight  float64
}

func NewHybridRetriever(
	bm25 *BM25Retriever,
	vectorStore port.VectorStore,
	embedder port.Embedder,
	chunkStore *store.BoltStore,
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

	candidateK := k * 3
	if candidateK < 20 {
		candidateK = 20
	}

	bm25Results, err := r.bm25.Search(query, candidateK)
	if err != nil {

		return r.vectorOnlySearch(query, k)
	}

	vectorResults, err := r.vectorSearch(query, candidateK)
	if err != nil {

		return bm25Results[:min(k, len(bm25Results))], nil
	}

	fused := r.rrfFuse(bm25Results, vectorResults)

	if len(fused) > k {
		fused = fused[:k]
	}

	return fused, nil
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

func (r *HybridRetriever) rrfFuse(bm25Results, vectorResults []domain.ScoredChunk) []domain.ScoredChunk {
	rrfScores := make(map[string]float64)
	chunkMap := make(map[string]domain.Chunk)

	for rank, result := range bm25Results {
		rrfScores[result.Chunk.ID] += r.bm25Weight / float64(r.rrfK+rank+1)
		chunkMap[result.Chunk.ID] = result.Chunk
	}

	vectorWeight := 1.0 - r.bm25Weight
	for rank, result := range vectorResults {
		rrfScores[result.Chunk.ID] += vectorWeight / float64(r.rrfK+rank+1)
		if _, exists := chunkMap[result.Chunk.ID]; !exists {
			chunkMap[result.Chunk.ID] = result.Chunk
		}
	}

	fused := make([]domain.ScoredChunk, 0, len(rrfScores))
	for id, score := range rrfScores {
		fused = append(fused, domain.ScoredChunk{
			Chunk: chunkMap[id],
			Score: score,
		})
	}

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
