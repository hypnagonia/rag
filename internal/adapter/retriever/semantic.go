package retriever

import (
	"fmt"

	"rag/internal/domain"
	"rag/internal/port"
)

type SemanticRetriever struct {
	vectorStore port.VectorStore
	embedder    port.Embedder
	chunkStore  port.IndexStore
}

func NewSemanticRetriever(
	vectorStore port.VectorStore,
	embedder port.Embedder,
	chunkStore port.IndexStore,
) *SemanticRetriever {
	return &SemanticRetriever{
		vectorStore: vectorStore,
		embedder:    embedder,
		chunkStore:  chunkStore,
	}
}

func (r *SemanticRetriever) Search(query string, k int) ([]domain.ScoredChunk, error) {
	if r.vectorStore == nil || r.embedder == nil {
		return nil, fmt.Errorf("semantic search not available: embeddings not configured")
	}

	embeddings, err := r.embedder.Embed([]string{query})
	if err != nil {
		return nil, fmt.Errorf("failed to embed query: %w", err)
	}
	if len(embeddings) == 0 {
		return nil, fmt.Errorf("embedding returned empty result")
	}

	results, err := r.vectorStore.Search(embeddings[0], k)
	if err != nil {
		return nil, fmt.Errorf("vector search failed: %w", err)
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
