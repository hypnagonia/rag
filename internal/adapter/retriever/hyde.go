package retriever

import (
	"fmt"

	"rag/internal/adapter/store"
	"rag/internal/domain"
	"rag/internal/port"
)

type HyDERetriever struct {
	llm         port.LLM
	embedder    port.Embedder
	vectorStore port.VectorStore
	chunkStore  *store.BoltStore
}

func NewHyDERetriever(
	llm port.LLM,
	embedder port.Embedder,
	vectorStore port.VectorStore,
	chunkStore *store.BoltStore,
) *HyDERetriever {
	return &HyDERetriever{
		llm:         llm,
		embedder:    embedder,
		vectorStore: vectorStore,
		chunkStore:  chunkStore,
	}
}

func (r *HyDERetriever) Search(query string, k int) ([]domain.ScoredChunk, error) {
	if r.llm == nil || r.embedder == nil || r.vectorStore == nil {
		return nil, fmt.Errorf("HyDE requires LLM, embedder, and vector store")
	}

	hypothetical, err := r.generateHypothetical(query)
	if err != nil {
		return nil, fmt.Errorf("failed to generate hypothetical: %w", err)
	}

	embeddings, err := r.embedder.Embed([]string{hypothetical})
	if err != nil {
		return nil, fmt.Errorf("failed to embed hypothetical: %w", err)
	}
	if len(embeddings) == 0 {
		return nil, fmt.Errorf("no embedding generated")
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

func (r *HyDERetriever) generateHypothetical(query string) (string, error) {
	systemPrompt := `You are a code documentation assistant. Given a question about code,
write a short code snippet or documentation excerpt that would answer the question.
Focus on being realistic - write code or documentation that might actually exist in a codebase.
Keep it concise (100-200 words max). Do not explain - just write the hypothetical code/docs.`

	userPrompt := fmt.Sprintf("Question: %s\n\nWrite a hypothetical code snippet or documentation that answers this:", query)

	return r.llm.GenerateWithSystem(systemPrompt, userPrompt)
}

func (r *HyDERetriever) SearchWithFallback(query string, k int) ([]domain.ScoredChunk, error) {

	results, err := r.Search(query, k)
	if err == nil && len(results) > 0 {
		return results, nil
	}

	if r.embedder == nil || r.vectorStore == nil {
		return nil, fmt.Errorf("no embedder or vector store available for fallback")
	}

	embeddings, err := r.embedder.Embed([]string{query})
	if err != nil {
		return nil, err
	}
	if len(embeddings) == 0 {
		return nil, nil
	}

	vectorResults, err := r.vectorStore.Search(embeddings[0], k)
	if err != nil {
		return nil, err
	}

	chunks := make([]domain.ScoredChunk, 0, len(vectorResults))
	for _, result := range vectorResults {
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
