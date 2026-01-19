package usecase

import (
	"rag/internal/adapter/retriever"
	"rag/internal/domain"
	"rag/internal/port"
)

// RetrieveUseCase handles search and retrieval operations.
type RetrieveUseCase struct {
	retriever         port.Retriever
	mmrReranker       *retriever.MMRReranker
	minScoreThreshold float64 // Filter results below this score (0 = disabled)
}

// NewRetrieveUseCase creates a new retrieve use case.
func NewRetrieveUseCase(
	retriever port.Retriever,
	mmrReranker *retriever.MMRReranker,
	minScoreThreshold float64,
) *RetrieveUseCase {
	return &RetrieveUseCase{
		retriever:         retriever,
		mmrReranker:       mmrReranker,
		minScoreThreshold: minScoreThreshold,
	}
}

// Retrieve searches for chunks matching the query.
func (u *RetrieveUseCase) Retrieve(query string, topK int) ([]domain.ScoredChunk, error) {

	candidates, err := u.retriever.Search(query, topK*2)
	if err != nil {
		return nil, err
	}

	if len(candidates) == 0 {
		return nil, nil
	}

	results := u.mmrReranker.Rerank(candidates, topK)

	if u.minScoreThreshold > 0 {
		results = u.filterByThreshold(results)
	}

	return results, nil
}

// filterByThreshold removes results below the minimum score threshold.
func (u *RetrieveUseCase) filterByThreshold(results []domain.ScoredChunk) []domain.ScoredChunk {
	filtered := make([]domain.ScoredChunk, 0, len(results))
	for _, r := range results {
		if r.Score >= u.minScoreThreshold {
			filtered = append(filtered, r)
		}
	}
	return filtered
}

// RetrieveWithoutMMR searches without MMR reranking (for testing/debugging).
func (u *RetrieveUseCase) RetrieveWithoutMMR(query string, topK int) ([]domain.ScoredChunk, error) {
	return u.retriever.Search(query, topK)
}

// ScoredChunkResult is a simplified result for CLI output.
type ScoredChunkResult struct {
	Path      string  `json:"path"`
	StartLine int     `json:"start_line"`
	EndLine   int     `json:"end_line"`
	Score     float64 `json:"score"`
	Text      string  `json:"text"`
}
