package port

import "rag/internal/domain"

type DiversityReranker interface {
	Rerank(chunks []domain.ScoredChunk, k int) []domain.ScoredChunk
}
