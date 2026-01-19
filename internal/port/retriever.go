package port

import "rag/internal/domain"

type Retriever interface {
	Search(query string, k int) ([]domain.ScoredChunk, error)
}
