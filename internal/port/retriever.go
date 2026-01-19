package port

import "rag/internal/domain"

// Retriever defines the interface for searching indexed content.
type Retriever interface {
	// Search searches for chunks matching the query and returns top-k results.
	Search(query string, k int) ([]domain.ScoredChunk, error)
}
