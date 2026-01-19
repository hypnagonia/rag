package port

import "rag/internal/domain"

// Packer defines the interface for packing chunks into compressed context.
type Packer interface {
	// Pack packs the scored chunks into a context that fits the token budget.
	Pack(query string, chunks []domain.ScoredChunk, budget int) (domain.PackedContext, error)
}
