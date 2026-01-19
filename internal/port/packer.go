package port

import "rag/internal/domain"

type Packer interface {
	Pack(query string, chunks []domain.ScoredChunk, budget int) (domain.PackedContext, error)
}
