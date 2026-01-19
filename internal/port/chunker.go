package port

import "rag/internal/domain"

type Chunker interface {
	Chunk(doc domain.Document, content string) ([]domain.Chunk, error)
}
