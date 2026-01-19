package port

import "rag/internal/domain"

// Chunker defines the interface for splitting documents into chunks.
type Chunker interface {
	// Chunk splits a document's content into chunks.
	Chunk(doc domain.Document, content string) ([]domain.Chunk, error)
}
