package port

import "rag/internal/domain"

// IndexStore defines the interface for the index storage.
type IndexStore interface {
	// PutDoc stores a document.
	PutDoc(doc domain.Document) error
	// GetDoc retrieves a document by ID.
	GetDoc(id string) (domain.Document, error)
	// DeleteDoc deletes a document and its associated chunks.
	DeleteDoc(id string) error
	// ListDocs lists all documents.
	ListDocs() ([]domain.Document, error)

	// PutChunk stores a chunk.
	PutChunk(chunk domain.Chunk) error
	// GetChunk retrieves a chunk by ID.
	GetChunk(id string) (domain.Chunk, error)
	// GetChunksByDoc retrieves all chunks for a document.
	GetChunksByDoc(docID string) ([]domain.Chunk, error)
	// DeleteChunksByDoc deletes all chunks for a document.
	DeleteChunksByDoc(docID string) error

	// PutPosting adds a term posting.
	PutPosting(term string, chunkID string, tf int) error
	// GetPostings retrieves all postings for a term.
	GetPostings(term string) ([]domain.Posting, error)
	// DeletePostings deletes all postings for a chunk.
	DeletePostings(chunkID string, terms []string) error

	// GetStats retrieves corpus statistics.
	GetStats() (domain.Stats, error)
	// UpdateStats updates corpus statistics.
	UpdateStats(stats domain.Stats) error

	// Close closes the store.
	Close() error
}
