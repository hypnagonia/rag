package port

import "rag/internal/domain"

type IndexStore interface {
	PutDoc(doc domain.Document) error

	GetDoc(id string) (domain.Document, error)

	DeleteDoc(id string) error

	ListDocs() ([]domain.Document, error)

	PutChunk(chunk domain.Chunk) error

	GetChunk(id string) (domain.Chunk, error)

	GetChunksByDoc(docID string) ([]domain.Chunk, error)

	DeleteChunksByDoc(docID string) error

	PutPosting(term string, chunkID string, tf int) error

	GetPostings(term string) ([]domain.Posting, error)

	DeletePostings(chunkID string, terms []string) error

	GetStats() (domain.Stats, error)

	UpdateStats(stats domain.Stats) error

	BatchIndex(files []IndexedFile) error

	Close() error
}

type IndexedFile struct {
	Doc      domain.Document
	Chunks   []domain.Chunk
	Postings map[string]map[string]int
}
