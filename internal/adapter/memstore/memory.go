package memstore

import (
	"fmt"
	"sync"

	"rag/internal/domain"
	"rag/internal/port"
)

type MemoryStore struct {
	mu        sync.RWMutex
	docs      map[string]domain.Document
	chunks    map[string]domain.Chunk
	docChunks map[string][]string
	postings  map[string][]domain.Posting
	stats     domain.Stats
}

func NewMemoryStore() *MemoryStore {
	return &MemoryStore{
		docs:      make(map[string]domain.Document),
		chunks:    make(map[string]domain.Chunk),
		docChunks: make(map[string][]string),
		postings:  make(map[string][]domain.Posting),
	}
}

func (s *MemoryStore) PutDoc(doc domain.Document) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.docs[doc.ID] = doc
	return nil
}

func (s *MemoryStore) GetDoc(id string) (domain.Document, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	doc, ok := s.docs[id]
	if !ok {
		return domain.Document{}, fmt.Errorf("document not found: %s", id)
	}
	return doc, nil
}

func (s *MemoryStore) DeleteDoc(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.docs, id)
	return nil
}

func (s *MemoryStore) ListDocs() ([]domain.Document, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	docs := make([]domain.Document, 0, len(s.docs))
	for _, doc := range s.docs {
		docs = append(docs, doc)
	}
	return docs, nil
}

func (s *MemoryStore) PutChunk(chunk domain.Chunk) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.chunks[chunk.ID] = chunk
	s.docChunks[chunk.DocID] = append(s.docChunks[chunk.DocID], chunk.ID)
	return nil
}

func (s *MemoryStore) GetChunk(id string) (domain.Chunk, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	chunk, ok := s.chunks[id]
	if !ok {
		return domain.Chunk{}, fmt.Errorf("chunk not found: %s", id)
	}
	return chunk, nil
}

func (s *MemoryStore) GetChunksByDoc(docID string) ([]domain.Chunk, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	chunkIDs := s.docChunks[docID]
	chunks := make([]domain.Chunk, 0, len(chunkIDs))
	for _, id := range chunkIDs {
		if chunk, ok := s.chunks[id]; ok {
			chunks = append(chunks, chunk)
		}
	}
	return chunks, nil
}

func (s *MemoryStore) DeleteChunksByDoc(docID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	chunkIDs := s.docChunks[docID]
	for _, id := range chunkIDs {
		delete(s.chunks, id)
	}
	delete(s.docChunks, docID)
	return nil
}

func (s *MemoryStore) PutPosting(term string, chunkID string, tf int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.postings[term] = append(s.postings[term], domain.Posting{
		ChunkID: chunkID,
		TF:      tf,
	})
	return nil
}

func (s *MemoryStore) GetPostings(term string) ([]domain.Posting, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.postings[term], nil
}

func (s *MemoryStore) DeletePostings(chunkID string, terms []string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, term := range terms {
		filtered := make([]domain.Posting, 0)
		for _, p := range s.postings[term] {
			if p.ChunkID != chunkID {
				filtered = append(filtered, p)
			}
		}
		if len(filtered) == 0 {
			delete(s.postings, term)
		} else {
			s.postings[term] = filtered
		}
	}
	return nil
}

func (s *MemoryStore) GetStats() (domain.Stats, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.stats, nil
}

func (s *MemoryStore) UpdateStats(stats domain.Stats) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.stats = stats
	return nil
}

func (s *MemoryStore) BatchIndex(files []port.IndexedFile) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, file := range files {
		s.docs[file.Doc.ID] = file.Doc

		for _, chunk := range file.Chunks {
			s.chunks[chunk.ID] = chunk
			s.docChunks[chunk.DocID] = append(s.docChunks[chunk.DocID], chunk.ID)
		}

		for term, chunkPostings := range file.Postings {
			for chunkID, tf := range chunkPostings {
				s.postings[term] = append(s.postings[term], domain.Posting{
					ChunkID: chunkID,
					TF:      tf,
				})
			}
		}
	}

	return nil
}

func (s *MemoryStore) Close() error {
	return nil
}
