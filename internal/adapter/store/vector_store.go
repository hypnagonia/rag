package store

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"sync"

	"go.etcd.io/bbolt"
	"rag/internal/port"
)

var (
	bucketVectors = []byte("vectors")
)

// BoltVectorStore implements VectorStore using BoltDB for persistence.
// Uses brute-force search for simplicity; can be replaced with HNSW for larger indexes.
type BoltVectorStore struct {
	db        *bbolt.DB
	dimension int
	mu        sync.RWMutex
	// In-memory cache for fast search
	vectors map[string]vectorEntry
}

type vectorEntry struct {
	vector   []float32
	metadata map[string]string
}

type storedVector struct {
	Vector   []float32         `json:"v"`
	Metadata map[string]string `json:"m,omitempty"`
}

// NewBoltVectorStore creates a new BoltDB-backed vector store.
func NewBoltVectorStore(db *bbolt.DB, dimension int) (*BoltVectorStore, error) {
	// Create bucket
	err := db.Update(func(tx *bbolt.Tx) error {
		_, err := tx.CreateBucketIfNotExists(bucketVectors)
		return err
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create vectors bucket: %w", err)
	}

	store := &BoltVectorStore{
		db:        db,
		dimension: dimension,
		vectors:   make(map[string]vectorEntry),
	}

	// Load existing vectors into memory
	if err := store.loadVectors(); err != nil {
		return nil, fmt.Errorf("failed to load vectors: %w", err)
	}

	return store, nil
}

// loadVectors loads all vectors from BoltDB into memory.
func (s *BoltVectorStore) loadVectors() error {
	return s.db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket(bucketVectors)
		if b == nil {
			return nil
		}

		return b.ForEach(func(k, v []byte) error {
			var stored storedVector
			if err := json.Unmarshal(v, &stored); err != nil {
				return nil // Skip corrupted entries
			}
			s.vectors[string(k)] = vectorEntry{
				vector:   stored.Vector,
				metadata: stored.Metadata,
			}
			return nil
		})
	})
}

// Upsert adds or updates vectors in the store.
func (s *BoltVectorStore) Upsert(items []port.VectorItem) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.db.Update(func(tx *bbolt.Tx) error {
		b := tx.Bucket(bucketVectors)
		if b == nil {
			return fmt.Errorf("vectors bucket not found")
		}

		for _, item := range items {
			if len(item.Vector) != s.dimension {
				return fmt.Errorf("vector dimension mismatch: expected %d, got %d", s.dimension, len(item.Vector))
			}

			stored := storedVector{
				Vector:   item.Vector,
				Metadata: item.Metadata,
			}
			data, err := json.Marshal(stored)
			if err != nil {
				return err
			}

			if err := b.Put([]byte(item.ID), data); err != nil {
				return err
			}

			// Update in-memory cache
			s.vectors[item.ID] = vectorEntry{
				vector:   item.Vector,
				metadata: item.Metadata,
			}
		}

		return nil
	})
}

// Search finds the k nearest vectors to the query using cosine similarity.
func (s *BoltVectorStore) Search(query []float32, k int) ([]port.VectorResult, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if len(query) != s.dimension {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d", s.dimension, len(query))
	}

	if len(s.vectors) == 0 {
		return nil, nil
	}

	// Calculate similarity for all vectors (brute force)
	type scored struct {
		id       string
		score    float64
		metadata map[string]string
	}

	scores := make([]scored, 0, len(s.vectors))
	for id, entry := range s.vectors {
		sim := cosineSimilarity(query, entry.vector)
		scores = append(scores, scored{
			id:       id,
			score:    sim,
			metadata: entry.metadata,
		})
	}

	// Sort by score descending
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	// Return top-k
	if k > len(scores) {
		k = len(scores)
	}

	results := make([]port.VectorResult, k)
	for i := 0; i < k; i++ {
		results[i] = port.VectorResult{
			ID:       scores[i].id,
			Score:    scores[i].score,
			Metadata: scores[i].metadata,
		}
	}

	return results, nil
}

// Delete removes vectors by their IDs.
func (s *BoltVectorStore) Delete(ids []string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.db.Update(func(tx *bbolt.Tx) error {
		b := tx.Bucket(bucketVectors)
		if b == nil {
			return nil
		}

		for _, id := range ids {
			if err := b.Delete([]byte(id)); err != nil {
				return err
			}
			delete(s.vectors, id)
		}

		return nil
	})
}

// Count returns the number of vectors in the store.
func (s *BoltVectorStore) Count() (int, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.vectors), nil
}

// cosineSimilarity calculates the cosine similarity between two vectors.
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}
