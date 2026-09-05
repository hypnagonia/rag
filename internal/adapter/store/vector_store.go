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
	bucketVectors     = []byte("vectors")
	bucketVectorMeta  = []byte("vector_meta")
	keyVectorMetaInfo = []byte("info")
)

type BoltVectorStore struct {
	db        *bbolt.DB
	dimension int
	mu        sync.RWMutex

	vectors map[string]vectorEntry
}

type vectorEntry struct {
	vector   []float32
	norm     float64
	metadata map[string]string
}

type storedVector struct {
	Vector   []float32         `json:"v"`
	Metadata map[string]string `json:"m,omitempty"`
}

type VectorMeta struct {
	Model     string `json:"model"`
	Dimension int    `json:"dimension"`
}

func NewBoltVectorStore(db *bbolt.DB, dimension int) (*BoltVectorStore, error) {
	if dimension <= 0 {
		return nil, fmt.Errorf("vector store dimension must be positive, got %d", dimension)
	}

	err := db.Update(func(tx *bbolt.Tx) error {
		if _, err := tx.CreateBucketIfNotExists(bucketVectors); err != nil {
			return err
		}
		_, err := tx.CreateBucketIfNotExists(bucketVectorMeta)
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

	if err := store.loadVectors(); err != nil {
		return nil, fmt.Errorf("failed to load vectors: %w", err)
	}

	return store, nil
}

func (s *BoltVectorStore) loadVectors() error {
	return s.db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket(bucketVectors)
		if b == nil {
			return nil
		}

		return b.ForEach(func(k, v []byte) error {
			var stored storedVector
			if err := json.Unmarshal(v, &stored); err != nil {
				return nil
			}
			if len(stored.Vector) != s.dimension {
				return nil
			}
			s.vectors[string(k)] = vectorEntry{
				vector:   stored.Vector,
				norm:     l2Norm(stored.Vector),
				metadata: stored.Metadata,
			}
			return nil
		})
	})
}

func (s *BoltVectorStore) Meta() (*VectorMeta, error) {
	var meta *VectorMeta
	err := s.db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket(bucketVectorMeta)
		if b == nil {
			return nil
		}
		data := b.Get(keyVectorMetaInfo)
		if data == nil {
			return nil
		}
		var m VectorMeta
		if err := json.Unmarshal(data, &m); err != nil {
			return err
		}
		meta = &m
		return nil
	})
	return meta, err
}

func (s *BoltVectorStore) SetMeta(meta VectorMeta) error {
	data, err := json.Marshal(meta)
	if err != nil {
		return err
	}
	return s.db.Update(func(tx *bbolt.Tx) error {
		b, err := tx.CreateBucketIfNotExists(bucketVectorMeta)
		if err != nil {
			return err
		}
		return b.Put(keyVectorMetaInfo, data)
	})
}

func (s *BoltVectorStore) Upsert(items []port.VectorItem) error {
	if len(items) == 0 {
		return nil
	}

	for _, item := range items {
		if len(item.Vector) != s.dimension {
			return fmt.Errorf("vector dimension mismatch for %s: expected %d, got %d", item.ID, s.dimension, len(item.Vector))
		}
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	err := s.db.Update(func(tx *bbolt.Tx) error {
		b := tx.Bucket(bucketVectors)
		if b == nil {
			return fmt.Errorf("vectors bucket not found")
		}

		for _, item := range items {
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
		}

		return nil
	})
	if err != nil {
		return err
	}

	for _, item := range items {
		s.vectors[item.ID] = vectorEntry{
			vector:   item.Vector,
			norm:     l2Norm(item.Vector),
			metadata: item.Metadata,
		}
	}

	return nil
}

func (s *BoltVectorStore) Search(query []float32, k int) ([]port.VectorResult, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if len(query) != s.dimension {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d (index was built with a different embedding model - re-run 'rag index')", s.dimension, len(query))
	}

	if len(s.vectors) == 0 || k <= 0 {
		return nil, nil
	}

	queryNorm := l2Norm(query)
	if queryNorm == 0 {
		return nil, nil
	}

	results := make([]port.VectorResult, 0, len(s.vectors))
	for id, entry := range s.vectors {
		results = append(results, port.VectorResult{
			ID:       id,
			Score:    cosine(query, queryNorm, entry),
			Metadata: entry.metadata,
		})
	}

	sortResults(results)

	if k > len(results) {
		k = len(results)
	}

	return results[:k], nil
}

func (s *BoltVectorStore) SearchSubset(query []float32, ids []string) ([]port.VectorResult, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if len(query) != s.dimension {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d (index was built with a different embedding model - re-run 'rag index')", s.dimension, len(query))
	}

	queryNorm := l2Norm(query)
	if queryNorm == 0 {
		return nil, nil
	}

	results := make([]port.VectorResult, 0, len(ids))
	for _, id := range ids {
		entry, exists := s.vectors[id]
		if !exists {
			continue
		}
		results = append(results, port.VectorResult{
			ID:       id,
			Score:    cosine(query, queryNorm, entry),
			Metadata: entry.metadata,
		})
	}

	sortResults(results)

	return results, nil
}

func (s *BoltVectorStore) Delete(ids []string) error {
	if len(ids) == 0 {
		return nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	err := s.db.Update(func(tx *bbolt.Tx) error {
		b := tx.Bucket(bucketVectors)
		if b == nil {
			return nil
		}

		for _, id := range ids {
			if err := b.Delete([]byte(id)); err != nil {
				return err
			}
		}

		return nil
	})
	if err != nil {
		return err
	}

	for _, id := range ids {
		delete(s.vectors, id)
	}

	return nil
}

func (s *BoltVectorStore) IDs() ([]string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	ids := make([]string, 0, len(s.vectors))
	for id := range s.vectors {
		ids = append(ids, id)
	}
	return ids, nil
}

func (s *BoltVectorStore) Count() (int, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.vectors), nil
}

func sortResults(results []port.VectorResult) {
	sort.Slice(results, func(i, j int) bool {
		if results[i].Score == results[j].Score {
			return results[i].ID < results[j].ID
		}
		return results[i].Score > results[j].Score
	})
}

func cosine(query []float32, queryNorm float64, entry vectorEntry) float64 {
	if entry.norm == 0 || len(entry.vector) != len(query) {
		return 0
	}

	var dot float64
	for i := range query {
		dot += float64(query[i]) * float64(entry.vector[i])
	}

	return dot / (queryNorm * entry.norm)
}

func l2Norm(v []float32) float64 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	return math.Sqrt(sum)
}
