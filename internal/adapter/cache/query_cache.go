package cache

import (
	"crypto/sha256"
	"encoding/hex"
	"sync"
	"time"

	"rag/internal/domain"
)

type QueryCache struct {
	mu       sync.RWMutex
	entries  map[string]*cacheEntry
	order    []string
	maxSize  int
	ttl      time.Duration
	indexGen uint64
}

type cacheEntry struct {
	results   []domain.ScoredChunk
	timestamp time.Time
	indexGen  uint64
}

func NewQueryCache(maxSize int, ttl time.Duration) *QueryCache {
	if maxSize <= 0 {
		maxSize = 100
	}
	if ttl <= 0 {
		ttl = 5 * time.Minute
	}
	return &QueryCache{
		entries: make(map[string]*cacheEntry),
		order:   make([]string, 0, maxSize),
		maxSize: maxSize,
		ttl:     ttl,
	}
}

func cacheKey(query string, topK int) string {
	data := []byte(query)
	data = append(data, byte(topK>>8), byte(topK))
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:16])
}

func (c *QueryCache) Get(query string, topK int) ([]domain.ScoredChunk, bool) {
	c.mu.RLock()
	key := cacheKey(query, topK)
	entry, exists := c.entries[key]
	currentGen := c.indexGen
	c.mu.RUnlock()

	if !exists {
		return nil, false
	}

	if time.Since(entry.timestamp) > c.ttl {
		c.mu.Lock()
		delete(c.entries, key)
		c.removeFromOrder(key)
		c.mu.Unlock()
		return nil, false
	}

	if entry.indexGen != currentGen {
		c.mu.Lock()
		delete(c.entries, key)
		c.removeFromOrder(key)
		c.mu.Unlock()
		return nil, false
	}

	c.mu.Lock()
	c.moveToEnd(key)
	c.mu.Unlock()

	return entry.results, true
}

func (c *QueryCache) Put(query string, topK int, results []domain.ScoredChunk) {
	c.mu.Lock()
	defer c.mu.Unlock()

	key := cacheKey(query, topK)

	if _, exists := c.entries[key]; exists {

		c.entries[key] = &cacheEntry{
			results:   results,
			timestamp: time.Now(),
			indexGen:  c.indexGen,
		}
		c.moveToEnd(key)
		return
	}

	if len(c.entries) >= c.maxSize {
		c.evictOldest()
	}

	c.entries[key] = &cacheEntry{
		results:   results,
		timestamp: time.Now(),
		indexGen:  c.indexGen,
	}
	c.order = append(c.order, key)
}

func (c *QueryCache) Invalidate() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.entries = make(map[string]*cacheEntry)
	c.order = c.order[:0]
	c.indexGen++
}

func (c *QueryCache) Size() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.entries)
}

func (c *QueryCache) evictOldest() {
	if len(c.order) == 0 {
		return
	}
	oldest := c.order[0]
	c.order = c.order[1:]
	delete(c.entries, oldest)
}

func (c *QueryCache) moveToEnd(key string) {
	c.removeFromOrder(key)
	c.order = append(c.order, key)
}

func (c *QueryCache) removeFromOrder(key string) {
	for i, k := range c.order {
		if k == key {
			c.order = append(c.order[:i], c.order[i+1:]...)
			return
		}
	}
}

type CachedRetriever struct {
	retriever Retriever
	cache     *QueryCache
}

type Retriever interface {
	Search(query string, k int) ([]domain.ScoredChunk, error)
}

func NewCachedRetriever(retriever Retriever, cache *QueryCache) *CachedRetriever {
	return &CachedRetriever{
		retriever: retriever,
		cache:     cache,
	}
}

func (r *CachedRetriever) Search(query string, k int) ([]domain.ScoredChunk, error) {

	if results, hit := r.cache.Get(query, k); hit {
		return results, nil
	}

	results, err := r.retriever.Search(query, k)
	if err != nil {
		return nil, err
	}

	r.cache.Put(query, k, results)

	return results, nil
}
