package cache

import (
	"crypto/sha256"
	"encoding/hex"
	"sync"
	"time"

	"rag/internal/domain"
)

// QueryCache provides caching for search queries with LRU eviction and TTL.
type QueryCache struct {
	mu       sync.RWMutex
	entries  map[string]*cacheEntry
	order    []string // For LRU tracking
	maxSize  int
	ttl      time.Duration
	indexGen uint64 // Generation counter to invalidate on re-index
}

type cacheEntry struct {
	results   []domain.ScoredChunk
	timestamp time.Time
	indexGen  uint64
}

// NewQueryCache creates a new query cache.
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

// cacheKey generates a cache key from query and parameters.
func cacheKey(query string, topK int) string {
	data := []byte(query)
	data = append(data, byte(topK>>8), byte(topK))
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:16])
}

// Get retrieves cached results for a query.
func (c *QueryCache) Get(query string, topK int) ([]domain.ScoredChunk, bool) {
	c.mu.RLock()
	key := cacheKey(query, topK)
	entry, exists := c.entries[key]
	currentGen := c.indexGen
	c.mu.RUnlock()

	if !exists {
		return nil, false
	}

	// Check TTL
	if time.Since(entry.timestamp) > c.ttl {
		c.mu.Lock()
		delete(c.entries, key)
		c.removeFromOrder(key)
		c.mu.Unlock()
		return nil, false
	}

	// Check if index was rebuilt since caching
	if entry.indexGen != currentGen {
		c.mu.Lock()
		delete(c.entries, key)
		c.removeFromOrder(key)
		c.mu.Unlock()
		return nil, false
	}

	// Move to end of LRU order (most recently used)
	c.mu.Lock()
	c.moveToEnd(key)
	c.mu.Unlock()

	return entry.results, true
}

// Put stores results in the cache.
func (c *QueryCache) Put(query string, topK int, results []domain.ScoredChunk) {
	c.mu.Lock()
	defer c.mu.Unlock()

	key := cacheKey(query, topK)

	// Check if already exists
	if _, exists := c.entries[key]; exists {
		// Update existing
		c.entries[key] = &cacheEntry{
			results:   results,
			timestamp: time.Now(),
			indexGen:  c.indexGen,
		}
		c.moveToEnd(key)
		return
	}

	// Evict oldest if at capacity
	if len(c.entries) >= c.maxSize {
		c.evictOldest()
	}

	// Add new entry
	c.entries[key] = &cacheEntry{
		results:   results,
		timestamp: time.Now(),
		indexGen:  c.indexGen,
	}
	c.order = append(c.order, key)
}

// Invalidate clears all cached queries (call after re-indexing).
func (c *QueryCache) Invalidate() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.entries = make(map[string]*cacheEntry)
	c.order = c.order[:0]
	c.indexGen++
}

// Size returns the current number of cached entries.
func (c *QueryCache) Size() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.entries)
}

// evictOldest removes the oldest entry (must be called with lock held).
func (c *QueryCache) evictOldest() {
	if len(c.order) == 0 {
		return
	}
	oldest := c.order[0]
	c.order = c.order[1:]
	delete(c.entries, oldest)
}

// moveToEnd moves a key to the end of the LRU order (must be called with lock held).
func (c *QueryCache) moveToEnd(key string) {
	c.removeFromOrder(key)
	c.order = append(c.order, key)
}

// removeFromOrder removes a key from the order slice (must be called with lock held).
func (c *QueryCache) removeFromOrder(key string) {
	for i, k := range c.order {
		if k == key {
			c.order = append(c.order[:i], c.order[i+1:]...)
			return
		}
	}
}

// CachedRetriever wraps a retriever with caching.
type CachedRetriever struct {
	retriever Retriever
	cache     *QueryCache
}

// Retriever interface for search operations.
type Retriever interface {
	Search(query string, k int) ([]domain.ScoredChunk, error)
}

// NewCachedRetriever creates a new cached retriever.
func NewCachedRetriever(retriever Retriever, cache *QueryCache) *CachedRetriever {
	return &CachedRetriever{
		retriever: retriever,
		cache:     cache,
	}
}

// Search performs a cached search.
func (r *CachedRetriever) Search(query string, k int) ([]domain.ScoredChunk, error) {
	// Check cache first
	if results, hit := r.cache.Get(query, k); hit {
		return results, nil
	}

	// Cache miss - perform actual search
	results, err := r.retriever.Search(query, k)
	if err != nil {
		return nil, err
	}

	// Store in cache
	r.cache.Put(query, k, results)

	return results, nil
}
