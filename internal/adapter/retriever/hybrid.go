package retriever

import (
	"sort"
	"sync"

	"rag/internal/domain"
	"rag/internal/port"
)

const (
	defaultRRFK    = 60
	minCandidates  = 50
	candidateRatio = 4
)

type HybridRetriever struct {
	bm25        port.Retriever
	vectorStore port.VectorStore
	embedder    port.Embedder
	chunkStore  port.IndexStore
	rrfK        int
	bm25Weight  float64

	mu    sync.Mutex
	stats HybridStats
}

type HybridStats struct {
	BM25Candidates   int
	VectorCandidates int
	Fused            int
	BM25Error        error
	VectorError      error
}

type HybridBuilder struct {
	bm25        port.Retriever
	vectorStore port.VectorStore
	embedder    port.Embedder
	chunkStore  port.IndexStore
	rrfK        int
	bm25Weight  float64
}

func NewHybridBuilder() *HybridBuilder {
	return &HybridBuilder{
		rrfK:       defaultRRFK,
		bm25Weight: 0.5,
	}
}

func (b *HybridBuilder) BM25(r port.Retriever) *HybridBuilder {
	b.bm25 = r
	return b
}

func (b *HybridBuilder) VectorStore(vs port.VectorStore) *HybridBuilder {
	b.vectorStore = vs
	return b
}

func (b *HybridBuilder) Embedder(e port.Embedder) *HybridBuilder {
	b.embedder = e
	return b
}

func (b *HybridBuilder) ChunkStore(s port.IndexStore) *HybridBuilder {
	b.chunkStore = s
	return b
}

func (b *HybridBuilder) RRFK(k int) *HybridBuilder {
	if k > 0 {
		b.rrfK = k
	}
	return b
}

func (b *HybridBuilder) BM25Weight(w float64) *HybridBuilder {
	if w >= 0 && w <= 1 {
		b.bm25Weight = w
	}
	return b
}

func (b *HybridBuilder) Build() *HybridRetriever {
	return &HybridRetriever{
		bm25:        b.bm25,
		vectorStore: b.vectorStore,
		embedder:    b.embedder,
		chunkStore:  b.chunkStore,
		rrfK:        b.rrfK,
		bm25Weight:  b.bm25Weight,
	}
}

func (r *HybridRetriever) Stats() HybridStats {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.stats
}

func (r *HybridRetriever) Search(query string, k int) ([]domain.ScoredChunk, error) {
	if k <= 0 {
		return nil, nil
	}

	poolSize := k * candidateRatio
	if poolSize < minCandidates {
		poolSize = minCandidates
	}

	var (
		bm25Results   []domain.ScoredChunk
		vectorResults []domain.ScoredChunk
		bm25Err       error
		vectorErr     error
		wg            sync.WaitGroup
	)

	if r.bm25 != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			bm25Results, bm25Err = r.bm25.Search(query, poolSize)
		}()
	}

	if r.vectorStore != nil && r.embedder != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			vectorResults, vectorErr = r.vectorSearch(query, poolSize)
		}()
	}

	wg.Wait()

	r.mu.Lock()
	r.stats = HybridStats{
		BM25Candidates:   len(bm25Results),
		VectorCandidates: len(vectorResults),
		BM25Error:        bm25Err,
		VectorError:      vectorErr,
	}
	r.mu.Unlock()

	if len(bm25Results) == 0 && len(vectorResults) == 0 {
		if bm25Err != nil {
			return nil, bm25Err
		}
		if vectorErr != nil {
			return nil, vectorErr
		}
		return nil, nil
	}

	fused := r.fuseRRF(bm25Results, vectorResults)

	r.mu.Lock()
	r.stats.Fused = len(fused)
	r.mu.Unlock()

	if len(fused) > k {
		fused = fused[:k]
	}

	return fused, nil
}

func (r *HybridRetriever) vectorSearch(query string, k int) ([]domain.ScoredChunk, error) {
	embeddings, err := r.embedder.Embed([]string{query})
	if err != nil {
		return nil, err
	}
	if len(embeddings) == 0 || len(embeddings[0]) == 0 {
		return nil, nil
	}

	results, err := r.vectorStore.Search(embeddings[0], k)
	if err != nil {
		return nil, err
	}

	chunks := make([]domain.ScoredChunk, 0, len(results))
	for _, result := range results {
		chunk, err := r.chunkStore.GetChunk(result.ID)
		if err != nil {
			continue
		}
		chunks = append(chunks, domain.ScoredChunk{
			Chunk: chunk,
			Score: result.Score,
		})
	}

	return chunks, nil
}

func (r *HybridRetriever) fuseRRF(bm25Results, vectorResults []domain.ScoredChunk) []domain.ScoredChunk {
	vectorWeight := 1.0 - r.bm25Weight

	scores := make(map[string]float64)
	chunks := make(map[string]domain.Chunk)
	order := make([]string, 0, len(bm25Results)+len(vectorResults))

	accumulate := func(results []domain.ScoredChunk, weight float64) {
		for rank, result := range results {
			id := result.Chunk.ID
			if _, seen := chunks[id]; !seen {
				chunks[id] = result.Chunk
				order = append(order, id)
			}
			scores[id] += weight / float64(r.rrfK+rank+1)
		}
	}

	accumulate(bm25Results, r.bm25Weight)
	accumulate(vectorResults, vectorWeight)

	fused := make([]domain.ScoredChunk, 0, len(order))
	for _, id := range order {
		fused = append(fused, domain.ScoredChunk{
			Chunk: chunks[id],
			Score: scores[id],
		})
	}

	sort.SliceStable(fused, func(i, j int) bool {
		return fused[i].Score > fused[j].Score
	})

	return fused
}
