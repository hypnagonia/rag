package retriever

import (
	"errors"
	"fmt"
	"testing"

	"rag/internal/domain"
	"rag/internal/port"
)

type stubRetriever struct {
	results []domain.ScoredChunk
	err     error
}

func (s *stubRetriever) Search(query string, k int) ([]domain.ScoredChunk, error) {
	if s.err != nil {
		return nil, s.err
	}
	if k < len(s.results) {
		return s.results[:k], nil
	}
	return s.results, nil
}

type stubVectorStore struct {
	ranking []string
	err     error
}

func (s *stubVectorStore) Upsert(items []port.VectorItem) error { return nil }

func (s *stubVectorStore) Search(query []float32, k int) ([]port.VectorResult, error) {
	if s.err != nil {
		return nil, s.err
	}
	results := make([]port.VectorResult, 0, len(s.ranking))
	for i, id := range s.ranking {
		if i >= k {
			break
		}
		results = append(results, port.VectorResult{ID: id, Score: 1.0 - float64(i)*0.1})
	}
	return results, nil
}

func (s *stubVectorStore) SearchSubset(query []float32, ids []string) ([]port.VectorResult, error) {
	return nil, nil
}

func (s *stubVectorStore) Delete(ids []string) error { return nil }
func (s *stubVectorStore) IDs() ([]string, error)    { return s.ranking, nil }
func (s *stubVectorStore) Count() (int, error)       { return len(s.ranking), nil }

type stubEmbedder struct {
	err error
}

func (e *stubEmbedder) Embed(texts []string) ([][]float32, error) {
	if e.err != nil {
		return nil, e.err
	}
	out := make([][]float32, len(texts))
	for i := range texts {
		out[i] = []float32{1, 0, 0}
	}
	return out, nil
}

func (e *stubEmbedder) Dimension() int    { return 3 }
func (e *stubEmbedder) ModelName() string { return "stub" }

type stubChunkStore struct {
	chunks map[string]domain.Chunk
}

func (s *stubChunkStore) GetChunk(id string) (domain.Chunk, error) {
	chunk, ok := s.chunks[id]
	if !ok {
		return domain.Chunk{}, fmt.Errorf("chunk %s not found", id)
	}
	return chunk, nil
}

func (s *stubChunkStore) PutDoc(doc domain.Document) error          { return nil }
func (s *stubChunkStore) GetDoc(id string) (domain.Document, error) { return domain.Document{}, nil }
func (s *stubChunkStore) DeleteDoc(id string) error                 { return nil }
func (s *stubChunkStore) ListDocs() ([]domain.Document, error)      { return nil, nil }
func (s *stubChunkStore) PutChunk(chunk domain.Chunk) error         { return nil }
func (s *stubChunkStore) GetChunksByDoc(id string) ([]domain.Chunk, error) {
	return nil, nil
}
func (s *stubChunkStore) DeleteChunksByDoc(docID string) error { return nil }
func (s *stubChunkStore) PutPosting(term, chunkID string, tf int) error {
	return nil
}
func (s *stubChunkStore) GetPostings(term string) ([]domain.Posting, error) { return nil, nil }
func (s *stubChunkStore) DeletePostings(chunkID string, terms []string) error {
	return nil
}
func (s *stubChunkStore) GetStats() (domain.Stats, error)           { return domain.Stats{}, nil }
func (s *stubChunkStore) UpdateStats(stats domain.Stats) error      { return nil }
func (s *stubChunkStore) BatchIndex(files []port.IndexedFile) error { return nil }
func (s *stubChunkStore) Close() error                              { return nil }

func chunk(id string) domain.Chunk {
	return domain.Chunk{ID: id, DocID: "doc", Text: id}
}

func scored(ids ...string) []domain.ScoredChunk {
	out := make([]domain.ScoredChunk, 0, len(ids))
	for i, id := range ids {
		out = append(out, domain.ScoredChunk{Chunk: chunk(id), Score: 10 - float64(i)})
	}
	return out
}

func chunkStoreFor(ids ...string) *stubChunkStore {
	chunks := make(map[string]domain.Chunk, len(ids))
	for _, id := range ids {
		chunks[id] = chunk(id)
	}
	return &stubChunkStore{chunks: chunks}
}

func ids(results []domain.ScoredChunk) []string {
	out := make([]string, 0, len(results))
	for _, r := range results {
		out = append(out, r.Chunk.ID)
	}
	return out
}

func TestHybridSurfacesVectorOnlyResults(t *testing.T) {
	hybrid := NewHybridBuilder().
		BM25(&stubRetriever{results: scored("lex1", "lex2")}).
		VectorStore(&stubVectorStore{ranking: []string{"vec1", "vec2"}}).
		Embedder(&stubEmbedder{}).
		ChunkStore(chunkStoreFor("vec1", "vec2")).
		Build()

	results, err := hybrid.Search("query", 10)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	found := map[string]bool{}
	for _, id := range ids(results) {
		found[id] = true
	}

	for _, want := range []string{"lex1", "lex2", "vec1", "vec2"} {
		if !found[want] {
			t.Errorf("expected %q in fused results, got %v", want, ids(results))
		}
	}
}

func TestHybridRanksTopOfBothArmsFirst(t *testing.T) {
	hybrid := NewHybridBuilder().
		BM25(&stubRetriever{results: scored("shared", "lexOnly")}).
		VectorStore(&stubVectorStore{ranking: []string{"shared", "vecOnly"}}).
		Embedder(&stubEmbedder{}).
		ChunkStore(chunkStoreFor("shared", "vecOnly")).
		Build()

	results, err := hybrid.Search("query", 10)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results) != 3 {
		t.Fatalf("expected 3 unique chunks, got %d (%v)", len(results), ids(results))
	}
	if results[0].Chunk.ID != "shared" {
		t.Errorf("a chunk ranked first by both arms should win, got %v", ids(results))
	}
}

func TestHybridDeduplicatesChunksAcrossArms(t *testing.T) {
	hybrid := NewHybridBuilder().
		BM25(&stubRetriever{results: scored("a", "b")}).
		VectorStore(&stubVectorStore{ranking: []string{"a", "b"}}).
		Embedder(&stubEmbedder{}).
		ChunkStore(chunkStoreFor("a", "b")).
		Build()

	results, _ := hybrid.Search("query", 10)

	seen := map[string]int{}
	for _, id := range ids(results) {
		seen[id]++
	}
	for id, n := range seen {
		if n > 1 {
			t.Errorf("chunk %q appeared %d times in fused results", id, n)
		}
	}
}

func TestHybridBM25WeightOneIgnoresVectorArm(t *testing.T) {
	hybrid := NewHybridBuilder().
		BM25(&stubRetriever{results: scored("lex")}).
		VectorStore(&stubVectorStore{ranking: []string{"vec"}}).
		Embedder(&stubEmbedder{}).
		ChunkStore(chunkStoreFor("vec")).
		BM25Weight(1.0).
		Build()

	results, _ := hybrid.Search("query", 10)

	if results[0].Chunk.ID != "lex" {
		t.Errorf("with bm25_weight=1.0 the BM25 hit should rank first, got %v", ids(results))
	}
	for _, r := range results {
		if r.Chunk.ID == "vec" && r.Score != 0 {
			t.Errorf("vector-only chunk should contribute 0 at bm25_weight=1.0, got %f", r.Score)
		}
	}
}

func TestHybridFallsBackToBM25WhenEmbedderFails(t *testing.T) {
	hybrid := NewHybridBuilder().
		BM25(&stubRetriever{results: scored("lex1", "lex2")}).
		VectorStore(&stubVectorStore{ranking: []string{"vec1"}}).
		Embedder(&stubEmbedder{err: errors.New("ollama unreachable")}).
		ChunkStore(chunkStoreFor("vec1")).
		Build()

	results, err := hybrid.Search("query", 10)
	if err != nil {
		t.Fatalf("expected graceful fallback, got error: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected the 2 BM25 results, got %v", ids(results))
	}

	stats := hybrid.Stats()
	if stats.VectorError == nil {
		t.Error("expected the vector arm failure to be recorded in stats")
	}
	if stats.BM25Candidates != 2 {
		t.Errorf("expected 2 bm25 candidates in stats, got %d", stats.BM25Candidates)
	}
}

func TestHybridReturnsVectorResultsWhenBM25Empty(t *testing.T) {
	hybrid := NewHybridBuilder().
		BM25(&stubRetriever{}).
		VectorStore(&stubVectorStore{ranking: []string{"vec1", "vec2"}}).
		Embedder(&stubEmbedder{}).
		ChunkStore(chunkStoreFor("vec1", "vec2")).
		Build()

	results, err := hybrid.Search("query", 10)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected vector results when BM25 finds nothing, got %v", ids(results))
	}
}

func TestHybridSkipsVectorHitsWithMissingChunks(t *testing.T) {
	hybrid := NewHybridBuilder().
		BM25(&stubRetriever{results: scored("lex")}).
		VectorStore(&stubVectorStore{ranking: []string{"orphan", "vec"}}).
		Embedder(&stubEmbedder{}).
		ChunkStore(chunkStoreFor("vec")).
		Build()

	results, err := hybrid.Search("query", 10)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	for _, id := range ids(results) {
		if id == "orphan" {
			t.Error("a vector whose chunk was deleted must not be returned")
		}
	}
}

func TestHybridRespectsK(t *testing.T) {
	hybrid := NewHybridBuilder().
		BM25(&stubRetriever{results: scored("a", "b", "c", "d")}).
		VectorStore(&stubVectorStore{ranking: []string{"e", "f"}}).
		Embedder(&stubEmbedder{}).
		ChunkStore(chunkStoreFor("e", "f")).
		Build()

	results, _ := hybrid.Search("query", 3)
	if len(results) != 3 {
		t.Errorf("expected exactly 3 results, got %d", len(results))
	}
}

func TestHybridErrorsWhenBothArmsFail(t *testing.T) {
	hybrid := NewHybridBuilder().
		BM25(&stubRetriever{err: errors.New("store closed")}).
		VectorStore(&stubVectorStore{err: errors.New("vector store closed")}).
		Embedder(&stubEmbedder{}).
		ChunkStore(chunkStoreFor()).
		Build()

	if _, err := hybrid.Search("query", 10); err == nil {
		t.Error("expected an error when both retrieval arms fail")
	}
}
