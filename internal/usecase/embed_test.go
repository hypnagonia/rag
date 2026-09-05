package usecase

import (
	"errors"
	"strings"
	"testing"

	"rag/internal/adapter/memstore"
	"rag/internal/domain"
	"rag/internal/port"
)

type fakeEmbedder struct {
	calls     [][]string
	dimension int
	err       error
}

func (e *fakeEmbedder) Embed(texts []string) ([][]float32, error) {
	if e.err != nil {
		return nil, e.err
	}
	e.calls = append(e.calls, append([]string(nil), texts...))

	out := make([][]float32, len(texts))
	for i := range texts {
		vec := make([]float32, e.dimension)
		vec[0] = float32(len(texts[i]))
		out[i] = vec
	}
	return out, nil
}

func (e *fakeEmbedder) Dimension() int    { return e.dimension }
func (e *fakeEmbedder) ModelName() string { return "fake" }

func (e *fakeEmbedder) embeddedCount() int {
	n := 0
	for _, call := range e.calls {
		n += len(call)
	}
	return n
}

type fakeVectorStore struct {
	vectors map[string][]float32
}

func newFakeVectorStore() *fakeVectorStore {
	return &fakeVectorStore{vectors: map[string][]float32{}}
}

func (s *fakeVectorStore) Upsert(items []port.VectorItem) error {
	for _, item := range items {
		s.vectors[item.ID] = item.Vector
	}
	return nil
}

func (s *fakeVectorStore) Search(query []float32, k int) ([]port.VectorResult, error) {
	return nil, nil
}

func (s *fakeVectorStore) SearchSubset(query []float32, ids []string) ([]port.VectorResult, error) {
	return nil, nil
}

func (s *fakeVectorStore) Delete(ids []string) error {
	for _, id := range ids {
		delete(s.vectors, id)
	}
	return nil
}

func (s *fakeVectorStore) IDs() ([]string, error) {
	out := make([]string, 0, len(s.vectors))
	for id := range s.vectors {
		out = append(out, id)
	}
	return out, nil
}

func (s *fakeVectorStore) Count() (int, error) { return len(s.vectors), nil }

func seedStore(t *testing.T, docs map[string][]domain.Chunk) *memstore.MemoryStore {
	t.Helper()

	st := memstore.NewMemoryStore()
	for docID, chunks := range docs {
		if err := st.PutDoc(domain.Document{ID: docID, Path: "/src/" + docID + ".go"}); err != nil {
			t.Fatalf("put doc failed: %v", err)
		}
		for _, c := range chunks {
			c.DocID = docID
			if err := st.PutChunk(c); err != nil {
				t.Fatalf("put chunk failed: %v", err)
			}
		}
	}
	return st
}

func buildEmbedUC(t *testing.T, st port.IndexStore, emb port.Embedder, vs port.VectorStore, force bool) *EmbedUseCase {
	t.Helper()

	uc, err := NewEmbedBuilder().
		Store(st).
		Embedder(emb).
		VectorStore(vs).
		BatchSize(2).
		Force(force).
		Build()
	if err != nil {
		t.Fatalf("build failed: %v", err)
	}
	return uc
}

func TestEmbedSyncEmbedsAllChunksOnFirstRun(t *testing.T) {
	st := seedStore(t, map[string][]domain.Chunk{
		"doc1": {{ID: "c1", Text: "alpha", StartLine: 1, EndLine: 2}, {ID: "c2", Text: "beta", StartLine: 3, EndLine: 4}},
	})
	emb := &fakeEmbedder{dimension: 4}
	vs := newFakeVectorStore()

	result, err := buildEmbedUC(t, st, emb, vs, false).Sync(nil)
	if err != nil {
		t.Fatalf("sync failed: %v", err)
	}

	if result.Embedded != 2 {
		t.Errorf("expected 2 embedded, got %d", result.Embedded)
	}
	if result.Reused != 0 {
		t.Errorf("expected 0 reused on first run, got %d", result.Reused)
	}
	if count, _ := vs.Count(); count != 2 {
		t.Errorf("expected 2 stored vectors, got %d", count)
	}
}

func TestEmbedSyncReusesExistingVectorsOnSecondRun(t *testing.T) {
	st := seedStore(t, map[string][]domain.Chunk{
		"doc1": {{ID: "c1", Text: "alpha"}, {ID: "c2", Text: "beta"}},
	})
	emb := &fakeEmbedder{dimension: 4}
	vs := newFakeVectorStore()

	if _, err := buildEmbedUC(t, st, emb, vs, false).Sync(nil); err != nil {
		t.Fatalf("first sync failed: %v", err)
	}
	firstCallCount := emb.embeddedCount()

	result, err := buildEmbedUC(t, st, emb, vs, false).Sync(nil)
	if err != nil {
		t.Fatalf("second sync failed: %v", err)
	}

	if result.Embedded != 0 {
		t.Errorf("expected 0 re-embedded on unchanged index, got %d", result.Embedded)
	}
	if result.Reused != 2 {
		t.Errorf("expected 2 reused, got %d", result.Reused)
	}
	if emb.embeddedCount() != firstCallCount {
		t.Errorf("second sync must not call the embedder again (%d -> %d texts)", firstCallCount, emb.embeddedCount())
	}
}

func TestEmbedSyncOnlyEmbedsNewChunks(t *testing.T) {
	st := seedStore(t, map[string][]domain.Chunk{
		"doc1": {{ID: "c1", Text: "alpha"}},
	})
	emb := &fakeEmbedder{dimension: 4}
	vs := newFakeVectorStore()

	if _, err := buildEmbedUC(t, st, emb, vs, false).Sync(nil); err != nil {
		t.Fatalf("first sync failed: %v", err)
	}

	if err := st.PutChunk(domain.Chunk{ID: "c2", DocID: "doc1", Text: "beta"}); err != nil {
		t.Fatalf("put chunk failed: %v", err)
	}

	emb.calls = nil
	result, err := buildEmbedUC(t, st, emb, vs, false).Sync(nil)
	if err != nil {
		t.Fatalf("second sync failed: %v", err)
	}

	if result.Embedded != 1 || result.Reused != 1 {
		t.Errorf("expected 1 embedded and 1 reused, got %d and %d", result.Embedded, result.Reused)
	}
	if emb.embeddedCount() != 1 {
		t.Errorf("expected exactly 1 text sent to the embedder, got %d", emb.embeddedCount())
	}
}

func TestEmbedSyncDeletesVectorsForRemovedChunks(t *testing.T) {
	st := seedStore(t, map[string][]domain.Chunk{
		"doc1": {{ID: "c1", Text: "alpha"}},
	})
	emb := &fakeEmbedder{dimension: 4}
	vs := newFakeVectorStore()

	if _, err := buildEmbedUC(t, st, emb, vs, false).Sync(nil); err != nil {
		t.Fatalf("first sync failed: %v", err)
	}

	if err := vs.Upsert([]port.VectorItem{{ID: "stale", Vector: make([]float32, 4)}}); err != nil {
		t.Fatalf("seed stale vector failed: %v", err)
	}

	result, err := buildEmbedUC(t, st, emb, vs, false).Sync(nil)
	if err != nil {
		t.Fatalf("second sync failed: %v", err)
	}

	if result.Deleted != 1 {
		t.Errorf("expected 1 stale vector deleted, got %d", result.Deleted)
	}
	if _, exists := vs.vectors["stale"]; exists {
		t.Error("orphaned vector was not deleted")
	}
}

func TestEmbedSyncForceReEmbedsEverything(t *testing.T) {
	st := seedStore(t, map[string][]domain.Chunk{
		"doc1": {{ID: "c1", Text: "alpha"}, {ID: "c2", Text: "beta"}},
	})
	emb := &fakeEmbedder{dimension: 4}
	vs := newFakeVectorStore()

	if _, err := buildEmbedUC(t, st, emb, vs, false).Sync(nil); err != nil {
		t.Fatalf("first sync failed: %v", err)
	}

	emb.calls = nil
	result, err := buildEmbedUC(t, st, emb, vs, true).Sync(nil)
	if err != nil {
		t.Fatalf("force sync failed: %v", err)
	}

	if result.Embedded != 2 {
		t.Errorf("expected force to re-embed 2 chunks, got %d", result.Embedded)
	}
	if emb.embeddedCount() != 2 {
		t.Errorf("expected 2 texts sent to the embedder, got %d", emb.embeddedCount())
	}
}

func TestEmbedSyncIncludesFilePathInEmbeddedText(t *testing.T) {
	st := seedStore(t, map[string][]domain.Chunk{
		"doc1": {{ID: "c1", Text: "func Handler()", StartLine: 10, EndLine: 20}},
	})
	emb := &fakeEmbedder{dimension: 4}

	if _, err := buildEmbedUC(t, st, emb, newFakeVectorStore(), false).Sync(nil); err != nil {
		t.Fatalf("sync failed: %v", err)
	}

	if len(emb.calls) == 0 || len(emb.calls[0]) == 0 {
		t.Fatal("embedder was never called")
	}

	text := emb.calls[0][0]
	if !strings.Contains(text, "/src/doc1.go:10-20") {
		t.Errorf("expected embedded text to carry file path and line range, got %q", text)
	}
	if !strings.Contains(text, "func Handler()") {
		t.Errorf("expected embedded text to carry chunk body, got %q", text)
	}
}

func TestEmbedSyncReportsEmbedderFailure(t *testing.T) {
	st := seedStore(t, map[string][]domain.Chunk{
		"doc1": {{ID: "c1", Text: "alpha"}},
	})
	emb := &fakeEmbedder{dimension: 4, err: errors.New("connection refused")}

	_, err := buildEmbedUC(t, st, emb, newFakeVectorStore(), false).Sync(nil)
	if err == nil {
		t.Fatal("expected sync to report the embedder failure")
	}
	if !strings.Contains(err.Error(), "connection refused") {
		t.Errorf("expected the underlying cause in the error, got %v", err)
	}
}

func TestEmbedBuilderRequiresDependencies(t *testing.T) {
	if _, err := NewEmbedBuilder().Embedder(&fakeEmbedder{dimension: 4}).VectorStore(newFakeVectorStore()).Build(); err == nil {
		t.Error("expected a missing store to be rejected")
	}
	if _, err := NewEmbedBuilder().Store(memstore.NewMemoryStore()).VectorStore(newFakeVectorStore()).Build(); err == nil {
		t.Error("expected a missing embedder to be rejected")
	}
	if _, err := NewEmbedBuilder().Store(memstore.NewMemoryStore()).Embedder(&fakeEmbedder{dimension: 4}).Build(); err == nil {
		t.Error("expected a missing vector store to be rejected")
	}
}

func TestEmbedSyncSkipsBlankChunks(t *testing.T) {
	st := seedStore(t, map[string][]domain.Chunk{
		"doc1": {
			{ID: "c1", Text: "func Handler()"},
			{ID: "blank", Text: "   \n\t"},
			{ID: "empty", Text: ""},
		},
	})
	emb := &fakeEmbedder{dimension: 4}
	vs := newFakeVectorStore()

	result, err := buildEmbedUC(t, st, emb, vs, false).Sync(nil)
	if err != nil {
		t.Fatalf("sync failed: %v", err)
	}

	if result.Embedded != 1 {
		t.Errorf("expected only the non-blank chunk to be embedded, got %d", result.Embedded)
	}
	if _, exists := vs.vectors["blank"]; exists {
		t.Error("a whitespace-only chunk should not get a vector")
	}
	if _, exists := vs.vectors["empty"]; exists {
		t.Error("an empty chunk should not get a vector")
	}
}
