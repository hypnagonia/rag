package store

import (
	"path/filepath"
	"testing"

	"go.etcd.io/bbolt"
	"rag/internal/port"
)

func newTestVectorStore(t *testing.T, dimension int) (*BoltVectorStore, *bbolt.DB) {
	t.Helper()

	db, err := bbolt.Open(filepath.Join(t.TempDir(), "test.db"), 0600, nil)
	if err != nil {
		t.Fatalf("failed to open db: %v", err)
	}
	t.Cleanup(func() { db.Close() })

	vs, err := NewBoltVectorStore(db, dimension)
	if err != nil {
		t.Fatalf("failed to create vector store: %v", err)
	}
	return vs, db
}

func TestVectorStoreUpsertAndSearchRanksByCosine(t *testing.T) {
	vs, _ := newTestVectorStore(t, 3)

	err := vs.Upsert([]port.VectorItem{
		{ID: "exact", Vector: []float32{1, 0, 0}},
		{ID: "orthogonal", Vector: []float32{0, 1, 0}},
		{ID: "opposite", Vector: []float32{-1, 0, 0}},
	})
	if err != nil {
		t.Fatalf("upsert failed: %v", err)
	}

	results, err := vs.Search([]float32{1, 0, 0}, 3)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}

	wantOrder := []string{"exact", "orthogonal", "opposite"}
	for i, want := range wantOrder {
		if results[i].ID != want {
			t.Errorf("result %d: expected %q, got %q", i, want, results[i].ID)
		}
	}

	if results[0].Score < 0.999 {
		t.Errorf("expected identical vector to score ~1.0, got %f", results[0].Score)
	}
	if results[2].Score > -0.999 {
		t.Errorf("expected opposite vector to score ~-1.0, got %f", results[2].Score)
	}
}

func TestVectorStoreSearchIsMagnitudeInvariant(t *testing.T) {
	vs, _ := newTestVectorStore(t, 3)

	if err := vs.Upsert([]port.VectorItem{
		{ID: "short", Vector: []float32{1, 1, 0}},
		{ID: "long", Vector: []float32{100, 100, 0}},
	}); err != nil {
		t.Fatalf("upsert failed: %v", err)
	}

	results, err := vs.Search([]float32{1, 1, 0}, 2)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if diff := results[0].Score - results[1].Score; diff > 1e-6 || diff < -1e-6 {
		t.Errorf("cosine should ignore magnitude, got %f vs %f", results[0].Score, results[1].Score)
	}
}

func TestVectorStoreRejectsWrongDimension(t *testing.T) {
	vs, _ := newTestVectorStore(t, 3)

	if err := vs.Upsert([]port.VectorItem{{ID: "a", Vector: []float32{1, 0}}}); err == nil {
		t.Error("expected upsert to reject a 2-dim vector in a 3-dim store")
	}

	count, _ := vs.Count()
	if count != 0 {
		t.Errorf("rejected upsert should not be stored, got count %d", count)
	}

	if _, err := vs.Search([]float32{1, 0}, 1); err == nil {
		t.Error("expected search to reject a 2-dim query in a 3-dim store")
	}
}

func TestVectorStoreUpsertIsAtomicAcrossBatch(t *testing.T) {
	vs, _ := newTestVectorStore(t, 3)

	err := vs.Upsert([]port.VectorItem{
		{ID: "good", Vector: []float32{1, 0, 0}},
		{ID: "bad", Vector: []float32{1, 0}},
	})
	if err == nil {
		t.Fatal("expected batch with a malformed vector to fail")
	}

	count, _ := vs.Count()
	if count != 0 {
		t.Errorf("failed batch should write nothing, got count %d", count)
	}
}

func TestVectorStorePersistsAcrossReopen(t *testing.T) {
	vs, db := newTestVectorStore(t, 3)

	if err := vs.Upsert([]port.VectorItem{{ID: "a", Vector: []float32{1, 2, 3}}}); err != nil {
		t.Fatalf("upsert failed: %v", err)
	}
	if err := vs.SetMeta(VectorMeta{Model: "test-model", Dimension: 3}); err != nil {
		t.Fatalf("set meta failed: %v", err)
	}

	reopened, err := NewBoltVectorStore(db, 3)
	if err != nil {
		t.Fatalf("reopen failed: %v", err)
	}

	count, _ := reopened.Count()
	if count != 1 {
		t.Errorf("expected 1 persisted vector, got %d", count)
	}

	meta, err := reopened.Meta()
	if err != nil {
		t.Fatalf("meta read failed: %v", err)
	}
	if meta == nil || meta.Model != "test-model" || meta.Dimension != 3 {
		t.Errorf("expected persisted meta {test-model 3}, got %+v", meta)
	}
}

func TestVectorStoreSkipsVectorsOfWrongDimensionOnLoad(t *testing.T) {
	vs, db := newTestVectorStore(t, 3)

	if err := vs.Upsert([]port.VectorItem{{ID: "a", Vector: []float32{1, 2, 3}}}); err != nil {
		t.Fatalf("upsert failed: %v", err)
	}

	stale, err := NewBoltVectorStore(db, 768)
	if err != nil {
		t.Fatalf("reopen failed: %v", err)
	}

	count, _ := stale.Count()
	if count != 0 {
		t.Errorf("a 3-dim vector must not load into a 768-dim store, got count %d", count)
	}
}

func TestVectorStoreDeleteRemovesFromMemoryAndDisk(t *testing.T) {
	vs, db := newTestVectorStore(t, 3)

	if err := vs.Upsert([]port.VectorItem{
		{ID: "a", Vector: []float32{1, 0, 0}},
		{ID: "b", Vector: []float32{0, 1, 0}},
	}); err != nil {
		t.Fatalf("upsert failed: %v", err)
	}

	if err := vs.Delete([]string{"a"}); err != nil {
		t.Fatalf("delete failed: %v", err)
	}

	results, _ := vs.Search([]float32{1, 0, 0}, 10)
	for _, r := range results {
		if r.ID == "a" {
			t.Error("deleted vector still returned by search")
		}
	}

	reopened, _ := NewBoltVectorStore(db, 3)
	count, _ := reopened.Count()
	if count != 1 {
		t.Errorf("expected 1 vector after delete, got %d", count)
	}
}

func TestVectorStoreIDs(t *testing.T) {
	vs, _ := newTestVectorStore(t, 3)

	if err := vs.Upsert([]port.VectorItem{
		{ID: "a", Vector: []float32{1, 0, 0}},
		{ID: "b", Vector: []float32{0, 1, 0}},
	}); err != nil {
		t.Fatalf("upsert failed: %v", err)
	}

	ids, err := vs.IDs()
	if err != nil {
		t.Fatalf("IDs failed: %v", err)
	}
	if len(ids) != 2 {
		t.Fatalf("expected 2 ids, got %d", len(ids))
	}

	seen := map[string]bool{}
	for _, id := range ids {
		seen[id] = true
	}
	if !seen["a"] || !seen["b"] {
		t.Errorf("expected ids a and b, got %v", ids)
	}
}

func TestVectorStoreSearchSubsetOnlyReturnsRequestedIDs(t *testing.T) {
	vs, _ := newTestVectorStore(t, 3)

	if err := vs.Upsert([]port.VectorItem{
		{ID: "a", Vector: []float32{1, 0, 0}},
		{ID: "b", Vector: []float32{0, 1, 0}},
		{ID: "c", Vector: []float32{0, 0, 1}},
	}); err != nil {
		t.Fatalf("upsert failed: %v", err)
	}

	results, err := vs.SearchSubset([]float32{1, 0, 0}, []string{"b", "c", "missing"})
	if err != nil {
		t.Fatalf("subset search failed: %v", err)
	}

	if len(results) != 2 {
		t.Fatalf("expected 2 results (unknown ids skipped), got %d", len(results))
	}
	for _, r := range results {
		if r.ID == "a" || r.ID == "missing" {
			t.Errorf("unexpected id in subset results: %s", r.ID)
		}
	}
}

func TestVectorStoreZeroVectorScoresZeroInsteadOfNaN(t *testing.T) {
	vs, _ := newTestVectorStore(t, 3)

	if err := vs.Upsert([]port.VectorItem{{ID: "zero", Vector: []float32{0, 0, 0}}}); err != nil {
		t.Fatalf("upsert failed: %v", err)
	}

	results, err := vs.Search([]float32{1, 0, 0}, 1)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].Score != 0 {
		t.Errorf("zero vector should score 0, got %f", results[0].Score)
	}
}

func TestVectorStoreRejectsNonPositiveDimension(t *testing.T) {
	db, err := bbolt.Open(filepath.Join(t.TempDir(), "test.db"), 0600, nil)
	if err != nil {
		t.Fatalf("failed to open db: %v", err)
	}
	defer db.Close()

	if _, err := NewBoltVectorStore(db, 0); err == nil {
		t.Error("expected a zero dimension to be rejected")
	}
}
