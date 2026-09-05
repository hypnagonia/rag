package retriever

import (
	"errors"
	"strings"
	"testing"

	"rag/internal/domain"
)

type countingLLM struct {
	calls    int
	response string
	err      error
}

func (l *countingLLM) Generate(prompt string) (string, error) {
	return l.GenerateWithSystem("", prompt)
}

func (l *countingLLM) GenerateWithSystem(system, user string) (string, error) {
	l.calls++
	if l.err != nil {
		return "", l.err
	}
	return l.response, nil
}

func (l *countingLLM) ModelName() string { return "counting" }

type recordingRetriever struct {
	queries []string
	results []domain.ScoredChunk
}

func (r *recordingRetriever) Search(query string, k int) ([]domain.ScoredChunk, error) {
	r.queries = append(r.queries, query)
	return r.results, nil
}

type memCache struct {
	data map[string]string
}

func newMemCache() *memCache { return &memCache{data: map[string]string{}} }

func (c *memCache) Get(q string) (string, bool) { v, ok := c.data[q]; return v, ok }
func (c *memCache) Put(q, h string) error       { c.data[q] = h; return nil }

func TestHyDEMakesExactlyOneLLMCallPerSearch(t *testing.T) {
	llm := &countingLLM{response: "Eddard Stark was beheaded on the steps of Baelor."}
	inner := &recordingRetriever{results: scored("a")}

	hyde, err := NewHyDEBuilder().Inner(inner).LLM(llm).Build()
	if err != nil {
		t.Fatalf("build failed: %v", err)
	}

	if _, err := hyde.Search("How did Ned Stark die", 5); err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if llm.calls != 1 {
		t.Errorf("expected exactly 1 LLM call, got %d", llm.calls)
	}
	if hyde.Stats().LLMCalls != 1 {
		t.Errorf("stats should report 1 call, got %d", hyde.Stats().LLMCalls)
	}
}

func TestHyDECacheHitCostsZeroLLMCalls(t *testing.T) {
	llm := &countingLLM{response: "hypothetical answer"}
	cache := newMemCache()
	inner := &recordingRetriever{results: scored("a")}

	build := func() *HyDERetriever {
		h, err := NewHyDEBuilder().Inner(inner).LLM(llm).Cache(cache).Build()
		if err != nil {
			t.Fatalf("build failed: %v", err)
		}
		return h
	}

	if _, err := build().Search("same question", 5); err != nil {
		t.Fatalf("first search failed: %v", err)
	}
	if llm.calls != 1 {
		t.Fatalf("expected 1 call after first search, got %d", llm.calls)
	}

	second := build()
	if _, err := second.Search("same question", 5); err != nil {
		t.Fatalf("second search failed: %v", err)
	}

	if llm.calls != 1 {
		t.Errorf("a cached query must not call the LLM again, got %d total calls", llm.calls)
	}
	if !second.Stats().CacheHit {
		t.Error("stats should report a cache hit")
	}
	if second.Stats().LLMCalls != 0 {
		t.Errorf("cache hit should report 0 llm calls, got %d", second.Stats().LLMCalls)
	}
}

func TestHyDEAppendsHypotheticalToInnerQuery(t *testing.T) {
	llm := &countingLLM{response: "Ser Ilyn Payne struck off his head with Ice."}
	inner := &recordingRetriever{results: scored("a")}

	hyde, _ := NewHyDEBuilder().Inner(inner).LLM(llm).Build()
	if _, err := hyde.Search("How did Ned Stark die", 5); err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(inner.queries) != 1 {
		t.Fatalf("expected 1 inner search, got %d", len(inner.queries))
	}
	q := inner.queries[0]
	if !strings.Contains(q, "How did Ned Stark die") {
		t.Errorf("inner query must keep the original question, got %q", q)
	}
	if !strings.Contains(q, "Ser Ilyn Payne") {
		t.Errorf("inner query must include the hypothetical, got %q", q)
	}
}

func TestHyDEFallsBackToPlainQueryWhenLLMFails(t *testing.T) {
	llm := &countingLLM{err: errors.New("402 insufficient balance")}
	inner := &recordingRetriever{results: scored("a", "b")}

	hyde, _ := NewHyDEBuilder().Inner(inner).LLM(llm).Build()
	results, err := hyde.Search("How did Ned Stark die", 5)
	if err != nil {
		t.Fatalf("HyDE must degrade gracefully, got error: %v", err)
	}
	if len(results) != 2 {
		t.Errorf("expected the inner retriever's results, got %d", len(results))
	}
	if inner.queries[0] != "How did Ned Stark die" {
		t.Errorf("fallback must use the plain query, got %q", inner.queries[0])
	}
	if hyde.Stats().Err == nil {
		t.Error("stats should record the LLM failure")
	}
}

func TestHyDEFallsBackWhenLLMReturnsEmpty(t *testing.T) {
	llm := &countingLLM{response: "   "}
	inner := &recordingRetriever{results: scored("a")}

	hyde, _ := NewHyDEBuilder().Inner(inner).LLM(llm).Build()
	if _, err := hyde.Search("q", 5); err != nil {
		t.Fatalf("search failed: %v", err)
	}
	if inner.queries[0] != "q" {
		t.Errorf("empty hypothetical should fall back to the plain query, got %q", inner.queries[0])
	}
}

func TestHyDEBuilderRequiresInnerAndLLM(t *testing.T) {
	if _, err := NewHyDEBuilder().LLM(&countingLLM{}).Build(); err == nil {
		t.Error("expected a missing inner retriever to be rejected")
	}
	if _, err := NewHyDEBuilder().Inner(&recordingRetriever{}).Build(); err == nil {
		t.Error("expected a missing LLM to be rejected")
	}
}
