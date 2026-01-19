package retriever

import (
	"testing"

	"rag/internal/domain"
)

func TestMMRReranking(t *testing.T) {
	reranker := NewMMRReranker(0.7, 0.9)

	candidates := []domain.ScoredChunk{
		{
			Chunk: domain.Chunk{
				ID:     "c1",
				Tokens: []string{"auth", "login", "user", "password"},
			},
			Score: 1.0,
		},
		{
			Chunk: domain.Chunk{
				ID:     "c2",
				Tokens: []string{"auth", "login", "user", "session"},
			},
			Score: 0.9,
		},
		{
			Chunk: domain.Chunk{
				ID:     "c3",
				Tokens: []string{"database", "query", "sql", "connection"},
			},
			Score: 0.8,
		},
		{
			Chunk: domain.Chunk{
				ID:     "c4",
				Tokens: []string{"auth", "jwt", "token", "oauth"},
			},
			Score: 0.7,
		},
	}

	results := reranker.Rerank(candidates, 3)

	if len(results) == 0 {
		t.Fatal("expected results from MMR reranking")
	}

	if results[0].Chunk.ID != "c1" {
		t.Errorf("expected c1 as first result, got %s", results[0].Chunk.ID)
	}

	hasC3BeforeC2 := false
	c3Idx, c2Idx := -1, -1
	for i, r := range results {
		if r.Chunk.ID == "c3" {
			c3Idx = i
		}
		if r.Chunk.ID == "c2" {
			c2Idx = i
		}
	}

	if c3Idx != -1 && (c2Idx == -1 || c3Idx < c2Idx) {
		hasC3BeforeC2 = true
	}

	if !hasC3BeforeC2 && c2Idx != -1 && c3Idx != -1 {
		t.Error("expected MMR to prioritize diverse results (c3) over similar results (c2)")
	}
}

func TestMMRDeduplication(t *testing.T) {

	reranker := NewMMRReranker(0.5, 0.3)

	candidates := []domain.ScoredChunk{
		{
			Chunk: domain.Chunk{
				ID:     "c1",
				Tokens: []string{"a", "b", "c"},
			},
			Score: 1.0,
		},
		{
			Chunk: domain.Chunk{
				ID:     "c2",
				Tokens: []string{"a", "b", "c"},
			},
			Score: 0.9,
		},
	}

	results := reranker.Rerank(candidates, 2)

	if len(results) != 1 {
		t.Errorf("expected 1 result after dedup, got %d", len(results))
	}

	if results[0].Chunk.ID != "c1" {
		t.Errorf("expected c1 (highest score), got %s", results[0].Chunk.ID)
	}
}

func TestMMREmptyCandidates(t *testing.T) {
	reranker := NewMMRReranker(0.7, 0.8)

	results := reranker.Rerank(nil, 10)
	if results != nil {
		t.Errorf("expected nil for empty candidates, got %v", results)
	}

	results = reranker.Rerank([]domain.ScoredChunk{}, 10)
	if results != nil {
		t.Errorf("expected nil for empty slice, got %v", results)
	}
}

func TestJaccardSimilarity(t *testing.T) {
	tests := []struct {
		name     string
		a        []string
		b        []string
		expected float64
	}{
		{
			name:     "identical",
			a:        []string{"a", "b", "c"},
			b:        []string{"a", "b", "c"},
			expected: 1.0,
		},
		{
			name:     "no overlap",
			a:        []string{"a", "b", "c"},
			b:        []string{"d", "e", "f"},
			expected: 0.0,
		},
		{
			name:     "half overlap",
			a:        []string{"a", "b"},
			b:        []string{"b", "c"},
			expected: 1.0 / 3.0,
		},
		{
			name:     "empty a",
			a:        []string{},
			b:        []string{"a", "b"},
			expected: 0.0,
		},
		{
			name:     "empty b",
			a:        []string{"a", "b"},
			b:        []string{},
			expected: 0.0,
		},
		{
			name:     "both empty",
			a:        []string{},
			b:        []string{},
			expected: 1.0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := JaccardSimilarity(tc.a, tc.b)
			if !floatEquals(result, tc.expected, 0.001) {
				t.Errorf("JaccardSimilarity(%v, %v) = %f, expected %f", tc.a, tc.b, result, tc.expected)
			}
		})
	}
}

func floatEquals(a, b, tolerance float64) bool {
	diff := a - b
	if diff < 0 {
		diff = -diff
	}
	return diff < tolerance
}
