package retriever

import (
	"os"
	"testing"

	"rag/internal/adapter/analyzer"
	"rag/internal/adapter/store"
	"rag/internal/domain"
)

func TestBM25Scoring(t *testing.T) {

	tmpDir, err := os.MkdirTemp("", "bm25_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	st, err := store.NewBoltStore(tmpDir + "/test.db")
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	tokenizer := analyzer.NewTokenizer(true)

	testChunks := []struct {
		id     string
		text   string
		tokens []string
	}{
		{
			id:     "chunk1",
			text:   "This is a test document about authentication and login",
			tokens: tokenizer.Tokenize("This is a test document about authentication and login"),
		},
		{
			id:     "chunk2",
			text:   "Database connection pooling and query optimization",
			tokens: tokenizer.Tokenize("Database connection pooling and query optimization"),
		},
		{
			id:     "chunk3",
			text:   "User authentication with JWT tokens and OAuth",
			tokens: tokenizer.Tokenize("User authentication with JWT tokens and OAuth"),
		},
	}

	for _, tc := range testChunks {
		chunk := domain.Chunk{
			ID:        tc.id,
			DocID:     "doc1",
			StartLine: 1,
			EndLine:   10,
			Tokens:    tc.tokens,
			Text:      tc.text,
		}
		if err := st.PutChunk(chunk); err != nil {
			t.Fatal(err)
		}

		tf := make(map[string]int)
		for _, token := range tc.tokens {
			tf[token]++
		}
		for term, count := range tf {
			if err := st.PutPosting(term, tc.id, count); err != nil {
				t.Fatal(err)
			}
		}
	}

	totalTokens := 0
	for _, tc := range testChunks {
		totalTokens += len(tc.tokens)
	}
	stats := domain.Stats{
		TotalDocs:   1,
		TotalChunks: 3,
		AvgChunkLen: float64(totalTokens) / 3.0,
	}
	if err := st.UpdateStats(stats); err != nil {
		t.Fatal(err)
	}

	retriever := NewBM25Retriever(st, tokenizer, 1.2, 0.75, 0)

	results, err := retriever.Search("authentication", 10)
	if err != nil {
		t.Fatal(err)
	}

	if len(results) == 0 {
		t.Fatal("expected results for 'authentication' query")
	}

	hasAuth := false
	for _, r := range results[:2] {
		if r.Chunk.ID == "chunk1" || r.Chunk.ID == "chunk3" {
			hasAuth = true
			break
		}
	}
	if !hasAuth {
		t.Error("expected authentication-related chunks to be in top results")
	}

	results, err = retriever.Search("database", 10)
	if err != nil {
		t.Fatal(err)
	}

	if len(results) == 0 {
		t.Fatal("expected results for 'database' query")
	}

	if results[0].Chunk.ID != "chunk2" {
		t.Errorf("expected chunk2 to be top result for 'database', got %s", results[0].Chunk.ID)
	}
}

func TestBM25EmptyQuery(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "bm25_empty_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	st, err := store.NewBoltStore(tmpDir + "/test.db")
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	tokenizer := analyzer.NewTokenizer(true)
	retriever := NewBM25Retriever(st, tokenizer, 1.2, 0.75, 0)

	results, err := retriever.Search("", 10)
	if err != nil {
		t.Fatal(err)
	}

	if len(results) != 0 {
		t.Errorf("expected no results for empty query, got %d", len(results))
	}
}

func TestBM25NoMatches(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "bm25_nomatch_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	st, err := store.NewBoltStore(tmpDir + "/test.db")
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	tokenizer := analyzer.NewTokenizer(true)

	chunk := domain.Chunk{
		ID:        "chunk1",
		DocID:     "doc1",
		StartLine: 1,
		EndLine:   10,
		Tokens:    tokenizer.Tokenize("hello world"),
		Text:      "hello world",
	}
	st.PutChunk(chunk)
	st.PutPosting("hello", "chunk1", 1)
	st.PutPosting("world", "chunk1", 1)
	st.UpdateStats(domain.Stats{TotalDocs: 1, TotalChunks: 1, AvgChunkLen: 2})

	retriever := NewBM25Retriever(st, tokenizer, 1.2, 0.75, 0)

	results, err := retriever.Search("zzzznonexistent", 10)
	if err != nil {
		t.Fatal(err)
	}

	if len(results) != 0 {
		t.Errorf("expected no results for non-matching query, got %d", len(results))
	}
}
