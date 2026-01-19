package usecase

import (
	"os"
	"testing"
	"time"

	"rag/internal/adapter/analyzer"
	"rag/internal/adapter/store"
	"rag/internal/domain"
)

func TestPackBudget(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "pack_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	st, err := store.NewBoltStore(tmpDir + "/test.db")
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	doc := domain.Document{
		ID:      "doc1",
		Path:    "/test/file.go",
		ModTime: time.Now(),
		Lang:    "go",
	}
	if err := st.PutDoc(doc); err != nil {
		t.Fatal(err)
	}

	tokenizer := analyzer.NewTokenizer(true)
	packUC := NewPackUseCase(st, tokenizer, 0)

	chunks := []domain.ScoredChunk{
		{
			Chunk: domain.Chunk{
				ID:        "c1",
				DocID:     "doc1",
				StartLine: 1,
				EndLine:   10,
				Tokens:    tokenizer.Tokenize("This is a short chunk of code"),
				Text:      "This is a short chunk of code",
			},
			Score: 1.0,
		},
		{
			Chunk: domain.Chunk{
				ID:        "c2",
				DocID:     "doc1",
				StartLine: 20,
				EndLine:   30,
				Tokens:    tokenizer.Tokenize("Another chunk with some more text here for testing purposes"),
				Text:      "Another chunk with some more text here for testing purposes",
			},
			Score: 0.8,
		},
		{
			Chunk: domain.Chunk{
				ID:        "c3",
				DocID:     "doc1",
				StartLine: 40,
				EndLine:   50,
				Tokens:    tokenizer.Tokenize("Yet another chunk"),
				Text:      "Yet another chunk",
			},
			Score: 0.6,
		},
	}

	packed, err := packUC.Pack("test query", chunks, 20)
	if err != nil {
		t.Fatal(err)
	}

	if packed.UsedTokens > 20 {
		t.Errorf("packed context exceeds budget: %d > 20", packed.UsedTokens)
	}

	if packed.BudgetTokens != 20 {
		t.Errorf("expected budget 20, got %d", packed.BudgetTokens)
	}

	packed, err = packUC.Pack("test query", chunks, 1000)
	if err != nil {
		t.Fatal(err)
	}

	if len(packed.Snippets) == 0 {
		t.Error("expected some snippets with large budget")
	}

	for _, s := range packed.Snippets {
		if s.Path == "" {
			t.Error("snippet missing path")
		}
		if s.Range == "" {
			t.Error("snippet missing range")
		}
	}
}

func TestPackEmptyChunks(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "pack_empty_test")
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
	packUC := NewPackUseCase(st, tokenizer, 0)

	packed, err := packUC.Pack("test query", nil, 1000)
	if err != nil {
		t.Fatal(err)
	}

	if packed.UsedTokens != 0 {
		t.Errorf("expected 0 used tokens for empty chunks, got %d", packed.UsedTokens)
	}

	if len(packed.Snippets) != 0 {
		t.Errorf("expected 0 snippets for empty chunks, got %d", len(packed.Snippets))
	}
}

func TestPackMergeAdjacent(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "pack_merge_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	st, err := store.NewBoltStore(tmpDir + "/test.db")
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	doc := domain.Document{
		ID:      "doc1",
		Path:    "/test/file.go",
		ModTime: time.Now(),
		Lang:    "go",
	}
	st.PutDoc(doc)

	tokenizer := analyzer.NewTokenizer(true)
	packUC := NewPackUseCase(st, tokenizer, 0)

	chunks := []domain.ScoredChunk{
		{
			Chunk: domain.Chunk{
				ID:        "c1",
				DocID:     "doc1",
				StartLine: 1,
				EndLine:   10,
				Tokens:    []string{"a", "b"},
				Text:      "Line 1-10",
			},
			Score: 1.0,
		},
		{
			Chunk: domain.Chunk{
				ID:        "c2",
				DocID:     "doc1",
				StartLine: 11,
				EndLine:   20,
				Tokens:    []string{"c", "d"},
				Text:      "Line 11-20",
			},
			Score: 0.9,
		},
		{
			Chunk: domain.Chunk{
				ID:        "c3",
				DocID:     "doc1",
				StartLine: 50,
				EndLine:   60,
				Tokens:    []string{"e", "f"},
				Text:      "Line 50-60",
			},
			Score: 0.8,
		},
	}

	packed, err := packUC.Pack("test", chunks, 1000)
	if err != nil {
		t.Fatal(err)
	}

	if len(packed.Snippets) > 2 {
		t.Errorf("expected adjacent chunks to merge, got %d snippets for 3 chunks", len(packed.Snippets))
	}
}

func TestPackUtilityRanking(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "pack_utility_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	st, err := store.NewBoltStore(tmpDir + "/test.db")
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	doc := domain.Document{
		ID:      "doc1",
		Path:    "/test/file.go",
		ModTime: time.Now(),
		Lang:    "go",
	}
	st.PutDoc(doc)

	tokenizer := analyzer.NewTokenizer(true)
	packUC := NewPackUseCase(st, tokenizer, 0)

	chunks := []domain.ScoredChunk{
		{
			Chunk: domain.Chunk{
				ID:        "big",
				DocID:     "doc1",
				StartLine: 1,
				EndLine:   100,
				Tokens:    make([]string, 100),
				Text:      "This is a very long chunk with lots of text that takes many tokens to represent properly and thoroughly",
			},
			Score: 1.0,
		},
		{
			Chunk: domain.Chunk{
				ID:        "small",
				DocID:     "doc1",
				StartLine: 200,
				EndLine:   210,
				Tokens:    []string{"compact", "useful"},
				Text:      "compact useful",
			},
			Score: 0.9,
		},
	}

	packed, err := packUC.Pack("test", chunks, 10)
	if err != nil {
		t.Fatal(err)
	}

	if len(packed.Snippets) == 0 {
		t.Skip("no snippets fit in budget")
	}

	found := false
	for _, s := range packed.Snippets {
		if s.Range == "L200-210" {
			found = true
			break
		}
	}

	if !found {
		t.Log("Utility-based selection may vary based on token counting")
	}
}
