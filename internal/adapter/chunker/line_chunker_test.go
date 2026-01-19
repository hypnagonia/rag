package chunker

import (
	"strings"
	"testing"

	"rag/internal/adapter/analyzer"
	"rag/internal/domain"
)

func TestLineChunkerBasic(t *testing.T) {
	tokenizer := analyzer.NewTokenizer(false)
	chunker := NewLineChunker(50, 10, tokenizer)

	doc := domain.Document{
		ID:   "doc1",
		Path: "/test/file.go",
	}

	content := `package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}

func helper() {
    // some helper function
    return
}`

	chunks, err := chunker.Chunk(doc, content)
	if err != nil {
		t.Fatal(err)
	}

	if len(chunks) == 0 {
		t.Fatal("expected at least one chunk")
	}

	for _, chunk := range chunks {
		if chunk.ID == "" {
			t.Error("chunk has empty ID")
		}
		if chunk.DocID != "doc1" {
			t.Errorf("expected DocID 'doc1', got '%s'", chunk.DocID)
		}
		if chunk.StartLine < 1 {
			t.Errorf("invalid StartLine: %d", chunk.StartLine)
		}
		if chunk.EndLine < chunk.StartLine {
			t.Errorf("EndLine (%d) < StartLine (%d)", chunk.EndLine, chunk.StartLine)
		}
		if chunk.Text == "" {
			t.Error("chunk has empty text")
		}
	}
}

func TestLineChunkerBoundaries(t *testing.T) {
	tokenizer := analyzer.NewTokenizer(false)

	chunker := NewLineChunker(10, 2, tokenizer)

	doc := domain.Document{
		ID:   "doc1",
		Path: "/test/file.go",
	}

	lines := []string{
		"Line one",
		"Line two",
		"Line three",
		"Line four",
		"Line five",
		"Line six",
		"Line seven",
		"Line eight",
	}
	content := strings.Join(lines, "\n")

	chunks, err := chunker.Chunk(doc, content)
	if err != nil {
		t.Fatal(err)
	}

	allText := ""
	for _, chunk := range chunks {
		if !strings.HasPrefix(allText, "") {
			allText += "\n"
		}

		if !strings.Contains(allText, chunk.Text) {
			allText += chunk.Text
		}
	}

	for _, line := range lines {
		found := false
		for _, chunk := range chunks {
			if strings.Contains(chunk.Text, line) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("line '%s' not found in any chunk", line)
		}
	}
}

func TestLineChunkerOverlap(t *testing.T) {
	tokenizer := analyzer.NewTokenizer(false)

	chunker := NewLineChunker(5, 2, tokenizer)

	doc := domain.Document{
		ID:   "doc1",
		Path: "/test/file.go",
	}

	content := "Line1\nLine2\nLine3\nLine4\nLine5"

	chunks, err := chunker.Chunk(doc, content)
	if err != nil {
		t.Fatal(err)
	}

	if len(chunks) < 2 {
		t.Skip("need at least 2 chunks to test overlap")
	}

	for i := 0; i < len(chunks)-1; i++ {
		current := chunks[i]
		next := chunks[i+1]

		if next.StartLine > current.EndLine+1 {
			t.Errorf("no overlap between chunk %d (ends at %d) and chunk %d (starts at %d)",
				i, current.EndLine, i+1, next.StartLine)
		}
	}
}

func TestLineChunkerEmptyContent(t *testing.T) {
	tokenizer := analyzer.NewTokenizer(false)
	chunker := NewLineChunker(50, 10, tokenizer)

	doc := domain.Document{
		ID:   "doc1",
		Path: "/test/empty.go",
	}

	chunks, err := chunker.Chunk(doc, "")
	if err != nil {
		t.Fatal(err)
	}

	if len(chunks) > 1 {
		t.Errorf("expected 0 or 1 chunks for empty content, got %d", len(chunks))
	}
}

func TestLineChunkerSingleLine(t *testing.T) {
	tokenizer := analyzer.NewTokenizer(false)
	chunker := NewLineChunker(50, 10, tokenizer)

	doc := domain.Document{
		ID:   "doc1",
		Path: "/test/single.go",
	}

	content := "Just a single line of code"

	chunks, err := chunker.Chunk(doc, content)
	if err != nil {
		t.Fatal(err)
	}

	if len(chunks) != 1 {
		t.Errorf("expected 1 chunk for single line, got %d", len(chunks))
	}

	if chunks[0].Text != content {
		t.Errorf("expected chunk text to match content")
	}

	if chunks[0].StartLine != 1 || chunks[0].EndLine != 1 {
		t.Errorf("expected lines 1-1, got %d-%d", chunks[0].StartLine, chunks[0].EndLine)
	}
}

func TestLineChunkerLongLine(t *testing.T) {
	tokenizer := analyzer.NewTokenizer(false)

	chunker := NewLineChunker(5, 0, tokenizer)

	doc := domain.Document{
		ID:   "doc1",
		Path: "/test/long.go",
	}

	content := "This is a very long line with many many words that will exceed the token limit"

	chunks, err := chunker.Chunk(doc, content)
	if err != nil {
		t.Fatal(err)
	}

	if len(chunks) == 0 {
		t.Error("expected at least one chunk even for oversized line")
	}

	if chunks[0].Text != content {
		t.Error("chunk should contain the full oversized line")
	}
}

func TestChunkIDUniqueness(t *testing.T) {
	tokenizer := analyzer.NewTokenizer(false)
	chunker := NewLineChunker(10, 2, tokenizer)

	doc := domain.Document{
		ID:   "doc1",
		Path: "/test/file.go",
	}

	content := "Line1\nLine2\nLine3\nLine4\nLine5\nLine6\nLine7\nLine8"

	chunks, err := chunker.Chunk(doc, content)
	if err != nil {
		t.Fatal(err)
	}

	ids := make(map[string]bool)
	for _, chunk := range chunks {
		if ids[chunk.ID] {
			t.Errorf("duplicate chunk ID: %s", chunk.ID)
		}
		ids[chunk.ID] = true
	}
}
