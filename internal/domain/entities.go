package domain

import "time"

// Document represents an indexed file.
type Document struct {
	ID      string
	Path    string
	ModTime time.Time
	Lang    string
}

// Chunk represents a portion of a document.
type Chunk struct {
	ID        string
	DocID     string
	StartLine int
	EndLine   int
	Tokens    []string
	Text      string
}

// Query represents a search query.
type Query struct {
	Text string
}

// ScoredChunk is a chunk with a relevance score.
type ScoredChunk struct {
	Chunk Chunk
	Score float64
}

// PackedContext is the compressed context output for LLM consumption.
type PackedContext struct {
	Query         string        `json:"query"`
	BudgetTokens  int           `json:"budget_tokens"`
	UsedTokens    int           `json:"used_tokens"`
	Snippets      []Snippet     `json:"snippets"`
	OpenQuestions []string      `json:"open_questions,omitempty"`
	Assumptions   []string      `json:"assumptions,omitempty"`
}

// Snippet is a code/text snippet with citation metadata.
type Snippet struct {
	Path  string `json:"path"`
	Range string `json:"range"`
	Why   string `json:"why"`
	Text  string `json:"text"`
}

// Posting represents a term occurrence in a chunk.
type Posting struct {
	ChunkID string
	TF      int
}

// Stats holds corpus-level statistics for BM25.
type Stats struct {
	TotalDocs   int
	TotalChunks int
	AvgChunkLen float64
}
