package domain

import "time"

type Document struct {
	ID      string
	Path    string
	ModTime time.Time
	Lang    string
}

type Chunk struct {
	ID        string
	DocID     string
	StartLine int
	EndLine   int
	Tokens    []string
	Text      string
}

type Query struct {
	Text string
}

type ScoredChunk struct {
	Chunk Chunk
	Score float64
}

type PackedContext struct {
	Query         string    `json:"query"`
	BudgetTokens  int       `json:"budget_tokens"`
	UsedTokens    int       `json:"used_tokens"`
	Snippets      []Snippet `json:"snippets"`
	OpenQuestions []string  `json:"open_questions,omitempty"`
	Assumptions   []string  `json:"assumptions,omitempty"`
}

type Snippet struct {
	Path  string `json:"path"`
	Range string `json:"range"`
	Why   string `json:"why"`
	Text  string `json:"text"`
}

type Posting struct {
	ChunkID string
	TF      int
}

type Stats struct {
	TotalDocs   int
	TotalChunks int
	AvgChunkLen float64
}

type Symbol struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Type      string `json:"type"`
	DocID     string `json:"doc_id"`
	Line      int    `json:"line"`
	Signature string `json:"signature,omitempty"`
	ChunkID   string `json:"chunk_id,omitempty"`
}

type CallGraphEntry struct {
	CallerID string `json:"caller_id"`
	CalleeID string `json:"callee_id"`
	Line     int    `json:"line"`
}

type ChunkMetadata struct {
	Type      string   `json:"type,omitempty"`
	Name      string   `json:"name,omitempty"`
	Signature string   `json:"signature,omitempty"`
	Symbols   []string `json:"symbols,omitempty"`
	Imports   []string `json:"imports,omitempty"`
	Calls     []string `json:"calls,omitempty"`
	CalledBy  []string `json:"called_by,omitempty"`
	ParentID  string   `json:"parent_id,omitempty"`
}
