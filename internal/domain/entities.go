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
	Query         string    `json:"query"`
	BudgetTokens  int       `json:"budget_tokens"`
	UsedTokens    int       `json:"used_tokens"`
	Snippets      []Snippet `json:"snippets"`
	OpenQuestions []string  `json:"open_questions,omitempty"`
	Assumptions   []string  `json:"assumptions,omitempty"`
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

// Symbol represents a code symbol (function, type, method, etc.).
type Symbol struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Type      string `json:"type"` // "function", "method", "type", "interface", "struct", "variable", "constant"
	DocID     string `json:"doc_id"`
	Line      int    `json:"line"`
	Signature string `json:"signature,omitempty"`
	ChunkID   string `json:"chunk_id,omitempty"`
}

// CallGraphEntry represents a caller-callee relationship.
type CallGraphEntry struct {
	CallerID string `json:"caller_id"` // Symbol ID of the caller
	CalleeID string `json:"callee_id"` // Symbol ID of the callee (or name if external)
	Line     int    `json:"line"`      // Line where the call occurs
}

// ChunkMetadata contains enriched metadata for a chunk.
type ChunkMetadata struct {
	Type      string   `json:"type,omitempty"`      // "function", "class", "comment", "mixed"
	Name      string   `json:"name,omitempty"`      // Primary symbol name if applicable
	Signature string   `json:"signature,omitempty"` // Function/method signature
	Symbols   []string `json:"symbols,omitempty"`   // Symbol IDs contained in this chunk
	Imports   []string `json:"imports,omitempty"`   // Import paths
	Calls     []string `json:"calls,omitempty"`     // Functions/methods called
	CalledBy  []string `json:"called_by,omitempty"` // Functions/methods that call this
	ParentID  string   `json:"parent_id,omitempty"` // Parent chunk ID (for nested structures)
}
