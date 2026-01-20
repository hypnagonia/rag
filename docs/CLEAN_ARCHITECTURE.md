# Clean Architecture Refactor Analysis

## Current Architecture Map

```
rag/
├── cmd/rag/main.go           # Entry point, minimal wiring
├── config/                   # Configuration loading
├── internal/
│   ├── domain/entities.go    # Core entities (Document, Chunk, etc.)
│   ├── port/                 # Interfaces (mostly clean)
│   ├── usecase/              # Business logic (has violations)
│   ├── adapter/              # Implementations
│   │   ├── store/            # BoltDB storage
│   │   ├── retriever/        # BM25, Hybrid, Semantic
│   │   ├── embedding/        # OpenAI, Ollama, etc.
│   │   ├── chunker/          # Line, Composite chunkers
│   │   ├── analyzer/         # Tokenizer, stemmer
│   │   ├── fs/               # File walker
│   │   └── cache/            # Query cache
│   └── cli/                  # Cobra command handlers
└── examples/                 # Example applications
```

## Boundary Violations Found

### 1. UseCase → Adapter Dependencies (CRITICAL)

**File: `internal/usecase/index.go`**
```go
import (
    "rag/internal/adapter/analyzer"  // ❌ concrete adapter
    "rag/internal/adapter/fs"        // ❌ concrete adapter
    "rag/internal/adapter/store"     // ❌ concrete adapter
)

type IndexUseCase struct {
    store     *store.BoltStore       // ❌ concrete type
    walker    *fs.Walker             // ❌ concrete type
    tokenizer *analyzer.Tokenizer    // ❌ concrete type
}
```

**File: `internal/usecase/retrieve.go`**
```go
import (
    "rag/internal/adapter/retriever"  // ❌ concrete adapter
)

type RetrieveUseCase struct {
    mmrReranker *retriever.MMRReranker  // ❌ concrete type
}
```

### 2. Transport Concerns in UseCase Layer

**File: `internal/usecase/retrieve.go`**
```go
type ScoredChunkResult struct {
    Path      string  `json:"path"`      // ❌ JSON tags are transport concern
    StartLine int     `json:"start_line"`
    ...
}
```

### 3. Business Logic in CLI Handlers

**File: `internal/cli/index.go`**
- `generateEmbeddings()` function (lines 186-293) contains business logic
- Embedder provider selection logic should be in usecase or factory

### 4. Domain Entities with JSON Tags

**File: `internal/domain/entities.go`**
- `PackedContext`, `Snippet`, `Symbol` have JSON tags
- Minor violation, acceptable if same structure used for both domain and API

---

## Target Structure

```
rag/
├── cmd/
│   └── rag/
│       └── main.go              # Wiring ONLY
│
├── config/
│   └── config.go                # Config loading (unchanged)
│
├── internal/
│   ├── domain/                  # Core business entities
│   │   ├── document.go
│   │   ├── chunk.go
│   │   ├── query.go
│   │   └── errors.go            # NEW: domain errors
│   │
│   ├── usecase/                 # Business logic (verb-based)
│   │   ├── index_files.go       # IndexFiles use case
│   │   ├── search_chunks.go     # SearchChunks use case
│   │   ├── pack_context.go      # PackContext use case
│   │   └── generate_embeddings.go # NEW: moved from CLI
│   │
│   ├── port/                    # Interfaces (consumer-defined)
│   │   ├── store.go             # IndexStore, VectorStore
│   │   ├── retriever.go         # Retriever
│   │   ├── reranker.go          # NEW: MMRReranker interface
│   │   ├── embedder.go          # Embedder
│   │   ├── chunker.go           # Chunker
│   │   ├── tokenizer.go         # Tokenizer interface
│   │   └── walker.go            # NEW: FileWalker interface
│   │
│   ├── adapter/                 # Interface implementations
│   │   ├── store/
│   │   ├── retriever/
│   │   ├── embedding/
│   │   ├── chunker/
│   │   ├── analyzer/
│   │   ├── fs/
│   │   └── cache/
│   │
│   └── cli/                     # Transport layer (thin handlers)
│       ├── root.go
│       ├── index.go             # Parse args → call usecase → format output
│       ├── query.go
│       └── pack.go
│
└── examples/
```

---

## Use-Case List

| Use Case | Verb | Inputs | Outputs | Errors |
|----------|------|--------|---------|--------|
| `IndexFiles` | Index documents | root path, options | IndexResult | ErrInvalidPath, ErrStoreFailed |
| `SearchChunks` | Search indexed chunks | query, topK, options | []ScoredChunk | ErrNoIndex, ErrSearchFailed |
| `PackContext` | Pack context for LLM | query, budget | PackedContext | ErrInvalidBudget |
| `GenerateEmbeddings` | Create embeddings | chunks | count | ErrEmbeddingFailed |

---

## Ports/Interfaces List

### Existing (keep)
- `port.IndexStore` - document/chunk CRUD
- `port.VectorStore` - vector storage
- `port.Retriever` - search interface
- `port.Embedder` - embedding generation
- `port.Chunker` - content chunking

### New (extract)
```go
// port/reranker.go
type Reranker interface {
    Rerank(chunks []domain.ScoredChunk, k int) []domain.ScoredChunk
}

// port/tokenizer.go
type Tokenizer interface {
    Tokenize(text string) []string
    TokenizeWithPositions(text string) []TokenPosition
}

// port/walker.go
type FileWalker interface {
    Walk(root string) ([]FileInfo, error)
}

type FileInfo struct {
    Path    string
    ModTime int64
}
```

---

## Refactor Plan (Ordered Commits)

### Step 1: Extract Tokenizer Interface
- Create `port/tokenizer.go` with interface
- Update `IndexUseCase` to use interface
- No behavior change

### Step 2: Extract FileWalker Interface
- Create `port/walker.go` with interface
- Move `fs.FileInfo` to port or domain
- Update `IndexUseCase` to use interface

### Step 3: Extract Reranker Interface
- Create `port/reranker.go` with interface
- Update `RetrieveUseCase` to use interface
- MMRReranker implements the interface

### Step 4: Fix IndexUseCase Dependencies
- Change `*store.BoltStore` → `port.IndexStore`
- Remove concrete adapter imports
- Update constructor

### Step 5: Move Embedding Logic to UseCase
- Create `usecase/generate_embeddings.go`
- Move logic from `cli/index.go`
- CLI calls usecase

### Step 6: Extract Domain Errors
- Create `domain/errors.go`
- Define `ErrNotFound`, `ErrInvalidInput`, etc.
- Adapters wrap/translate errors

### Step 7: Clean Up Transport Concerns
- Move `ScoredChunkResult` to CLI layer
- Create DTOs where needed

---

## Vertical Slice: SearchChunks (Query Command)

### Current Flow
```
cli/query.go → usecase/retrieve.go → adapter/retriever/bm25.go
                                   → adapter/retriever/mmr.go
                                   → adapter/store/boltdb.go
```

### Target Flow (after refactor)
```
cli/query.go (thin)
    ↓ calls
usecase.SearchChunks(ctx, query, opts)
    ↓ uses ports
    port.Retriever.Search()
    port.Reranker.Rerank()
    ↓ returns
    []domain.ScoredChunk
    ↓
cli/query.go formats output
```

---

## First Refactor: Extract Reranker Interface

This is the simplest fix with highest impact.

### Files Changed
1. `internal/port/reranker.go` (NEW)
2. `internal/usecase/retrieve.go` (MODIFY)
3. `internal/adapter/retriever/mmr.go` (unchanged, already implements)

### Code

**internal/port/reranker.go**
```go
package port

import "rag/internal/domain"

type Reranker interface {
    Rerank(chunks []domain.ScoredChunk, k int) []domain.ScoredChunk
}
```

**internal/usecase/retrieve.go** (updated)
```go
package usecase

import (
    "rag/internal/domain"
    "rag/internal/port"
)

type RetrieveUseCase struct {
    retriever         port.Retriever
    reranker          port.Reranker  // ← interface, not concrete
    minScoreThreshold float64
}

func NewRetrieveUseCase(
    retriever port.Retriever,
    reranker port.Reranker,  // ← interface
    minScoreThreshold float64,
) *RetrieveUseCase {
    return &RetrieveUseCase{
        retriever:         retriever,
        reranker:          reranker,
        minScoreThreshold: minScoreThreshold,
    }
}
```

### Why This Change
- Removes direct dependency on `adapter/retriever` package
- UseCase now depends only on ports (interfaces)
- MMRReranker already has the right method signature
- Zero behavior change, all tests pass

---

## Test Examples

### Unit Test for SearchChunks UseCase
```go
func TestRetrieveUseCase_Retrieve(t *testing.T) {
    tests := []struct {
        name          string
        query         string
        topK          int
        searchResults []domain.ScoredChunk
        rerankResults []domain.ScoredChunk
        threshold     float64
        want          []domain.ScoredChunk
        wantErr       bool
    }{
        {
            name:  "returns reranked results",
            query: "test query",
            topK:  5,
            searchResults: []domain.ScoredChunk{
                {Chunk: domain.Chunk{ID: "1"}, Score: 0.9},
                {Chunk: domain.Chunk{ID: "2"}, Score: 0.8},
            },
            rerankResults: []domain.ScoredChunk{
                {Chunk: domain.Chunk{ID: "2"}, Score: 0.85},
                {Chunk: domain.Chunk{ID: "1"}, Score: 0.80},
            },
            want: []domain.ScoredChunk{
                {Chunk: domain.Chunk{ID: "2"}, Score: 0.85},
                {Chunk: domain.Chunk{ID: "1"}, Score: 0.80},
            },
        },
        {
            name:          "filters by threshold",
            query:         "test",
            topK:          5,
            threshold:     0.5,
            searchResults: []domain.ScoredChunk{{Score: 0.6}, {Score: 0.3}},
            rerankResults: []domain.ScoredChunk{{Score: 0.6}, {Score: 0.3}},
            want:          []domain.ScoredChunk{{Score: 0.6}},
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            mockRetriever := &MockRetriever{results: tt.searchResults}
            mockReranker := &MockReranker{results: tt.rerankResults}

            uc := NewRetrieveUseCase(mockRetriever, mockReranker, tt.threshold)
            got, err := uc.Retrieve(tt.query, tt.topK)

            if (err != nil) != tt.wantErr {
                t.Errorf("error = %v, wantErr %v", err, tt.wantErr)
            }
            if !reflect.DeepEqual(got, tt.want) {
                t.Errorf("got %v, want %v", got, tt.want)
            }
        })
    }
}

type MockRetriever struct {
    results []domain.ScoredChunk
}

func (m *MockRetriever) Search(query string, k int) ([]domain.ScoredChunk, error) {
    return m.results, nil
}

type MockReranker struct {
    results []domain.ScoredChunk
}

func (m *MockReranker) Rerank(chunks []domain.ScoredChunk, k int) []domain.ScoredChunk {
    return m.results
}
```
