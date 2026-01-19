# CLAUDE.md

## Build & Test Commands

```bash
go build -o rag ./cmd/rag    # Build CLI
go test ./...                 # Run all tests
go test -v ./internal/...     # Verbose tests for internal packages
go mod tidy                   # Update dependencies
```

## CLI Commands

```bash
rag index [path]              # Index files, creates .rag/index.db
rag query -q "search"         # Search with BM25 + MMR
rag pack -q "question" -b 4000  # Pack context for LLM consumption
```

## Architecture

**Layered hexagonal architecture:**

```
cmd/rag/main.go           → Entrypoint, delegates to internal/cli
config/config.go          → YAML config loading (rag.yaml)
internal/
  cli/                    → Cobra command handlers
  domain/entities.go      → Core entities (Document, Chunk, ScoredChunk, Query, PackedContext)
  port/                   → Interfaces (IndexStore, Retriever, Chunker, Tokenizer, Embedder)
  usecase/                → Business logic orchestration (index, retrieve, pack)
  adapter/                → Implementations
    store/boltdb.go       → BoltDB persistence (.rag/index.db)
    retriever/bm25.go     → BM25 scoring with path boosting
    retriever/mmr.go      → MMR diversity reranking
    retriever/hybrid.go   → BM25 + vector hybrid search (RRF fusion)
    chunker/              → Line-based and composite chunking
    analyzer/             → Tokenization, Porter stemming, symbol extraction
    embedding/            → Ollama/OpenAI embedding providers
```

## Key Interfaces (internal/port/)

- **IndexStore** - Document/chunk/posting persistence (BoltDB impl)
- **Retriever** - Search interface returning ScoredChunks
- **Chunker** - Split documents into token-aware chunks
- **Embedder** - Generate vector embeddings
- **VectorStore** - Vector similarity search

## Retrieval Pipeline

1. **BM25** scores chunks: `score = Σ IDF(t) × (tf × (k1+1)) / (tf + k1 × (1-b + b×|c|/avgDl))`
2. **Path boosting** adds score for query terms matching file path
3. **MMR** reranks for diversity: `MMR(c) = λ × relevance - (1-λ) × max_similarity`
4. **Hybrid mode** fuses BM25 + vector results via RRF: `score = Σ 1/(k + rank)`
5. **Packing** selects by utility (score/tokens), merges adjacent chunks

## Configuration (rag.yaml)

Key settings:
- `index.chunk_tokens` (512) - Max tokens per chunk
- `index.k1` (1.2), `index.b` (0.75) - BM25 parameters
- `retrieve.top_k` (20), `retrieve.mmr_lambda` (0.7)
- `embedding.provider` (ollama), `embedding.model` (nomic-embed-text)
- `pack.token_budget` (4000)

## Agentic RAG Example

```bash
cd examples/agentic-rag
export DEEPSEEK_API_KEY=...
go run main.go -q "question" -index /path/to/indexed -expand
```

Flags: `-fast` (1 LLM call), `-expand` (query expansion), `-v` (verbose)

## Guidelines
- Do not commit Claude as project's coauthor
- Never add comments to generated code!
- Do not create constructors with many arguments - Builder pattern is preferable 
- Do git command 
  - create a new branch 
  - add all changes and commit them 
  - git push
  - create a remote PR
  - merge it to main using gh
  - delete the merged branch
  - switch back to main branch
  - pull updated main branch
