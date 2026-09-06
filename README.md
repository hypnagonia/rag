# RAG Context Compressor

A tool for indexing and searching text using hybrid search (BM25 + vector embeddings), with MMR deduplication and context packing for LLM consumption. Runs as CLI or in-browser via WebAssembly.

## Example: Querying Game of Thrones (5 books, ~2M tokens)

```bash
$ rag query -q "How did Ned Stark die" -index ./books -expand
```
> Ned Stark was executed by beheading after being accused of treason. King Joffrey,
> despite initially suggesting Ned could take the black, ordered his execution.
> Ser Ilyn Payne, the King's Justice, carried out the sentence at the steps of the
> Great Sept of Baelor [1.txt:16974-16996].
>
> **~3k tokens** (with RAG) vs **~2M tokens** (without RAG)

```bash
$ rag query -q "Joffrey death" -index ./books -expand
```
> Joffrey was murdered by poison at his own wedding feast. The poison used is
> identified as "the strangler," a rare substance that causes the throat muscles
> to clench, shutting off the windpipe and turning the victim's face purple
> [2.txt:L472-495]. During Tyrion's trial, Grand Maester Pycelle confirms that
> the strangler was used to kill Joffrey [3.txt:L22842-22868].
>
> **~4k tokens** (with RAG) vs **~2M tokens** (without RAG)

```bash
$ rag query -q "Red Wedding Robb Stark murdered" -index ./books -expand
```
> Robb Stark was betrayed and murdered by the Freys and Boltons at the Twins
> during his uncle's wedding, an event known as the Red Wedding [4.txt:21098-21138].
>
> **~3k tokens** (with RAG) vs **~2M tokens** (without RAG)

```bash
$ rag query -q "How did Drogo die" -index ./books -expand
```
> Drogo died after being placed in a comatose state by a bloodmagic ritual
> performed by Mirri Maz Duur. The ritual involved sacrificing his horse and
> using its blood, but it left Drogo alive yet unresponsive [1.txt:L16544-16620].
> Mirri Maz Duur states that Drogo will only return to his former self under
> impossible conditions, implying he will never recover [1.txt:L17745-17772].
>
> **~5k tokens** (with RAG) vs **~2M tokens** (without RAG)

<sub>Responses generated using DeepSeek with hybrid search (BM25 + embeddings)</sub>

---

## Installation

```bash
go build -o rag ./cmd/rag
```

## Quick Start

First, create a `rag.yaml` config in your content directory:

```yaml
# books/rag.yaml
index:
  includes:
    - "**/*.txt"
  chunk_tokens: 512
  stemming: true

retrieve:
  top_k: 20
  mmr_lambda: 0.7

pack:
  token_budget: 4000
```

```bash
# Index a directory
rag index /path/to/books

# Search for relevant code
rag query -q "authentication handler"

# Pack context for LLM consumption
rag pack -q "how does auth work" -b 4000 -o context.json

# Generate a prompt for manual LLM orchestration
rag runprompt --runtime --ctx context.json -q "Explain the auth flow"
```

## Commands

### `rag index <path>`

Index files in a directory for later retrieval. Creates a `.rag/index.db` file.

```bash
rag index .                      # Index current directory
rag index /path/to/project       # Index specific directory
```

**Flags:**
- `-d, --dir` - Root directory (default: current directory)
- `--config` - Path to config file (default: `./rag.yaml`)
- `--force-embed` - Re-embed every chunk instead of reusing existing vectors

### `rag query -q "<question>"`

Search indexed files using BM25 retrieval with MMR deduplication.

```bash
rag query -q "database connection"
rag query -q "error handling" --top-k 10 --json
rag query -q "how to handle errors" --semantic
```

**Flags:**
- `-q, --query` - Search query (required)
- `-k, --top-k` - Number of results (default from config)
- `--json` - Output as JSON
- `--no-mmr` - Disable MMR reranking
- `--semantic` - Use embedding-only search (no BM25); mutually exclusive with `--lexical`
- `--lexical` - Use BM25-only search (no embeddings)
- `--explain` - Print which retrieval arms ran and how many candidates each produced
- `--hyde` - Expand the query with one LLM-generated hypothetical answer (1 API call, cached)
- `-c, --context` - Expand results by N lines before/after

### `rag ask -q "<question>"`

Retrieve context and have a hosted LLM answer the question, with citations. This is the
full pipeline: hybrid retrieval plus generation.

```bash
rag ask -q "how does authentication work"
rag ask -q "how does authentication work" --fast    # exactly one LLM call
rag ask -q "how does authentication work" --hyde    # better retrieval, cached probe
```

**Flags:**
- `-q, --query` - Question (required)
- `--fast` - Search once and answer: exactly one LLM call
- `--expand` - Expand the query with the LLM first (+1 call)
- `--hyde` - Expand retrieval with a hypothetical answer (+1 call, cached)
- `--max-iters` - Maximum retrieve/evaluate rounds (default 2)
- `--semantic` - Vector search only, no BM25
- `--lexical` - BM25 only, no embeddings
- `-k, --top-k`, `-b, --budget`, `--explain`

Every run prints a stats block: retrieval mode, rounds used, LLM calls, and input/output/total
tokens (reported by the API when the provider returns a usage field).

The API key is read from `.env` automatically - see the `llm:` section under Configuration.

### `rag pack -q "<question>"`

Pack relevant chunks into compressed context that fits a token budget.

```bash
rag pack -q "authentication flow" -b 2000
rag pack -q "API endpoints" -o context.json
rag pack -q "session handling" --lexical
```

**Flags:**
- `-q, --query` - Search query (required)
- `-b, --budget` - Token budget (default from config)
- `-o, --output` - Output file (default: stdout)
- `-k, --top-k` - Candidate pool size

### `rag runprompt`

Generate formatted prompts from templates for manual LLM orchestration.

```bash
# Runtime prompt for question answering
rag runprompt --runtime --ctx context.json -q "How does auth work?"

# Builder prompt for context compression
rag runprompt --builder --ctx context.json
```

**Flags:**
- `--runtime` - Use runtime (answering) prompt template
- `--builder` - Use builder (compression) prompt template
- `--ctx` - Path to packed context JSON file (required)
- `-q, --query` - Override query for runtime prompt

## Configuration

Create a `rag.yaml` file in your project root:

```yaml
index:
  includes:
    - "**/*.go"
    - "**/*.py"
    - "**/*.js"
    - "**/*.ts"
    - "**/*.md"
  excludes:
    - "**/node_modules/**"
    - "**/vendor/**"
    - "**/.git/**"
  stemming: true
  chunk_tokens: 512
  chunk_overlap: 50
  k1: 1.2
  b: 0.75

retrieve:
  top_k: 20
  mmr_lambda: 0.7
  dedup_jaccard: 0.8

pack:
  token_budget: 4000
  output: json

logging:
  level: info
```

### Configuration Options

| Section | Option | Description | Default |
|---------|--------|-------------|---------|
| `index` | `includes` | Glob patterns for files to index | Common code extensions |
| `index` | `excludes` | Glob patterns to exclude | node_modules, vendor, .git |
| `index` | `stemming` | Enable Porter stemming | `true` |
| `index` | `chunk_tokens` | Max tokens per chunk | `512` |
| `index` | `chunk_overlap` | Token overlap between chunks | `50` |
| `index` | `k1` | BM25 k1 parameter | `1.2` |
| `index` | `b` | BM25 b parameter | `0.75` |
| `retrieve` | `top_k` | Default number of results | `20` |
| `retrieve` | `mmr_lambda` | MMR relevance vs diversity (0-1) | `0.7` |
| `retrieve` | `dedup_jaccard` | Jaccard threshold for dedup | `0.8` |
| `pack` | `token_budget` | Default token budget | `4000` |

### Hybrid Search (BM25 + Vector Embeddings)

To enable semantic search alongside BM25 keyword search, install [Ollama](https://ollama.ai/) and pull an embedding model:

```bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama server
ollama serve

# Pull embedding model
ollama pull nomic-embed-text
```

Then add embedding config to your `rag.yaml`:

```yaml
embedding:
  enabled: true
  provider: ollama
  model: nomic-embed-text
  dimension: 768

retrieve:
  hybrid_enabled: true
  rrf_k: 60          # RRF fusion parameter
  bm25_weight: 0.5   # Balance between BM25 and vector (0-1)
```

Re-index to generate embeddings:

```bash
rag index /path/to/content
```

Hybrid search runs both arms independently and fuses their rankings with Reciprocal Rank Fusion:

```
score(c) = bm25_weight / (rrf_k + rank_bm25) + (1 - bm25_weight) / (rrf_k + rank_vector)
```

Because the arms run independently, a chunk that only the vector arm finds still reaches
the results — BM25 does not gate the candidate pool. Use `--explain` to see how many
candidates each arm produced:

```bash
rag query -q "how are sessions validated" --explain
# retrieval: hybrid (bm25 + vector, RRF) (model=nomic-embed-text, vectors=1842)
# candidates: bm25=32 vector=80 fused=97
```

If embeddings are unavailable (provider down, index not embedded, model changed), query
and pack print a warning and fall back to BM25 rather than failing silently.

`rag pack` uses the same retrieval path as `rag query`, so hybrid search applies there too.

### Tuning vector quality

Retrieval quality depends far more on the embedding model and chunk size than on the
fusion parameters. Measured on a ~7.5MB prose corpus (14 questions, recall@10):

| Setting | Vector recall@10 |
|---|---|
| `nomic-embed-text`, ~2200-char chunks | 2/14 |
| `mxbai-embed-large`, ~540-char chunks | 6/14 |

Raising `k` shows the answers are being found but ranked low - vector recall@200 is 10/14.
If you need them in the top 10, add a reranking stage over a deep candidate pool; tuning
`bm25_weight` does not help (recall@10 was flat at 6/14 across 0.2-0.65).

`embedding.include_path` prepends `path:startLine-endLine` to each embedded chunk. This
helps for code, where the path carries real signal, and hurts for prose - on the corpus
above it cost 0.12 MRR. It defaults to `true`; set it to `false` for prose.

Changing `embedding.model`, `embedding.dimension`, or `embedding.include_path` invalidates
stored vectors. The index records what it was embedded with and re-embeds automatically.

### HyDE query expansion

`--hyde` makes exactly **one** LLM call per query to write a hypothetical answer passage,
appends it to the query, and searches with both. The generated text is cached in the index,
so repeating a query costs zero API calls. If the LLM fails, the search silently falls back
to the plain query.

Generation runs against a hosted API only - there is deliberately no local-LLM provider:

```yaml
llm:
  provider: deepseek        # or openai
  model: deepseek-chat
  api_key_env: DEEPSEEK_API_KEY
  max_tokens: 400
```

> **Note:** RRF scores are much smaller than BM25 scores (typically `0.005`-`0.03`). If you
> set `retrieve.min_score_threshold`, tune it for whichever mode you actually run — a
> threshold picked for BM25 scores will filter out every hybrid result.

**Embeddings are incremental.** Re-running `rag index` only embeds chunks that do not
already have a vector, and drops vectors for chunks that no longer exist. Use
`rag index --force-embed` to re-embed everything. Changing `embedding.model` triggers a
rebuild, discarding vectors from the old model.

### Semantic-Only Search

Use `--semantic` flag to search using only vector embeddings (no BM25 keyword matching):

```bash
rag query -q "a noble man betrayed by those he trusted" --semantic
```

Semantic search is useful for:
- Natural language questions (e.g., "how to handle errors gracefully")
- Conceptual queries where exact keywords may not appear
- Finding related content even when terminology differs

Requires embeddings to be enabled and indexed (see Hybrid Search section above).

## How It Works

### Indexing

1. Walks directory with glob patterns
2. Checks file modification times for incremental updates
3. Splits files into line-based chunks with token awareness
4. Tokenizes with optional Porter stemming
5. Builds inverted index with term frequencies
6. Stores in BoltDB (`.rag/index.db`)

### Retrieval

1. Tokenizes and stems query
2. Scores chunks using BM25:
   ```
   score(q,c) = Σ IDF(t) × (tf × (k1+1)) / (tf + k1 × (1-b + b×|c|/avgDl))
   ```
3. Applies MMR for diversity:
   ```
   MMR(c) = λ × relevance(c) - (1-λ) × max_similarity(c, selected)
   ```
4. Returns ranked, deduplicated results

### Packing

1. Calculates utility = score / token_count
2. Greedily selects chunks by utility until budget exhausted
3. Merges adjacent chunks from same file
4. Outputs JSON with citations (path, line range, relevance)

## Output Format

### Packed Context JSON

```json
{
  "query": "authentication",
  "budget_tokens": 4000,
  "used_tokens": 1250,
  "snippets": [
    {
      "path": "/src/auth/handler.go",
      "range": "L45-89",
      "why": "BM25 score: 2.34",
      "text": "func Authenticate(..."
    }
  ]
}
```

## WebAssembly (Browser)

RAG can run entirely in the browser via WebAssembly (BM25 search only, no embeddings).

### Build WASM

```bash
make build-wasm
# Or manually:
GOOS=js GOARCH=wasm go build -o examples/wasm/rag.wasm ./cmd/wasm
```

### Run Demo

```bash
cd examples/wasm
python3 -m http.server 8080
# Open http://localhost:8080
```

### JavaScript API

```javascript
// Index content
ragIndex("file.txt", "Your text content here...")

// Search (returns JSON string)
const results = JSON.parse(ragQuery("search term", 5))

// Clear index
ragClear()

// Get statistics
const stats = JSON.parse(ragStats())
```

See [examples/wasm/README.md](examples/wasm/README.md) for details.

---

## Architecture

```
cmd/rag/main.go          # Entrypoint
cmd/wasm/main.go         # WASM entrypoint
internal/
├── domain/              # Core entities (Document, Chunk, etc.)
├── port/                # Interfaces (IndexStore, Retriever, etc.)
├── usecase/             # Business logic
│   ├── index.go         # Indexing orchestration
│   ├── retrieve.go      # Search with BM25 + MMR
│   └── pack.go          # Context packing
└── adapter/
    ├── fs/              # File system walker
    ├── store/           # BoltDB implementation
    ├── analyzer/        # Tokenizer + Porter stemmer
    ├── chunker/         # Line-based chunking
    └── retriever/       # BM25 + MMR implementations
```

## License

MIT
