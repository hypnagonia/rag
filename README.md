# RAG Context Compressor

A CLI tool for indexing local files using BM25 lexical retrieval, ranking results with MMR deduplication, and packing compressed context with citations for LLM consumption.

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

---

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

---

```bash
$ rag query -q "Red Wedding Robb Stark murdered" -index ./books -expand
```
> Robb Stark was betrayed and murdered by the Freys and Boltons at the Twins
> during his uncle's wedding, an event known as the Red Wedding [4.txt:21098-21138].
>
> **~3k tokens** (with RAG) vs **~2M tokens** (without RAG)

---

```bash
$ rag query -q "How did Drogo die" -index ./books -expand
```
> Drogo died after being placed in a comatose state by a bloodmagic ritual
> performed by Mirri Maz Duur. The ritual involved sacrificing his horse and
> using its blood, but it left Drogo alive yet unresponsive [125804431.txt:L16544-16620].
> Mirri Maz Duur states that Drogo will only return to his former self under
> impossible conditions, implying he will never recover [1.txt:L17745-17772].
>
> **~5k tokens** (with RAG) vs **~2M tokens** (without RAG)

---

## Installation

```bash
go build -o rag ./cmd/rag
```

## Quick Start

```bash
# Index a directory
rag index /path/to/project

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

### `rag query -q "<question>"`

Search indexed files using BM25 retrieval with MMR deduplication.

```bash
rag query -q "database connection"
rag query -q "error handling" --top-k 10 --json
```

**Flags:**
- `-q, --query` - Search query (required)
- `-k, --top-k` - Number of results (default from config)
- `--json` - Output as JSON
- `--no-mmr` - Disable MMR reranking

### `rag pack -q "<question>"`

Pack relevant chunks into compressed context that fits a token budget.

```bash
rag pack -q "authentication flow" -b 2000
rag pack -q "API endpoints" -o context.json
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

## Architecture

```
cmd/rag/main.go          # Entrypoint
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
