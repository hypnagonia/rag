# Agentic RAG Example

Shows how to drive the `rag` library from your own Go program: open an index, build a
hybrid retriever, and run the agentic answer loop.

The same loop is available as a first-class command — `rag ask` — which also supports
`--hyde`. Use this example when you want to embed the loop in your own code; use
`rag ask` when you just want an answer.

## Workflow

1. **Query expansion** (optional, `-expand`): one LLM call produces alternative queries
2. **Search**: hybrid BM25 + vector retrieval for each query, accumulating unique chunks
3. **Context evaluation**: one LLM call judges whether the packed context answers the question
4. **Iteration**: if insufficient, the suggested queries drive another round
5. **Answer**: one LLM call writes the answer with `[file:lines]` citations

`-fast` skips steps 1, 3 and 4 entirely: one search, one LLM call.

## Usage

Index your content first:

```bash
rag index /path/to/your/content
```

The API key is read from `.env` (searched upward from `-index` and the working
directory) or from the environment.

```bash
# one LLM call
go run . -q "what is the main theme" -index /path/to/content -fast

# iterative refinement, up to 2 rounds
go run . -q "explain the key concepts" -index /path/to/content

# query expansion as well
go run . -q "compare chapters 1 and 3" -index /path/to/content -expand -v
```

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `-q` | – | Question (required) |
| `-index` | `.` | Indexed directory; also selects its `rag.yaml` |
| `-k` | 10 | Chunks retrieved per search |
| `-budget` | 4000 | Context token budget |
| `-max-iters` | 2 | Maximum retrieve/evaluate rounds |
| `-fast` | false | One search, one LLM call |
| `-expand` | false | LLM query expansion (+1 call) |
| `-v` | false | Print retrieval mode and per-round progress |

## Cost

Every run prints what it used:

```
   Back-and-forth rounds:  1 of 2 max
   LLM calls:              1
   Total tokens:           1903  (reported by the API)
```

Token counts come from the provider's `usage` field when it reports one; the line says
which source was used. Cheapest run is `-fast`; most expensive is `-expand` with several
rounds.

Generation runs against a hosted provider only (DeepSeek or OpenAI), configured under
`llm:` in `rag.yaml`. Embeddings may be local.
