# Agentic RAG Example

A general-purpose agentic RAG (Retrieval-Augmented Generation) tool that works with any indexed content: books, articles, documentation, or any text.

## Workflow

1. **Query Expansion** (optional): LLM generates alternative search queries
2. **Search**: Executes searches against the RAG index
3. **Context Evaluation**: LLM evaluates if retrieved context is sufficient
4. **Iteration**: If insufficient, LLM suggests new queries or context expansion
5. **Answer**: Generates a final answer based on retrieved passages

## Usage

First, index your content:
```bash
rag index /path/to/your/content
```

Then run the agentic search:
```bash
# Fast mode - minimal tokens (recommended for simple queries)
export DEEPSEEK_API_KEY=your-key
go run main.go -q "what is the main theme" -index /path/to/content -fast

# Standard mode - iterative refinement
go run main.go -q "explain the key concepts" -index /path/to/content

# Full agentic mode - query expansion + iteration
go run main.go -q "compare chapters 1 and 3" -index /path/to/content -expand

# With different providers
go run main.go -q "summarize" -provider openai -model gpt-4o-mini
go run main.go -q "summarize" -provider local -model llama2
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `-q` | (required) | Search query |
| `-index` | `.` | Path to indexed directory |
| `-provider` | `deepseek` | LLM provider: `deepseek`, `openai`, `local` |
| `-model` | `deepseek-chat` | Model name |
| `-base-url` | (auto) | Custom API base URL |
| `-api-key` | (env var) | API key (or use env var) |
| `-k` | `10` | Results per query |
| `-max-iters` | `2` | Maximum search iterations |
| `-budget` | `4000` | Token budget for context packing |
| `-fast` | `false` | Fast mode: 1 LLM call only |
| `-expand` | `false` | Use LLM to expand queries |
| `-v` | `false` | Verbose output |

## Modes and Token Usage

| Mode | LLM Calls | Est. Tokens | Use Case |
|------|-----------|-------------|----------|
| `-fast` | 1 | ~1k | Simple factual queries |
| default | 2-3 | ~2-3k | Most queries |
| `-expand` | 4-6 | ~4-6k | Complex/ambiguous queries |

## Supported Providers

| Provider | Base URL | API Key Env Var |
|----------|----------|-----------------|
| `deepseek` | `https://api.deepseek.com/v1` | `DEEPSEEK_API_KEY` |
| `openai` | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `local` | `http://localhost:11434/v1` | (none) |

For other OpenAI-compatible APIs, use `-base-url` and `-api-key` flags.


