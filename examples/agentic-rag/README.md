# Agentic RAG Example

This example demonstrates an agentic RAG (Retrieval-Augmented Generation) workflow that iteratively refines searches using an LLM.

## Workflow

1. **Query Expansion**: LLM generates alternative search queries from your original question
2. **Search**: Executes searches against the RAG index with all queries
3. **Context Evaluation**: LLM evaluates if retrieved context is sufficient
4. **Iteration**: If insufficient, LLM suggests new queries and the process repeats
5. **Results**: Returns the most relevant code sections

## Usage

First, index your codebase:
```bash
rag index /path/to/your/project
```

Then run the agentic search:
```bash
# Using DeepSeek (default)
export DEEPSEEK_API_KEY=your-key
go run main.go -q "how does authentication work" -index /path/to/your/project

# Using OpenAI
export OPENAI_API_KEY=your-key
go run main.go -q "database connection handling" -provider openai -model gpt-4o-mini

# Using local Ollama
go run main.go -q "error handling patterns" -provider local -model llama2

# Verbose mode to see the agentic process
go run main.go -q "how are requests routed" -v
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
| `-max-iters` | `3` | Maximum search iterations |
| `-v` | `false` | Verbose output |

## Supported Providers

| Provider | Base URL | API Key Env Var |
|----------|----------|-----------------|
| `deepseek` | `https://api.deepseek.com/v1` | `DEEPSEEK_API_KEY` |
| `openai` | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `local` | `http://localhost:11434/v1` | (none) |

For other OpenAI-compatible APIs, use `-base-url` and `-api-key` flags.

## Example Output

```
🚀 Starting agentic RAG search...

🔍 Original query: how does authentication work
📝 Expanded queries: [how does authentication work, auth middleware implementation, login handler user verification]

--- Iteration 1 ---
📚 Found 15 unique chunks
🤔 LLM decision: sufficient=true, reason=Found auth middleware and login handlers

============================================================
📋 RESULTS for: how does authentication work
============================================================

Found 15 relevant code sections:

─── [1] internal/auth/middleware.go:L12-45 (score: 8.234) ───
func AuthMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        token := r.Header.Get("Authorization")
        ...
```
