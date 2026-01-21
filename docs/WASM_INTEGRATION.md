# RAG WASM Integration Spec

Specification for integrating RAG search into browser-based AI agents and applications.

## Overview

The RAG WASM module provides in-browser BM25 search with MMR deduplication. No server required - all indexing and search runs client-side in the browser.

## Requirements

- Modern browser with WebAssembly support
- ~5MB download for WASM binary
- Memory scales with indexed content (~1KB per chunk)

## Files

```
rag.wasm        # WASM binary (build with: make build-wasm)
wasm_exec.js    # Go WASM runtime (from Go installation)
```

## Integration

### 1. Load WASM Module

```html
<script src="wasm_exec.js"></script>
<script>
const go = new Go();

async function initRAG() {
    const result = await WebAssembly.instantiateStreaming(
        fetch("rag.wasm"),
        go.importObject
    );
    go.run(result.instance);
    // RAG functions now available globally
}

initRAG();
</script>
```

### 2. Wait for Ready

```javascript
// Poll for availability
function waitForRAG() {
    return new Promise((resolve) => {
        const check = () => {
            if (typeof ragIndex !== 'undefined') {
                resolve();
            } else {
                setTimeout(check, 50);
            }
        };
        check();
    });
}

await waitForRAG();
```

## API Reference

### `ragIndex(filename: string, content: string): string`

Index text content under a filename.

**Parameters:**
- `filename` - Identifier for the content (e.g., "doc.txt", "chat_history.md")
- `content` - Text content to index

**Returns:** JSON string
```typescript
type IndexResult = {
    success: boolean;
    chunks: number;      // Number of chunks created
    filename: string;
} | {
    error: string;
}
```

**Example:**
```javascript
const result = JSON.parse(ragIndex("notes.txt", "Meeting notes from Monday..."));
if (result.success) {
    console.log(`Indexed ${result.chunks} chunks`);
}
```

---

### `ragQuery(query: string, topK?: number): string`

Search indexed content.

**Parameters:**
- `query` - Search query (keywords work best)
- `topK` - Maximum results to return (default: 5)

**Returns:** JSON string
```typescript
type QueryResult = {
    query: string;
    results: Array<{
        path: string;       // Filename
        startLine: number;  // Starting line in original content
        endLine: number;    // Ending line
        score: number;      // BM25 relevance score
        text: string;       // Matched chunk text
    }>;
} | {
    error: string;
}
```

**Example:**
```javascript
const result = JSON.parse(ragQuery("meeting Monday", 3));
for (const r of result.results) {
    console.log(`[${r.path}:${r.startLine}-${r.endLine}] Score: ${r.score}`);
    console.log(r.text);
}
```

---

### `ragClear(): string`

Clear all indexed content.

**Returns:** JSON string
```typescript
type ClearResult = { success: boolean }
```

**Example:**
```javascript
ragClear();
```

---

### `ragStats(): string`

Get index statistics.

**Returns:** JSON string
```typescript
type StatsResult = {
    totalDocs: number;      // Number of indexed documents
    totalChunks: number;    // Total chunks across all docs
    avgChunkLen: number;    // Average tokens per chunk
    files: string[];        // List of indexed filenames
}
```

**Example:**
```javascript
const stats = JSON.parse(ragStats());
console.log(`${stats.totalDocs} docs, ${stats.totalChunks} chunks`);
```

## AI Agent Integration Patterns

### Pattern 1: Chat History Search

Index conversation history for context retrieval:

```javascript
// After each message exchange
function indexMessage(role, content, timestamp) {
    const filename = `chat_${timestamp}.txt`;
    const formatted = `[${role}]: ${content}`;
    ragIndex(filename, formatted);
}

// Before generating response, find relevant context
function getRelevantContext(userQuery) {
    const result = JSON.parse(ragQuery(userQuery, 5));
    return result.results.map(r => r.text).join("\n\n");
}
```

### Pattern 2: Document Q&A

Load documents and answer questions:

```javascript
// Index uploaded documents
async function indexDocument(file) {
    const content = await file.text();
    return JSON.parse(ragIndex(file.name, content));
}

// Build context for LLM
function buildPrompt(question) {
    const results = JSON.parse(ragQuery(question, 5));

    const context = results.results
        .map(r => `[Source: ${r.path}:${r.startLine}-${r.endLine}]\n${r.text}`)
        .join("\n\n---\n\n");

    return `Based on the following context, answer the question.

Context:
${context}

Question: ${question}

Answer:`;
}
```

### Pattern 3: Knowledge Base Agent

Maintain persistent knowledge:

```javascript
class KnowledgeBase {
    async addKnowledge(topic, content) {
        const result = JSON.parse(ragIndex(`kb_${topic}.md`, content));
        return result.success;
    }

    search(query, limit = 5) {
        const result = JSON.parse(ragQuery(query, limit));
        return result.results;
    }

    getContext(query, tokenBudget = 2000) {
        const results = this.search(query, 10);
        let context = [];
        let tokens = 0;

        for (const r of results) {
            const chunkTokens = r.text.split(/\s+/).length;
            if (tokens + chunkTokens > tokenBudget) break;
            context.push(r);
            tokens += chunkTokens;
        }

        return context;
    }

    clear() {
        ragClear();
    }
}
```

### Pattern 4: Streaming Index

Index content as it arrives:

```javascript
class StreamingIndex {
    constructor(docName) {
        this.docName = docName;
        this.buffer = [];
        this.chunkSize = 500; // words
    }

    append(text) {
        this.buffer.push(text);
        const combined = this.buffer.join(" ");
        const words = combined.split(/\s+/);

        if (words.length >= this.chunkSize) {
            this.flush();
        }
    }

    flush() {
        if (this.buffer.length === 0) return;
        const content = this.buffer.join(" ");
        ragIndex(`${this.docName}_${Date.now()}.txt`, content);
        this.buffer = [];
    }
}
```

## Search Tips

### Effective Queries

BM25 works best with keyword-based queries:

```javascript
// Good - specific keywords
ragQuery("authentication JWT token")
ragQuery("database connection pool")
ragQuery("error handling retry")

// Less effective - natural language
ragQuery("how do I authenticate users")
ragQuery("what happens when the database fails")
```

### Query Preprocessing

Improve results by extracting keywords:

```javascript
function extractKeywords(naturalQuery) {
    // Remove stop words, keep nouns/verbs
    const stopWords = new Set(['how', 'do', 'i', 'what', 'is', 'the', 'a', 'an']);
    return naturalQuery
        .toLowerCase()
        .split(/\s+/)
        .filter(w => !stopWords.has(w) && w.length > 2)
        .join(" ");
}

const query = extractKeywords("How do I handle authentication errors?");
// "handle authentication errors"
```

## Limitations

| Limitation | Details |
|------------|---------|
| No persistence | Index lost on page refresh |
| No semantic search | BM25 keyword matching only |
| Memory bound | All data in browser memory |
| Single thread | No Web Workers support (Go WASM limitation) |

## Performance

| Operation | ~Time (10K chunks) |
|-----------|-------------------|
| Index 1 doc | 5-20ms |
| Query | 10-50ms |
| Clear | <1ms |

## Error Handling

All functions return JSON. Check for `error` field:

```javascript
function safeQuery(query) {
    try {
        const result = JSON.parse(ragQuery(query, 5));
        if (result.error) {
            console.error("RAG error:", result.error);
            return [];
        }
        return result.results;
    } catch (e) {
        console.error("Parse error:", e);
        return [];
    }
}
```

## TypeScript Definitions

```typescript
declare function ragIndex(filename: string, content: string): string;
declare function ragQuery(query: string, topK?: number): string;
declare function ragClear(): string;
declare function ragStats(): string;

interface IndexResult {
    success?: boolean;
    chunks?: number;
    filename?: string;
    error?: string;
}

interface SearchResult {
    path: string;
    startLine: number;
    endLine: number;
    score: number;
    text: string;
}

interface QueryResult {
    query: string;
    results: SearchResult[];
    error?: string;
}

interface StatsResult {
    totalDocs: number;
    totalChunks: number;
    avgChunkLen: number;
    files: string[];
}
```
