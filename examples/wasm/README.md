# RAG WebAssembly Demo

Run BM25 search with MMR deduplication entirely in the browser.

## Build

```bash
# From project root
make build-wasm

# Or manually
GOOS=js GOARCH=wasm go build -o examples/wasm/rag.wasm ./cmd/wasm
```

## Run

```bash
cd examples/wasm
python3 -m http.server 8080
```

Open http://localhost:8080

## JavaScript API

### `ragIndex(filename, content)`

Index text content.

```javascript
const result = JSON.parse(ragIndex("document.txt", "Your text content..."))
// { "success": true, "chunks": 5, "filename": "document.txt" }
```

### `ragQuery(query, topK)`

Search indexed content. Returns top K results ranked by BM25 with MMR deduplication.

```javascript
const result = JSON.parse(ragQuery("search term", 5))
// {
//   "query": "search term",
//   "results": [
//     {
//       "path": "document.txt",
//       "startLine": 10,
//       "endLine": 15,
//       "score": 2.34,
//       "text": "matching content..."
//     }
//   ]
// }
```

### `ragClear()`

Clear the entire index.

```javascript
ragClear()
```

### `ragStats()`

Get index statistics.

```javascript
const stats = JSON.parse(ragStats())
// {
//   "totalDocs": 3,
//   "totalChunks": 15,
//   "avgChunkLen": 45.2,
//   "files": ["doc1.txt", "doc2.txt", "doc3.txt"]
// }
```

## Limitations

- **No embeddings/semantic search** - BM25 keyword search only
- **In-memory storage** - Index is lost on page refresh
- **~5MB download** - WASM binary size

## Files

```
examples/wasm/
├── index.html      # Demo UI
├── rag.wasm        # WASM binary (build output, gitignored)
├── wasm_exec.js    # Go WASM runtime
└── README.md       # This file
```
