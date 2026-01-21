//go:build js && wasm

package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"syscall/js"
	"time"

	"rag/internal/adapter/analyzer"
	"rag/internal/adapter/chunker"
	"rag/internal/adapter/memstore"
	"rag/internal/adapter/retriever"
	"rag/internal/domain"
	"rag/internal/port"
)

var (
	store     *memstore.MemoryStore
	tokenizer *analyzer.Tokenizer
	chk       port.Chunker
	bm25      *retriever.BM25Retriever
	mmr       *retriever.MMRReranker
)

func init() {
	store = memstore.NewMemoryStore()
	tokenizer = analyzer.NewTokenizer(true)
	chk = chunker.NewLineChunker(256, 50, tokenizer)
	bm25 = retriever.NewBM25Retriever(store, tokenizer, 1.2, 0.75, 0.3)
	mmr = retriever.NewMMRReranker(0.7, 0.8)
}

func main() {
	c := make(chan struct{})

	js.Global().Set("ragIndex", js.FuncOf(indexContent))
	js.Global().Set("ragQuery", js.FuncOf(queryContent))
	js.Global().Set("ragClear", js.FuncOf(clearIndex))
	js.Global().Set("ragStats", js.FuncOf(getStats))

	<-c
}

func indexContent(this js.Value, args []js.Value) interface{} {
	if len(args) < 2 {
		return makeError("usage: ragIndex(filename, content)")
	}

	filename := args[0].String()
	content := args[1].String()

	docID := generateDocID(filename)
	doc := domain.Document{
		ID:      docID,
		Path:    filename,
		ModTime: time.Now(),
		Lang:    "text",
	}

	chunks, err := chk.Chunk(doc, content)
	if err != nil {
		return makeError("chunking failed: " + err.Error())
	}

	postings := make(map[string]map[string]int)
	totalTokens := 0

	for _, chunk := range chunks {
		tf := make(map[string]int)
		for _, token := range chunk.Tokens {
			tf[token]++
		}
		for term, count := range tf {
			if postings[term] == nil {
				postings[term] = make(map[string]int)
			}
			postings[term][chunk.ID] = count
		}
		totalTokens += len(chunk.Tokens)
	}

	err = store.BatchIndex([]port.IndexedFile{{
		Doc:      doc,
		Chunks:   chunks,
		Postings: postings,
	}})
	if err != nil {
		return makeError("indexing failed: " + err.Error())
	}

	docs, _ := store.ListDocs()
	stats := domain.Stats{
		TotalDocs:   len(docs),
		TotalChunks: countChunks(),
		AvgChunkLen: float64(totalTokens) / float64(len(chunks)),
	}
	store.UpdateStats(stats)

	return makeResult(map[string]interface{}{
		"success":  true,
		"chunks":   len(chunks),
		"filename": filename,
	})
}

func queryContent(this js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return makeError("usage: ragQuery(query, [topK])")
	}

	query := args[0].String()
	topK := 5
	if len(args) > 1 {
		topK = args[1].Int()
	}

	candidates, err := bm25.Search(query, topK*2)
	if err != nil {
		return makeError("search failed: " + err.Error())
	}

	if len(candidates) == 0 {
		return makeResult(map[string]interface{}{
			"results": []interface{}{},
			"query":   query,
		})
	}

	results := mmr.Rerank(candidates, topK)

	output := make([]map[string]interface{}, 0, len(results))
	for _, r := range results {
		doc, _ := store.GetDoc(r.Chunk.DocID)
		output = append(output, map[string]interface{}{
			"path":      doc.Path,
			"startLine": r.Chunk.StartLine,
			"endLine":   r.Chunk.EndLine,
			"score":     r.Score,
			"text":      r.Chunk.Text,
		})
	}

	return makeResult(map[string]interface{}{
		"results": output,
		"query":   query,
	})
}

func clearIndex(this js.Value, args []js.Value) interface{} {
	store = memstore.NewMemoryStore()
	bm25 = retriever.NewBM25Retriever(store, tokenizer, 1.2, 0.75, 0.3)
	return makeResult(map[string]interface{}{
		"success": true,
	})
}

func getStats(this js.Value, args []js.Value) interface{} {
	stats, _ := store.GetStats()
	docs, _ := store.ListDocs()

	filenames := make([]string, len(docs))
	for i, doc := range docs {
		filenames[i] = doc.Path
	}

	return makeResult(map[string]interface{}{
		"totalDocs":   stats.TotalDocs,
		"totalChunks": stats.TotalChunks,
		"avgChunkLen": stats.AvgChunkLen,
		"files":       filenames,
	})
}

func countChunks() int {
	docs, _ := store.ListDocs()
	total := 0
	for _, doc := range docs {
		chunks, _ := store.GetChunksByDoc(doc.ID)
		total += len(chunks)
	}
	return total
}

func generateDocID(path string) string {
	hash := sha256.Sum256([]byte(path))
	return hex.EncodeToString(hash[:8])
}

func makeError(msg string) interface{} {
	result, _ := json.Marshal(map[string]interface{}{
		"error": msg,
	})
	return string(result)
}

func makeResult(data map[string]interface{}) interface{} {
	result, _ := json.Marshal(data)
	return string(result)
}
