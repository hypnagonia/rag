package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"rag/config"
	"rag/internal/adapter/embedding"
	"rag/internal/adapter/store"
	"rag/internal/port"
)

func main() {
	indexPath := flag.String("index", ".", "Path to indexed directory")
	query := flag.String("q", "", "Query to test")
	topK := flag.Int("k", 10, "Number of results")
	flag.Parse()

	if *query == "" {
		fmt.Println("Usage: go run cmd/benchmark/main.go -index ./tmp -q \"query\"")
		fmt.Println("\nTests:")
		fmt.Println("  1. Embedding infrastructure (model connection, vector store)")
		fmt.Println("  2. Semantic similarity (query vs results)")
		fmt.Println("  3. Synonym handling (finds related concepts)")
		os.Exit(1)
	}

	cfg, err := config.LoadFromDir(*indexPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading config: %v\n", err)
		os.Exit(1)
	}

	dbPath := config.IndexDBPath(*indexPath)
	st, err := store.NewBoltStore(dbPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error opening index: %v\n", err)
		os.Exit(1)
	}
	defer st.Close()

	embedder, vectorStore, err := setupEmbedding(st, cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Semantic search not available: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("SEMANTIC SEARCH BENCHMARK")
	fmt.Println(strings.Repeat("=", 70))

	count, _ := vectorStore.Count()
	fmt.Printf("Embeddings indexed: %d\n", count)
	fmt.Printf("Model: %s (%s)\n", cfg.Embedding.Model, cfg.Embedding.Provider)
	fmt.Printf("Dimension: %d\n", embedder.Dimension())
	fmt.Println()

	fmt.Printf("Query: \"%s\"\n", *query)
	fmt.Println(strings.Repeat("-", 70))

	queryVec, err := embedder.Embed([]string{*query})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Embedding error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Query embedded: %d dimensions\n\n", len(queryVec[0]))

	results, err := vectorStore.Search(queryVec[0], *topK)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Search error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Top %d semantic matches:\n\n", len(results))

	totalScore := 0.0
	for i, r := range results {
		chunk, _ := st.GetChunk(r.ID)
		doc, _ := st.GetDoc(chunk.DocID)

		preview := chunk.Text
		if len(preview) > 150 {
			preview = preview[:150] + "..."
		}
		preview = strings.ReplaceAll(preview, "\n", " ")

		similarity := r.Score
		totalScore += similarity

		rating := "LOW"
		if similarity > 0.7 {
			rating = "HIGH"
		} else if similarity > 0.5 {
			rating = "GOOD"
		} else if similarity > 0.3 {
			rating = "OK"
		}

		fmt.Printf("%d. [%s %.3f] %s:L%d-%d\n", i+1, rating, similarity, shortPath(doc.Path), chunk.StartLine, chunk.EndLine)
		fmt.Printf("   %s\n\n", preview)
	}

	avgScore := totalScore / float64(len(results))
	fmt.Println(strings.Repeat("=", 70))
	fmt.Printf("QUALITY METRICS:\n")
	fmt.Printf("  Average similarity: %.3f\n", avgScore)
	fmt.Printf("  Top-1 similarity:   %.3f\n", results[0].Score)

	if avgScore > 0.5 {
		fmt.Println("  Status: GOOD - semantic search working well")
	} else if avgScore > 0.3 {
		fmt.Println("  Status: OK - results are somewhat related")
	} else {
		fmt.Println("  Status: POOR - may need better embeddings or re-indexing")
	}
}

func shortPath(path string) string {
	parts := strings.Split(path, "/")
	if len(parts) > 2 {
		return parts[len(parts)-1]
	}
	return path
}

func setupEmbedding(st *store.BoltStore, cfg *config.Config) (port.Embedder, port.VectorStore, error) {
	if !cfg.Embedding.Enabled {
		return nil, nil, fmt.Errorf("embeddings not enabled in config")
	}

	var embedder port.Embedder
	var err error

	switch cfg.Embedding.Provider {
	case "ollama":
		embedder, err = embedding.NewOllamaEmbedder(cfg.Embedding.Model, cfg.Embedding.BaseURL)
	case "openai":
		embedder, err = embedding.NewOpenAIEmbedder(cfg.Embedding.APIKeyEnv, cfg.Embedding.Model)
	default:
		return nil, nil, fmt.Errorf("unsupported provider: %s", cfg.Embedding.Provider)
	}
	if err != nil {
		return nil, nil, fmt.Errorf("embedder init failed: %w", err)
	}

	vectorStore, err := store.NewBoltVectorStore(st.DB(), embedder.Dimension())
	if err != nil {
		return nil, nil, fmt.Errorf("vector store failed: %w", err)
	}

	count, _ := vectorStore.Count()
	if count == 0 {
		return nil, nil, fmt.Errorf("no embeddings - run 'rag index' with embedding.enabled=true")
	}

	return embedder, vectorStore, nil
}
