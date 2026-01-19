package cli

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"rag/config"
	"rag/internal/adapter/analyzer"
	"rag/internal/adapter/embedding"
	"rag/internal/adapter/retriever"
	"rag/internal/adapter/store"
	"rag/internal/domain"
	"rag/internal/port"
	"rag/internal/usecase"
)

var (
	queryText    string
	queryTopK    int
	queryJSON    bool
	queryNoMMR   bool
	queryContext int
)

var queryCmd = &cobra.Command{
	Use:   "query",
	Short: "Search indexed files",
	Long: `Search for relevant code chunks using BM25 retrieval with MMR deduplication.

Examples:
  rag query -q "authentication handler"
  rag query -q "database connection" --top-k 10 --json`,
	RunE: runQuery,
}

func init() {
	rootCmd.AddCommand(queryCmd)
	queryCmd.Flags().StringVarP(&queryText, "query", "q", "", "search query (required)")
	queryCmd.Flags().IntVarP(&queryTopK, "top-k", "k", 0, "number of results (default from config)")
	queryCmd.Flags().BoolVar(&queryJSON, "json", false, "output as JSON")
	queryCmd.Flags().BoolVar(&queryNoMMR, "no-mmr", false, "disable MMR reranking")
	queryCmd.Flags().IntVarP(&queryContext, "context", "c", 0, "expand results by N lines before/after")
	queryCmd.MarkFlagRequired("query")
}

func runQuery(cmd *cobra.Command, args []string) error {
	cfg := GetConfig()
	rootDir := GetRootDir()

	// Check if index exists
	dbPath := config.IndexDBPath(rootDir)
	if _, err := os.Stat(dbPath); os.IsNotExist(err) {
		return fmt.Errorf("no index found. Run 'rag index' first")
	}

	// Open store
	st, err := store.NewBoltStore(dbPath)
	if err != nil {
		return fmt.Errorf("failed to open index: %w", err)
	}
	defer st.Close()

	// Create tokenizer
	tokenizer := analyzer.NewTokenizer(cfg.Index.Stemming)

	// Create BM25 retriever
	bm25 := retriever.NewBM25Retriever(st, tokenizer, cfg.Index.K1, cfg.Index.B, cfg.Retrieve.PathBoostWeight)
	mmr := retriever.NewMMRReranker(cfg.Retrieve.MMRLambda, cfg.Retrieve.DedupJaccard)

	// Check if hybrid retrieval is enabled and set up
	var searchRetriever port.Retriever = bm25
	if cfg.Retrieve.HybridEnabled && cfg.Embedding.Enabled {
		embedder, vectorStore, err := setupHybridRetrieval(st, cfg)
		if err != nil {
			fmt.Printf("Warning: hybrid retrieval unavailable: %v\n", err)
		} else {
			searchRetriever = retriever.NewHybridRetriever(
				bm25, vectorStore, embedder, st,
				cfg.Retrieve.RRFK, cfg.Retrieve.BM25Weight,
			)
		}
	}

	// Create retrieve use case
	retrieveUC := usecase.NewRetrieveUseCase(searchRetriever, mmr, cfg.Retrieve.MinScoreThreshold)

	// Determine top-k
	topK := cfg.Retrieve.TopK
	if queryTopK > 0 {
		topK = queryTopK
	}

	// Execute search
	var chunks []domain.ScoredChunk
	if queryNoMMR {
		chunks, err = retrieveUC.RetrieveWithoutMMR(queryText, topK)
	} else {
		chunks, err = retrieveUC.Retrieve(queryText, topK)
	}
	if err != nil {
		return fmt.Errorf("search failed: %w", err)
	}

	// Build results with optional context expansion
	var results []usecase.ScoredChunkResult
	for _, c := range chunks {
		doc, _ := st.GetDoc(c.Chunk.DocID)
		startLine := c.Chunk.StartLine
		endLine := c.Chunk.EndLine
		text := c.Chunk.Text

		// Expand context if requested
		if queryContext > 0 {
			newStart, newEnd, expandedText, err := expandContext(doc.Path, startLine, endLine, queryContext)
			if err == nil && expandedText != "" {
				startLine = newStart
				endLine = newEnd
				text = expandedText
			}
		}

		results = append(results, usecase.ScoredChunkResult{
			Path:      doc.Path,
			StartLine: startLine,
			EndLine:   endLine,
			Score:     c.Score,
			Text:      text,
		})
	}

	// Output results
	if queryJSON {
		output, _ := json.MarshalIndent(results, "", "  ")
		fmt.Println(string(output))
	} else {
		if len(results) == 0 {
			fmt.Println("No results found.")
			return nil
		}
		fmt.Printf("Found %d results for: %s\n\n", len(results), queryText)
		for i, r := range results {
			fmt.Printf("--- [%d] %s:L%d-%d (score: %.2f) ---\n", i+1, r.Path, r.StartLine, r.EndLine, r.Score)
			// Truncate long text for display (unless context expansion is used)
			text := r.Text
			if queryContext == 0 && len(text) > 500 {
				text = text[:500] + "..."
			}
			fmt.Println(text)
			fmt.Println()
		}
	}

	return nil
}

// expandContext reads additional lines from the source file around the chunk.
func expandContext(path string, startLine, endLine, extraLines int) (newStart, newEnd int, text string, err error) {
	if extraLines <= 0 {
		return startLine, endLine, "", nil
	}

	file, err := os.Open(path)
	if err != nil {
		return startLine, endLine, "", err
	}
	defer file.Close()

	// Calculate expanded range
	newStart = startLine - extraLines
	if newStart < 1 {
		newStart = 1
	}
	newEnd = endLine + extraLines

	// Read lines
	scanner := bufio.NewScanner(file)
	var lines []string
	lineNum := 0
	for scanner.Scan() {
		lineNum++
		if lineNum >= newStart && lineNum <= newEnd {
			lines = append(lines, scanner.Text())
		}
		if lineNum > newEnd {
			break
		}
	}

	if err := scanner.Err(); err != nil {
		return startLine, endLine, "", err
	}

	// Adjust newEnd if file was shorter
	actualEnd := newStart + len(lines) - 1
	if actualEnd < newEnd {
		newEnd = actualEnd
	}

	return newStart, newEnd, joinLines(lines), nil
}

func joinLines(lines []string) string {
	result := ""
	for i, line := range lines {
		if i > 0 {
			result += "\n"
		}
		result += line
	}
	return result
}

// setupHybridRetrieval creates the embedder and vector store for hybrid search.
func setupHybridRetrieval(st *store.BoltStore, cfg *config.Config) (port.Embedder, port.VectorStore, error) {
	var embedder port.Embedder
	var err error

	switch cfg.Embedding.Provider {
	case "openai":
		embedder, err = embedding.NewOpenAIEmbedder(cfg.Embedding.APIKeyEnv, cfg.Embedding.Model)
	case "deepseek":
		embedder, err = embedding.NewDeepSeekEmbedder(cfg.Embedding.APIKeyEnv, cfg.Embedding.Model)
	case "jina":
		embedder, err = embedding.NewJinaEmbedder(cfg.Embedding.APIKeyEnv, cfg.Embedding.Model)
	case "ollama":
		embedder, err = embedding.NewOllamaEmbedder(cfg.Embedding.Model, cfg.Embedding.BaseURL)
	case "mock":
		embedder = embedding.NewMockEmbedder(cfg.Embedding.Dimension)
	default:
		return nil, nil, fmt.Errorf("unsupported embedding provider: %s", cfg.Embedding.Provider)
	}
	if err != nil {
		return nil, nil, err
	}

	vectorStore, err := store.NewBoltVectorStore(st.DB(), embedder.Dimension())
	if err != nil {
		return nil, nil, err
	}

	// Check if vector store has any vectors
	count, err := vectorStore.Count()
	if err != nil {
		return nil, nil, err
	}
	if count == 0 {
		return nil, nil, fmt.Errorf("no embeddings found - run 'rag index' first with embedding.enabled=true")
	}

	return embedder, vectorStore, nil
}
