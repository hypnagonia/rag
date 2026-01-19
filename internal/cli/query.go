package cli

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"rag/config"
	"rag/internal/adapter/analyzer"
	"rag/internal/adapter/retriever"
	"rag/internal/adapter/store"
	"rag/internal/usecase"
)

var (
	queryText   string
	queryTopK   int
	queryJSON   bool
	queryNoMMR  bool
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

	// Create retrievers
	bm25 := retriever.NewBM25Retriever(st, tokenizer, cfg.Index.K1, cfg.Index.B)
	mmr := retriever.NewMMRReranker(cfg.Retrieve.MMRLambda, cfg.Retrieve.DedupJaccard)

	// Create retrieve use case
	retrieveUC := usecase.NewRetrieveUseCase(bm25, mmr)

	// Determine top-k
	topK := cfg.Retrieve.TopK
	if queryTopK > 0 {
		topK = queryTopK
	}

	// Execute search
	var results []usecase.ScoredChunkResult
	if queryNoMMR {
		chunks, err := retrieveUC.RetrieveWithoutMMR(queryText, topK)
		if err != nil {
			return fmt.Errorf("search failed: %w", err)
		}
		for _, c := range chunks {
			doc, _ := st.GetDoc(c.Chunk.DocID)
			results = append(results, usecase.ScoredChunkResult{
				Path:      doc.Path,
				StartLine: c.Chunk.StartLine,
				EndLine:   c.Chunk.EndLine,
				Score:     c.Score,
				Text:      c.Chunk.Text,
			})
		}
	} else {
		chunks, err := retrieveUC.Retrieve(queryText, topK)
		if err != nil {
			return fmt.Errorf("search failed: %w", err)
		}
		for _, c := range chunks {
			doc, _ := st.GetDoc(c.Chunk.DocID)
			results = append(results, usecase.ScoredChunkResult{
				Path:      doc.Path,
				StartLine: c.Chunk.StartLine,
				EndLine:   c.Chunk.EndLine,
				Score:     c.Score,
				Text:      c.Chunk.Text,
			})
		}
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
			// Truncate long text for display
			text := r.Text
			if len(text) > 500 {
				text = text[:500] + "..."
			}
			fmt.Println(text)
			fmt.Println()
		}
	}

	return nil
}
