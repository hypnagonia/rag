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
	packQuery  string
	packBudget int
	packOutput string
	packTopK   int
)

var packCmd = &cobra.Command{
	Use:   "pack",
	Short: "Pack relevant context for LLM consumption",
	Long: `Search and pack relevant code chunks into a compressed context
that fits within a token budget, including citations.

Examples:
  rag pack -q "how does authentication work"
  rag pack -q "database layer" -b 2000 -o context.json`,
	RunE: runPack,
}

func init() {
	rootCmd.AddCommand(packCmd)
	packCmd.Flags().StringVarP(&packQuery, "query", "q", "", "search query (required)")
	packCmd.Flags().IntVarP(&packBudget, "budget", "b", 0, "token budget (default from config)")
	packCmd.Flags().StringVarP(&packOutput, "output", "o", "", "output file (default: stdout)")
	packCmd.Flags().IntVarP(&packTopK, "top-k", "k", 0, "candidate pool size (default from config)")
	packCmd.MarkFlagRequired("query")
}

func runPack(cmd *cobra.Command, args []string) error {
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

	// Create use cases
	retrieveUC := usecase.NewRetrieveUseCase(bm25, mmr)
	packUC := usecase.NewPackUseCase(st, tokenizer)

	// Determine parameters
	topK := cfg.Retrieve.TopK
	if packTopK > 0 {
		topK = packTopK
	}

	budget := cfg.Pack.TokenBudget
	if packBudget > 0 {
		budget = packBudget
	}

	// Retrieve candidates
	chunks, err := retrieveUC.Retrieve(packQuery, topK)
	if err != nil {
		return fmt.Errorf("retrieval failed: %w", err)
	}

	if len(chunks) == 0 {
		fmt.Fprintln(os.Stderr, "No relevant content found.")
		return nil
	}

	// Pack context
	packed, err := packUC.Pack(packQuery, chunks, budget)
	if err != nil {
		return fmt.Errorf("packing failed: %w", err)
	}

	// Output
	output, err := json.MarshalIndent(packed, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal output: %w", err)
	}

	if packOutput != "" {
		if err := os.WriteFile(packOutput, output, 0644); err != nil {
			return fmt.Errorf("failed to write output file: %w", err)
		}
		fmt.Printf("Context packed to: %s\n", packOutput)
		fmt.Printf("  Snippets: %d\n", len(packed.Snippets))
		fmt.Printf("  Tokens:   %d / %d\n", packed.UsedTokens, packed.BudgetTokens)
	} else {
		fmt.Println(string(output))
	}

	return nil
}
