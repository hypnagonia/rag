package cli

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"rag/config"
	"rag/internal/adapter/analyzer"
	"rag/internal/adapter/llm"
	"rag/internal/adapter/retriever"
	"rag/internal/adapter/store"
	"rag/internal/port"
	"rag/internal/usecase"
)

var (
	askQuery    string
	askTopK     int
	askBudget   int
	askMaxIters int
	askFast     bool
	askExpand   bool
	askHyDE     bool
	askLexical  bool
	askSemantic bool
	askExplain  bool
)

var askCmd = &cobra.Command{
	Use:   "ask",
	Short: "Answer a question from the index using an LLM",
	Long: `Retrieve relevant context and have an LLM answer the question, with citations.

Retrieval uses the same hybrid BM25 + vector path as 'rag query'. The LLM is
a hosted provider configured under 'llm:' in rag.yaml.

Examples:
  rag ask -q "how does authentication work"
  rag ask -q "how does authentication work" --fast      # one LLM call
  rag ask -q "how does authentication work" --hyde      # better retrieval, +1 call`,
	RunE: runAsk,
}

func init() {
	rootCmd.AddCommand(askCmd)
	askCmd.Flags().StringVarP(&askQuery, "query", "q", "", "question to answer (required)")
	askCmd.Flags().IntVarP(&askTopK, "top-k", "k", 0, "chunks to retrieve per search (default from config)")
	askCmd.Flags().IntVarP(&askBudget, "budget", "b", 0, "context token budget (default from config)")
	askCmd.Flags().IntVar(&askMaxIters, "max-iters", 2, "maximum retrieve/evaluate rounds")
	askCmd.Flags().BoolVar(&askFast, "fast", false, "search once and answer: exactly one LLM call")
	askCmd.Flags().BoolVar(&askExpand, "expand", false, "expand the query with the LLM first (+1 call)")
	askCmd.Flags().BoolVar(&askHyDE, "hyde", false, "expand retrieval with a hypothetical answer (+1 call, cached)")
	askCmd.Flags().BoolVar(&askLexical, "lexical", false, "BM25 only, no embeddings")
	askCmd.Flags().BoolVar(&askSemantic, "semantic", false, "vector search only, no BM25")
	askCmd.Flags().BoolVar(&askExplain, "explain", false, "print retrieval diagnostics")
	askCmd.MarkFlagRequired("query")
}

func runAsk(cmd *cobra.Command, args []string) error {
	cfg := GetConfig()
	rootDir := GetRootDir()

	dbPath := config.IndexDBPath(rootDir)
	if _, err := os.Stat(dbPath); os.IsNotExist(err) {
		return fmt.Errorf("no index found. Run 'rag index' first")
	}

	st, err := store.NewBoltStore(dbPath)
	if err != nil {
		return fmt.Errorf("failed to open index: %w", err)
	}
	defer st.Close()

	tokenizer := analyzer.NewTokenizer(cfg.Index.Stemming)

	if askLexical && askSemantic {
		return fmt.Errorf("--lexical and --semantic are mutually exclusive")
	}

	mode := ModeAuto
	if askSemantic {
		mode = ModeSemantic
	} else if askLexical {
		mode = ModeLexical
	}

	searchRetriever, plan, err := buildRetriever(st, cfg, tokenizer, mode)
	if err != nil {
		return err
	}
	if askHyDE {
		searchRetriever, plan, err = wrapWithHyDE(st, cfg, searchRetriever, plan)
		if err != nil {
			return err
		}
	}
	if askExplain || plan.Warning != "" {
		fmt.Fprintln(os.Stderr, plan.Describe())
	}

	client, err := llm.NewBuilder().
		Provider(cfg.LLM.Provider).
		Model(cfg.LLM.Model).
		APIKeyEnv(cfg.LLM.APIKeyEnv).
		BaseURL(cfg.LLM.BaseURL).
		MaxTokens(cfg.LLM.MaxTokens).
		Build()
	if err != nil {
		return fmt.Errorf("ask needs an LLM: %w", err)
	}

	topK := cfg.Retrieve.TopK
	if askTopK > 0 {
		topK = askTopK
	}
	budget := cfg.Pack.TokenBudget
	if askBudget > 0 {
		budget = askBudget
	}

	mmr := retriever.NewMMRReranker(cfg.Retrieve.MMRLambda, cfg.Retrieve.DedupJaccard)
	retrieveUC := usecase.NewRetrieveUseCase(searchRetriever, mmr, cfg.Retrieve.MinScoreThreshold)
	packUC := usecase.NewPackUseCase(st, tokenizer, cfg.Pack.RecencyBoost)

	askUC, err := usecase.NewAskBuilder().
		Retriever(retrieveUC).
		Packer(packUC).
		LLM(client).
		MaxIterations(askMaxIters).
		TopK(topK).
		Budget(budget).
		Fast(askFast).
		ExpandQuery(askExpand).
		Build()
	if err != nil {
		return err
	}

	progress := func(stage usecase.AskStage, iteration int, detail string) {
		fmt.Fprintf(os.Stderr, "[%d] %s\n", iteration, stage)
	}

	result, err := askUC.Ask(askQuery, progress)
	if err != nil {
		return err
	}

	fmt.Println()
	fmt.Println(result.Answer)

	printAskStats(plan, client, result, askMaxIters)
	return nil
}

func printAskStats(plan RetrievalPlan, client port.LLM, result *usecase.AskResult, maxIters int) {
	line := "──────────────────────────────────────────────────────────────────────"

	fmt.Printf("\n%s\n", line)
	fmt.Printf("PIPELINE STATS\n")
	fmt.Printf("%s\n", line)
	fmt.Printf("   Retrieval:              %s\n", plan.Mode)
	if plan.Model != "" {
		fmt.Printf("   Embedding model:        %s (%d vectors)\n", plan.Model, plan.Vectors)
	}
	fmt.Printf("   LLM model:              %s\n", client.ModelName())
	fmt.Printf("   Back-and-forth rounds:  %d of %d max\n", result.IterationsUsed, maxIters)
	fmt.Printf("   LLM calls:              %d\n", result.LLMCalls)
	fmt.Printf("   Search queries used:    %d\n", len(result.QueriesUsed))
	fmt.Printf("   Chunks retrieved:       %d\n", len(result.Chunks))
	fmt.Printf("   Context packed:         %d / %d tokens (%d snippets)\n",
		result.Context.UsedTokens, result.Context.BudgetTokens, len(result.Context.Snippets))

	if reporter, ok := client.(*llm.Client); ok {
		s := reporter.Stats()
		source := "estimated from characters"
		if s.ReportedByAPI {
			source = "reported by the API"
		}
		fmt.Printf("   Input tokens:           %s\n", formatThousands(s.InputTokens))
		fmt.Printf("   Output tokens:          %s\n", formatThousands(s.OutputTokens))
		fmt.Printf("   Total tokens:           %s  (%s)\n", formatThousands(s.InputTokens+s.OutputTokens), source)
	}
	fmt.Printf("%s\n", line)
}

func formatThousands(n int) string {
	sign := ""
	if n < 0 {
		sign = "-"
		n = -n
	}

	digits := fmt.Sprintf("%d", n)
	var out []byte
	for i, c := range []byte(digits) {
		if i > 0 && (len(digits)-i)%3 == 0 {
			out = append(out, ',')
		}
		out = append(out, c)
	}

	return sign + string(out)
}
