package main

import (
	"flag"
	"fmt"
	"os"

	"rag/config"
	"rag/internal/adapter/analyzer"
	"rag/internal/adapter/embedding"
	"rag/internal/adapter/llm"
	"rag/internal/adapter/retriever"
	"rag/internal/adapter/store"
	"rag/internal/port"
	"rag/internal/usecase"
)

func main() {
	query := flag.String("q", "", "Question to answer (required)")
	indexPath := flag.String("index", ".", "Path to an indexed directory")
	topK := flag.Int("k", 10, "Chunks to retrieve per search")
	budget := flag.Int("budget", 4000, "Context token budget")
	maxIters := flag.Int("max-iters", 2, "Maximum retrieve/evaluate rounds")
	fast := flag.Bool("fast", false, "Search once and answer: exactly one LLM call")
	expand := flag.Bool("expand", false, "Expand the query with the LLM first (+1 call)")
	verbose := flag.Bool("v", false, "Print progress")
	flag.Parse()

	cwd, _ := os.Getwd()
	config.LoadDotEnv(*indexPath, cwd)

	if *query == "" {
		fmt.Fprintln(os.Stderr, "Usage: go run . -q \"your question\" -index /path/to/indexed")
		flag.PrintDefaults()
		os.Exit(1)
	}

	if err := run(*query, *indexPath, *topK, *budget, *maxIters, *fast, *expand, *verbose); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func run(query, indexPath string, topK, budget, maxIters int, fast, expand, verbose bool) error {
	cfg, err := config.LoadFromDir(indexPath)
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}

	st, err := store.NewBoltStore(config.IndexDBPath(indexPath))
	if err != nil {
		return fmt.Errorf("failed to open index: %w", err)
	}
	defer st.Close()

	tokenizer := analyzer.NewTokenizer(cfg.Index.Stemming)
	bm25 := retriever.NewBM25Retriever(st, tokenizer, cfg.Index.K1, cfg.Index.B, cfg.Retrieve.PathBoostWeight)

	searchRetriever, mode := buildRetriever(st, cfg, bm25)
	if verbose {
		fmt.Fprintf(os.Stderr, "retrieval: %s\n", mode)
	}

	built, err := llm.NewBuilder().
		Provider(cfg.LLM.Provider).
		Model(cfg.LLM.Model).
		APIKeyEnv(cfg.LLM.APIKeyEnv).
		BaseURL(cfg.LLM.BaseURL).
		MaxTokens(cfg.LLM.MaxTokens).
		Build()
	if err != nil {
		return err
	}

	client := built.(*llm.Client)

	mmr := retriever.NewMMRReranker(cfg.Retrieve.MMRLambda, cfg.Retrieve.DedupJaccard)

	askUC, err := usecase.NewAskBuilder().
		Retriever(usecase.NewRetrieveUseCase(searchRetriever, mmr, cfg.Retrieve.MinScoreThreshold)).
		Packer(usecase.NewPackUseCase(st, tokenizer, cfg.Pack.RecencyBoost)).
		LLM(client).
		MaxIterations(maxIters).
		TopK(topK).
		Budget(budget).
		Fast(fast).
		ExpandQuery(expand).
		Build()
	if err != nil {
		return err
	}

	var progress usecase.AskProgress
	if verbose {
		progress = func(stage usecase.AskStage, iteration int, detail string) {
			fmt.Fprintf(os.Stderr, "[round %d] %s\n", iteration, stage)
		}
	}

	result, err := askUC.Ask(query, progress)
	if err != nil {
		return err
	}

	fmt.Printf("\n%s\n", result.Answer)
	printStats(mode, client, result, maxIters)

	return nil
}

func buildRetriever(st *store.BoltStore, cfg *config.Config, bm25 port.Retriever) (port.Retriever, string) {
	if !cfg.Retrieve.HybridEnabled || !cfg.Embedding.Enabled {
		return bm25, "BM25 only"
	}

	embedder, err := embedding.NewBuilder().
		Provider(cfg.Embedding.Provider).
		Model(cfg.Embedding.Model).
		APIKeyEnv(cfg.Embedding.APIKeyEnv).
		BaseURL(cfg.Embedding.BaseURL).
		Dimension(cfg.Embedding.Dimension).
		BatchSize(cfg.Embedding.BatchSize).
		Build()
	if err != nil {
		return bm25, fmt.Sprintf("BM25 only (embedder unavailable: %v)", err)
	}

	vectorStore, err := store.NewBoltVectorStore(st.DB(), embedder.Dimension())
	if err != nil {
		return bm25, fmt.Sprintf("BM25 only (vector store unavailable: %v)", err)
	}
	if count, _ := vectorStore.Count(); count == 0 {
		return bm25, "BM25 only (no embeddings indexed)"
	}

	hybrid := retriever.NewHybridBuilder().
		BM25(bm25).
		VectorStore(vectorStore).
		Embedder(embedder).
		ChunkStore(st).
		RRFK(cfg.Retrieve.RRFK).
		BM25Weight(cfg.Retrieve.BM25Weight).
		Build()

	return hybrid, fmt.Sprintf("hybrid: BM25 + %s, RRF (bm25_weight=%.2f)", embedder.ModelName(), cfg.Retrieve.BM25Weight)
}

func printStats(mode string, client *llm.Client, result *usecase.AskResult, maxIters int) {
	line := "──────────────────────────────────────────────────────────────────────"
	s := client.Stats()

	source := "estimated from characters"
	if s.ReportedByAPI {
		source = "reported by the API"
	}

	fmt.Printf("\n%s\n", line)
	fmt.Printf("PIPELINE STATS\n")
	fmt.Printf("%s\n", line)
	fmt.Printf("   Retrieval:              %s\n", mode)
	fmt.Printf("   LLM model:              %s\n", client.ModelName())
	fmt.Printf("   Back-and-forth rounds:  %d of %d max\n", result.IterationsUsed, maxIters)
	fmt.Printf("   LLM calls:              %d\n", result.LLMCalls)
	fmt.Printf("   Search queries used:    %d\n", len(result.QueriesUsed))
	fmt.Printf("   Chunks retrieved:       %d\n", len(result.Chunks))
	fmt.Printf("   Context packed:         %d / %d tokens (%d snippets)\n",
		result.Context.UsedTokens, result.Context.BudgetTokens, len(result.Context.Snippets))
	fmt.Printf("   Input tokens:           %d\n", s.InputTokens)
	fmt.Printf("   Output tokens:          %d\n", s.OutputTokens)
	fmt.Printf("   Total tokens:           %d  (%s)\n", s.InputTokens+s.OutputTokens, source)
	fmt.Printf("%s\n", line)
}
