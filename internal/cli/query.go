package cli

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"
	"golang.org/x/term"
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
	queryText     string
	queryTopK     int
	queryJSON     bool
	queryNoMMR    bool
	queryContext  int
	querySemantic bool
)

var queryCmd = &cobra.Command{
	Use:   "query",
	Short: "Search indexed files",
	Long: `Search for relevant code chunks using BM25 retrieval with MMR deduplication.

Search modes:
  - Default: BM25 keyword search (or hybrid if configured)
  - --semantic: Uses only embedding/vector search (requires embeddings enabled)

Examples:
  rag query -q "authentication handler"
  rag query -q "database connection" --top-k 10 --json
  rag query -q "how to handle errors" --semantic`,
	RunE: runQuery,
}

func init() {
	rootCmd.AddCommand(queryCmd)
	queryCmd.Flags().StringVarP(&queryText, "query", "q", "", "search query (required)")
	queryCmd.Flags().IntVarP(&queryTopK, "top-k", "k", 0, "number of results (default from config)")
	queryCmd.Flags().BoolVar(&queryJSON, "json", false, "output as JSON")
	queryCmd.Flags().BoolVar(&queryNoMMR, "no-mmr", false, "disable MMR reranking")
	queryCmd.Flags().IntVarP(&queryContext, "context", "c", 0, "expand results by N lines before/after")
	queryCmd.Flags().BoolVar(&querySemantic, "semantic", false, "use only embedding/vector search (no BM25)")
	queryCmd.MarkFlagRequired("query")
}

func runQuery(cmd *cobra.Command, args []string) error {
	cfg := GetConfig()
	rootDir := GetRootDir()

	dbPath := config.IndexDBPath(rootDir)
	if _, err := os.Stat(dbPath); os.IsNotExist(err) {
		if askYesNo("No index found. Index this directory?") {
			if err := runIndex(cmd, []string{rootDir}); err != nil {
				return fmt.Errorf("indexing failed: %w", err)
			}
		} else {
			return fmt.Errorf("no index found. Run 'rag index' first")
		}
	}

	st, err := store.NewBoltStore(dbPath)
	if err != nil {
		return fmt.Errorf("failed to open index: %w", err)
	}
	defer st.Close()

	changedFiles := checkForChanges(st)
	if len(changedFiles) > 0 {
		fmt.Printf("Detected %d changed file(s):\n", len(changedFiles))
		for i, f := range changedFiles {
			if i >= 5 {
				fmt.Printf("  ... and %d more\n", len(changedFiles)-5)
				break
			}
			fmt.Printf("  - %s\n", f)
		}
		if askYesNo("Reindex?") {
			st.Close()
			if err := runIndex(cmd, []string{rootDir}); err != nil {
				return fmt.Errorf("reindexing failed: %w", err)
			}
			st, err = store.NewBoltStore(dbPath)
			if err != nil {
				return fmt.Errorf("failed to reopen index: %w", err)
			}
			defer st.Close()
		}
	}

	tokenizer := analyzer.NewTokenizer(cfg.Index.Stemming)

	bm25 := retriever.NewBM25Retriever(st, tokenizer, cfg.Index.K1, cfg.Index.B, cfg.Retrieve.PathBoostWeight)
	mmr := retriever.NewMMRReranker(cfg.Retrieve.MMRLambda, cfg.Retrieve.DedupJaccard)

	var searchRetriever port.Retriever = bm25

	if querySemantic {
		if !cfg.Embedding.Enabled {
			return fmt.Errorf("semantic search requires embeddings. Enable in rag.yaml:\n  embedding:\n    enabled: true\n    provider: ollama\n    model: nomic-embed-text")
		}
		embedder, vectorStore, err := setupHybridRetrieval(st, cfg)
		if err != nil {
			return fmt.Errorf("semantic search unavailable: %v", err)
		}
		searchRetriever = retriever.NewSemanticRetriever(vectorStore, embedder, st)
	} else if cfg.Retrieve.HybridEnabled && cfg.Embedding.Enabled {
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

	retrieveUC := usecase.NewRetrieveUseCase(searchRetriever, mmr, cfg.Retrieve.MinScoreThreshold)

	topK := cfg.Retrieve.TopK
	if queryTopK > 0 {
		topK = queryTopK
	}

	var chunks []domain.ScoredChunk
	if queryNoMMR {
		chunks, err = retrieveUC.RetrieveWithoutMMR(queryText, topK)
	} else {
		chunks, err = retrieveUC.Retrieve(queryText, topK)
	}
	if err != nil {
		return fmt.Errorf("search failed: %w", err)
	}

	var results []usecase.ScoredChunkResult
	for _, c := range chunks {
		doc, _ := st.GetDoc(c.Chunk.DocID)
		startLine := c.Chunk.StartLine
		endLine := c.Chunk.EndLine
		text := c.Chunk.Text

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

func expandContext(path string, startLine, endLine, extraLines int) (newStart, newEnd int, text string, err error) {
	if extraLines <= 0 {
		return startLine, endLine, "", nil
	}

	file, err := os.Open(path)
	if err != nil {
		return startLine, endLine, "", err
	}
	defer file.Close()

	newStart = startLine - extraLines
	if newStart < 1 {
		newStart = 1
	}
	newEnd = endLine + extraLines

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

	count, err := vectorStore.Count()
	if err != nil {
		return nil, nil, err
	}
	if count == 0 {
		return nil, nil, fmt.Errorf("no embeddings found - run 'rag index' first with embedding.enabled=true")
	}

	return embedder, vectorStore, nil
}

func askYesNo(prompt string) bool {
	if !term.IsTerminal(int(os.Stdin.Fd())) {
		fmt.Printf("%s [auto: yes]\n", prompt)
		return true
	}
	reader := bufio.NewReader(os.Stdin)
	fmt.Printf("%s [Y/n]: ", prompt)
	response, err := reader.ReadString('\n')
	if err != nil {
		return true
	}
	response = strings.TrimSpace(strings.ToLower(response))
	return response != "n" && response != "no"
}

func checkForChanges(st *store.BoltStore) []string {
	docs, err := st.ListDocs()
	if err != nil {
		return nil
	}

	var changed []string
	for _, doc := range docs {
		info, err := os.Stat(doc.Path)
		if err != nil {
			if os.IsNotExist(err) {
				changed = append(changed, doc.Path+" (deleted)")
			}
			continue
		}
		if info.ModTime().Unix() > doc.ModTime.Unix() {
			changed = append(changed, doc.Path)
		}
	}
	return changed
}
