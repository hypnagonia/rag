package cli

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/schollz/progressbar/v3"
	"github.com/spf13/cobra"
	"rag/config"
	"rag/internal/adapter/analyzer"
	"rag/internal/adapter/chunker"
	"rag/internal/adapter/embedding"
	"rag/internal/adapter/fs"
	"rag/internal/adapter/store"
	"rag/internal/port"
	"rag/internal/usecase"
)

var indexCmd = &cobra.Command{
	Use:   "index [path]",
	Short: "Index files for retrieval",
	Long: `Index files in the specified directory for later retrieval.
The index is stored in .rag/index.db within the target directory.

Examples:
  rag index .                 # Index current directory
  rag index /path/to/project  # Index specific directory`,
	Args: cobra.MaximumNArgs(1),
	RunE: runIndex,
}

func init() {
	rootCmd.AddCommand(indexCmd)
}

func runIndex(cmd *cobra.Command, args []string) error {
	// Determine path to index
	path := GetRootDir()
	if len(args) > 0 {
		var err error
		path, err = filepath.Abs(args[0])
		if err != nil {
			return fmt.Errorf("invalid path: %w", err)
		}
	}

	// Verify path exists
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("path does not exist: %w", err)
	}
	if !info.IsDir() {
		return fmt.Errorf("path is not a directory: %s", path)
	}

	cfg := GetConfig()

	// Ensure .rag directory exists
	if err := config.EnsureRAGDir(path); err != nil {
		return fmt.Errorf("failed to create .rag directory: %w", err)
	}

	// Create store
	dbPath := config.IndexDBPath(path)
	st, err := store.NewBoltStore(dbPath)
	if err != nil {
		return fmt.Errorf("failed to open index store: %w", err)
	}
	defer st.Close()

	// Check for schema migration or rebuild
	migrationResult, err := st.CheckMigration(cfg)
	if err != nil {
		return fmt.Errorf("failed to check migration: %w", err)
	}

	if migrationResult.NeedsRebuild {
		fmt.Printf("Index rebuild required: %s\n", migrationResult.Reason)
		fmt.Println("Clearing existing index...")
		if err := st.Clear(); err != nil {
			return fmt.Errorf("failed to clear index: %w", err)
		}
	} else if migrationResult.NeedsMigration {
		fmt.Printf("Running schema migration: %s\n", migrationResult.Reason)
		if err := st.Migrate(cfg); err != nil {
			return fmt.Errorf("migration failed: %w", err)
		}
	}

	// Create tokenizer
	tokenizer := analyzer.NewTokenizer(cfg.Index.Stemming)

	// Create walker with configured patterns
	walker := fs.NewWalker(cfg.Index.Includes, cfg.Index.Excludes)

	// Create chunker (use composite chunker if AST chunking is enabled)
	var chk port.Chunker
	if cfg.Index.ASTChunking {
		chk = chunker.NewCompositeChunker(cfg.Index.ChunkTokens, cfg.Index.ChunkOverlap, tokenizer, true)
	} else {
		chk = chunker.NewLineChunker(cfg.Index.ChunkTokens, cfg.Index.ChunkOverlap, tokenizer)
	}

	// Create index use case
	indexUC := usecase.NewIndexUseCase(st, walker, chk, tokenizer)

	fmt.Printf("Scanning %s...\n", path)

	// Create progress bar (will be initialized once we know total files)
	var bar *progressbar.ProgressBar
	var barMu sync.Mutex
	var startTime time.Time
	var initialized bool

	progressCallback := func(processed, total int, currentFile string) {
		barMu.Lock()
		defer barMu.Unlock()

		if !initialized {
			startTime = time.Now()
			bar = progressbar.NewOptions(total,
				progressbar.OptionEnableColorCodes(true),
				progressbar.OptionShowBytes(false),
				progressbar.OptionSetWidth(40),
				progressbar.OptionShowCount(),
				progressbar.OptionSetDescription("[cyan]Indexing[reset]"),
				progressbar.OptionSetTheme(progressbar.Theme{
					Saucer:        "[green]=[reset]",
					SaucerHead:    "[green]>[reset]",
					SaucerPadding: " ",
					BarStart:      "[",
					BarEnd:        "]",
				}),
				progressbar.OptionOnCompletion(func() {
					fmt.Println()
				}),
			)
			initialized = true
		}

		bar.Set(processed)

		// Calculate and display ETA
		if processed > 0 {
			elapsed := time.Since(startTime)
			rate := float64(processed) / elapsed.Seconds()
			remaining := total - processed
			if rate > 0 {
				eta := time.Duration(float64(remaining)/rate) * time.Second
				bar.Describe(fmt.Sprintf("[cyan]Indexing[reset] ETA: %s", formatDuration(eta)))
			}
		}
	}

	result, err := indexUC.Index(path, progressCallback)
	if err != nil {
		return fmt.Errorf("indexing failed: %w", err)
	}

	// Update schema info after successful indexing
	if err := st.Migrate(cfg); err != nil {
		return fmt.Errorf("failed to update schema info: %w", err)
	}

	// Generate embeddings if enabled
	var embeddingsGenerated int
	fmt.Printf("\nEmbedding config: enabled=%v, provider=%s, model=%s\n", cfg.Embedding.Enabled, cfg.Embedding.Provider, cfg.Embedding.Model)
	if cfg.Embedding.Enabled {
		embeddingsGenerated, err = generateEmbeddings(st, cfg)
		if err != nil {
			fmt.Printf("\nWarning: embedding generation failed: %v\n", err)
		}
	}

	// Print results
	fmt.Printf("\nIndexing complete:\n")
	fmt.Printf("  Files indexed:  %d\n", result.FilesIndexed)
	fmt.Printf("  Files skipped:  %d (unchanged)\n", result.FilesSkipped)
	fmt.Printf("  Files deleted:  %d (removed)\n", result.FilesDeleted)
	fmt.Printf("  Chunks created: %d\n", result.ChunksCreated)
	if embeddingsGenerated > 0 {
		fmt.Printf("  Embeddings:     %d\n", embeddingsGenerated)
	}

	if len(result.Errors) > 0 {
		fmt.Printf("\nWarnings:\n")
		for _, e := range result.Errors {
			fmt.Printf("  - %s\n", e)
		}
	}

	fmt.Printf("\nIndex stored at: %s\n", dbPath)
	return nil
}

// generateEmbeddings generates vector embeddings for all chunks.
func generateEmbeddings(st *store.BoltStore, cfg *config.Config) (int, error) {
	// Create embedder based on config
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
		return 0, fmt.Errorf("unsupported embedding provider: %s", cfg.Embedding.Provider)
	}
	if err != nil {
		return 0, fmt.Errorf("failed to create embedder: %w", err)
	}

	// Create vector store
	vectorStore, err := store.NewBoltVectorStore(st.DB(), embedder.Dimension())
	if err != nil {
		return 0, fmt.Errorf("failed to create vector store: %w", err)
	}

	// Get all chunks that need embeddings
	docs, err := st.ListDocs()
	if err != nil {
		return 0, err
	}

	var allChunks []struct {
		id   string
		text string
	}

	for _, doc := range docs {
		chunks, err := st.GetChunksByDoc(doc.ID)
		if err != nil {
			continue
		}
		for _, chunk := range chunks {
			allChunks = append(allChunks, struct {
				id   string
				text string
			}{chunk.ID, chunk.Text})
		}
	}

	if len(allChunks) == 0 {
		return 0, nil
	}

	fmt.Printf("\nGenerating embeddings for %d chunks...\n", len(allChunks))

	// Process in batches
	batchSize := cfg.Embedding.BatchSize
	if batchSize <= 0 {
		batchSize = 100
	}

	bar := progressbar.NewOptions(len(allChunks),
		progressbar.OptionEnableColorCodes(true),
		progressbar.OptionShowBytes(false),
		progressbar.OptionSetWidth(40),
		progressbar.OptionShowCount(),
		progressbar.OptionSetDescription("[cyan]Embedding[reset]"),
		progressbar.OptionOnCompletion(func() {
			fmt.Println()
		}),
	)

	generated := 0
	for i := 0; i < len(allChunks); i += batchSize {
		end := i + batchSize
		if end > len(allChunks) {
			end = len(allChunks)
		}
		batch := allChunks[i:end]

		// Extract texts for embedding
		texts := make([]string, len(batch))
		for j, c := range batch {
			texts[j] = c.text
		}

		// Generate embeddings
		embeddings, err := embedder.Embed(texts)
		if err != nil {
			return generated, fmt.Errorf("embedding batch failed: %w", err)
		}

		// Store vectors
		items := make([]port.VectorItem, len(batch))
		for j, c := range batch {
			items[j] = port.VectorItem{
				ID:     c.id,
				Vector: embeddings[j],
			}
		}

		if err := vectorStore.Upsert(items); err != nil {
			return generated, fmt.Errorf("failed to store vectors: %w", err)
		}

		generated += len(batch)
		bar.Set(generated)
	}

	return generated, nil
}

// formatDuration formats a duration in a human-readable way.
func formatDuration(d time.Duration) string {
	if d < time.Second {
		return "<1s"
	}
	if d < time.Minute {
		return fmt.Sprintf("%ds", int(d.Seconds()))
	}
	if d < time.Hour {
		m := int(d.Minutes())
		s := int(d.Seconds()) % 60
		return fmt.Sprintf("%dm%ds", m, s)
	}
	h := int(d.Hours())
	m := int(d.Minutes()) % 60
	return fmt.Sprintf("%dh%dm", h, m)
}
