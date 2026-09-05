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
	indexCmd.Flags().BoolVar(&indexForceEmbed, "force-embed", false, "re-embed every chunk instead of reusing existing vectors")
}

var indexForceEmbed bool

func runIndex(cmd *cobra.Command, args []string) error {

	path := GetRootDir()
	if len(args) > 0 {
		var err error
		path, err = filepath.Abs(args[0])
		if err != nil {
			return fmt.Errorf("invalid path: %w", err)
		}
	}

	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("path does not exist: %w", err)
	}
	if !info.IsDir() {
		return fmt.Errorf("path is not a directory: %s", path)
	}

	cfg := GetConfig()

	if err := config.EnsureRAGDir(path); err != nil {
		return fmt.Errorf("failed to create .rag directory: %w", err)
	}

	dbPath := config.IndexDBPath(path)
	st, err := store.NewBoltStore(dbPath)
	if err != nil {
		return fmt.Errorf("failed to open index store: %w", err)
	}
	defer st.Close()

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

	tokenizer := analyzer.NewTokenizer(cfg.Index.Stemming)

	walker := fs.NewWalker(cfg.Index.Includes, cfg.Index.Excludes)

	var chk port.Chunker
	if cfg.Index.ASTChunking {
		chk = chunker.NewCompositeChunker(cfg.Index.ChunkTokens, cfg.Index.ChunkOverlap, tokenizer, true)
	} else {
		chk = chunker.NewLineChunker(cfg.Index.ChunkTokens, cfg.Index.ChunkOverlap, tokenizer)
	}

	indexUC := usecase.NewIndexUseCase(st, walker, chk, tokenizer)

	fmt.Printf("Scanning %s...\n", path)

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

	if err := st.Migrate(cfg); err != nil {
		return fmt.Errorf("failed to update schema info: %w", err)
	}

	var embedResult *usecase.EmbedResult
	if cfg.Embedding.Enabled {
		fmt.Printf("\nEmbeddings: provider=%s model=%s\n", cfg.Embedding.Provider, cfg.Embedding.Model)
		embedResult, err = generateEmbeddings(st, cfg)
		if err != nil {
			fmt.Printf("\nWarning: embedding generation failed: %v\n", err)
		}
	}

	fmt.Printf("\nIndexing complete:\n")
	fmt.Printf("  Files indexed:  %d\n", result.FilesIndexed)
	fmt.Printf("  Files skipped:  %d (unchanged)\n", result.FilesSkipped)
	fmt.Printf("  Files deleted:  %d (removed)\n", result.FilesDeleted)
	fmt.Printf("  Chunks created: %d\n", result.ChunksCreated)
	if embedResult != nil {
		fmt.Printf("  Embeddings:     %d new, %d reused, %d stale removed (%d total chunks)\n",
			embedResult.Embedded, embedResult.Reused, embedResult.Deleted, embedResult.TotalChunks)
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

func generateEmbeddings(st *store.BoltStore, cfg *config.Config) (*usecase.EmbedResult, error) {
	embedder, err := buildEmbedder(cfg)
	if err != nil {
		return nil, err
	}

	vectorStore, err := openVectorStore(st, embedder)
	if err != nil {
		return nil, err
	}

	meta, err := vectorStore.Meta()
	if err != nil {
		return nil, err
	}
	want := currentVectorMeta(embedder)
	if meta != nil && (meta.Model != want.Model || meta.Dimension != want.Dimension) {
		fmt.Printf("Embedding model changed (%s/%d -> %s/%d), discarding old vectors\n",
			meta.Model, meta.Dimension, want.Model, want.Dimension)
		ids, err := vectorStore.IDs()
		if err != nil {
			return nil, err
		}
		if err := vectorStore.Delete(ids); err != nil {
			return nil, err
		}
	}

	embedUC, err := usecase.NewEmbedBuilder().
		Store(st).
		Embedder(embedder).
		VectorStore(vectorStore).
		BatchSize(cfg.Embedding.BatchSize).
		Force(indexForceEmbed).
		Build()
	if err != nil {
		return nil, err
	}

	var bar *progressbar.ProgressBar
	progress := func(embedded, total int) {
		if bar == nil {
			fmt.Printf("\nGenerating embeddings for %d chunks (%s)...\n", total, embedder.ModelName())
			bar = progressbar.NewOptions(total,
				progressbar.OptionEnableColorCodes(true),
				progressbar.OptionShowBytes(false),
				progressbar.OptionSetWidth(40),
				progressbar.OptionShowCount(),
				progressbar.OptionSetDescription("[cyan]Embedding[reset]"),
				progressbar.OptionOnCompletion(func() {
					fmt.Println()
				}),
			)
		}
		bar.Set(embedded)
	}

	result, err := embedUC.Sync(progress)
	if err != nil {
		return result, err
	}

	if err := vectorStore.SetMeta(want); err != nil {
		return result, fmt.Errorf("failed to record embedding metadata: %w", err)
	}

	return result, nil
}

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
