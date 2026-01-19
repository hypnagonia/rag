package cli

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"
	"rag/config"
	"rag/internal/adapter/analyzer"
	"rag/internal/adapter/chunker"
	"rag/internal/adapter/fs"
	"rag/internal/adapter/store"
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

	// Create tokenizer
	tokenizer := analyzer.NewTokenizer(cfg.Index.Stemming)

	// Create walker with configured patterns
	walker := fs.NewWalker(cfg.Index.Includes, cfg.Index.Excludes)

	// Create chunker
	chk := chunker.NewLineChunker(cfg.Index.ChunkTokens, cfg.Index.ChunkOverlap, tokenizer)

	// Create and run index use case
	indexUC := usecase.NewIndexUseCase(st, walker, chk, tokenizer)

	fmt.Printf("Indexing %s...\n", path)
	result, err := indexUC.Index(path)
	if err != nil {
		return fmt.Errorf("indexing failed: %w", err)
	}

	// Print results
	fmt.Printf("\nIndexing complete:\n")
	fmt.Printf("  Files indexed:  %d\n", result.FilesIndexed)
	fmt.Printf("  Files skipped:  %d (unchanged)\n", result.FilesSkipped)
	fmt.Printf("  Files deleted:  %d (removed)\n", result.FilesDeleted)
	fmt.Printf("  Chunks created: %d\n", result.ChunksCreated)

	if len(result.Errors) > 0 {
		fmt.Printf("\nWarnings:\n")
		for _, e := range result.Errors {
			fmt.Printf("  - %s\n", e)
		}
	}

	fmt.Printf("\nIndex stored at: %s\n", dbPath)
	return nil
}
