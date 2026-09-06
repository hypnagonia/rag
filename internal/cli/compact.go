package cli

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"rag/config"
	"rag/internal/adapter/store"
)

var compactEncoding string

var compactCmd = &cobra.Command{
	Use:   "compact",
	Short: "Shrink the index on disk",
	Long: `Rewrite legacy JSON vector records as packed binary and reclaim free pages.

Vectors written before the binary format used roughly three times the space they
needed. This converts them in place - no re-embedding, no quality change - and
then compacts the database file.`,
	RunE: runCompact,
}

func init() {
	rootCmd.AddCommand(compactCmd)
	compactCmd.Flags().StringVar(&compactEncoding, "encoding", "", "re-encode vectors: float32, float16 or int8 (float16 and int8 are lossy)")
}

func runCompact(cmd *cobra.Command, args []string) error {
	rootDir := GetRootDir()
	dbPath := config.IndexDBPath(rootDir)

	if _, err := os.Stat(dbPath); os.IsNotExist(err) {
		return fmt.Errorf("no index found at %s", dbPath)
	}

	rewritten, err := rewriteLegacyVectors(dbPath)
	if err != nil {
		return err
	}
	if rewritten > 0 {
		fmt.Printf("Rewrote %d vector records as packed binary\n", rewritten)
	}

	if compactEncoding != "" {
		converted, breakdown, err := reencodeVectors(dbPath, compactEncoding)
		if err != nil {
			return err
		}
		fmt.Printf("Re-encoded %d vectors as %s %v\n", converted, compactEncoding, breakdown)
	}

	before, after, err := store.CompactFile(dbPath)
	if err != nil {
		return fmt.Errorf("compaction failed: %w", err)
	}

	fmt.Printf("Index: %s -> %s", formatBytes(before), formatBytes(after))
	if before > 0 && after < before {
		fmt.Printf("  (%.1f%% smaller)", 100*float64(before-after)/float64(before))
	}
	fmt.Println()

	return nil
}

func rewriteLegacyVectors(dbPath string) (int, error) {
	cfg := GetConfig()
	if !cfg.Embedding.Enabled {
		return 0, nil
	}

	st, err := store.NewBoltStore(dbPath)
	if err != nil {
		return 0, fmt.Errorf("failed to open index: %w", err)
	}
	defer st.Close()

	meta, err := readVectorMeta(st)
	if err != nil || meta == nil || meta.Dimension <= 0 {
		return 0, nil
	}

	vectorStore, err := store.NewBoltVectorStore(st.DB(), meta.Dimension)
	if err != nil {
		return 0, nil
	}

	if vectorStore.LegacyRecordCount() == 0 {
		return 0, nil
	}

	return vectorStore.RewriteLegacyRecords()
}

func readVectorMeta(st *store.BoltStore) (*store.VectorMeta, error) {
	probe, err := store.NewBoltVectorStore(st.DB(), 1)
	if err != nil {
		return nil, err
	}
	return probe.Meta()
}

func formatBytes(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for n/div >= unit && exp < 3 {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(n)/float64(div), "KMG"[exp])
}

func reencodeVectors(dbPath, encoding string) (int, map[string]int, error) {
	switch encoding {
	case store.EncodingFloat32, store.EncodingFloat16, store.EncodingInt8:
	default:
		return 0, nil, fmt.Errorf("unknown encoding %q (want float32, float16 or int8)", encoding)
	}

	st, err := store.NewBoltStore(dbPath)
	if err != nil {
		return 0, nil, err
	}
	defer st.Close()

	meta, err := readVectorMeta(st)
	if err != nil || meta == nil || meta.Dimension <= 0 {
		return 0, nil, nil
	}

	vectorStore, err := store.NewBoltVectorStore(st.DB(), meta.Dimension)
	if err != nil {
		return 0, nil, err
	}

	converted, err := vectorStore.ReencodeAs(encoding)
	if err != nil {
		return 0, nil, err
	}

	return converted, vectorStore.EncodingBreakdown(), nil
}
