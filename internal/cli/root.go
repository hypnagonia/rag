package cli

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"rag/config"
)

var (
	cfgFile string
	cfg     *config.Config
	rootDir string
)

var rootCmd = &cobra.Command{
	Use:   "rag",
	Short: "RAG Context Compressor - Index and retrieve code for LLM consumption",
	Long: `RAG is a CLI tool that indexes local files using BM25 for lexical retrieval,
retrieves and ranks results with MMR deduplication, and packs compressed context
with citations for LLM consumption.

Example usage:
  rag index .                    # Index current directory
  rag query -q "authentication"  # Search for relevant code
  rag pack -q "how auth works"   # Pack context for LLM`,
	PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
		var err error

		if rootDir == "" {
			rootDir, err = os.Getwd()
			if err != nil {
				return fmt.Errorf("failed to get working directory: %w", err)
			}
		}

		if cfgFile != "" {
			cfg, err = config.Load(cfgFile)
		} else {
			cfg, err = config.LoadFromDir(rootDir)
		}
		if err != nil {
			return fmt.Errorf("failed to load config: %w", err)
		}

		return nil
	},
}

func Execute() {
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

func init() {
	rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default is ./rag.yaml)")
	rootCmd.PersistentFlags().StringVarP(&rootDir, "dir", "d", "", "root directory (default is current directory)")
}

func GetConfig() *config.Config {
	return cfg
}

func GetRootDir() string {
	return rootDir
}
