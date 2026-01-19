package config

import (
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

// Config holds all configuration for the RAG tool.
type Config struct {
	Index     IndexConfig     `yaml:"index"`
	Retrieve  RetrieveConfig  `yaml:"retrieve"`
	Pack      PackConfig      `yaml:"pack"`
	Embedding EmbeddingConfig `yaml:"embedding"`
	Logging   LoggingConfig   `yaml:"logging"`
}

// EmbeddingConfig holds embedding configuration.
type EmbeddingConfig struct {
	Enabled   bool   `yaml:"enabled"`
	Provider  string `yaml:"provider"`   // "openai", "voyage", "local", "mock"
	Model     string `yaml:"model"`      // e.g., "text-embedding-3-small"
	APIKeyEnv string `yaml:"api_key_env"` // Environment variable for API key
	Dimension int    `yaml:"dimension"`
	BatchSize int    `yaml:"batch_size"`
}

// IndexConfig holds indexing configuration.
type IndexConfig struct {
	Includes     []string `yaml:"includes"`
	Excludes     []string `yaml:"excludes"`
	Language     string   `yaml:"language"`
	Stemming     bool     `yaml:"stemming"`
	ChunkTokens  int      `yaml:"chunk_tokens"`
	ChunkOverlap int      `yaml:"chunk_overlap"`
	K1           float64  `yaml:"k1"`
	B            float64  `yaml:"b"`
	ASTChunking  bool     `yaml:"ast_chunking"`
}

// RetrieveConfig holds retrieval configuration.
type RetrieveConfig struct {
	TopK              int     `yaml:"top_k"`
	MMRLambda         float64 `yaml:"mmr_lambda"`
	DedupJaccard      float64 `yaml:"dedup_jaccard"`
	PathBoostWeight   float64 `yaml:"path_boost_weight"`
	HybridEnabled     bool    `yaml:"hybrid_enabled"`
	RRFK              int     `yaml:"rrf_k"`
	BM25Weight        float64 `yaml:"bm25_weight"`
	MinScoreThreshold float64 `yaml:"min_score_threshold"` // Filter results below this score (0 = disabled)
}

// PackConfig holds context packing configuration.
type PackConfig struct {
	TokenBudget  int    `yaml:"token_budget"`
	RecencyBoost float64 `yaml:"recency_boost"`
	Summarize    bool   `yaml:"summarize"`
	Output       string `yaml:"output"`
}

// LoggingConfig holds logging configuration.
type LoggingConfig struct {
	Level string `yaml:"level"`
}

// DefaultConfig returns the default configuration.
func DefaultConfig() *Config {
	return &Config{
		Index: IndexConfig{
			Includes:     []string{"**/*.go", "**/*.py", "**/*.js", "**/*.ts", "**/*.java", "**/*.c", "**/*.cpp", "**/*.h", "**/*.rs", "**/*.md", "**/*.txt", "**/*_test.go", "**/test_*.py", "**/*_test.py", "**/*.test.js", "**/*.test.ts", "**/*.spec.js", "**/*.spec.ts", "**/*Test.java"},
			Excludes:     []string{"**/node_modules/**", "**/vendor/**", "**/.git/**", "**/dist/**", "**/build/**", "**/__pycache__/**", "**/*.min.js"},
			Language:     "auto",
			Stemming:     true,
			ChunkTokens:  512,
			ChunkOverlap: 50,
			K1:           1.2,
			B:            0.75,
			ASTChunking:  true,
		},
		Retrieve: RetrieveConfig{
			TopK:            20,
			MMRLambda:       0.7,
			DedupJaccard:    0.8,
			PathBoostWeight: 0.3,
			HybridEnabled:   false, // Disabled by default (requires API key)
			RRFK:            60,
			BM25Weight:      0.5,
		},
		Embedding: EmbeddingConfig{
			Enabled:   false,
			Provider:  "openai",
			Model:     "text-embedding-3-small",
			APIKeyEnv: "OPENAI_API_KEY",
			Dimension: 1536,
			BatchSize: 100,
		},
		Pack: PackConfig{
			TokenBudget:  4000,
			RecencyBoost: 0.1,
			Summarize:    false,
			Output:       "json",
		},
		Logging: LoggingConfig{
			Level: "info",
		},
	}
}

// Load loads configuration from a YAML file.
func Load(path string) (*Config, error) {
	cfg := DefaultConfig()

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return cfg, nil // Return defaults if no config file
		}
		return nil, err
	}

	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, err
	}

	return cfg, nil
}

// LoadFromDir loads configuration from a directory (looks for rag.yaml).
func LoadFromDir(dir string) (*Config, error) {
	// Try rag.yaml in the directory
	path := filepath.Join(dir, "rag.yaml")
	if _, err := os.Stat(path); err == nil {
		return Load(path)
	}

	// Try .rag/config.yaml
	path = filepath.Join(dir, ".rag", "config.yaml")
	if _, err := os.Stat(path); err == nil {
		return Load(path)
	}

	// Return defaults
	return DefaultConfig(), nil
}

// Save saves configuration to a YAML file.
func (c *Config) Save(path string) error {
	data, err := yaml.Marshal(c)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// IndexDBPath returns the path to the index database.
func IndexDBPath(dir string) string {
	return filepath.Join(dir, ".rag", "index.db")
}

// EnsureRAGDir ensures the .rag directory exists.
func EnsureRAGDir(dir string) error {
	ragDir := filepath.Join(dir, ".rag")
	return os.MkdirAll(ragDir, 0755)
}
