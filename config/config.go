package config

import (
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Index     IndexConfig     `yaml:"index"`
	Retrieve  RetrieveConfig  `yaml:"retrieve"`
	Pack      PackConfig      `yaml:"pack"`
	Embedding EmbeddingConfig `yaml:"embedding"`
	Logging   LoggingConfig   `yaml:"logging"`
}

type EmbeddingConfig struct {
	Enabled   bool   `yaml:"enabled"`
	Provider  string `yaml:"provider"`
	Model     string `yaml:"model"`
	APIKeyEnv string `yaml:"api_key_env"`
	BaseURL   string `yaml:"base_url"`
	Dimension int    `yaml:"dimension"`
	BatchSize int    `yaml:"batch_size"`
}

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

type RetrieveConfig struct {
	TopK              int     `yaml:"top_k"`
	MMRLambda         float64 `yaml:"mmr_lambda"`
	DedupJaccard      float64 `yaml:"dedup_jaccard"`
	PathBoostWeight   float64 `yaml:"path_boost_weight"`
	HybridEnabled     bool    `yaml:"hybrid_enabled"`
	RRFK              int     `yaml:"rrf_k"`
	BM25Weight        float64 `yaml:"bm25_weight"`
	MinScoreThreshold float64 `yaml:"min_score_threshold"`
}

type PackConfig struct {
	TokenBudget  int     `yaml:"token_budget"`
	RecencyBoost float64 `yaml:"recency_boost"`
	Summarize    bool    `yaml:"summarize"`
	Output       string  `yaml:"output"`
}

type LoggingConfig struct {
	Level string `yaml:"level"`
}

func DefaultConfig() *Config {
	return &Config{
		Index: IndexConfig{
			Includes:     []string{"***.py", "**/*.js", "**/*.ts", "**/*.java", "**/*.c", "**/*.cpp", "**/*.h", "**/*.rs", "**/*.md", "**/*.txt", "**/*_test.go", "**/test_*.py", "**/*_test.py", "**/*.test.js", "**/*.test.ts", "**/*.spec.js", "**/*.spec.ts", "**/*Test.java"},
			Excludes:     []string{"**/node_modulesvendor/**", "**/.git/**", "**/dist/**", "**/build/**", "**/__pycache__/**", "**/*.min.js"},
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
			HybridEnabled:   false,
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

func Load(path string) (*Config, error) {
	cfg := DefaultConfig()

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return cfg, nil
		}
		return nil, err
	}

	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, err
	}

	return cfg, nil
}

func LoadFromDir(dir string) (*Config, error) {

	path := filepath.Join(dir, "rag.yaml")
	if _, err := os.Stat(path); err == nil {
		return Load(path)
	}

	path = filepath.Join(dir, ".rag", "config.yaml")
	if _, err := os.Stat(path); err == nil {
		return Load(path)
	}

	return DefaultConfig(), nil
}

func (c *Config) Save(path string) error {
	data, err := yaml.Marshal(c)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

func IndexDBPath(dir string) string {
	return filepath.Join(dir, ".rag", "index.db")
}

func EnsureRAGDir(dir string) error {
	ragDir := filepath.Join(dir, ".rag")
	return os.MkdirAll(ragDir, 0755)
}
