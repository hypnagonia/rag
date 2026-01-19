package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	if cfg.Index.ChunkTokens != 512 {
		t.Errorf("expected ChunkTokens=512, got %d", cfg.Index.ChunkTokens)
	}
	if cfg.Index.K1 != 1.2 {
		t.Errorf("expected K1=1.2, got %f", cfg.Index.K1)
	}
	if cfg.Index.B != 0.75 {
		t.Errorf("expected B=0.75, got %f", cfg.Index.B)
	}
	if cfg.Retrieve.TopK != 20 {
		t.Errorf("expected TopK=20, got %d", cfg.Retrieve.TopK)
	}
	if cfg.Pack.TokenBudget != 4000 {
		t.Errorf("expected TokenBudget=4000, got %d", cfg.Pack.TokenBudget)
	}
}

func TestLoad_NonExistent(t *testing.T) {
	cfg, err := Load("/nonexistent/path/config.yaml")
	if err != nil {
		t.Errorf("expected no error for non-existent file, got %v", err)
	}
	if cfg == nil {
		t.Error("expected default config, got nil")
	}
}

func TestLoad_ValidYAML(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "rag.yaml")

	content := `
index:
  chunk_tokens: 256
  stemming: false
retrieve:
  top_k: 10
`
	if err := os.WriteFile(configPath, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	cfg, err := Load(configPath)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if cfg.Index.ChunkTokens != 256 {
		t.Errorf("expected ChunkTokens=256, got %d", cfg.Index.ChunkTokens)
	}
	if cfg.Index.Stemming != false {
		t.Errorf("expected Stemming=false, got %v", cfg.Index.Stemming)
	}
	if cfg.Retrieve.TopK != 10 {
		t.Errorf("expected TopK=10, got %d", cfg.Retrieve.TopK)
	}
}

func TestLoadFromDir(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "rag.yaml")

	content := `
pack:
  token_budget: 8000
`
	if err := os.WriteFile(configPath, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	cfg, err := LoadFromDir(tmpDir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if cfg.Pack.TokenBudget != 8000 {
		t.Errorf("expected TokenBudget=8000, got %d", cfg.Pack.TokenBudget)
	}
}

func TestIndexDBPath(t *testing.T) {
	path := IndexDBPath("/home/user/project")
	expected := filepath.Join("/home/user/project", ".rag", "index.db")
	if path != expected {
		t.Errorf("expected %s, got %s", expected, path)
	}
}
