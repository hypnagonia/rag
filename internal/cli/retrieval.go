package cli

import (
	"fmt"
	"strings"

	"rag/config"
	"rag/internal/adapter/embedding"
	"rag/internal/adapter/retriever"
	"rag/internal/adapter/store"
	"rag/internal/port"
)

func buildEmbedder(cfg *config.Config) (port.Embedder, error) {
	embedder, err := embedding.NewBuilder().
		Provider(cfg.Embedding.Provider).
		Model(cfg.Embedding.Model).
		APIKeyEnv(cfg.Embedding.APIKeyEnv).
		BaseURL(cfg.Embedding.BaseURL).
		Dimension(cfg.Embedding.Dimension).
		BatchSize(cfg.Embedding.BatchSize).
		Build()
	if err != nil {
		return nil, fmt.Errorf("failed to create embedder: %w", err)
	}

	if embedder.Dimension() <= 0 {
		return nil, fmt.Errorf("could not determine embedding dimension for model %q - is the %s provider reachable?", cfg.Embedding.Model, cfg.Embedding.Provider)
	}

	return embedder, nil
}

func openVectorStore(st *store.BoltStore, embedder port.Embedder) (*store.BoltVectorStore, error) {
	vectorStore, err := store.NewBoltVectorStore(st.DB(), embedder.Dimension())
	if err != nil {
		return nil, fmt.Errorf("failed to open vector store: %w", err)
	}
	return vectorStore, nil
}

func currentVectorMeta(embedder port.Embedder) store.VectorMeta {
	return store.VectorMeta{
		Model:     embedder.ModelName(),
		Dimension: embedder.Dimension(),
	}
}

func checkVectorMeta(vectorStore *store.BoltVectorStore, embedder port.Embedder) error {
	meta, err := vectorStore.Meta()
	if err != nil {
		return err
	}
	if meta == nil {
		return nil
	}

	want := currentVectorMeta(embedder)
	if meta.Model != want.Model || meta.Dimension != want.Dimension {
		return fmt.Errorf("index was embedded with %s (%d dims) but config asks for %s (%d dims) - re-run 'rag index'",
			meta.Model, meta.Dimension, want.Model, want.Dimension)
	}

	return nil
}

func setupVectorRetrieval(st *store.BoltStore, cfg *config.Config) (port.Embedder, *store.BoltVectorStore, error) {
	if !cfg.Embedding.Enabled {
		return nil, nil, fmt.Errorf("embeddings are disabled - set embedding.enabled: true in rag.yaml")
	}

	embedder, err := buildEmbedder(cfg)
	if err != nil {
		return nil, nil, err
	}

	vectorStore, err := openVectorStore(st, embedder)
	if err != nil {
		return nil, nil, err
	}

	if err := checkVectorMeta(vectorStore, embedder); err != nil {
		return nil, nil, err
	}

	count, err := vectorStore.Count()
	if err != nil {
		return nil, nil, err
	}
	if count == 0 {
		return nil, nil, fmt.Errorf("no embeddings found - run 'rag index' with embedding.enabled=true")
	}

	return embedder, vectorStore, nil
}

type RetrievalMode int

const (
	ModeAuto RetrievalMode = iota
	ModeLexical
	ModeSemantic
)

type RetrievalPlan struct {
	Mode    string
	Model   string
	Vectors int
	Warning string

	hybrid *retriever.HybridRetriever
}

func (p RetrievalPlan) Describe() string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("retrieval: %s", p.Mode))
	if p.Model != "" {
		b.WriteString(fmt.Sprintf(" (model=%s, vectors=%d)", p.Model, p.Vectors))
	}
	if p.Warning != "" {
		b.WriteString(fmt.Sprintf("\nwarning: %s", p.Warning))
	}
	return b.String()
}

func (p RetrievalPlan) DescribeStats() string {
	if p.hybrid == nil {
		return fmt.Sprintf("candidates: single-arm (%s)", p.Mode)
	}

	stats := p.hybrid.Stats()
	var b strings.Builder
	b.WriteString(fmt.Sprintf("candidates: bm25=%d vector=%d fused=%d", stats.BM25Candidates, stats.VectorCandidates, stats.Fused))
	if stats.BM25Error != nil {
		b.WriteString(fmt.Sprintf("\nbm25 arm failed: %v", stats.BM25Error))
	}
	if stats.VectorError != nil {
		b.WriteString(fmt.Sprintf("\nvector arm failed: %v", stats.VectorError))
	}
	return b.String()
}

func buildRetriever(st *store.BoltStore, cfg *config.Config, tokenizer port.Tokenizer, mode RetrievalMode) (port.Retriever, RetrievalPlan, error) {
	bm25 := retriever.NewBM25Retriever(st, tokenizer, cfg.Index.K1, cfg.Index.B, cfg.Retrieve.PathBoostWeight)

	if mode == ModeLexical {
		return bm25, RetrievalPlan{Mode: "bm25"}, nil
	}

	wantVectors := mode == ModeSemantic || (cfg.Retrieve.HybridEnabled && cfg.Embedding.Enabled)
	if !wantVectors {
		return bm25, RetrievalPlan{Mode: "bm25"}, nil
	}

	embedder, vectorStore, err := setupVectorRetrieval(st, cfg)
	if err != nil {
		if mode == ModeSemantic {
			return nil, RetrievalPlan{}, fmt.Errorf("semantic search unavailable: %w", err)
		}
		return bm25, RetrievalPlan{Mode: "bm25", Warning: fmt.Sprintf("hybrid retrieval unavailable, falling back to BM25: %v", err)}, nil
	}

	count, _ := vectorStore.Count()

	if mode == ModeSemantic {
		plan := RetrievalPlan{Mode: "semantic", Model: embedder.ModelName(), Vectors: count}
		return retriever.NewSemanticRetriever(vectorStore, embedder, st), plan, nil
	}

	hybrid := retriever.NewHybridBuilder().
		BM25(bm25).
		VectorStore(vectorStore).
		Embedder(embedder).
		ChunkStore(st).
		RRFK(cfg.Retrieve.RRFK).
		BM25Weight(cfg.Retrieve.BM25Weight).
		Build()

	plan := RetrievalPlan{
		Mode:    "hybrid (bm25 + vector, RRF)",
		Model:   embedder.ModelName(),
		Vectors: count,
		hybrid:  hybrid,
	}

	return hybrid, plan, nil
}
