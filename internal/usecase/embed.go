package usecase

import (
	"fmt"
	"strings"

	"rag/internal/port"
)

const (
	defaultEmbedBatchSize = 100
	maxEmbedTextRunes     = 8000
)

type EmbedUseCase struct {
	store       port.IndexStore
	embedder    port.Embedder
	vectorStore port.VectorStore
	batchSize   int
	force       bool
	includePath bool
}

type EmbedBuilder struct {
	store       port.IndexStore
	embedder    port.Embedder
	vectorStore port.VectorStore
	batchSize   int
	force       bool
	includePath bool
}

func NewEmbedBuilder() *EmbedBuilder {
	return &EmbedBuilder{batchSize: defaultEmbedBatchSize, includePath: true}
}

func (b *EmbedBuilder) Store(s port.IndexStore) *EmbedBuilder {
	b.store = s
	return b
}

func (b *EmbedBuilder) Embedder(e port.Embedder) *EmbedBuilder {
	b.embedder = e
	return b
}

func (b *EmbedBuilder) VectorStore(vs port.VectorStore) *EmbedBuilder {
	b.vectorStore = vs
	return b
}

func (b *EmbedBuilder) BatchSize(size int) *EmbedBuilder {
	if size > 0 {
		b.batchSize = size
	}
	return b
}

func (b *EmbedBuilder) Force(force bool) *EmbedBuilder {
	b.force = force
	return b
}

func (b *EmbedBuilder) IncludePath(include bool) *EmbedBuilder {
	b.includePath = include
	return b
}

func (b *EmbedBuilder) Build() (*EmbedUseCase, error) {
	if b.store == nil {
		return nil, fmt.Errorf("embed use case requires an index store")
	}
	if b.embedder == nil {
		return nil, fmt.Errorf("embed use case requires an embedder")
	}
	if b.vectorStore == nil {
		return nil, fmt.Errorf("embed use case requires a vector store")
	}

	return &EmbedUseCase{
		store:       b.store,
		embedder:    b.embedder,
		vectorStore: b.vectorStore,
		batchSize:   b.batchSize,
		force:       b.force,
		includePath: b.includePath,
	}, nil
}

type EmbedResult struct {
	TotalChunks int
	Embedded    int
	Reused      int
	Deleted     int
}

type EmbedProgressCallback func(embedded, total int)

type pendingChunk struct {
	id   string
	text string
}

func (u *EmbedUseCase) Sync(progress EmbedProgressCallback) (*EmbedResult, error) {
	pending, currentIDs, err := u.collectChunks()
	if err != nil {
		return nil, err
	}

	result := &EmbedResult{TotalChunks: len(currentIDs)}

	existingIDs, err := u.vectorStore.IDs()
	if err != nil {
		return nil, fmt.Errorf("failed to list existing vectors: %w", err)
	}

	var orphans []string
	existing := make(map[string]struct{}, len(existingIDs))
	for _, id := range existingIDs {
		existing[id] = struct{}{}
		if _, ok := currentIDs[id]; !ok {
			orphans = append(orphans, id)
		}
	}

	if len(orphans) > 0 {
		if err := u.vectorStore.Delete(orphans); err != nil {
			return nil, fmt.Errorf("failed to delete stale vectors: %w", err)
		}
		result.Deleted = len(orphans)
	}

	todo := pending
	if !u.force {
		todo = make([]pendingChunk, 0, len(pending))
		for _, chunk := range pending {
			if _, ok := existing[chunk.id]; ok {
				result.Reused++
				continue
			}
			todo = append(todo, chunk)
		}
	}

	if len(todo) == 0 {
		return result, nil
	}

	if err := u.embedAll(todo, result, progress); err != nil {
		return result, err
	}

	return result, nil
}

func (u *EmbedUseCase) collectChunks() ([]pendingChunk, map[string]struct{}, error) {
	docs, err := u.store.ListDocs()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to list documents: %w", err)
	}

	var pending []pendingChunk
	currentIDs := make(map[string]struct{})

	for _, doc := range docs {
		chunks, err := u.store.GetChunksByDoc(doc.ID)
		if err != nil {
			continue
		}
		for _, chunk := range chunks {
			if strings.TrimSpace(chunk.Text) == "" {
				continue
			}
			currentIDs[chunk.ID] = struct{}{}
			pending = append(pending, pendingChunk{
				id:   chunk.ID,
				text: u.embeddingText(doc.Path, chunk.StartLine, chunk.EndLine, chunk.Text),
			})
		}
	}

	return pending, currentIDs, nil
}

func (u *EmbedUseCase) embedAll(todo []pendingChunk, result *EmbedResult, progress EmbedProgressCallback) error {
	for i := 0; i < len(todo); i += u.batchSize {
		end := i + u.batchSize
		if end > len(todo) {
			end = len(todo)
		}
		batch := todo[i:end]

		texts := make([]string, len(batch))
		for j, c := range batch {
			texts[j] = c.text
		}

		embeddings, err := u.embedder.Embed(texts)
		if err != nil {
			return fmt.Errorf("embedding batch failed: %w", err)
		}
		if len(embeddings) != len(batch) {
			return fmt.Errorf("embedder returned %d vectors for %d chunks", len(embeddings), len(batch))
		}

		items := make([]port.VectorItem, len(batch))
		for j, c := range batch {
			items[j] = port.VectorItem{ID: c.id, Vector: embeddings[j]}
		}

		if err := u.vectorStore.Upsert(items); err != nil {
			return fmt.Errorf("failed to store vectors: %w", err)
		}

		result.Embedded += len(batch)
		if progress != nil {
			progress(result.Embedded, len(todo))
		}
	}

	return nil
}

func (u *EmbedUseCase) embeddingText(path string, startLine, endLine int, text string) string {
	body := truncateRunes(text, maxEmbedTextRunes)
	if !u.includePath {
		return body
	}

	var b strings.Builder
	b.WriteString(fmt.Sprintf("%s:%d-%d\n", path, startLine, endLine))
	b.WriteString(body)
	return b.String()
}

func truncateRunes(s string, limit int) string {
	if len(s) <= limit {
		return s
	}
	runes := []rune(s)
	if len(runes) <= limit {
		return s
	}
	return string(runes[:limit])
}
