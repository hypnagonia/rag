package usecase

import (
	"fmt"
	"sort"
	"time"

	"rag/internal/adapter/analyzer"
	"rag/internal/adapter/store"
	"rag/internal/domain"
)

// PackUseCase handles context packing operations.
type PackUseCase struct {
	store        *store.BoltStore
	tokenizer    *analyzer.Tokenizer
	recencyBoost float64
}

// NewPackUseCase creates a new pack use case.
func NewPackUseCase(store *store.BoltStore, tokenizer *analyzer.Tokenizer, recencyBoost float64) *PackUseCase {
	return &PackUseCase{
		store:        store,
		tokenizer:    tokenizer,
		recencyBoost: recencyBoost,
	}
}

// Pack packs scored chunks into a context that fits the token budget.
func (u *PackUseCase) Pack(query string, chunks []domain.ScoredChunk, budget int) (domain.PackedContext, error) {
	if len(chunks) == 0 {
		return domain.PackedContext{
			Query:        query,
			BudgetTokens: budget,
			UsedTokens:   0,
			Snippets:     []domain.Snippet{},
		}, nil
	}

	// Calculate utility for each chunk: score * recency_factor / tokens
	type rankedChunk struct {
		chunk   domain.ScoredChunk
		utility float64
		tokens  int
	}

	// Find the most recent modification time for recency calculation
	var maxModTime time.Time
	docModTimes := make(map[string]time.Time)
	if u.recencyBoost > 0 {
		for _, c := range chunks {
			if _, exists := docModTimes[c.Chunk.DocID]; !exists {
				if doc, err := u.store.GetDoc(c.Chunk.DocID); err == nil {
					docModTimes[c.Chunk.DocID] = doc.ModTime
					if doc.ModTime.After(maxModTime) {
						maxModTime = doc.ModTime
					}
				}
			}
		}
	}

	// Calculate max age for normalization (use 30 days as reference window)
	maxAgeDays := 30.0

	ranked := make([]rankedChunk, 0, len(chunks))
	for _, c := range chunks {
		tokens := u.tokenizer.CountTokens(c.Chunk.Text)
		if tokens == 0 {
			tokens = 1
		}

		// Calculate recency factor (1.0 for most recent, decaying for older files)
		recencyFactor := 1.0
		if u.recencyBoost > 0 && !maxModTime.IsZero() {
			if modTime, exists := docModTimes[c.Chunk.DocID]; exists {
				ageDays := maxModTime.Sub(modTime).Hours() / 24.0
				if ageDays < 0 {
					ageDays = 0
				}
				// Normalize age to [0, 1] range (capped at maxAgeDays)
				normalizedAge := ageDays / maxAgeDays
				if normalizedAge > 1 {
					normalizedAge = 1
				}
				// Recency factor: 1 for newest, (1 - recencyBoost) for oldest
				recencyFactor = 1.0 + u.recencyBoost*(1.0-normalizedAge) - u.recencyBoost*normalizedAge
			}
		}

		// Utility = relevance score * recency factor / token cost
		utility := (c.Score * recencyFactor) / float64(tokens)
		ranked = append(ranked, rankedChunk{
			chunk:   c,
			utility: utility,
			tokens:  tokens,
		})
	}

	// Sort by utility (best value first)
	sort.Slice(ranked, func(i, j int) bool {
		return ranked[i].utility > ranked[j].utility
	})

	// Greedy selection until budget is exhausted
	selected := make([]domain.ScoredChunk, 0)
	usedTokens := 0

	for _, rc := range ranked {
		if usedTokens+rc.tokens > budget {
			continue // Skip if it would exceed budget
		}
		selected = append(selected, rc.chunk)
		usedTokens += rc.tokens
	}

	// Try to merge adjacent chunks from same file
	merged := u.mergeAdjacentChunks(selected)

	// Build snippets
	snippets := make([]domain.Snippet, 0, len(merged))
	for _, sc := range merged {
		doc, err := u.store.GetDoc(sc.Chunk.DocID)
		if err != nil {
			continue
		}
		snippet := domain.Snippet{
			Path:  doc.Path,
			Range: fmt.Sprintf("L%d-%d", sc.Chunk.StartLine, sc.Chunk.EndLine),
			Why:   fmt.Sprintf("BM25 score: %.2f", sc.Score),
			Text:  sc.Chunk.Text,
		}
		snippets = append(snippets, snippet)
	}

	// Recalculate used tokens after merging
	usedTokens = 0
	for _, s := range snippets {
		usedTokens += u.tokenizer.CountTokens(s.Text)
	}

	return domain.PackedContext{
		Query:        query,
		BudgetTokens: budget,
		UsedTokens:   usedTokens,
		Snippets:     snippets,
	}, nil
}

// mergeAdjacentChunks merges adjacent chunks from the same document.
func (u *PackUseCase) mergeAdjacentChunks(chunks []domain.ScoredChunk) []domain.ScoredChunk {
	if len(chunks) <= 1 {
		return chunks
	}

	// Group by document
	byDoc := make(map[string][]domain.ScoredChunk)
	for _, c := range chunks {
		byDoc[c.Chunk.DocID] = append(byDoc[c.Chunk.DocID], c)
	}

	result := make([]domain.ScoredChunk, 0, len(chunks))

	for _, docChunks := range byDoc {
		// Sort by start line
		sort.Slice(docChunks, func(i, j int) bool {
			return docChunks[i].Chunk.StartLine < docChunks[j].Chunk.StartLine
		})

		// Merge adjacent chunks
		i := 0
		for i < len(docChunks) {
			merged := docChunks[i]
			j := i + 1

			// Try to merge with following chunks
			for j < len(docChunks) {
				next := docChunks[j]
				// Check if adjacent (or overlapping)
				if next.Chunk.StartLine <= merged.Chunk.EndLine+1 {
					// Merge
					merged.Chunk.EndLine = maxInt(merged.Chunk.EndLine, next.Chunk.EndLine)
					merged.Chunk.Text = merged.Chunk.Text + "\n" + next.Chunk.Text
					merged.Chunk.Tokens = append(merged.Chunk.Tokens, next.Chunk.Tokens...)
					merged.Score = maxFloat(merged.Score, next.Score)
					j++
				} else {
					break
				}
			}

			result = append(result, merged)
			i = j
		}
	}

	return result
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func maxFloat(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
