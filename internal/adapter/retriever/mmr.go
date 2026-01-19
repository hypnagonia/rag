package retriever

import (
	"rag/internal/domain"
)

// MMRReranker implements Maximal Marginal Relevance for result diversification.
type MMRReranker struct {
	lambda        float64
	dedupJaccard  float64
}

// NewMMRReranker creates a new MMR reranker.
func NewMMRReranker(lambda, dedupJaccard float64) *MMRReranker {
	return &MMRReranker{
		lambda:       lambda,
		dedupJaccard: dedupJaccard,
	}
}

// Rerank applies MMR to diversify the results.
// MMR(c) = λ * relevance(c) - (1-λ) * max_similarity(c, selected)
func (r *MMRReranker) Rerank(candidates []domain.ScoredChunk, k int) []domain.ScoredChunk {
	if len(candidates) == 0 {
		return nil
	}

	if k > len(candidates) {
		k = len(candidates)
	}

	// Normalize scores to [0, 1] for fair comparison
	maxScore := candidates[0].Score
	for _, c := range candidates {
		if c.Score > maxScore {
			maxScore = c.Score
		}
	}
	if maxScore == 0 {
		maxScore = 1
	}

	selected := make([]domain.ScoredChunk, 0, k)
	remaining := make([]domain.ScoredChunk, len(candidates))
	copy(remaining, candidates)

	for len(selected) < k && len(remaining) > 0 {
		bestIdx := -1
		bestMMR := -1e9

		for i, candidate := range remaining {
			// Normalized relevance score
			relevance := candidate.Score / maxScore

			// Maximum similarity to already selected items
			maxSim := 0.0
			for _, sel := range selected {
				sim := jaccardSimilarity(candidate.Chunk.Tokens, sel.Chunk.Tokens)
				if sim > maxSim {
					maxSim = sim
				}
			}

			// MMR score
			mmr := r.lambda*relevance - (1-r.lambda)*maxSim

			// Also apply deduplication threshold
			if maxSim > r.dedupJaccard {
				continue // Skip if too similar to an already selected item
			}

			if mmr > bestMMR {
				bestMMR = mmr
				bestIdx = i
			}
		}

		if bestIdx == -1 {
			// All remaining candidates are too similar, stop
			break
		}

		// Add best candidate to selected
		selected = append(selected, remaining[bestIdx])

		// Remove from remaining
		remaining = append(remaining[:bestIdx], remaining[bestIdx+1:]...)
	}

	return selected
}

// jaccardSimilarity computes the Jaccard similarity between two token sets.
func jaccardSimilarity(a, b []string) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 1.0
	}
	if len(a) == 0 || len(b) == 0 {
		return 0.0
	}

	setA := make(map[string]struct{}, len(a))
	for _, t := range a {
		setA[t] = struct{}{}
	}

	setB := make(map[string]struct{}, len(b))
	for _, t := range b {
		setB[t] = struct{}{}
	}

	intersection := 0
	for t := range setA {
		if _, exists := setB[t]; exists {
			intersection++
		}
	}

	union := len(setA) + len(setB) - intersection
	if union == 0 {
		return 0.0
	}

	return float64(intersection) / float64(union)
}

// JaccardSimilarity is exported for testing.
func JaccardSimilarity(a, b []string) float64 {
	return jaccardSimilarity(a, b)
}
