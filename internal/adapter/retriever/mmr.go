package retriever

import (
	"rag/internal/domain"
)

type MMRReranker struct {
	lambda       float64
	dedupJaccard float64
}

func NewMMRReranker(lambda, dedupJaccard float64) *MMRReranker {
	return &MMRReranker{
		lambda:       lambda,
		dedupJaccard: dedupJaccard,
	}
}

func (r *MMRReranker) Rerank(candidates []domain.ScoredChunk, k int) []domain.ScoredChunk {
	if len(candidates) == 0 {
		return nil
	}

	if k > len(candidates) {
		k = len(candidates)
	}

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

			relevance := candidate.Score / maxScore

			maxSim := 0.0
			for _, sel := range selected {
				sim := jaccardSimilarity(candidate.Chunk.Tokens, sel.Chunk.Tokens)
				if sim > maxSim {
					maxSim = sim
				}
			}

			mmr := r.lambda*relevance - (1-r.lambda)*maxSim

			if maxSim > r.dedupJaccard {
				continue
			}

			if mmr > bestMMR {
				bestMMR = mmr
				bestIdx = i
			}
		}

		if bestIdx == -1 {

			break
		}

		selected = append(selected, remaining[bestIdx])

		remaining = append(remaining[:bestIdx], remaining[bestIdx+1:]...)
	}

	return selected
}

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

func JaccardSimilarity(a, b []string) float64 {
	return jaccardSimilarity(a, b)
}
