package retriever

import (
	"math"
	"path/filepath"
	"sort"
	"strings"

	"rag/internal/adapter/analyzer"
	"rag/internal/adapter/store"
	"rag/internal/domain"
)

type BM25Retriever struct {
	store           *store.BoltStore
	tokenizer       *analyzer.Tokenizer
	k1              float64
	b               float64
	pathBoostWeight float64
}

func NewBM25Retriever(store *store.BoltStore, tokenizer *analyzer.Tokenizer, k1, b, pathBoostWeight float64) *BM25Retriever {
	return &BM25Retriever{
		store:           store,
		tokenizer:       tokenizer,
		k1:              k1,
		b:               b,
		pathBoostWeight: pathBoostWeight,
	}
}

func (r *BM25Retriever) Search(query string, k int) ([]domain.ScoredChunk, error) {
	queryTokens := r.tokenizer.Tokenize(query)
	if len(queryTokens) == 0 {
		return nil, nil
	}

	stats, err := r.store.GetStats()
	if err != nil {
		return nil, err
	}

	if stats.TotalChunks == 0 {
		return nil, nil
	}

	queryTokenSet := make(map[string]struct{}, len(queryTokens))
	for _, t := range queryTokens {
		queryTokenSet[t] = struct{}{}
	}

	chunkScores := make(map[string]float64)
	chunkLengths := make(map[string]int)
	chunkDocIDs := make(map[string]string)

	for _, term := range queryTokens {
		postings, err := r.store.GetPostings(term)
		if err != nil {
			continue
		}

		n := float64(len(postings))
		N := float64(stats.TotalChunks)
		idf := math.Log((N-n+0.5)/(n+0.5) + 1)

		for _, posting := range postings {

			if _, exists := chunkLengths[posting.ChunkID]; !exists {
				chunk, err := r.store.GetChunk(posting.ChunkID)
				if err != nil {
					continue
				}
				chunkLengths[posting.ChunkID] = len(chunk.Tokens)
				chunkDocIDs[posting.ChunkID] = chunk.DocID
			}

			dl := float64(chunkLengths[posting.ChunkID])
			avgDl := stats.AvgChunkLen
			tf := float64(posting.TF)

			score := idf * (tf * (r.k1 + 1)) / (tf + r.k1*(1-r.b+r.b*dl/avgDl))
			chunkScores[posting.ChunkID] += score
		}
	}

	docPathBoosts := make(map[string]float64)

	results := make([]domain.ScoredChunk, 0, len(chunkScores))
	for chunkID, score := range chunkScores {
		chunk, err := r.store.GetChunk(chunkID)
		if err != nil {
			continue
		}

		finalScore := score
		if r.pathBoostWeight > 0 {
			docID := chunkDocIDs[chunkID]
			pathBoost, exists := docPathBoosts[docID]
			if !exists {
				doc, err := r.store.GetDoc(docID)
				if err == nil {
					pathBoost = r.calculatePathBoost(doc.Path, queryTokenSet)
					docPathBoosts[docID] = pathBoost
				}
			}
			finalScore = score * (1 + pathBoost*r.pathBoostWeight)
		}

		results = append(results, domain.ScoredChunk{
			Chunk: chunk,
			Score: finalScore,
		})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	if len(results) > k {
		results = results[:k]
	}

	return results, nil
}

func (r *BM25Retriever) calculatePathBoost(path string, queryTokenSet map[string]struct{}) float64 {
	pathTokens := tokenizePath(path)
	if len(pathTokens) == 0 || len(queryTokenSet) == 0 {
		return 0
	}

	matches := 0
	for _, pt := range pathTokens {
		if _, exists := queryTokenSet[pt]; exists {
			matches++
		}
	}

	return float64(matches) / float64(len(queryTokenSet))
}

func tokenizePath(path string) []string {

	path = strings.TrimPrefix(path, "/")

	var tokens []string
	parts := strings.Split(path, string(filepath.Separator))
	for _, part := range parts {

		subparts := strings.Split(part, ".")
		for _, sp := range subparts {

			for _, token := range strings.FieldsFunc(sp, func(r rune) bool {
				return r == '_' || r == '-'
			}) {
				token = strings.ToLower(token)
				if len(token) >= 2 {
					tokens = append(tokens, token)
				}
			}
		}
	}
	return tokens
}

func ComputeBM25Score(queryTokens []string, chunk domain.Chunk, stats domain.Stats, k1, b float64) float64 {

	chunkTF := make(map[string]int)
	for _, token := range chunk.Tokens {
		chunkTF[token]++
	}

	score := 0.0
	dl := float64(len(chunk.Tokens))
	avgDl := stats.AvgChunkLen
	N := float64(stats.TotalChunks)

	for _, term := range queryTokens {
		tf, exists := chunkTF[term]
		if !exists {
			continue
		}

		n := 1.0
		idf := math.Log((N-n+0.5)/(n+0.5) + 1)

		tfFloat := float64(tf)
		termScore := idf * (tfFloat * (k1 + 1)) / (tfFloat + k1*(1-b+b*dl/avgDl))
		score += termScore
	}

	return score
}
