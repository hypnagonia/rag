package retriever

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"time"

	"rag/internal/domain"
	"rag/internal/port"
)

// CohereReranker implements cross-encoder reranking using Cohere's API.
type CohereReranker struct {
	apiKey string
	model  string
	client *http.Client
}

// Cohere API types
type cohereRerankRequest struct {
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
	Model     string   `json:"model"`
	TopN      int      `json:"top_n,omitempty"`
}

type cohereRerankResponse struct {
	Results []cohereRerankResult `json:"results"`
}

type cohereRerankResult struct {
	Index          int     `json:"index"`
	RelevanceScore float64 `json:"relevance_score"`
}

// NewCohereReranker creates a new Cohere reranker.
func NewCohereReranker(apiKeyEnv, model string) (*CohereReranker, error) {
	apiKey := os.Getenv(apiKeyEnv)
	if apiKey == "" {
		return nil, fmt.Errorf("API key not found in environment variable: %s", apiKeyEnv)
	}

	if model == "" {
		model = "rerank-english-v3.0"
	}

	return &CohereReranker{
		apiKey: apiKey,
		model:  model,
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}, nil
}

// Rerank scores and reorders documents based on query relevance.
func (r *CohereReranker) Rerank(query string, documents []string) ([]port.RerankedResult, error) {
	if len(documents) == 0 {
		return nil, nil
	}

	// Cohere has a limit of 1000 documents per request
	const maxDocs = 1000
	if len(documents) > maxDocs {
		documents = documents[:maxDocs]
	}

	reqBody := cohereRerankRequest{
		Query:     query,
		Documents: documents,
		Model:     r.model,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequest("POST", "https://api.cohere.ai/v1/rerank", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+r.apiKey)

	resp, err := r.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API returned status %d: %s", resp.StatusCode, string(body))
	}

	var rerankResp cohereRerankResponse
	if err := json.Unmarshal(body, &rerankResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	results := make([]port.RerankedResult, len(rerankResp.Results))
	for i, res := range rerankResp.Results {
		results[i] = port.RerankedResult{
			Index: res.Index,
			Score: res.RelevanceScore,
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results, nil
}

// ModelName returns the model name.
func (r *CohereReranker) ModelName() string {
	return r.model
}

// RerankedRetriever wraps a retriever and applies reranking to results.
type RerankedRetriever struct {
	retriever port.Retriever
	reranker  port.Reranker
	topK      int // How many to rerank (should be > final k)
}

// NewRerankedRetriever creates a new reranked retriever.
func NewRerankedRetriever(retriever port.Retriever, reranker port.Reranker, topK int) *RerankedRetriever {
	if topK <= 0 {
		topK = 50
	}
	return &RerankedRetriever{
		retriever: retriever,
		reranker:  reranker,
		topK:      topK,
	}
}

// Search retrieves candidates and reranks them.
func (r *RerankedRetriever) Search(query string, k int) ([]domain.ScoredChunk, error) {

	candidates, err := r.retriever.Search(query, r.topK)
	if err != nil {
		return nil, err
	}

	if len(candidates) == 0 {
		return nil, nil
	}

	if r.reranker == nil {
		if len(candidates) > k {
			candidates = candidates[:k]
		}
		return candidates, nil
	}

	texts := make([]string, len(candidates))
	for i, c := range candidates {
		texts[i] = c.Chunk.Text
	}

	reranked, err := r.reranker.Rerank(query, texts)
	if err != nil {

		if len(candidates) > k {
			candidates = candidates[:k]
		}
		return candidates, nil
	}

	results := make([]domain.ScoredChunk, 0, min(k, len(reranked)))
	for i := 0; i < min(k, len(reranked)); i++ {
		idx := reranked[i].Index
		if idx < len(candidates) {
			results = append(results, domain.ScoredChunk{
				Chunk: candidates[idx].Chunk,
				Score: reranked[i].Score,
			})
		}
	}

	return results, nil
}

// SimpleReranker provides a simple TF-based reranking when no external reranker is available.
type SimpleReranker struct{}

// NewSimpleReranker creates a new simple reranker.
func NewSimpleReranker() *SimpleReranker {
	return &SimpleReranker{}
}

// Rerank performs simple term frequency-based reranking.
func (r *SimpleReranker) Rerank(query string, documents []string) ([]port.RerankedResult, error) {
	queryTerms := tokenizeSimple(query)
	if len(queryTerms) == 0 {

		results := make([]port.RerankedResult, len(documents))
		for i := range documents {
			results[i] = port.RerankedResult{Index: i, Score: 1.0 - float64(i)*0.01}
		}
		return results, nil
	}

	results := make([]port.RerankedResult, len(documents))
	for i, doc := range documents {
		score := calculateTermOverlap(queryTerms, doc)
		results[i] = port.RerankedResult{
			Index: i,
			Score: score,
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results, nil
}

// ModelName returns the model name.
func (r *SimpleReranker) ModelName() string {
	return "simple-tf"
}

// tokenizeSimple performs basic tokenization.
func tokenizeSimple(text string) map[string]int {
	terms := make(map[string]int)
	word := ""
	for _, r := range text {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' {
			word += string(r)
		} else {
			if len(word) >= 2 {
				terms[word]++
			}
			word = ""
		}
	}
	if len(word) >= 2 {
		terms[word]++
	}
	return terms
}

// calculateTermOverlap calculates overlap between query terms and document.
func calculateTermOverlap(queryTerms map[string]int, doc string) float64 {
	docTerms := tokenizeSimple(doc)
	if len(docTerms) == 0 {
		return 0
	}

	matches := 0
	for term := range queryTerms {
		if _, exists := docTerms[term]; exists {
			matches++
		}
	}

	return float64(matches) / float64(len(queryTerms))
}
