package embedding

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sync"
	"time"
)

const (
	defaultMaxBatch = 100
	defaultRetries  = 3
)

type OpenAIEmbedder struct {
	apiKey    string
	model     string
	baseURL   string
	dimension int
	maxBatch  int
	client    *http.Client

	probeOnce sync.Once
}

type embeddingRequest struct {
	Input []string `json:"input"`
	Model string   `json:"model"`
}

type embeddingResponse struct {
	Data  []embeddingData `json:"data"`
	Usage embeddingUsage  `json:"usage"`
	Error *apiError       `json:"error,omitempty"`
}

type embeddingData struct {
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

type embeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

type apiError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
}

func (e *OpenAIEmbedder) Embed(texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	maxBatch := e.maxBatch
	if maxBatch <= 0 {
		maxBatch = defaultMaxBatch
	}

	allEmbeddings := make([][]float32, 0, len(texts))

	for i := 0; i < len(texts); i += maxBatch {
		end := i + maxBatch
		if end > len(texts) {
			end = len(texts)
		}
		batch := texts[i:end]

		embeddings, err := e.embedBatchWithRetry(batch)
		if err != nil {
			return nil, err
		}
		allEmbeddings = append(allEmbeddings, embeddings...)
	}

	return allEmbeddings, nil
}

func (e *OpenAIEmbedder) embedBatchWithRetry(texts []string) ([][]float32, error) {
	var lastErr error

	for attempt := 0; attempt < defaultRetries; attempt++ {
		if attempt > 0 {
			time.Sleep(time.Duration(1<<uint(attempt-1)) * time.Second)
		}

		embeddings, retryable, err := e.embedBatch(texts)
		if err == nil {
			return embeddings, nil
		}
		lastErr = err
		if !retryable {
			return nil, err
		}
	}

	return nil, fmt.Errorf("embedding failed after %d attempts: %w", defaultRetries, lastErr)
}

func (e *OpenAIEmbedder) embedBatch(texts []string) ([][]float32, bool, error) {
	reqBody := embeddingRequest{
		Input: texts,
		Model: e.model,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, false, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequest("POST", e.baseURL+"/embeddings", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, false, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.apiKey)

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, true, fmt.Errorf("request to %s failed: %w", e.baseURL, err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, true, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		retryable := resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500
		return nil, retryable, fmt.Errorf("embedding API %s returned status %d: %s", e.baseURL, resp.StatusCode, truncate(string(body), 300))
	}

	var embResp embeddingResponse
	if err := json.Unmarshal(body, &embResp); err != nil {
		return nil, false, fmt.Errorf("failed to parse response (body: %s): %w", truncate(string(body), 200), err)
	}

	if embResp.Error != nil {
		return nil, false, fmt.Errorf("embedding API error: %s", embResp.Error.Message)
	}

	if len(embResp.Data) != len(texts) {
		return nil, false, fmt.Errorf("embedding API returned %d embeddings for %d inputs (model %q)", len(embResp.Data), len(texts), e.model)
	}

	embeddings := make([][]float32, len(texts))
	for _, data := range embResp.Data {
		if data.Index < 0 || data.Index >= len(embeddings) {
			return nil, false, fmt.Errorf("embedding API returned out-of-range index %d for %d inputs", data.Index, len(texts))
		}
		embeddings[data.Index] = data.Embedding
	}

	for i, emb := range embeddings {
		if len(emb) == 0 {
			return nil, false, fmt.Errorf("embedding API returned an empty vector at position %d (model %q)", i, e.model)
		}
		if len(emb) != len(embeddings[0]) {
			return nil, false, fmt.Errorf("embedding API returned inconsistent dimensions: %d vs %d", len(emb), len(embeddings[0]))
		}
	}

	return embeddings, false, nil
}

func (e *OpenAIEmbedder) Dimension() int {
	e.probeOnce.Do(func() {
		probe, _, err := e.embedBatch([]string{"dimension probe"})
		if err != nil || len(probe) == 0 || len(probe[0]) == 0 {
			return
		}
		e.dimension = len(probe[0])
	})
	return e.dimension
}

func (e *OpenAIEmbedder) ModelName() string {
	return e.model
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

type MockEmbedder struct {
	dimension int
}

func NewMockEmbedder(dimension int) *MockEmbedder {
	if dimension <= 0 {
		dimension = 64
	}
	return &MockEmbedder{dimension: dimension}
}

func (e *MockEmbedder) Embed(texts []string) ([][]float32, error) {
	embeddings := make([][]float32, len(texts))
	for i, text := range texts {
		embeddings[i] = hashBagOfWords(text, e.dimension)
	}
	return embeddings, nil
}

func (e *MockEmbedder) Dimension() int {
	return e.dimension
}

func (e *MockEmbedder) ModelName() string {
	return "mock"
}

func hashBagOfWords(text string, dimension int) []float32 {
	vec := make([]float32, dimension)

	word := make([]rune, 0, 32)
	flush := func() {
		if len(word) == 0 {
			return
		}
		var h uint32 = 2166136261
		for _, r := range word {
			if r >= 'A' && r <= 'Z' {
				r += 'a' - 'A'
			}
			h = (h ^ uint32(r)) * 16777619
		}
		vec[h%uint32(dimension)] += 1
		word = word[:0]
	}

	for _, r := range text {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') {
			word = append(word, r)
		} else {
			flush()
		}
	}
	flush()

	if allZero(vec) {
		vec[0] = 1
	}

	return vec
}

func allZero(v []float32) bool {
	for _, x := range v {
		if x != 0 {
			return false
		}
	}
	return true
}

func envAPIKey(apiKeyEnv string) (string, error) {
	key := os.Getenv(apiKeyEnv)
	if key == "" {
		return "", fmt.Errorf("API key not found in environment variable: %s", apiKeyEnv)
	}
	return key, nil
}
