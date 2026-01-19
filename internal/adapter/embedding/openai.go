package embedding

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

type OpenAIEmbedder struct {
	apiKey    string
	model     string
	baseURL   string
	dimension int
	client    *http.Client
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

func NewOpenAIEmbedder(apiKeyEnv, model string) (*OpenAIEmbedder, error) {
	return NewOpenAICompatibleEmbedder(apiKeyEnv, model, "https://api.openai.com/v1")
}

func NewDeepSeekEmbedder(apiKeyEnv, model string) (*OpenAIEmbedder, error) {
	return NewOpenAICompatibleEmbedder(apiKeyEnv, model, "https://api.deepseek.com/v1")
}

func NewJinaEmbedder(apiKeyEnv, model string) (*OpenAIEmbedder, error) {
	return NewOpenAICompatibleEmbedder(apiKeyEnv, model, "https://api.jina.ai/v1")
}

func NewOllamaEmbedder(model, baseURL string) (*OpenAIEmbedder, error) {
	if baseURL == "" {
		baseURL = "http://localhost:11434/v1"
	}

	dimension := 768
	switch model {
	case "nomic-embed-text":
		dimension = 768
	case "mxbai-embed-large":
		dimension = 1024
	case "all-minilm":
		dimension = 384
	}

	return &OpenAIEmbedder{
		apiKey:    "ollama",
		model:     model,
		baseURL:   baseURL,
		dimension: dimension,
		client: &http.Client{
			Timeout: 120 * time.Second,
		},
	}, nil
}

func NewOpenAICompatibleEmbedder(apiKeyEnv, model, baseURL string) (*OpenAIEmbedder, error) {
	apiKey := os.Getenv(apiKeyEnv)
	if apiKey == "" {
		return nil, fmt.Errorf("API key not found in environment variable: %s", apiKeyEnv)
	}

	dimension := 1536
	switch model {
	case "text-embedding-3-small":
		dimension = 1536
	case "text-embedding-3-large":
		dimension = 3072
	case "text-embedding-ada-002":
		dimension = 1536

	case "jina-embeddings-v3":
		dimension = 1024
	case "jina-embeddings-v4":
		dimension = 2048
	}

	return &OpenAIEmbedder{
		apiKey:    apiKey,
		model:     model,
		baseURL:   baseURL,
		dimension: dimension,
		client: &http.Client{
			Timeout: 60 * time.Second,
		},
	}, nil
}

func (e *OpenAIEmbedder) Embed(texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	const maxBatch = 100
	var allEmbeddings [][]float32

	for i := 0; i < len(texts); i += maxBatch {
		end := i + maxBatch
		if end > len(texts) {
			end = len(texts)
		}
		batch := texts[i:end]

		embeddings, err := e.embedBatch(batch)
		if err != nil {
			return nil, err
		}
		allEmbeddings = append(allEmbeddings, embeddings...)
	}

	return allEmbeddings, nil
}

func (e *OpenAIEmbedder) embedBatch(texts []string) ([][]float32, error) {
	reqBody := embeddingRequest{
		Input: texts,
		Model: e.model,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequest("POST", e.baseURL+"/embeddings", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.apiKey)

	resp, err := e.client.Do(req)
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

	var embResp embeddingResponse
	if err := json.Unmarshal(body, &embResp); err != nil {
		bodyPreview := string(body)
		if len(bodyPreview) > 200 {
			bodyPreview = bodyPreview[:200]
		}
		return nil, fmt.Errorf("failed to parse response (body: %s): %w", bodyPreview, err)
	}

	if embResp.Error != nil {
		return nil, fmt.Errorf("API error: %s", embResp.Error.Message)
	}

	embeddings := make([][]float32, len(texts))
	for _, data := range embResp.Data {
		if data.Index < len(embeddings) {
			embeddings[data.Index] = data.Embedding
		}
	}

	return embeddings, nil
}

func (e *OpenAIEmbedder) Dimension() int {
	return e.dimension
}

func (e *OpenAIEmbedder) ModelName() string {
	return e.model
}

type MockEmbedder struct {
	dimension int
}

func NewMockEmbedder(dimension int) *MockEmbedder {
	return &MockEmbedder{dimension: dimension}
}

func (e *MockEmbedder) Embed(texts []string) ([][]float32, error) {
	embeddings := make([][]float32, len(texts))
	for i := range texts {
		embeddings[i] = make([]float32, e.dimension)

		for j, r := range texts[i] {
			if j < e.dimension {
				embeddings[i][j] = float32(r) / 1000.0
			}
		}
	}
	return embeddings, nil
}

func (e *MockEmbedder) Dimension() int {
	return e.dimension
}

func (e *MockEmbedder) ModelName() string {
	return "mock"
}
