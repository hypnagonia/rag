package embedding

import (
	"fmt"
	"net/http"
	"strings"
	"time"

	"rag/internal/port"
)

const (
	ProviderOpenAI   = "openai"
	ProviderDeepSeek = "deepseek"
	ProviderJina     = "jina"
	ProviderOllama   = "ollama"
	ProviderMock     = "mock"
)

var providerDefaults = map[string]string{
	ProviderOpenAI:   "https://api.openai.com/v1",
	ProviderDeepSeek: "https://api.deepseek.com/v1",
	ProviderJina:     "https://api.jina.ai/v1",
	ProviderOllama:   "http://localhost:11434/v1",
}

var modelDimensions = map[string]int{
	"text-embedding-3-small": 1536,
	"text-embedding-3-large": 3072,
	"text-embedding-ada-002": 1536,
	"jina-embeddings-v3":     1024,
	"jina-embeddings-v4":     2048,
	"nomic-embed-text":       768,
	"mxbai-embed-large":      1024,
	"all-minilm":             384,
}

type Builder struct {
	provider  string
	model     string
	apiKeyEnv string
	baseURL   string
	dimension int
	batchSize int
	timeout   time.Duration
}

func NewBuilder() *Builder {
	return &Builder{
		provider: ProviderOpenAI,
		timeout:  120 * time.Second,
	}
}

func (b *Builder) Provider(provider string) *Builder {
	b.provider = strings.ToLower(strings.TrimSpace(provider))
	return b
}

func (b *Builder) Model(model string) *Builder {
	b.model = model
	return b
}

func (b *Builder) APIKeyEnv(env string) *Builder {
	b.apiKeyEnv = env
	return b
}

func (b *Builder) BaseURL(url string) *Builder {
	b.baseURL = strings.TrimRight(strings.TrimSpace(url), "/")
	return b
}

func (b *Builder) Dimension(dimension int) *Builder {
	b.dimension = dimension
	return b
}

func (b *Builder) BatchSize(size int) *Builder {
	b.batchSize = size
	return b
}

func (b *Builder) Timeout(timeout time.Duration) *Builder {
	b.timeout = timeout
	return b
}

func (b *Builder) Build() (port.Embedder, error) {
	if b.provider == ProviderMock {
		dimension := b.dimension
		if dimension <= 0 {
			dimension = 64
		}
		return NewMockEmbedder(dimension), nil
	}

	defaultURL, known := providerDefaults[b.provider]
	if !known {
		return nil, fmt.Errorf("unsupported embedding provider: %q (supported: openai, deepseek, jina, ollama, mock)", b.provider)
	}

	if b.model == "" {
		return nil, fmt.Errorf("embedding model is required for provider %q", b.provider)
	}

	baseURL := b.baseURL
	if baseURL == "" {
		baseURL = defaultURL
	}

	apiKey := "ollama"
	if b.provider != ProviderOllama {
		var err error
		apiKey, err = envAPIKey(b.apiKeyEnv)
		if err != nil {
			return nil, err
		}
	}

	dimension := b.dimension
	if dimension <= 0 {
		dimension = modelDimensions[b.model]
	}

	return &OpenAIEmbedder{
		apiKey:    apiKey,
		model:     b.model,
		baseURL:   baseURL,
		dimension: dimension,
		maxBatch:  b.batchSize,
		client:    &http.Client{Timeout: b.timeout},
	}, nil
}
