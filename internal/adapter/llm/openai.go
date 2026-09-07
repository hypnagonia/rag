package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"rag/internal/port"
)

const (
	ProviderOpenAI   = "openai"
	ProviderDeepSeek = "deepseek"
)

var providerDefaults = map[string]string{
	ProviderOpenAI:   "https://api.openai.com/v1",
	ProviderDeepSeek: "https://api.deepseek.com/v1",
}

type Client struct {
	apiKey      string
	model       string
	baseURL     string
	maxTokens   int
	temperature float64
	client      *http.Client

	mu    sync.Mutex
	stats Stats
}

type Stats struct {
	Calls         int
	InputTokens   int
	OutputTokens  int
	ReportedByAPI bool
	InputChars    int
	OutputChars   int
}

func (c *Client) Stats() Stats {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.stats
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Temperature float64       `json:"temperature"`
	Stream      bool          `json:"stream"`
}

type chatResponse struct {
	Choices []struct {
		Message chatMessage `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

type Builder struct {
	provider    string
	model       string
	apiKeyEnv   string
	baseURL     string
	maxTokens   int
	temperature float64
	timeout     time.Duration
}

func NewBuilder() *Builder {
	return &Builder{
		provider:    ProviderOpenAI,
		maxTokens:   400,
		temperature: 0.3,
		timeout:     120 * time.Second,
	}
}

func (b *Builder) Provider(p string) *Builder {
	b.provider = strings.ToLower(strings.TrimSpace(p))
	return b
}

func (b *Builder) Model(m string) *Builder {
	b.model = m
	return b
}

func (b *Builder) APIKeyEnv(env string) *Builder {
	b.apiKeyEnv = env
	return b
}

func (b *Builder) BaseURL(u string) *Builder {
	b.baseURL = strings.TrimRight(strings.TrimSpace(u), "/")
	return b
}

func (b *Builder) MaxTokens(n int) *Builder {
	if n > 0 {
		b.maxTokens = n
	}
	return b
}

func (b *Builder) Temperature(t float64) *Builder {
	b.temperature = t
	return b
}

func (b *Builder) Timeout(d time.Duration) *Builder {
	b.timeout = d
	return b
}

func (b *Builder) Build() (port.LLM, error) {
	defaultURL, known := providerDefaults[b.provider]
	if !known {
		return nil, fmt.Errorf("unsupported LLM provider: %q (supported: openai, deepseek)", b.provider)
	}
	if b.model == "" {
		return nil, fmt.Errorf("LLM model is required for provider %q", b.provider)
	}

	baseURL := b.baseURL
	if baseURL == "" {
		baseURL = defaultURL
	}

	var apiKey string
	if b.apiKeyEnv != "" {
		apiKey = os.Getenv(b.apiKeyEnv)
		if apiKey == "" {
			return nil, fmt.Errorf("API key not found in environment variable: %s", b.apiKeyEnv)
		}
	} else if baseURL == defaultURL {
		return nil, fmt.Errorf("provider %q has no api_key_env: a keyless endpoint needs an explicit base_url", b.provider)
	}

	return &Client{
		apiKey:      apiKey,
		model:       b.model,
		baseURL:     baseURL,
		maxTokens:   b.maxTokens,
		temperature: b.temperature,
		client:      &http.Client{Timeout: b.timeout},
	}, nil
}

func (c *Client) Generate(prompt string) (string, error) {
	return c.chat([]chatMessage{{Role: "user", Content: prompt}})
}

func (c *Client) GenerateWithSystem(systemPrompt, userPrompt string) (string, error) {
	return c.chat([]chatMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	})
}

func (c *Client) ModelName() string {
	return c.model
}

func (c *Client) chat(messages []chatMessage) (string, error) {
	body, err := json.Marshal(chatRequest{
		Model:       c.model,
		Messages:    messages,
		MaxTokens:   c.maxTokens,
		Temperature: c.temperature,
	})
	if err != nil {
		return "", err
	}

	req, err := http.NewRequest("POST", c.baseURL+"/chat/completions", bytes.NewBuffer(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("request to %s failed: %w", c.baseURL, err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("LLM API %s returned status %d: %s", c.baseURL, resp.StatusCode, truncate(string(raw), 300))
	}

	var parsed chatResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return "", fmt.Errorf("failed to parse LLM response (body: %s): %w", truncate(string(raw), 200), err)
	}
	if parsed.Error != nil {
		return "", fmt.Errorf("LLM API error: %s", parsed.Error.Message)
	}
	if len(parsed.Choices) == 0 {
		return "", fmt.Errorf("LLM returned no choices")
	}

	output := stripReasoning(parsed.Choices[0].Message.Content)

	var inputChars int
	for _, m := range messages {
		inputChars += len(m.Content)
	}

	c.mu.Lock()
	c.stats.Calls++
	c.stats.InputChars += inputChars
	c.stats.OutputChars += len(output)
	if parsed.Usage.TotalTokens > 0 {
		c.stats.ReportedByAPI = true
		c.stats.InputTokens += parsed.Usage.PromptTokens
		c.stats.OutputTokens += parsed.Usage.CompletionTokens
	} else {
		c.stats.InputTokens += inputChars / 4
		c.stats.OutputTokens += len(output) / 4
	}
	c.mu.Unlock()

	return output, nil
}

func stripReasoning(s string) string {
	for {
		start := strings.Index(s, "<think>")
		if start < 0 {
			break
		}
		end := strings.Index(s, "</think>")
		if end < 0 || end < start {
			s = s[:start]
			break
		}
		s = s[:start] + s[end+len("</think>"):]
	}
	return strings.TrimSpace(s)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
