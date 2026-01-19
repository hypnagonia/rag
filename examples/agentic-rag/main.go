// Agentic RAG Example
//
// This example demonstrates an agentic RAG workflow:
// 1. Accept a search query from the user
// 2. Use an LLM to expand/refine the query
// 3. Search the RAG index
// 4. Ask LLM if the context is sufficient or needs expansion
// 5. Iterate until satisfied, then present results
//
// Usage:
//   export DEEPSEEK_API_KEY=your-key  # or OPENAI_API_KEY
//   go run main.go -q "how does authentication work" -index /path/to/project
//
// Supported LLM providers (OpenAI-compatible APIs):
//   - DeepSeek: -provider deepseek -model deepseek-chat
//   - OpenAI:   -provider openai -model gpt-4o-mini
//   - Local:    -provider local -base-url http://localhost:11434/v1

package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"rag/config"
	"rag/internal/adapter/analyzer"
	"rag/internal/adapter/retriever"
	"rag/internal/adapter/store"
	"rag/internal/domain"
	"rag/internal/usecase"
)

// LLMClient provides a generic OpenAI-compatible LLM client
type LLMClient struct {
	baseURL string
	apiKey  string
	model   string
	client  *http.Client
}

// ChatMessage represents a message in the chat format
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatRequest is the request format for chat completions
type ChatRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	Temperature float64       `json:"temperature,omitempty"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
}

// ChatResponse is the response format from chat completions
type ChatResponse struct {
	Choices []struct {
		Message ChatMessage `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// Provider configurations
var providers = map[string]struct {
	baseURL   string
	keyEnvVar string
}{
	"deepseek": {"https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"},
	"openai":   {"https://api.openai.com/v1", "OPENAI_API_KEY"},
	"local":    {"http://localhost:11434/v1", ""},
}

// NewLLMClient creates a new LLM client for the specified provider
func NewLLMClient(provider, model, baseURL, apiKey string) (*LLMClient, error) {
	p, ok := providers[provider]
	if !ok && baseURL == "" {
		return nil, fmt.Errorf("unknown provider: %s (use -base-url for custom endpoints)", provider)
	}

	if baseURL == "" {
		baseURL = p.baseURL
	}

	if apiKey == "" && p.keyEnvVar != "" {
		apiKey = os.Getenv(p.keyEnvVar)
		if apiKey == "" {
			return nil, fmt.Errorf("API key not found. Set %s environment variable", p.keyEnvVar)
		}
	}

	return &LLMClient{
		baseURL: baseURL,
		apiKey:  apiKey,
		model:   model,
		client:  &http.Client{Timeout: 60 * time.Second},
	}, nil
}

// Chat sends a chat completion request
func (c *LLMClient) Chat(messages []ChatMessage) (string, error) {
	req := ChatRequest{
		Model:       c.model,
		Messages:    messages,
		Temperature: 0.7,
		MaxTokens:   2000,
	}

	jsonData, err := json.Marshal(req)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequest("POST", c.baseURL+"/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.client.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %w", err)
	}

	var chatResp ChatResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		return "", fmt.Errorf("failed to parse response: %w", err)
	}

	if chatResp.Error != nil {
		return "", fmt.Errorf("API error: %s", chatResp.Error.Message)
	}

	if len(chatResp.Choices) == 0 {
		return "", fmt.Errorf("no response from LLM")
	}

	return chatResp.Choices[0].Message.Content, nil
}

// Generate implements single-turn generation
func (c *LLMClient) Generate(prompt string) (string, error) {
	return c.Chat([]ChatMessage{{Role: "user", Content: prompt}})
}

// GenerateWithSystem implements generation with system prompt
func (c *LLMClient) GenerateWithSystem(systemPrompt, userPrompt string) (string, error) {
	return c.Chat([]ChatMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	})
}

// AgenticRAG orchestrates the agentic RAG workflow
type AgenticRAG struct {
	llm         *LLMClient
	retrieveUC  *usecase.RetrieveUseCase
	store       *store.BoltStore
	maxIters    int
	topK        int
	verbose     bool
}

// NewAgenticRAG creates a new agentic RAG instance
func NewAgenticRAG(llm *LLMClient, retrieveUC *usecase.RetrieveUseCase, store *store.BoltStore, maxIters, topK int, verbose bool) *AgenticRAG {
	return &AgenticRAG{
		llm:        llm,
		retrieveUC: retrieveUC,
		store:      store,
		maxIters:   maxIters,
		topK:       topK,
		verbose:    verbose,
	}
}

// SearchResult holds search results with metadata
type SearchResult struct {
	Query   string
	Chunks  []domain.ScoredChunk
	Context string
}

// Run executes the agentic RAG workflow
func (a *AgenticRAG) Run(originalQuery string) (*SearchResult, error) {
	if a.verbose {
		fmt.Printf("\n🔍 Original query: %s\n", originalQuery)
	}

	// Step 1: Expand the query using LLM
	expandedQueries, err := a.expandQuery(originalQuery)
	if err != nil {
		if a.verbose {
			fmt.Printf("⚠️  Query expansion failed, using original: %v\n", err)
		}
		expandedQueries = []string{originalQuery}
	}

	if a.verbose {
		fmt.Printf("📝 Expanded queries: %v\n", expandedQueries)
	}

	// Collect all results
	allChunks := make(map[string]domain.ScoredChunk)

	for iter := 0; iter < a.maxIters; iter++ {
		if a.verbose {
			fmt.Printf("\n--- Iteration %d ---\n", iter+1)
		}

		// Step 2: Search with all queries
		for _, q := range expandedQueries {
			chunks, err := a.retrieveUC.Retrieve(q, a.topK)
			if err != nil {
				continue
			}
			for _, c := range chunks {
				if existing, ok := allChunks[c.Chunk.ID]; !ok || c.Score > existing.Score {
					allChunks[c.Chunk.ID] = c
				}
			}
		}

		if len(allChunks) == 0 {
			return nil, fmt.Errorf("no results found for any query")
		}

		// Build context from current results
		context := a.buildContext(allChunks)

		if a.verbose {
			fmt.Printf("📚 Found %d unique chunks\n", len(allChunks))
		}

		// Step 3: Ask LLM if context is sufficient
		decision, err := a.evaluateContext(originalQuery, context)
		if err != nil {
			if a.verbose {
				fmt.Printf("⚠️  Context evaluation failed: %v\n", err)
			}
			// Assume sufficient on error
			decision = &ContextDecision{Sufficient: true}
		}

		if a.verbose {
			fmt.Printf("🤔 LLM decision: sufficient=%v, reason=%s\n", decision.Sufficient, decision.Reason)
		}

		if decision.Sufficient {
			// We have enough context
			chunks := sortedChunks(allChunks)
			return &SearchResult{
				Query:   originalQuery,
				Chunks:  chunks,
				Context: context,
			}, nil
		}

		// Step 4: Need more context - generate new queries
		if len(decision.SuggestedQueries) > 0 {
			expandedQueries = decision.SuggestedQueries
			if a.verbose {
				fmt.Printf("🔄 New queries suggested: %v\n", expandedQueries)
			}
		} else {
			// No new suggestions, break
			break
		}
	}

	// Return what we have after max iterations
	chunks := sortedChunks(allChunks)
	return &SearchResult{
		Query:   originalQuery,
		Chunks:  chunks,
		Context: a.buildContext(allChunks),
	}, nil
}

// expandQuery uses LLM to generate expanded search queries
func (a *AgenticRAG) expandQuery(query string) ([]string, error) {
	systemPrompt := `You are a code search query expansion assistant. Given a user's search query about code, generate 2-3 alternative search queries that might help find relevant code.

Focus on:
- Synonyms and related programming terms
- Different ways to phrase the same concept
- Technical terms used in code (function names, variable names, etc.)

Output ONLY the queries, one per line. Include the original query as the first line. No explanations or numbering.`

	userPrompt := fmt.Sprintf("Expand this code search query: %s", query)

	response, err := a.llm.GenerateWithSystem(systemPrompt, userPrompt)
	if err != nil {
		return nil, err
	}

	// Parse response into queries
	queries := []string{}
	for _, line := range strings.Split(response, "\n") {
		line = strings.TrimSpace(line)
		if line != "" && !strings.HasPrefix(line, "-") && !strings.HasPrefix(line, "*") {
			// Remove numbering if present
			line = strings.TrimLeft(line, "0123456789. ")
			if line != "" {
				queries = append(queries, line)
			}
		}
	}

	if len(queries) == 0 {
		return []string{query}, nil
	}

	return queries, nil
}

// ContextDecision represents the LLM's evaluation of context sufficiency
type ContextDecision struct {
	Sufficient       bool
	Reason           string
	SuggestedQueries []string
}

// evaluateContext asks LLM if the current context is sufficient
func (a *AgenticRAG) evaluateContext(query, context string) (*ContextDecision, error) {
	systemPrompt := `You are evaluating whether retrieved code context is sufficient to answer a user's question.

Analyze the provided context and determine:
1. Is there enough information to answer the question?
2. If not, what additional searches might help?

Respond in this exact JSON format:
{
  "sufficient": true/false,
  "reason": "brief explanation",
  "suggested_queries": ["query1", "query2"]  // only if not sufficient
}

Be concise. Only mark as insufficient if clearly missing critical information.`

	userPrompt := fmt.Sprintf(`Question: %s

Retrieved Context:
%s

Is this context sufficient to answer the question?`, query, truncateContext(context, 6000))

	response, err := a.llm.GenerateWithSystem(systemPrompt, userPrompt)
	if err != nil {
		return nil, err
	}

	// Parse JSON response
	// Find JSON in response (might have markdown code blocks)
	jsonStr := extractJSON(response)

	var decision struct {
		Sufficient       bool     `json:"sufficient"`
		Reason           string   `json:"reason"`
		SuggestedQueries []string `json:"suggested_queries"`
	}

	if err := json.Unmarshal([]byte(jsonStr), &decision); err != nil {
		// Try to infer from text
		lower := strings.ToLower(response)
		sufficient := strings.Contains(lower, "sufficient") && !strings.Contains(lower, "not sufficient") && !strings.Contains(lower, "insufficient")
		return &ContextDecision{
			Sufficient: sufficient,
			Reason:     "Could not parse LLM response",
		}, nil
	}

	return &ContextDecision{
		Sufficient:       decision.Sufficient,
		Reason:           decision.Reason,
		SuggestedQueries: decision.SuggestedQueries,
	}, nil
}

// buildContext creates a text context from chunks
func (a *AgenticRAG) buildContext(chunks map[string]domain.ScoredChunk) string {
	var sb strings.Builder
	for _, c := range chunks {
		doc, _ := a.store.GetDoc(c.Chunk.DocID)
		sb.WriteString(fmt.Sprintf("=== %s (L%d-%d) ===\n", doc.Path, c.Chunk.StartLine, c.Chunk.EndLine))
		sb.WriteString(c.Chunk.Text)
		sb.WriteString("\n\n")
	}
	return sb.String()
}

// Helper functions

func sortedChunks(m map[string]domain.ScoredChunk) []domain.ScoredChunk {
	chunks := make([]domain.ScoredChunk, 0, len(m))
	for _, c := range m {
		chunks = append(chunks, c)
	}
	// Sort by score descending
	for i := 0; i < len(chunks); i++ {
		for j := i + 1; j < len(chunks); j++ {
			if chunks[j].Score > chunks[i].Score {
				chunks[i], chunks[j] = chunks[j], chunks[i]
			}
		}
	}
	return chunks
}

func truncateContext(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "\n... (truncated)"
}

func extractJSON(s string) string {
	// Try to find JSON in the response
	start := strings.Index(s, "{")
	end := strings.LastIndex(s, "}")
	if start >= 0 && end > start {
		return s[start : end+1]
	}
	return s
}

func main() {
	// Command line flags
	query := flag.String("q", "", "Search query (required)")
	indexPath := flag.String("index", ".", "Path to indexed directory")
	provider := flag.String("provider", "deepseek", "LLM provider: deepseek, openai, local")
	model := flag.String("model", "deepseek-chat", "Model name")
	baseURL := flag.String("base-url", "", "Custom API base URL (optional)")
	apiKey := flag.String("api-key", "", "API key (optional, uses env var if not set)")
	topK := flag.Int("k", 10, "Number of results per query")
	maxIters := flag.Int("max-iters", 3, "Maximum iterations")
	verbose := flag.Bool("v", false, "Verbose output")

	flag.Parse()

	if *query == "" {
		fmt.Println("Usage: go run main.go -q \"your query\" [options]")
		fmt.Println("\nOptions:")
		flag.PrintDefaults()
		fmt.Println("\nExamples:")
		fmt.Println("  go run main.go -q \"how does authentication work\"")
		fmt.Println("  go run main.go -q \"database connection\" -provider openai -model gpt-4o-mini")
		fmt.Println("  go run main.go -q \"error handling\" -index /path/to/project -v")
		os.Exit(1)
	}

	// Initialize LLM client
	llm, err := NewLLMClient(*provider, *model, *baseURL, *apiKey)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error initializing LLM: %v\n", err)
		os.Exit(1)
	}

	// Load config and open store
	cfg, err := config.LoadFromDir(*indexPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading config: %v\n", err)
		os.Exit(1)
	}

	dbPath := config.IndexDBPath(*indexPath)
	if _, err := os.Stat(dbPath); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "No index found at %s. Run 'rag index' first.\n", *indexPath)
		os.Exit(1)
	}

	st, err := store.NewBoltStore(dbPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error opening index: %v\n", err)
		os.Exit(1)
	}
	defer st.Close()

	// Create retriever
	tokenizer := analyzer.NewTokenizer(cfg.Index.Stemming)
	bm25 := retriever.NewBM25Retriever(st, tokenizer, cfg.Index.K1, cfg.Index.B, cfg.Retrieve.PathBoostWeight)
	mmr := retriever.NewMMRReranker(cfg.Retrieve.MMRLambda, cfg.Retrieve.DedupJaccard)
	retrieveUC := usecase.NewRetrieveUseCase(bm25, mmr, cfg.Retrieve.MinScoreThreshold)

	// Create and run agentic RAG
	agent := NewAgenticRAG(llm, retrieveUC, st, *maxIters, *topK, *verbose)

	fmt.Printf("🚀 Starting agentic RAG search...\n")
	result, err := agent.Run(*query)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	// Print results
	fmt.Printf("\n" + strings.Repeat("=", 60) + "\n")
	fmt.Printf("📋 RESULTS for: %s\n", result.Query)
	fmt.Printf(strings.Repeat("=", 60) + "\n\n")

	if len(result.Chunks) == 0 {
		fmt.Println("No results found.")
		return
	}

	fmt.Printf("Found %d relevant code sections:\n\n", len(result.Chunks))

	for i, c := range result.Chunks {
		if i >= 10 {
			fmt.Printf("\n... and %d more results\n", len(result.Chunks)-10)
			break
		}
		doc, _ := st.GetDoc(c.Chunk.DocID)
		fmt.Printf("─── [%d] %s:L%d-%d (score: %.3f) ───\n",
			i+1, doc.Path, c.Chunk.StartLine, c.Chunk.EndLine, c.Score)

		text := c.Chunk.Text
		if len(text) > 500 {
			text = text[:500] + "\n... (truncated)"
		}
		fmt.Println(text)
		fmt.Println()
	}
}
