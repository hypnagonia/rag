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

// Prompt templates for LLM interactions (kept short to minimize tokens)
const (
	promptExpandQuery = `Generate 2-3 code search queries for the given question. Output one query per line, no numbering.`

	promptEvaluateContext = `Given code snippets and a question, respond in JSON:
{"sufficient":bool,"reason":"brief","expand":["funcName"],"queries":["search term"]}
Use "expand" for specific items you see referenced. Use "queries" for new searches.`

	promptGenerateAnswer = `Answer the code question using ONLY the provided context. Be concise.`

	tokenMethodologyText = `
   Tokens are estimated using a ratio of ~4 characters per token.
   This is a conservative estimate for code, which typically has:
   - Shorter words than natural language
   - Many symbols and operators (each often = 1 token)
   - Indentation and whitespace

   Actual tokenization varies by model:
   - GPT models use BPE (Byte Pair Encoding)
   - DeepSeek uses similar subword tokenization
   - Common identifiers may be single tokens
   - Rare names get split into multiple tokens

   For precise counts, use the model's tokenizer:
   - OpenAI: tiktoken library
   - DeepSeek: their tokenizer API

   The comparison shows that RAG retrieves only relevant code chunks
   instead of sending the entire codebase, dramatically reducing:
   - Token usage (and thus API costs)
   - Latency (smaller context = faster processing)
   - Risk of hitting context length limits
`
)

// LLMClient provides a generic OpenAI-compatible LLM client
type LLMClient struct {
	baseURL string
	apiKey  string
	model   string
	client  *http.Client
	stats   LLMStats
}

// LLMStats tracks LLM usage statistics
type LLMStats struct {
	TotalCalls       int
	TotalInputChars  int
	TotalOutputChars int
	TotalInputTokens int  // estimated
	TotalOutputTokens int // estimated
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
	// Calculate input size
	inputChars := 0
	for _, msg := range messages {
		inputChars += len(msg.Content)
	}

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

	output := chatResp.Choices[0].Message.Content

	// Update stats
	c.stats.TotalCalls++
	c.stats.TotalInputChars += inputChars
	c.stats.TotalOutputChars += len(output)
	// Rough token estimate: ~4 chars per token for English
	c.stats.TotalInputTokens += inputChars / 4
	c.stats.TotalOutputTokens += len(output) / 4

	return output, nil
}

// GetStats returns the current LLM usage statistics
func (c *LLMClient) GetStats() LLMStats {
	return c.stats
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
	llm          *LLMClient
	retrieveUC   *usecase.RetrieveUseCase
	store        *store.BoltStore
	maxIters     int
	topK         int
	verbose      bool
	expandQuery  bool // whether to use LLM for query expansion
	fastMode     bool // skip iterative evaluation, just search and answer
	initialChunks int // chunks to start with (progressive context)
}

// NewAgenticRAG creates a new agentic RAG instance
func NewAgenticRAG(llm *LLMClient, retrieveUC *usecase.RetrieveUseCase, store *store.BoltStore, opts AgenticRAGOptions) *AgenticRAG {
	return &AgenticRAG{
		llm:          llm,
		retrieveUC:   retrieveUC,
		store:        store,
		maxIters:     opts.MaxIters,
		topK:         opts.TopK,
		verbose:      opts.Verbose,
		expandQuery:  opts.ExpandQuery,
		fastMode:     opts.FastMode,
		initialChunks: opts.InitialChunks,
	}
}

// AgenticRAGOptions contains configuration for AgenticRAG
type AgenticRAGOptions struct {
	MaxIters      int
	TopK          int
	Verbose       bool
	ExpandQuery   bool
	FastMode      bool
	InitialChunks int
}

// SearchResult holds search results with metadata
type SearchResult struct {
	Query   string
	Chunks  []domain.ScoredChunk
	Context string
	Answer  string // LLM-generated answer based on context
}

// CodebaseStats holds statistics about the indexed codebase
type CodebaseStats struct {
	TotalFiles      int
	TotalChunks     int
	TotalChars      int
	TotalTokensEst  int
}

// Run executes the agentic RAG workflow
func (a *AgenticRAG) Run(originalQuery string) (*SearchResult, error) {
	if a.verbose {
		fmt.Printf("\n%s\n", strings.Repeat("─", 70))
		fmt.Printf("🔍 ORIGINAL QUERY: %s\n", originalQuery)
		fmt.Printf("%s\n", strings.Repeat("─", 70))
	}

	// Step 1: Optionally expand the query using LLM
	var expandedQueries []string
	if a.expandQuery {
		if a.verbose {
			fmt.Printf("\n📤 [LLM] Requesting query expansion...\n")
		} else {
			fmt.Printf("📤 Expanding query...")
		}
		var err error
		expandedQueries, err = a.doExpandQuery(originalQuery)
		if err != nil {
			if a.verbose {
				fmt.Printf("⚠️  Query expansion failed: %v\n", err)
			} else {
				fmt.Printf(" failed\n")
			}
			expandedQueries = []string{originalQuery}
		} else {
			if a.verbose {
				fmt.Printf("📥 [LLM] Expanded queries:\n")
				for i, q := range expandedQueries {
					fmt.Printf("   %d. %s\n", i+1, q)
				}
			} else {
				fmt.Printf(" got %d queries\n", len(expandedQueries))
			}
		}
	} else {
		expandedQueries = []string{originalQuery}
	}

	// Fast mode: just search and answer, no iteration
	if a.fastMode {
		return a.runFastMode(originalQuery, expandedQueries)
	}

	// Collect all results
	allChunks := make(map[string]domain.ScoredChunk)

	for iter := 0; iter < a.maxIters; iter++ {
		if a.verbose {
			fmt.Printf("\n%s\n", strings.Repeat("═", 70))
			fmt.Printf("🔄 ITERATION %d of %d\n", iter+1, a.maxIters)
			fmt.Printf("%s\n", strings.Repeat("═", 70))
		}

		// Step 2: Search with all queries
		if a.verbose {
			fmt.Printf("\n🔎 Executing RAG searches:\n")
		} else {
			fmt.Printf("🔎 Searching [iter %d]...", iter+1)
		}
		for _, q := range expandedQueries {
			if a.verbose {
				fmt.Printf("\n   $ rag query -q \"%s\" -k %d\n", q, a.topK)
			}
			chunks, err := a.retrieveUC.Retrieve(q, a.topK)
			if err != nil {
				if a.verbose {
					fmt.Printf("     ❌ Error: %v\n", err)
				}
				continue
			}
			newCount := 0
			for _, c := range chunks {
				if existing, ok := allChunks[c.Chunk.ID]; !ok || c.Score > existing.Score {
					if !ok {
						newCount++
					}
					allChunks[c.Chunk.ID] = c
				}
			}
			if a.verbose {
				fmt.Printf("     ✓ Found %d results (%d new unique chunks)\n", len(chunks), newCount)
			}
		}

		if len(allChunks) == 0 {
			return nil, fmt.Errorf("no results found for any query")
		}

		if a.verbose {
			fmt.Printf("\n📚 Total unique chunks collected: %d\n", len(allChunks))
		} else {
			fmt.Printf(" found %d chunks\n", len(allChunks))
		}

		// Build context from current results
		context := a.buildContext(allChunks)

		// Step 3: Ask LLM if context is sufficient
		if a.verbose {
			fmt.Printf("\n📤 [LLM] Evaluating if context is sufficient...\n")
		} else {
			fmt.Printf("🤔 Evaluating context...")
		}
		decision, err := a.evaluateContext(originalQuery, context)
		if err != nil {
			if a.verbose {
				fmt.Printf("⚠️  Context evaluation failed: %v\n", err)
				fmt.Printf("   Assuming context is sufficient.\n")
			} else {
				fmt.Printf(" failed, proceeding\n")
			}
			decision = &ContextDecision{Sufficient: true, Reason: "Evaluation failed, proceeding with results"}
		} else if !a.verbose {
			if decision.Sufficient {
				fmt.Printf(" sufficient!\n")
			} else {
				fmt.Printf(" need more context\n")
			}
		}

		if a.verbose {
			fmt.Printf("📥 [LLM] Decision:\n")
			fmt.Printf("   • Sufficient: %v\n", decision.Sufficient)
			fmt.Printf("   • Reason: %s\n", decision.Reason)
			if len(decision.ExpandContext) > 0 {
				fmt.Printf("   • Expand context: %v\n", decision.ExpandContext)
			}
			if len(decision.SuggestedQueries) > 0 {
				fmt.Printf("   • Suggested queries: %v\n", decision.SuggestedQueries)
			}
		}

		if decision.Sufficient {
			if a.verbose {
				fmt.Printf("\n✅ Context is sufficient! Generating answer...\n")
			} else {
				fmt.Printf("💡 Generating answer...")
			}
			chunks := sortedChunks(allChunks)

			// Generate final answer
			answer, err := a.generateAnswer(originalQuery, context)
			if err != nil {
				if a.verbose {
					fmt.Printf("⚠️  Failed to generate answer: %v\n", err)
				} else {
					fmt.Printf(" failed\n")
				}
				answer = "(Answer generation failed)"
			} else if !a.verbose {
				fmt.Printf(" done\n")
			}

			return &SearchResult{
				Query:   originalQuery,
				Chunks:  chunks,
				Context: context,
				Answer:  answer,
			}, nil
		}

		// Step 4a: Handle context expansion requests first
		if len(decision.ExpandContext) > 0 {
			if a.verbose {
				fmt.Printf("\n📚 LLM requested context expansion:\n")
				for i, item := range decision.ExpandContext {
					fmt.Printf("   %d. %s\n", i+1, item)
				}
			} else {
				fmt.Printf("📚 Expanding context (%d items)...", len(decision.ExpandContext))
			}

			// Fetch the requested context items
			expandedChunks := a.expandContextItems(decision.ExpandContext)
			newCount := 0
			for _, c := range expandedChunks {
				if _, exists := allChunks[c.Chunk.ID]; !exists {
					allChunks[c.Chunk.ID] = c
					newCount++
				}
			}

			if a.verbose {
				fmt.Printf("   ✓ Added %d new chunks from expansion\n", newCount)
			} else {
				fmt.Printf(" +%d chunks\n", newCount)
			}

			// If we got new chunks, re-evaluate without consuming a query iteration
			if newCount > 0 {
				continue
			}
		}

		// Step 4b: Need more context - generate new queries
		if len(decision.SuggestedQueries) > 0 {
			if a.verbose {
				fmt.Printf("\n🔄 LLM suggested new queries:\n")
				for i, q := range decision.SuggestedQueries {
					fmt.Printf("   %d. %s\n", i+1, q)
				}
			}
			expandedQueries = decision.SuggestedQueries
		} else if len(decision.ExpandContext) == 0 {
			// Only stop if we had neither expansion nor queries
			if a.verbose {
				fmt.Printf("\n⚠️  No new queries or expansion suggested. Stopping iterations.\n")
			}
			break
		}
	}

	if a.verbose {
		fmt.Printf("\n⏹️  Max iterations reached. Generating answer with available context...\n")
	} else {
		fmt.Printf("💡 Generating answer...")
	}

	// Return what we have after max iterations
	chunks := sortedChunks(allChunks)
	context := a.buildContext(allChunks)

	// Generate final answer anyway
	answer, err := a.generateAnswer(originalQuery, context)
	if err != nil {
		if a.verbose {
			fmt.Printf("⚠️  Failed to generate answer: %v\n", err)
		} else {
			fmt.Printf(" failed\n")
		}
		answer = "(Answer generation failed)"
	} else if !a.verbose {
		fmt.Printf(" done\n")
	}

	return &SearchResult{
		Query:   originalQuery,
		Chunks:  chunks,
		Context: context,
		Answer:  answer,
	}, nil
}

// generateAnswer asks LLM to answer the question based on retrieved context
func (a *AgenticRAG) generateAnswer(query, context string) (string, error) {
	truncatedContext := truncateContext(context, 4000) // reduced from 8000
	userPrompt := fmt.Sprintf("Q: %s\n\nCode:\n%s", query, truncatedContext)

	if a.verbose {
		fmt.Printf("\n📤 [LLM] Generating final answer...\n")
		a.printPromptBox(promptGenerateAnswer, fmt.Sprintf("Question: %s\n\nCode Context: (%d chars, omitted)", query, len(truncatedContext)))
	}

	response, err := a.llm.GenerateWithSystem(promptGenerateAnswer, userPrompt)
	if err != nil {
		return "", err
	}

	if a.verbose {
		a.printResponseBox("LLM ANSWER", response)
	}

	return response, nil
}

// runFastMode executes a minimal token workflow: search once, answer once
func (a *AgenticRAG) runFastMode(originalQuery string, queries []string) (*SearchResult, error) {
	if !a.verbose {
		fmt.Printf("🔎 Searching...")
	}

	// Search with all queries
	allChunks := make(map[string]domain.ScoredChunk)
	for _, q := range queries {
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
		return nil, fmt.Errorf("no results found")
	}

	if !a.verbose {
		fmt.Printf(" %d chunks\n", len(allChunks))
		fmt.Printf("💡 Generating answer...")
	}

	// Build context and generate answer
	chunks := sortedChunks(allChunks)
	// Limit to top chunks for fast mode
	if len(chunks) > a.topK {
		chunks = chunks[:a.topK]
	}
	context := a.buildContextFromSlice(chunks)

	answer, err := a.generateAnswer(originalQuery, context)
	if err != nil {
		if !a.verbose {
			fmt.Printf(" failed\n")
		}
		return nil, err
	}

	if !a.verbose {
		fmt.Printf(" done\n")
	}

	return &SearchResult{
		Query:   originalQuery,
		Chunks:  chunks,
		Context: context,
		Answer:  answer,
	}, nil
}

// buildContextFromSlice creates context from a slice of chunks
func (a *AgenticRAG) buildContextFromSlice(chunks []domain.ScoredChunk) string {
	var sb strings.Builder
	for _, c := range chunks {
		doc, _ := a.store.GetDoc(c.Chunk.DocID)
		sb.WriteString(fmt.Sprintf("=== %s:L%d-%d ===\n", doc.Path, c.Chunk.StartLine, c.Chunk.EndLine))
		sb.WriteString(c.Chunk.Text)
		sb.WriteString("\n\n")
	}
	return sb.String()
}

// printPromptBox prints a formatted box with system and user prompts
func (a *AgenticRAG) printPromptBox(systemPrompt, userPrompt string) {
	fmt.Printf("\n   ┌── SYSTEM PROMPT ──────────────────────────────────────────────────\n")
	for _, line := range strings.Split(systemPrompt, "\n") {
		fmt.Printf("   │ %s\n", line)
	}
	fmt.Printf("   ├── USER PROMPT ────────────────────────────────────────────────────\n")
	for _, line := range strings.Split(userPrompt, "\n") {
		fmt.Printf("   │ %s\n", line)
	}
	fmt.Printf("   └───────────────────────────────────────────────────────────────────\n")
}

// printResponseBox prints a formatted box with LLM response
func (a *AgenticRAG) printResponseBox(title, response string) {
	fmt.Printf("\n   ┌── %s ", title)
	fmt.Printf("%s\n", strings.Repeat("─", 55-len(title)))
	for _, line := range strings.Split(response, "\n") {
		fmt.Printf("   │ %s\n", line)
	}
	fmt.Printf("   └───────────────────────────────────────────────────────────────────\n")
}

// doExpandQuery uses LLM to generate expanded search queries
func (a *AgenticRAG) doExpandQuery(query string) ([]string, error) {
	userPrompt := fmt.Sprintf("Expand this code search query: %s", query)

	if a.verbose {
		a.printPromptBox(promptExpandQuery, userPrompt)
	}

	response, err := a.llm.GenerateWithSystem(promptExpandQuery, userPrompt)
	if err != nil {
		return nil, err
	}

	if a.verbose {
		a.printResponseBox("LLM RESPONSE", response)
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
	Sufficient        bool
	Reason            string
	SuggestedQueries  []string
	ExpandContext     []string // requests to expand context (e.g., "fetch definition of funcX", "show file Y")
}

// evaluateContext asks LLM if the current context is sufficient
func (a *AgenticRAG) evaluateContext(query, context string) (*ContextDecision, error) {
	// Send truncated context - reduced from 6000 to 2000 for token savings
	truncatedContext := truncateContext(context, 2000)
	userPrompt := fmt.Sprintf("Q: %s\n\nCode:\n%s\n\nSufficient?", query, truncatedContext)

	if a.verbose {
		a.printPromptBox(promptEvaluateContext, fmt.Sprintf("Q: %s\n\nCode: (%d chars)", query, len(truncatedContext)))
	}

	response, err := a.llm.GenerateWithSystem(promptEvaluateContext, userPrompt)
	if err != nil {
		return nil, err
	}

	if a.verbose {
		a.printResponseBox("LLM RESPONSE", response)
	}

	// Parse JSON response
	// Find JSON in response (might have markdown code blocks)
	jsonStr := extractJSON(response)

	var decision struct {
		Sufficient       bool     `json:"sufficient"`
		Reason           string   `json:"reason"`
		ExpandContext    []string `json:"expand"`
		SuggestedQueries []string `json:"queries"`
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
		ExpandContext:    decision.ExpandContext,
		SuggestedQueries: decision.SuggestedQueries,
	}, nil
}

// expandContextItems searches for specific items requested by the LLM
func (a *AgenticRAG) expandContextItems(requests []string) []domain.ScoredChunk {
	var results []domain.ScoredChunk
	seen := make(map[string]bool)

	for _, req := range requests {
		if a.verbose {
			fmt.Printf("\n   🔎 Expanding context: %s\n", req)
		}

		// Search for the requested item
		chunks, err := a.retrieveUC.Retrieve(req, a.topK/2+1)
		if err != nil {
			if a.verbose {
				fmt.Printf("      ❌ Error: %v\n", err)
			}
			continue
		}

		added := 0
		for _, c := range chunks {
			if !seen[c.Chunk.ID] {
				seen[c.Chunk.ID] = true
				results = append(results, c)
				added++
			}
		}
		if a.verbose {
			fmt.Printf("      ✓ Added %d chunks\n", added)
		}
	}

	return results
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

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}

// getCodebaseStats calculates total size of indexed codebase
func getCodebaseStats(st *store.BoltStore) CodebaseStats {
	stats := CodebaseStats{}

	docs, err := st.ListDocs()
	if err != nil {
		return stats
	}
	stats.TotalFiles = len(docs)

	for _, doc := range docs {
		chunks, err := st.GetChunksByDoc(doc.ID)
		if err != nil {
			continue
		}
		stats.TotalChunks += len(chunks)
		for _, chunk := range chunks {
			stats.TotalChars += len(chunk.Text)
		}
	}

	// Estimate tokens: ~4 chars per token for code (conservative)
	stats.TotalTokensEst = stats.TotalChars / 4

	return stats
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
	topK := flag.Int("k", 5, "Number of results per query")
	maxIters := flag.Int("max-iters", 2, "Maximum iterations")
	verbose := flag.Bool("v", false, "Verbose output")
	fullOutput := flag.Bool("full", false, "Show full code content (no truncation)")
	maxResults := flag.Int("max-results", 10, "Maximum results to display (0 = all)")
	expand := flag.Bool("expand", false, "Use LLM to expand query (uses more tokens)")
	fast := flag.Bool("fast", false, "Fast mode: search once and answer (minimal tokens)")

	flag.Parse()

	if *query == "" {
		fmt.Println("Usage: go run main.go -q \"your query\" [options]")
		fmt.Println("\nOptions:")
		flag.PrintDefaults()
		fmt.Println("\nExamples:")
		fmt.Println("  go run main.go -q \"how does authentication work\"")
		fmt.Println("  go run main.go -q \"auth\" -fast              # minimal tokens")
		fmt.Println("  go run main.go -q \"auth\" -expand -v         # full agentic mode")
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
	agent := NewAgenticRAG(llm, retrieveUC, st, AgenticRAGOptions{
		MaxIters:      *maxIters,
		TopK:          *topK,
		Verbose:       *verbose,
		ExpandQuery:   *expand,
		FastMode:      *fast,
		InitialChunks: 3, // start with 3 chunks for progressive context
	})

	// Print startup info
	mode := "iterative"
	if *fast {
		mode = "fast (1 LLM call)"
	}
	fmt.Printf("🚀 Agentic RAG [%s]\n", mode)
	fmt.Printf("┌─────────────────────────────────────────────────────────────────────\n")
	fmt.Printf("│ LLM: %s/%s | Index: %s\n", *provider, *model, *indexPath)
	fmt.Printf("│ top-k: %d | expand: %v | fast: %v\n", *topK, *expand, *fast)
	fmt.Printf("└─────────────────────────────────────────────────────────────────────\n")

	result, err := agent.Run(*query)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	// Print results only in verbose mode
	if *verbose {
		fmt.Printf("\n" + strings.Repeat("=", 70) + "\n")
		fmt.Printf("📋 RESULTS for: %s\n", result.Query)
		fmt.Printf(strings.Repeat("=", 70) + "\n\n")

		if len(result.Chunks) == 0 {
			fmt.Println("No results found.")
		} else {
			displayCount := len(result.Chunks)
			if *maxResults > 0 && displayCount > *maxResults {
				displayCount = *maxResults
			}

			fmt.Printf("Found %d relevant code sections", len(result.Chunks))
			if displayCount < len(result.Chunks) {
				fmt.Printf(" (showing top %d)", displayCount)
			}
			fmt.Println(":")
			fmt.Println()

			for i := 0; i < displayCount; i++ {
				c := result.Chunks[i]
				doc, _ := st.GetDoc(c.Chunk.DocID)

				// Print header
				fmt.Printf("┌─── [%d] %s ───\n", i+1, doc.Path)
				fmt.Printf("│ Lines: %d-%d | Score: %.3f\n", c.Chunk.StartLine, c.Chunk.EndLine, c.Score)
				fmt.Printf("├" + strings.Repeat("─", 69) + "\n")

				// Print code content
				text := c.Chunk.Text
				if !*fullOutput && len(text) > 1000 {
					text = text[:1000] + "\n... (use -full to see complete content)"
				}

				// Add line prefix for readability
				lines := strings.Split(text, "\n")
				for _, line := range lines {
					fmt.Printf("│ %s\n", line)
				}
				fmt.Printf("└" + strings.Repeat("─", 69) + "\n\n")
			}

			if displayCount < len(result.Chunks) {
				fmt.Printf("... and %d more results (use -max-results 0 to show all)\n", len(result.Chunks)-displayCount)
			}
		}
	}

	// Print the answer
	fmt.Printf("\n%s\n", strings.Repeat("═", 70))
	fmt.Printf("💡 ANSWER\n")
	fmt.Printf("%s\n\n", strings.Repeat("═", 70))
	fmt.Println(result.Answer)

	// Always show token comparison summary
	codebaseStats := getCodebaseStats(st)
	llmStats := llm.GetStats()
	fullContextTokens := codebaseStats.TotalTokensEst + 100 // +100 for query/system prompt
	tokensUsed := llmStats.TotalInputTokens + llmStats.TotalOutputTokens

	fmt.Printf("\n%s\n", strings.Repeat("─", 70))
	fmt.Printf("📊 TOKEN USAGE: ~%s tokens (with RAG) vs ~%s tokens (without RAG)\n",
		formatNumber(tokensUsed), formatNumber(fullContextTokens))
	if fullContextTokens > 0 {
		reduction := float64(fullContextTokens) / float64(max(llmStats.TotalInputTokens, 1))
		fmt.Printf("   💰 RAG saved %.1fx tokens\n", reduction)
	}
	fmt.Printf("%s\n", strings.Repeat("─", 70))

	// Show detailed statistics only in verbose mode
	if *verbose {
		fmt.Printf("\n%s\n", strings.Repeat("─", 70))
		fmt.Printf("📊 LLM USAGE STATISTICS (detailed)\n")
		fmt.Printf("%s\n", strings.Repeat("─", 70))
		fmt.Printf("   Total LLM calls:        %d\n", llmStats.TotalCalls)
		fmt.Printf("   Input characters:       %s\n", formatNumber(llmStats.TotalInputChars))
		fmt.Printf("   Output characters:      %s\n", formatNumber(llmStats.TotalOutputChars))
		fmt.Printf("   Est. input tokens:      ~%s\n", formatNumber(llmStats.TotalInputTokens))
		fmt.Printf("   Est. output tokens:     ~%s\n", formatNumber(llmStats.TotalOutputTokens))
		fmt.Printf("   Est. total tokens:      ~%s\n", formatNumber(tokensUsed))

		// Print comparison with full codebase
		fmt.Printf("\n%s\n", strings.Repeat("─", 70))
		fmt.Printf("📈 RAG EFFICIENCY COMPARISON (detailed)\n")
		fmt.Printf("%s\n", strings.Repeat("─", 70))
		fmt.Printf("\n   📁 Indexed Codebase:\n")
		fmt.Printf("      Files:               %d\n", codebaseStats.TotalFiles)
		fmt.Printf("      Chunks:              %d\n", codebaseStats.TotalChunks)
		fmt.Printf("      Total characters:    %s\n", formatNumber(codebaseStats.TotalChars))
		fmt.Printf("      Est. tokens:         ~%s\n", formatNumber(codebaseStats.TotalTokensEst))

		fmt.Printf("\n   🎯 With RAG (actual usage):\n")
		fmt.Printf("      Context sent to LLM: %s chars\n", formatNumber(llmStats.TotalInputChars))
		fmt.Printf("      Est. tokens used:    ~%s\n", formatNumber(tokensUsed))

		fmt.Printf("\n   ❌ Without RAG (full codebase):\n")
		fmt.Printf("      Would need to send:  %s chars\n", formatNumber(codebaseStats.TotalChars))
		fmt.Printf("      Est. tokens needed:  ~%s\n", formatNumber(fullContextTokens))

		// Calculate savings
		if fullContextTokens > 0 {
			savings := float64(fullContextTokens-llmStats.TotalInputTokens) / float64(fullContextTokens) * 100
			reduction := float64(fullContextTokens) / float64(max(llmStats.TotalInputTokens, 1))
			fmt.Printf("\n   💰 SAVINGS:\n")
			fmt.Printf("      Token reduction:     %.1fx less tokens\n", reduction)
			fmt.Printf("      Percentage saved:    %.1f%%\n", savings)

			// Cost estimation (using typical pricing)
			// GPT-4o-mini: $0.15/1M input, $0.60/1M output
			// DeepSeek: ~$0.14/1M input, $0.28/1M output
			ragCostInput := float64(llmStats.TotalInputTokens) / 1000000 * 0.15
			ragCostOutput := float64(llmStats.TotalOutputTokens) / 1000000 * 0.60
			ragCost := ragCostInput + ragCostOutput

			fullCostInput := float64(fullContextTokens) / 1000000 * 0.15
			fullCostOutput := float64(llmStats.TotalOutputTokens) / 1000000 * 0.60 // same output
			fullCost := fullCostInput + fullCostOutput

			fmt.Printf("\n      Est. cost with RAG:  $%.6f\n", ragCost)
			fmt.Printf("      Est. cost without:   $%.6f\n", fullCost)
			if fullCost > 0 {
				fmt.Printf("      Cost savings:        $%.6f (%.1f%%)\n", fullCost-ragCost, (fullCost-ragCost)/fullCost*100)
			}
		}
	}

	// Show methodology only in verbose mode
	if *verbose {
		fmt.Printf("\n%s\n", strings.Repeat("─", 70))
		fmt.Printf("📝 TOKEN ESTIMATION METHODOLOGY\n")
		fmt.Printf("%s\n", strings.Repeat("─", 70))
		fmt.Print(tokenMethodologyText)
		fmt.Printf("%s\n", strings.Repeat("─", 70))
	}
}

func formatNumber(n int) string {
	if n >= 1000000 {
		return fmt.Sprintf("%dM", n/1000000)
	}
	if n >= 1000 {
		return fmt.Sprintf("%dk", n/1000)
	}
	return fmt.Sprintf("%d", n)
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
