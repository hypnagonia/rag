package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"rag/config"
	"rag/internal/adapter/analyzer"
	"rag/internal/adapter/embedding"
	"rag/internal/adapter/retriever"
	"rag/internal/adapter/store"
	"rag/internal/domain"
	"rag/internal/port"
	"rag/internal/usecase"
)

const (
	promptExpandQuery = `Generate 3 search queries to find relevant passages. Output one query per line, no numbering.
	Query must not contain the answer to the question!
	Rules:
	- Use full names
	- Include HOW events happened 
	- One query should focus on the scene itself with descriptive words
`

	promptEvaluateContext = `Given text passages and a question, respond in JSON:
{"sufficient":bool,"reason":"brief","expand":["topic"],"queries":["search"],"expandLines":["file:line"]}
- "expand": search for related topics mentioned in context
- "queries": new search terms
- "expandLines": show more lines around a passage, e.g. ["3.txt:17650"] to see lines before/after line 17650`

	promptGenerateAnswer = `Answer the question using ONLY the provided context. Be concise. ALWAYS cite sources as [filename:lines] for every fact you mention.`

	tokenMethodologyText = `
   Tokens are estimated using a ratio of ~4 characters per token.
   This is a reasonable estimate for English text. Actual tokenization
   varies by model (GPT uses BPE, others use similar subword methods).

   RAG retrieves only relevant passages instead of sending entire
   documents, dramatically reducing token usage and API costs.
`
)

type LLMClient struct {
	baseURL string
	apiKey  string
	model   string
	client  *http.Client
	stats   LLMStats
}

type LLMStats struct {
	TotalCalls        int
	TotalInputChars   int
	TotalOutputChars  int
	TotalInputTokens  int
	TotalOutputTokens int
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	Temperature float64       `json:"temperature,omitempty"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
}

type ChatResponse struct {
	Choices []struct {
		Message ChatMessage `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

var providers = map[string]struct {
	baseURL   string
	keyEnvVar string
}{
	"deepseek": {"https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"},
	"openai":   {"https://api.openai.com/v1", "OPENAI_API_KEY"},
	"local":    {"http://localhost:11434/v1", ""},
}

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

func (c *LLMClient) Chat(messages []ChatMessage) (string, error) {

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

	c.stats.TotalCalls++
	c.stats.TotalInputChars += inputChars
	c.stats.TotalOutputChars += len(output)

	c.stats.TotalInputTokens += inputChars / 4
	c.stats.TotalOutputTokens += len(output) / 4

	return output, nil
}

func (c *LLMClient) GetStats() LLMStats {
	return c.stats
}

func (c *LLMClient) Generate(prompt string) (string, error) {
	return c.Chat([]ChatMessage{{Role: "user", Content: prompt}})
}

func (c *LLMClient) GenerateWithSystem(systemPrompt, userPrompt string) (string, error) {
	return c.Chat([]ChatMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	})
}

type AgenticRAG struct {
	llm           *LLMClient
	retrieveUC    *usecase.RetrieveUseCase
	packUC        *usecase.PackUseCase
	store         *store.BoltStore
	indexPath     string
	expandedLines map[string]string
	maxIters      int
	topK          int
	tokenBudget   int
	verbose       bool
	expandQuery   bool
	fastMode      bool
}

func NewAgenticRAG(llm *LLMClient, retrieveUC *usecase.RetrieveUseCase, packUC *usecase.PackUseCase, store *store.BoltStore, opts AgenticRAGOptions) *AgenticRAG {
	return &AgenticRAG{
		llm:           llm,
		retrieveUC:    retrieveUC,
		packUC:        packUC,
		store:         store,
		indexPath:     opts.IndexPath,
		expandedLines: make(map[string]string),
		maxIters:      opts.MaxIters,
		topK:          opts.TopK,
		tokenBudget:   opts.TokenBudget,
		verbose:       opts.Verbose,
		expandQuery:   opts.ExpandQuery,
		fastMode:      opts.FastMode,
	}
}

type AgenticRAGOptions struct {
	IndexPath   string
	MaxIters    int
	TopK        int
	TokenBudget int
	Verbose     bool
	ExpandQuery bool
	FastMode    bool
}

type SearchResult struct {
	Query       string
	Chunks      []domain.ScoredChunk
	Context     string
	Answer      string
	QueriesUsed []string
}

type ContentStats struct {
	TotalDocs      int
	TotalChunks    int
	TotalChars     int
	TotalTokensEst int
}

func (a *AgenticRAG) Run(originalQuery string) (*SearchResult, error) {
	if a.verbose {
		fmt.Printf("\n%s\n", strings.Repeat("─", 70))
		fmt.Printf("ORIGINAL QUERY: %s\n", originalQuery)
		fmt.Printf("%s\n", strings.Repeat("─", 70))
	}

	var expandedQueries []string
	if a.expandQuery {
		if a.verbose {
			fmt.Printf("\n[LLM] Requesting query expansion...\n")
		} else {
			fmt.Printf("Expanding query...")
		}
		var err error
		expandedQueries, err = a.doExpandQuery(originalQuery)
		if err != nil {
			if a.verbose {
				fmt.Printf("Warning: Query expansion failed: %v\n", err)
			} else {
				fmt.Printf(" failed\n")
			}
			expandedQueries = []string{originalQuery}
		} else {
			if a.verbose {
				fmt.Printf("[LLM] Expanded queries:\n")
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

	if a.fastMode {
		return a.runFastMode(originalQuery, expandedQueries)
	}

	allChunks := make(map[string]domain.ScoredChunk)

	for iter := 0; iter < a.maxIters; iter++ {
		if a.verbose {
			fmt.Printf("\n%s\n", strings.Repeat("═", 70))
			fmt.Printf("ITERATION %d of %d\n", iter+1, a.maxIters)
			fmt.Printf("%s\n", strings.Repeat("═", 70))
		}

		if a.verbose {
			fmt.Printf("\nExecuting RAG searches:\n")
		} else {
			fmt.Printf("Searching [iter %d]...", iter+1)
		}
		for _, q := range expandedQueries {
			if a.verbose {
				fmt.Printf("\n   $ rag query -q \"%s\" -k %d\n", q, a.topK)
			}
			chunks, err := a.retrieveUC.Retrieve(q, a.topK)
			if err != nil {
				if a.verbose {
					fmt.Printf("     Error: %v\n", err)
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
				fmt.Printf("     Found %d results (%d new unique chunks)\n", len(chunks), newCount)
			}
		}

		if len(allChunks) == 0 {
			return nil, fmt.Errorf("no results found for any query")
		}

		if a.verbose {
			fmt.Printf("\nTotal unique chunks collected: %d\n", len(allChunks))
		} else {
			fmt.Printf(" found %d chunks\n", len(allChunks))
		}

		context := a.buildContext(allChunks, originalQuery)

		if a.verbose {
			fmt.Printf("\n[LLM] Evaluating if context is sufficient...\n")
		} else {
			fmt.Printf("Evaluating context...")
		}
		decision, err := a.evaluateContext(originalQuery, context)
		if err != nil {
			if a.verbose {
				fmt.Printf("Warning: Context evaluation failed: %v\n", err)
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
			fmt.Printf("[LLM] Decision:\n")
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
				fmt.Printf("\nContext is sufficient! Generating answer...\n")
			} else {
				fmt.Printf("Generating answer...")
			}
			chunks := sortedChunks(allChunks)

			answer, err := a.generateAnswer(originalQuery, context)
			if err != nil {
				if a.verbose {
					fmt.Printf("Warning: Failed to generate answer: %v\n", err)
				} else {
					fmt.Printf(" failed\n")
				}
				answer = "(Answer generation failed)"
			} else if !a.verbose {
				fmt.Printf(" done\n")
			}

			return &SearchResult{
				Query:       originalQuery,
				Chunks:      chunks,
				Context:     context,
				Answer:      answer,
				QueriesUsed: expandedQueries,
			}, nil
		}

		if len(decision.ExpandContext) > 0 {
			if a.verbose {
				fmt.Printf("\nLLM requested context expansion:\n")
				for i, item := range decision.ExpandContext {
					fmt.Printf("   %d. %s\n", i+1, item)
				}
			} else {
				fmt.Printf("Expanding context (%d items)...", len(decision.ExpandContext))
			}

			expandedChunks := a.expandContextItems(decision.ExpandContext)
			newCount := 0
			for _, c := range expandedChunks {
				if _, exists := allChunks[c.Chunk.ID]; !exists {
					allChunks[c.Chunk.ID] = c
					newCount++
				}
			}

			if a.verbose {
				fmt.Printf("   Added %d new chunks from expansion\n", newCount)
			} else {
				fmt.Printf(" +%d chunks\n", newCount)
			}

			if newCount > 0 {
				continue
			}
		}

		if len(decision.ExpandLines) > 0 {
			if a.verbose {
				fmt.Printf("\nLLM requested line expansion:\n")
				for i, item := range decision.ExpandLines {
					fmt.Printf("   %d. %s\n", i+1, item)
				}
			} else {
				fmt.Printf("Expanding lines (%d locations)...", len(decision.ExpandLines))
			}

			newCount := a.expandLinesAround(decision.ExpandLines, 50)

			if a.verbose {
				fmt.Printf("   Expanded %d file locations\n", newCount)
			} else {
				fmt.Printf(" +%d expanded\n", newCount)
			}

			if newCount > 0 {
				continue
			}
		}

		if len(decision.SuggestedQueries) > 0 {
			if a.verbose {
				fmt.Printf("\nLLM suggested new queries:\n")
				for i, q := range decision.SuggestedQueries {
					fmt.Printf("   %d. %s\n", i+1, q)
				}
			}
			expandedQueries = decision.SuggestedQueries
		} else if len(decision.ExpandContext) == 0 && len(decision.ExpandLines) == 0 {

			if a.verbose {
				fmt.Printf("\nNo new queries or expansion suggested. Stopping iterations.\n")
			}
			break
		}
	}

	if a.verbose {
		fmt.Printf("\n⏹️  Max iterations reached. Generating answer with available context...\n")
	} else {
		fmt.Printf("Generating answer...")
	}

	chunks := sortedChunks(allChunks)
	context := a.buildContext(allChunks, originalQuery)

	answer, err := a.generateAnswer(originalQuery, context)
	if err != nil {
		if a.verbose {
			fmt.Printf("Warning: Failed to generate answer: %v\n", err)
		} else {
			fmt.Printf(" failed\n")
		}
		answer = "(Answer generation failed)"
	} else if !a.verbose {
		fmt.Printf(" done\n")
	}

	return &SearchResult{
		Query:       originalQuery,
		Chunks:      chunks,
		Context:     context,
		Answer:      answer,
		QueriesUsed: expandedQueries,
	}, nil
}

func (a *AgenticRAG) generateAnswer(query, context string) (string, error) {
	truncatedContext := truncateContext(context, 8000)
	userPrompt := fmt.Sprintf("Q: %s\n\nContext:\n%s", query, truncatedContext)

	if a.verbose {
		fmt.Printf("\n[LLM] Generating final answer...\n")
		a.printPromptBox(promptGenerateAnswer, fmt.Sprintf("Question: %s\n\nContext: (%d chars, omitted)", query, len(truncatedContext)))
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

func (a *AgenticRAG) runFastMode(originalQuery string, queries []string) (*SearchResult, error) {
	if !a.verbose {
		fmt.Printf("Searching...")
	}

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
		fmt.Printf("Generating answer...")
	}

	chunks := sortedChunks(allChunks)

	if len(chunks) > a.topK {
		chunks = chunks[:a.topK]
	}
	context := a.buildContextFromSlice(chunks, originalQuery)

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
		Query:       originalQuery,
		Chunks:      chunks,
		Context:     context,
		Answer:      answer,
		QueriesUsed: queries,
	}, nil
}

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

func (a *AgenticRAG) printResponseBox(title, response string) {
	fmt.Printf("\n   ┌── %s ", title)
	fmt.Printf("%s\n", strings.Repeat("─", 55-len(title)))
	for _, line := range strings.Split(response, "\n") {
		fmt.Printf("   │ %s\n", line)
	}
	fmt.Printf("   └───────────────────────────────────────────────────────────────────\n")
}

func (a *AgenticRAG) doExpandQuery(query string) ([]string, error) {
	userPrompt := fmt.Sprintf("Expand this search query: %s", query)

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

	queries := []string{}
	for _, line := range strings.Split(response, "\n") {
		line = strings.TrimSpace(line)
		if line != "" && !strings.HasPrefix(line, "-") && !strings.HasPrefix(line, "*") {

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

type ContextDecision struct {
	Sufficient       bool
	Reason           string
	SuggestedQueries []string
	ExpandContext    []string
	ExpandLines      []string
}

func (a *AgenticRAG) evaluateContext(query, context string) (*ContextDecision, error) {

	truncatedContext := truncateContext(context, 6000)
	userPrompt := fmt.Sprintf("Q: %s\n\nContext:\n%s\n\nSufficient?", query, truncatedContext)

	if a.verbose {
		a.printPromptBox(promptEvaluateContext, fmt.Sprintf("Q: %s\n\nContext: (%d chars)", query, len(truncatedContext)))
	}

	response, err := a.llm.GenerateWithSystem(promptEvaluateContext, userPrompt)
	if err != nil {
		return nil, err
	}

	if a.verbose {
		a.printResponseBox("LLM RESPONSE", response)
	}

	jsonStr := extractJSON(response)

	var decision struct {
		Sufficient       bool     `json:"sufficient"`
		Reason           string   `json:"reason"`
		ExpandContext    []string `json:"expand"`
		SuggestedQueries []string `json:"queries"`
		ExpandLines      []string `json:"expandLines"`
	}

	if err := json.Unmarshal([]byte(jsonStr), &decision); err != nil {

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
		ExpandLines:      decision.ExpandLines,
	}, nil
}

func (a *AgenticRAG) expandContextItems(requests []string) []domain.ScoredChunk {
	var results []domain.ScoredChunk
	seen := make(map[string]bool)

	for _, req := range requests {
		if a.verbose {
			fmt.Printf("\n   Expanding context: %s\n", req)
		}

		chunks, err := a.retrieveUC.Retrieve(req, a.topK/2+1)
		if err != nil {
			if a.verbose {
				fmt.Printf("      Error: %v\n", err)
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
			fmt.Printf("      Added %d chunks\n", added)
		}
	}

	return results
}

func (a *AgenticRAG) expandLinesAround(requests []string, extraLines int) int {
	if extraLines <= 0 {
		extraLines = 50
	}

	added := 0
	for _, req := range requests {

		parts := strings.SplitN(req, ":", 2)
		if len(parts) != 2 {
			if a.verbose {
				fmt.Printf("   Warning: Invalid format: %s (expected file:line)\n", req)
			}
			continue
		}

		filename := parts[0]
		centerLine, err := strconv.Atoi(parts[1])
		if err != nil {
			if a.verbose {
				fmt.Printf("   Warning: Invalid line number: %s\n", parts[1])
			}
			continue
		}

		fullPath := filepath.Join(a.indexPath, filename)
		if _, err := os.Stat(fullPath); os.IsNotExist(err) {

			docs, _ := a.store.ListDocs()
			for _, doc := range docs {
				if strings.HasSuffix(doc.Path, filename) || strings.Contains(doc.Path, filename) {
					fullPath = doc.Path
					break
				}
			}
		}

		if a.verbose {
			fmt.Printf("   Expanding lines around %s:%d (±%d lines)\n", filename, centerLine, extraLines)
		}

		text, startLine, endLine, err := readLinesAround(fullPath, centerLine, extraLines)
		if err != nil {
			if a.verbose {
				fmt.Printf("      Error: %v\n", err)
			}
			continue
		}

		key := fmt.Sprintf("%s:%d", filename, centerLine)
		a.expandedLines[key] = fmt.Sprintf("=== %s:L%d-%d (expanded) ===\n%s", filename, startLine, endLine, text)
		added++

		if a.verbose {
			fmt.Printf("      Read lines %d-%d (%d chars)\n", startLine, endLine, len(text))
		}
	}

	return added
}

func readLinesAround(path string, centerLine, extraLines int) (string, int, int, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", 0, 0, err
	}
	defer file.Close()

	startLine := centerLine - extraLines
	if startLine < 1 {
		startLine = 1
	}
	endLine := centerLine + extraLines

	scanner := bufio.NewScanner(file)
	var lines []string
	lineNum := 0

	for scanner.Scan() {
		lineNum++
		if lineNum >= startLine && lineNum <= endLine {
			lines = append(lines, scanner.Text())
		}
		if lineNum > endLine {
			break
		}
	}

	if err := scanner.Err(); err != nil {
		return "", 0, 0, err
	}

	actualEnd := startLine + len(lines) - 1
	if actualEnd < endLine {
		endLine = actualEnd
	}

	return strings.Join(lines, "\n"), startLine, endLine, nil
}

func (a *AgenticRAG) buildContext(chunks map[string]domain.ScoredChunk, query string) string {

	chunkSlice := make([]domain.ScoredChunk, 0, len(chunks))
	for _, c := range chunks {
		chunkSlice = append(chunkSlice, c)
	}
	return a.buildContextFromSlice(chunkSlice, query)
}

func (a *AgenticRAG) buildContextFromSlice(chunks []domain.ScoredChunk, query string) string {

	packed, err := a.packUC.Pack(query, chunks, a.tokenBudget)
	if err != nil {

		var sb strings.Builder
		for _, c := range chunks {
			doc, _ := a.store.GetDoc(c.Chunk.DocID)
			sb.WriteString(fmt.Sprintf("=== %s:L%d-%d ===\n", doc.Path, c.Chunk.StartLine, c.Chunk.EndLine))
			sb.WriteString(c.Chunk.Text)
			sb.WriteString("\n\n")
		}

		for _, expanded := range a.expandedLines {
			sb.WriteString(expanded)
			sb.WriteString("\n\n")
		}
		return sb.String()
	}

	var sb strings.Builder
	for _, s := range packed.Snippets {
		sb.WriteString(fmt.Sprintf("=== %s:%s ===\n", s.Path, s.Range))
		sb.WriteString(s.Text)
		sb.WriteString("\n\n")
	}

	if len(a.expandedLines) > 0 {
		for _, expanded := range a.expandedLines {
			sb.WriteString(expanded)
			sb.WriteString("\n\n")
		}
		if a.verbose {
			fmt.Printf("   Added %d expanded line contexts\n", len(a.expandedLines))
		}
	}

	if a.verbose {
		fmt.Printf("   Packed: %d/%d tokens used (%d snippets)\n",
			packed.UsedTokens, packed.BudgetTokens, len(packed.Snippets))
		for i, s := range packed.Snippets {
			fmt.Printf("      %d. %s:%s (%.0f chars)\n", i+1, filepath.Base(s.Path), s.Range, float64(len(s.Text)))
		}
	}

	return sb.String()
}

func sortedChunks(m map[string]domain.ScoredChunk) []domain.ScoredChunk {
	chunks := make([]domain.ScoredChunk, 0, len(m))
	for _, c := range m {
		chunks = append(chunks, c)
	}

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

func getContentStats(st *store.BoltStore) ContentStats {
	stats := ContentStats{}

	docs, err := st.ListDocs()
	if err != nil {
		return stats
	}
	stats.TotalDocs = len(docs)

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

	stats.TotalTokensEst = stats.TotalChars / 4

	return stats
}

func extractJSON(s string) string {

	start := strings.Index(s, "{")
	end := strings.LastIndex(s, "}")
	if start >= 0 && end > start {
		return s[start : end+1]
	}
	return s
}

func setupHybridRetrieval(st *store.BoltStore, cfg *config.Config) (port.Embedder, port.VectorStore, error) {
	var embedder port.Embedder
	var err error

	switch cfg.Embedding.Provider {
	case "openai":
		embedder, err = embedding.NewOpenAIEmbedder(cfg.Embedding.APIKeyEnv, cfg.Embedding.Model)
	case "jina":
		embedder, err = embedding.NewJinaEmbedder(cfg.Embedding.APIKeyEnv, cfg.Embedding.Model)
	case "ollama":
		embedder, err = embedding.NewOllamaEmbedder(cfg.Embedding.Model, cfg.Embedding.BaseURL)
	case "mock":
		embedder = embedding.NewMockEmbedder(cfg.Embedding.Dimension)
	default:
		return nil, nil, fmt.Errorf("unsupported embedding provider: %s", cfg.Embedding.Provider)
	}
	if err != nil {
		return nil, nil, err
	}

	vectorStore, err := store.NewBoltVectorStore(st.DB(), embedder.Dimension())
	if err != nil {
		return nil, nil, err
	}

	count, err := vectorStore.Count()
	if err != nil {
		return nil, nil, err
	}
	if count == 0 {
		return nil, nil, fmt.Errorf("no embeddings found - run 'rag index' with embedding.enabled=true")
	}

	return embedder, vectorStore, nil
}

func main() {

	query := flag.String("q", "", "Search query (required)")
	indexPath := flag.String("index", ".", "Path to indexed directory")
	provider := flag.String("provider", "deepseek", "LLM provider: deepseek, openai, local")
	model := flag.String("model", "deepseek-chat", "Model name")
	baseURL := flag.String("base-url", "", "Custom API base URL (optional)")
	apiKey := flag.String("api-key", "", "API key (optional, uses env var if not set)")
	topK := flag.Int("k", 10, "Number of results per query")
	maxIters := flag.Int("max-iters", 2, "Maximum iterations")
	verbose := flag.Bool("v", false, "Verbose output")
	fullOutput := flag.Bool("full", false, "Show full content (no truncation)")
	maxResults := flag.Int("max-results", 10, "Maximum results to display (0 = all)")
	expand := flag.Bool("expand", false, "Use LLM to expand query (uses more tokens)")
	fast := flag.Bool("fast", false, "Fast mode: search once and answer (minimal tokens)")
	budget := flag.Int("budget", 4000, "Token budget for context packing")

	flag.Parse()

	if *query == "" {
		fmt.Println("Usage: go run main.go -q \"your query\" [options]")
		fmt.Println("\nOptions:")
		flag.PrintDefaults()
		fmt.Println("\nExamples:")
		fmt.Println("  go run main.go -q \"what is the main theme\"")
		fmt.Println("  go run main.go -q \"summary\" -fast           # minimal tokens")
		fmt.Println("  go run main.go -q \"explain chapter 3\" -expand -v  # full agentic mode")
		os.Exit(1)
	}

	llm, err := NewLLMClient(*provider, *model, *baseURL, *apiKey)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error initializing LLM: %v\n", err)
		os.Exit(1)
	}

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

	tokenizer := analyzer.NewTokenizer(cfg.Index.Stemming)
	bm25 := retriever.NewBM25Retriever(st, tokenizer, cfg.Index.K1, cfg.Index.B, cfg.Retrieve.PathBoostWeight)
	mmr := retriever.NewMMRReranker(cfg.Retrieve.MMRLambda, cfg.Retrieve.DedupJaccard)

	var searchRetriever port.Retriever = bm25
	if cfg.Retrieve.HybridEnabled && cfg.Embedding.Enabled {
		embedder, vectorStore, err := setupHybridRetrieval(st, cfg)
		if err != nil {
			if *verbose {
				fmt.Printf("Warning: Hybrid search unavailable: %v (using BM25 only)\n", err)
			}
		} else {
			searchRetriever = retriever.NewHybridRetriever(
				bm25, vectorStore, embedder, st,
				cfg.Retrieve.RRFK, cfg.Retrieve.BM25Weight,
			)
			if *verbose {
				fmt.Printf("Hybrid search enabled (BM25 + vector)\n")
			}
		}
	}

	retrieveUC := usecase.NewRetrieveUseCase(searchRetriever, mmr, cfg.Retrieve.MinScoreThreshold)

	packUC := usecase.NewPackUseCase(st, tokenizer, cfg.Pack.RecencyBoost)

	agent := NewAgenticRAG(llm, retrieveUC, packUC, st, AgenticRAGOptions{
		IndexPath:   *indexPath,
		MaxIters:    *maxIters,
		TopK:        *topK,
		TokenBudget: *budget,
		Verbose:     *verbose,
		ExpandQuery: *expand,
		FastMode:    *fast,
	})

	searchMode := "BM25"
	if cfg.Retrieve.HybridEnabled && cfg.Embedding.Enabled {
		if _, ok := searchRetriever.(*retriever.HybridRetriever); ok {
			searchMode = "Hybrid (BM25 + Vector)"
		}
	}

	mode := "iterative"
	if *fast {
		mode = "fast (1 LLM call)"
	}
	fmt.Printf("Agentic RAG [%s]\n", mode)
	fmt.Printf("┌─────────────────────────────────────────────────────────────────────\n")
	fmt.Printf("│ LLM: %s/%s | Search: %s\n", *provider, *model, searchMode)
	fmt.Printf("│ Index: %s | top-k: %d\n", *indexPath, *topK)
	fmt.Printf("└─────────────────────────────────────────────────────────────────────\n")

	result, err := agent.Run(*query)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

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

			fmt.Printf("Found %d relevant passages", len(result.Chunks))
			if displayCount < len(result.Chunks) {
				fmt.Printf(" (showing top %d)", displayCount)
			}
			fmt.Println(":")
			fmt.Println()

			for i := 0; i < displayCount; i++ {
				c := result.Chunks[i]
				doc, _ := st.GetDoc(c.Chunk.DocID)

				fmt.Printf("┌─── [%d] %s ───\n", i+1, doc.Path)
				fmt.Printf("│ Lines: %d-%d | Score: %.3f\n", c.Chunk.StartLine, c.Chunk.EndLine, c.Score)
				fmt.Printf("├" + strings.Repeat("─", 69) + "\n")

				text := c.Chunk.Text
				if !*fullOutput && len(text) > 1000 {
					text = text[:1000] + "\n... (use -full to see complete content)"
				}

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

	fmt.Printf("\n%s\n", strings.Repeat("═", 70))
	fmt.Printf("ANSWER\n")
	fmt.Printf("%s\n", strings.Repeat("═", 70))
	if (*verbose || *expand) && len(result.QueriesUsed) > 0 {
		fmt.Printf("Queries used:\n")
		for i, q := range result.QueriesUsed {
			fmt.Printf("  %d. %s\n", i+1, q)
		}
		fmt.Println()
	}
	fmt.Println(result.Answer)

	contentStats := getContentStats(st)
	llmStats := llm.GetStats()
	fullContextTokens := contentStats.TotalTokensEst + 100
	tokensUsed := llmStats.TotalInputTokens + llmStats.TotalOutputTokens

	fmt.Printf("\n%s\n", strings.Repeat("─", 70))
	fmt.Printf("📊 TOKEN USAGE: ~%s tokens (with RAG) vs ~%s tokens (without RAG)\n",
		formatNumber(tokensUsed), formatNumber(fullContextTokens))
	if fullContextTokens > 0 {
		reduction := float64(fullContextTokens) / float64(max(llmStats.TotalInputTokens, 1))
		fmt.Printf("   💰 RAG saved %.1fx tokens\n", reduction)
	}
	fmt.Printf("%s\n", strings.Repeat("─", 70))

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

		fmt.Printf("\n%s\n", strings.Repeat("─", 70))
		fmt.Printf("📈 RAG EFFICIENCY COMPARISON (detailed)\n")
		fmt.Printf("%s\n", strings.Repeat("─", 70))
		fmt.Printf("\n   📁 Indexed Content:\n")
		fmt.Printf("      Documents:           %d\n", contentStats.TotalDocs)
		fmt.Printf("      Chunks:              %d\n", contentStats.TotalChunks)
		fmt.Printf("      Total characters:    %s\n", formatNumber(contentStats.TotalChars))
		fmt.Printf("      Est. tokens:         ~%s\n", formatNumber(contentStats.TotalTokensEst))

		fmt.Printf("\n   🎯 With RAG (actual usage):\n")
		fmt.Printf("      Context sent to LLM: %s chars\n", formatNumber(llmStats.TotalInputChars))
		fmt.Printf("      Est. tokens used:    ~%s\n", formatNumber(tokensUsed))

		fmt.Printf("\n   ❌ Without RAG (full content):\n")
		fmt.Printf("      Would need to send:  %s chars\n", formatNumber(contentStats.TotalChars))
		fmt.Printf("      Est. tokens needed:  ~%s\n", formatNumber(fullContextTokens))

		if fullContextTokens > 0 {
			savings := float64(fullContextTokens-llmStats.TotalInputTokens) / float64(fullContextTokens) * 100
			reduction := float64(fullContextTokens) / float64(max(llmStats.TotalInputTokens, 1))
			fmt.Printf("\n   💰 SAVINGS:\n")
			fmt.Printf("      Token reduction:     %.1fx less tokens\n", reduction)
			fmt.Printf("      Percentage saved:    %.1f%%\n", savings)

			ragCostInput := float64(llmStats.TotalInputTokens) / 1000000 * 0.15
			ragCostOutput := float64(llmStats.TotalOutputTokens) / 1000000 * 0.60
			ragCost := ragCostInput + ragCostOutput

			fullCostInput := float64(fullContextTokens) / 1000000 * 0.15
			fullCostOutput := float64(llmStats.TotalOutputTokens) / 1000000 * 0.60
			fullCost := fullCostInput + fullCostOutput

			fmt.Printf("\n      Est. cost with RAG:  $%.6f\n", ragCost)
			fmt.Printf("      Est. cost without:   $%.6f\n", fullCost)
			if fullCost > 0 {
				fmt.Printf("      Cost savings:        $%.6f (%.1f%%)\n", fullCost-ragCost, (fullCost-ragCost)/fullCost*100)
			}
		}
	}

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
