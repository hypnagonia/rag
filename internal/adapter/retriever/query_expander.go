package retriever

import (
	"fmt"
	"strings"

	"rag/internal/port"
)

// QueryExpander expands queries using an LLM to generate related search terms.
type QueryExpander struct {
	llm port.LLM
}

// NewQueryExpander creates a new query expander.
func NewQueryExpander(llm port.LLM) *QueryExpander {
	return &QueryExpander{llm: llm}
}

// Expand generates expanded queries for better retrieval coverage.
// Returns the original query plus generated variants.
func (e *QueryExpander) Expand(query string) ([]string, error) {
	if e.llm == nil {
		return []string{query}, nil
	}

	systemPrompt := `You are a search query expansion assistant for a code search system.
Given a user's search query, generate 2-3 alternative queries that might help find relevant code.
Focus on:
- Synonyms and related programming terms
- Different ways to phrase the same concept
- Technical terms that might be used in code comments or function names

Output ONLY the alternative queries, one per line. Do not include explanations or numbering.`

	userPrompt := fmt.Sprintf("Original query: %s\n\nGenerate alternative search queries:", query)

	response, err := e.llm.GenerateWithSystem(systemPrompt, userPrompt)
	if err != nil {
		// Fall back to original query on error
		return []string{query}, nil
	}

	// Parse response into individual queries
	queries := []string{query} // Always include original
	lines := strings.Split(response, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		// Skip empty lines and lines that look like explanations
		if line == "" || strings.HasPrefix(line, "-") || strings.Contains(line, ":") {
			continue
		}
		// Remove common prefixes like "1.", "2.", etc.
		line = strings.TrimLeft(line, "0123456789. ")
		if line != "" && line != query {
			queries = append(queries, line)
		}
	}

	// Limit to 4 total queries
	if len(queries) > 4 {
		queries = queries[:4]
	}

	return queries, nil
}

// ExpandWithKeywords generates additional keyword-based expansions.
// This is a simpler, non-LLM approach for when LLM is unavailable.
func (e *QueryExpander) ExpandWithKeywords(query string) []string {
	queries := []string{query}

	// Common programming term expansions
	expansions := map[string][]string{
		"auth":       {"authentication", "login", "authorize"},
		"config":     {"configuration", "settings", "options"},
		"db":         {"database", "storage", "persistence"},
		"err":        {"error", "exception", "failure"},
		"func":       {"function", "method", "handler"},
		"init":       {"initialize", "setup", "bootstrap"},
		"msg":        {"message", "notification", "event"},
		"req":        {"request", "input", "query"},
		"resp":       {"response", "output", "result"},
		"util":       {"utility", "helper", "common"},
		"api":        {"endpoint", "route", "handler"},
		"test":       {"testing", "spec", "unit test"},
		"validate":   {"validation", "check", "verify"},
		"parse":      {"parsing", "decode", "deserialize"},
		"serialize":  {"encoding", "marshal", "format"},
		"async":      {"asynchronous", "concurrent", "parallel"},
		"cache":      {"caching", "memoize", "store"},
		"log":        {"logging", "logger", "trace"},
		"middleware": {"interceptor", "filter", "handler"},
	}

	lowerQuery := strings.ToLower(query)
	for abbrev, synonyms := range expansions {
		if strings.Contains(lowerQuery, abbrev) {
			for _, syn := range synonyms {
				expanded := strings.ReplaceAll(lowerQuery, abbrev, syn)
				if expanded != lowerQuery {
					queries = append(queries, expanded)
				}
			}
		}
	}

	// Limit expansions
	if len(queries) > 5 {
		queries = queries[:5]
	}

	return queries
}
