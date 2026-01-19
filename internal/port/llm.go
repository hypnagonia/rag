package port

// LLM represents a language model for text generation.
type LLM interface {
	// Generate generates text based on the prompt.
	Generate(prompt string) (string, error)

	// GenerateWithSystem generates text with a system prompt.
	GenerateWithSystem(systemPrompt, userPrompt string) (string, error)

	// ModelName returns the name of the model.
	ModelName() string
}

// Reranker scores query-document pairs for relevance.
type Reranker interface {
	// Rerank scores and reorders chunks based on query relevance.
	// Returns chunks sorted by relevance score (highest first).
	Rerank(query string, chunkTexts []string) ([]RerankedResult, error)

	// ModelName returns the name of the reranking model.
	ModelName() string
}

// RerankedResult represents a reranked document.
type RerankedResult struct {
	Index int     // Original index in the input slice
	Score float64 // Relevance score (higher is better)
}
