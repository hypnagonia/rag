package port

// Tokenizer defines the interface for text tokenization.
type Tokenizer interface {
	// Tokenize splits text into tokens with stemming and stopword removal.
	Tokenize(text string) []string
	// CountTokens returns the approximate token count for LLM budget estimation.
	CountTokens(text string) int
}
