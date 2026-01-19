package port

type LLM interface {
	Generate(prompt string) (string, error)

	GenerateWithSystem(systemPrompt, userPrompt string) (string, error)

	ModelName() string
}

type Reranker interface {
	Rerank(query string, chunkTexts []string) ([]RerankedResult, error)

	ModelName() string
}

type RerankedResult struct {
	Index int
	Score float64
}
