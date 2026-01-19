package port

type Embedder interface {
	Embed(texts []string) ([][]float32, error)

	Dimension() int

	ModelName() string
}

type VectorStore interface {
	Upsert(items []VectorItem) error

	Search(query []float32, k int) ([]VectorResult, error)

	Delete(ids []string) error

	Count() (int, error)
}

type VectorItem struct {
	ID       string
	Vector   []float32
	Metadata map[string]string
}

type VectorResult struct {
	ID       string
	Score    float64
	Metadata map[string]string
}
