package port

// Embedder generates vector embeddings for text.
type Embedder interface {
	// Embed generates embeddings for the given texts.
	// Returns a slice of vectors, one per input text.
	Embed(texts []string) ([][]float32, error)

	// Dimension returns the embedding vector dimension.
	Dimension() int

	// ModelName returns the name of the embedding model.
	ModelName() string
}

// VectorStore stores and searches embedding vectors.
type VectorStore interface {
	// Upsert adds or updates vectors in the store.
	Upsert(items []VectorItem) error

	// Search finds the k nearest vectors to the query.
	Search(query []float32, k int) ([]VectorResult, error)

	// Delete removes vectors by their IDs.
	Delete(ids []string) error

	// Count returns the number of vectors in the store.
	Count() (int, error)
}

// VectorItem represents a vector to be stored.
type VectorItem struct {
	ID       string            // Unique identifier (typically chunk ID)
	Vector   []float32         // Embedding vector
	Metadata map[string]string // Optional metadata
}

// VectorResult represents a search result.
type VectorResult struct {
	ID       string            // Chunk ID
	Score    float64           // Similarity score (higher is better)
	Metadata map[string]string // Stored metadata
}
