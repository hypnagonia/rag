package embedding

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func embeddingServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	return srv
}

func vectorsResponse(count, dimension int) embeddingResponse {
	resp := embeddingResponse{}
	for i := 0; i < count; i++ {
		vec := make([]float32, dimension)
		vec[0] = float32(i + 1)
		resp.Data = append(resp.Data, embeddingData{Embedding: vec, Index: i})
	}
	return resp
}

func testEmbedder(t *testing.T, url string) *OpenAIEmbedder {
	t.Helper()
	embedder, err := NewBuilder().Provider(ProviderOllama).Model("test-model").BaseURL(url).Timeout(5 * time.Second).Build()
	if err != nil {
		t.Fatalf("build failed: %v", err)
	}
	return embedder.(*OpenAIEmbedder)
}

func TestEmbedReturnsVectorsInInputOrder(t *testing.T) {
	srv := embeddingServer(t, func(w http.ResponseWriter, r *http.Request) {
		resp := embeddingResponse{Data: []embeddingData{
			{Embedding: []float32{3, 0}, Index: 2},
			{Embedding: []float32{1, 0}, Index: 0},
			{Embedding: []float32{2, 0}, Index: 1},
		}}
		json.NewEncoder(w).Encode(resp)
	})

	got, err := testEmbedder(t, srv.URL).Embed([]string{"a", "b", "c"})
	if err != nil {
		t.Fatalf("embed failed: %v", err)
	}

	for i, want := range []float32{1, 2, 3} {
		if got[i][0] != want {
			t.Errorf("position %d: expected %v, got %v", i, want, got[i][0])
		}
	}
}

func TestEmbedErrorsWhenProviderReturnsFewerVectorsThanInputs(t *testing.T) {
	srv := embeddingServer(t, func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(vectorsResponse(2, 4))
	})

	_, err := testEmbedder(t, srv.URL).Embed([]string{"a", "b", "c"})
	if err == nil {
		t.Fatal("expected an error when the provider returns 2 vectors for 3 inputs")
	}
	if !strings.Contains(err.Error(), "2 embeddings for 3 inputs") {
		t.Errorf("expected a count-mismatch message, got %v", err)
	}
}

func TestEmbedErrorsOnInconsistentDimensions(t *testing.T) {
	srv := embeddingServer(t, func(w http.ResponseWriter, r *http.Request) {
		resp := embeddingResponse{Data: []embeddingData{
			{Embedding: []float32{1, 2, 3}, Index: 0},
			{Embedding: []float32{1, 2}, Index: 1},
		}}
		json.NewEncoder(w).Encode(resp)
	})

	_, err := testEmbedder(t, srv.URL).Embed([]string{"a", "b"})
	if err == nil || !strings.Contains(err.Error(), "inconsistent dimensions") {
		t.Errorf("expected an inconsistent-dimension error, got %v", err)
	}
}

func TestEmbedSurfacesHTTPErrorBody(t *testing.T) {
	srv := embeddingServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		w.Write([]byte(`{"error":{"message":"model not found"}}`))
	})

	_, err := testEmbedder(t, srv.URL).Embed([]string{"a"})
	if err == nil {
		t.Fatal("expected an error for a 404 response")
	}
	if !strings.Contains(err.Error(), "model not found") {
		t.Errorf("expected the provider message in the error, got %v", err)
	}
}

func TestEmbedRetriesServerErrors(t *testing.T) {
	var attempts int32
	srv := embeddingServer(t, func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&attempts, 1) < 2 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		json.NewEncoder(w).Encode(vectorsResponse(1, 4))
	})

	got, err := testEmbedder(t, srv.URL).Embed([]string{"a"})
	if err != nil {
		t.Fatalf("expected a retry to succeed, got %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("expected 1 vector, got %d", len(got))
	}
	if atomic.LoadInt32(&attempts) < 2 {
		t.Errorf("expected at least 2 attempts, got %d", attempts)
	}
}

func TestEmbedDoesNotRetryClientErrors(t *testing.T) {
	var attempts int32
	srv := embeddingServer(t, func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&attempts, 1)
		w.WriteHeader(http.StatusBadRequest)
	})

	if _, err := testEmbedder(t, srv.URL).Embed([]string{"a"}); err == nil {
		t.Fatal("expected a 400 to fail")
	}
	if attempts != 1 {
		t.Errorf("a 400 must not be retried, got %d attempts", attempts)
	}
}

func TestDimensionProbedFromProvider(t *testing.T) {
	srv := embeddingServer(t, func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(vectorsResponse(1, 384))
	})

	embedder, err := NewBuilder().
		Provider(ProviderOllama).
		Model("unknown-model").
		BaseURL(srv.URL).
		Dimension(768).
		Build()
	if err != nil {
		t.Fatalf("build failed: %v", err)
	}

	if got := embedder.Dimension(); got != 384 {
		t.Errorf("expected the probed dimension 384 to win over the configured 768, got %d", got)
	}
}

func TestDimensionFallsBackWhenProviderUnreachable(t *testing.T) {
	embedder, err := NewBuilder().
		Provider(ProviderOllama).
		Model("nomic-embed-text").
		BaseURL("http://127.0.0.1:1").
		Timeout(500 * time.Millisecond).
		Build()
	if err != nil {
		t.Fatalf("build failed: %v", err)
	}

	if got := embedder.Dimension(); got != 768 {
		t.Errorf("expected the known-model fallback of 768, got %d", got)
	}
}

func TestEmbedRespectsBatchSize(t *testing.T) {
	var batches int32
	srv := embeddingServer(t, func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&batches, 1)
		var req embeddingRequest
		json.NewDecoder(r.Body).Decode(&req)
		json.NewEncoder(w).Encode(vectorsResponse(len(req.Input), 4))
	})

	embedder, err := NewBuilder().Provider(ProviderOllama).Model("test-model").BaseURL(srv.URL).BatchSize(2).Build()
	if err != nil {
		t.Fatalf("build failed: %v", err)
	}

	got, err := embedder.Embed([]string{"a", "b", "c", "d", "e"})
	if err != nil {
		t.Fatalf("embed failed: %v", err)
	}
	if len(got) != 5 {
		t.Fatalf("expected 5 vectors, got %d", len(got))
	}
	if batches != 3 {
		t.Errorf("expected 5 inputs at batch size 2 to make 3 requests, got %d", batches)
	}
}

func TestBuilderRejectsUnknownProvider(t *testing.T) {
	if _, err := NewBuilder().Provider("pinecone").Model("m").Build(); err == nil {
		t.Error("expected an unknown provider to be rejected")
	}
}

func TestBuilderRequiresAPIKeyForHostedProviders(t *testing.T) {
	t.Setenv("RAG_TEST_MISSING_KEY", "")
	if _, err := NewBuilder().Provider(ProviderOpenAI).Model("text-embedding-3-small").APIKeyEnv("RAG_TEST_MISSING_KEY").Build(); err == nil {
		t.Error("expected a missing API key to be rejected")
	}
}

func TestBuilderOllamaNeedsNoAPIKey(t *testing.T) {
	if _, err := NewBuilder().Provider(ProviderOllama).Model("nomic-embed-text").BaseURL("http://127.0.0.1:1").Build(); err != nil {
		t.Errorf("ollama should not require an API key, got %v", err)
	}
}

func TestMockEmbedderScoresSharedVocabularyHigher(t *testing.T) {
	mock := NewMockEmbedder(128)

	vecs, err := mock.Embed([]string{
		"func parse config yaml",
		"func parse config file",
		"database connection pool retry",
	})
	if err != nil {
		t.Fatalf("embed failed: %v", err)
	}

	similar := dot(vecs[0], vecs[1]) / (norm(vecs[0]) * norm(vecs[1]))
	different := dot(vecs[0], vecs[2]) / (norm(vecs[0]) * norm(vecs[2]))

	if similar <= different {
		t.Errorf("texts sharing vocabulary should score higher: %f vs %f", similar, different)
	}
}

func TestMockEmbedderProducesNonZeroVectors(t *testing.T) {
	vecs, err := NewMockEmbedder(64).Embed([]string{"", "   "})
	if err != nil {
		t.Fatalf("embed failed: %v", err)
	}
	for i, v := range vecs {
		if norm(v) == 0 {
			t.Errorf("vector %d has zero norm, which makes cosine similarity undefined", i)
		}
	}
}

func dot(a, b []float32) float64 {
	var sum float64
	for i := range a {
		sum += float64(a[i]) * float64(b[i])
	}
	return sum
}

func norm(a []float32) float64 {
	return sqrt(dot(a, a))
}

func sqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	guess := x
	for i := 0; i < 40; i++ {
		guess = 0.5 * (guess + x/guess)
	}
	return guess
}
