package llm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestBuilderRejectsLocalProviders(t *testing.T) {
	for _, provider := range []string{"ollama", "local", "llamacpp", "lmstudio"} {
		if _, err := NewBuilder().Provider(provider).Model("m").Build(); err == nil {
			t.Errorf("provider %q must be rejected: generation may not run against a local model", provider)
		}
	}
}

func TestBuilderRequiresAPIKey(t *testing.T) {
	t.Setenv("RAG_TEST_LLM_KEY", "")
	if _, err := NewBuilder().Provider(ProviderDeepSeek).Model("deepseek-chat").APIKeyEnv("RAG_TEST_LLM_KEY").Build(); err == nil {
		t.Error("expected a missing API key to be rejected")
	}
}

func TestBuilderDefaultsToOpenAI(t *testing.T) {
	t.Setenv("RAG_TEST_LLM_KEY", "sk-test")
	client, err := NewBuilder().Model("gpt-4o-mini").APIKeyEnv("RAG_TEST_LLM_KEY").Build()
	if err != nil {
		t.Fatalf("build failed: %v", err)
	}
	if c := client.(*Client); c.baseURL != providerDefaults[ProviderOpenAI] {
		t.Errorf("expected the OpenAI endpoint by default, got %q", c.baseURL)
	}
}

func TestBuilderAllowsKeylessGatewayWithExplicitBaseURL(t *testing.T) {
	client, err := NewBuilder().Model("deepseek-web").BaseURL("http://127.0.0.1:8787/v1").Build()
	if err != nil {
		t.Fatalf("a keyless gateway with an explicit base_url must build: %v", err)
	}
	if c := client.(*Client); c.apiKey != "" {
		t.Errorf("expected no API key, got %q", c.apiKey)
	}
}

func TestBuilderRejectsKeylessProviderDefault(t *testing.T) {
	if _, err := NewBuilder().Model("gpt-4o-mini").Build(); err == nil {
		t.Error("a keyless build against the provider default endpoint must be rejected")
	}
}

func newTestClient(t *testing.T, handler http.HandlerFunc) port_LLM {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	t.Setenv("RAG_TEST_LLM_KEY", "sk-test")

	client, err := NewBuilder().
		Provider(ProviderDeepSeek).
		Model("deepseek-chat").
		APIKeyEnv("RAG_TEST_LLM_KEY").
		BaseURL(srv.URL).
		Build()
	if err != nil {
		t.Fatalf("build failed: %v", err)
	}
	return client
}

type port_LLM interface {
	Generate(string) (string, error)
	GenerateWithSystem(string, string) (string, error)
	ModelName() string
}

func chatReply(content string) string {
	b, _ := json.Marshal(map[string]any{
		"choices": []map[string]any{{"message": map[string]string{"role": "assistant", "content": content}}},
	})
	return string(b)
}

func TestGenerateReturnsContent(t *testing.T) {
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer sk-test" {
			t.Errorf("missing bearer token, got %q", got)
		}
		w.Write([]byte(chatReply("Eddard Stark was beheaded.")))
	})

	got, err := c.Generate("How did Ned Stark die")
	if err != nil {
		t.Fatalf("generate failed: %v", err)
	}
	if got != "Eddard Stark was beheaded." {
		t.Errorf("unexpected content: %q", got)
	}
}

func TestGenerateStripsReasoningBlocks(t *testing.T) {
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(chatReply("<think>let me recall the scene</think>\nHe was beheaded with Ice.")))
	})

	got, err := c.Generate("q")
	if err != nil {
		t.Fatalf("generate failed: %v", err)
	}
	if strings.Contains(got, "<think>") || strings.Contains(got, "let me recall") {
		t.Errorf("reasoning block should be stripped, got %q", got)
	}
	if got != "He was beheaded with Ice." {
		t.Errorf("unexpected content: %q", got)
	}
}

func TestGenerateSurfacesAPIError(t *testing.T) {
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusPaymentRequired)
		w.Write([]byte(`{"error":{"message":"Insufficient Balance"}}`))
	})

	_, err := c.Generate("q")
	if err == nil {
		t.Fatal("expected an error for a 402 response")
	}
	if !strings.Contains(err.Error(), "Insufficient Balance") {
		t.Errorf("expected the provider message in the error, got %v", err)
	}
}

func TestGenerateErrorsWhenNoChoices(t *testing.T) {
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"choices":[]}`))
	})
	if _, err := c.Generate("q"); err == nil {
		t.Error("expected an error when the API returns no choices")
	}
}

func TestGenerateWithSystemSendsBothMessages(t *testing.T) {
	var got chatRequest
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&got)
		w.Write([]byte(chatReply("ok")))
	})

	if _, err := c.GenerateWithSystem("sys", "user"); err != nil {
		t.Fatalf("generate failed: %v", err)
	}
	if len(got.Messages) != 2 || got.Messages[0].Role != "system" || got.Messages[1].Role != "user" {
		t.Errorf("expected a system+user message pair, got %+v", got.Messages)
	}
}

func TestGenerateOmitsAuthorizationWhenKeyless(t *testing.T) {
	var hadAuth bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, hadAuth = r.Header["Authorization"]
		w.Write([]byte(chatReply("34")))
	}))
	defer srv.Close()

	client, err := NewBuilder().Model("deepseek-web").BaseURL(srv.URL).Build()
	if err != nil {
		t.Fatalf("build failed: %v", err)
	}
	if _, err := client.Generate("2 + 2 * 16."); err != nil {
		t.Fatalf("generate failed: %v", err)
	}
	if hadAuth {
		t.Error("a keyless client must not send an Authorization header")
	}
}
