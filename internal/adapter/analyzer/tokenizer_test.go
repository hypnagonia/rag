package analyzer

import (
	"testing"
)

func TestTokenizer_Tokenize_WithStemming(t *testing.T) {
	tok := NewTokenizer(true)

	tokens := tok.Tokenize("running dogs are playing")
	if len(tokens) != 3 {
		t.Errorf("expected 3 tokens, got %d: %v", len(tokens), tokens)
	}

	hasRun := false
	for _, token := range tokens {
		if token == "run" {
			hasRun = true
		}
	}
	if !hasRun {
		t.Errorf("expected 'running' to be stemmed to 'run', got %v", tokens)
	}
}

func TestTokenizer_Tokenize_WithoutStemming(t *testing.T) {
	tok := NewTokenizer(false)

	tokens := tok.Tokenize("running dogs are playing")
	if len(tokens) != 3 {
		t.Errorf("expected 3 tokens, got %d: %v", len(tokens), tokens)
	}

	hasRunning := false
	for _, token := range tokens {
		if token == "running" {
			hasRunning = true
		}
	}
	if !hasRunning {
		t.Errorf("expected 'running' to remain unstemmed, got %v", tokens)
	}
}

func TestTokenizer_StopwordRemoval(t *testing.T) {
	tok := NewTokenizer(false)

	tokens := tok.Tokenize("the quick brown fox")
	for _, token := range tokens {
		if token == "the" {
			t.Errorf("stopword 'the' should be removed, got %v", tokens)
		}
	}
}

func TestTokenizer_ShortWordRemoval(t *testing.T) {
	tok := NewTokenizer(false)

	tokens := tok.Tokenize("a I go to")
	for _, token := range tokens {
		if len(token) < 2 {
			t.Errorf("short word should be removed: %s", token)
		}
	}
}

func TestTokenizer_CountTokens(t *testing.T) {
	tok := NewTokenizer(false)

	count := tok.CountTokens("hello world this is a test")
	if count == 0 {
		t.Error("expected non-zero token count")
	}
	if count < 6 {
		t.Errorf("expected count >= 6 words, got %d", count)
	}
}

func TestTokenizer_EmptyInput(t *testing.T) {
	tok := NewTokenizer(true)

	tokens := tok.Tokenize("")
	if len(tokens) != 0 {
		t.Errorf("expected 0 tokens for empty input, got %d", len(tokens))
	}

	count := tok.CountTokens("")
	if count != 0 {
		t.Errorf("expected 0 count for empty input, got %d", count)
	}
}

func TestSplitWords(t *testing.T) {
	tests := []struct {
		input    string
		expected int
	}{
		{"hello world", 2},
		{"hello_world", 1},
		{"hello-world", 2},
		{"func(x, y)", 3},
		{"CamelCase", 1},
		{"snake_case_name", 1},
		{"123numbers456", 1},
	}

	for _, tt := range tests {
		words := splitWords(tt.input)
		if len(words) != tt.expected {
			t.Errorf("splitWords(%q) = %d words, want %d: %v", tt.input, len(words), tt.expected, words)
		}
	}
}
