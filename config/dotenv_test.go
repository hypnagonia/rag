package config

import (
	"os"
	"path/filepath"
	"testing"
)

func writeDotEnv(t *testing.T, dir, content string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, DotEnvFileName), []byte(content), 0600); err != nil {
		t.Fatal(err)
	}
}

func TestLoadDotEnvSetsVariables(t *testing.T) {
	dir := t.TempDir()
	writeDotEnv(t, dir, "RAG_TEST_PLAIN=value1\n")

	t.Setenv("RAG_TEST_PLAIN", "")
	os.Unsetenv("RAG_TEST_PLAIN")

	loaded := LoadDotEnv(dir)
	if len(loaded) != 1 {
		t.Fatalf("expected 1 loaded file, got %v", loaded)
	}
	if got := os.Getenv("RAG_TEST_PLAIN"); got != "value1" {
		t.Errorf("expected value1, got %q", got)
	}
}

func TestLoadDotEnvHandlesExportPrefixAndQuotes(t *testing.T) {
	dir := t.TempDir()
	writeDotEnv(t, dir, `
# a comment
export RAG_TEST_EXPORTED=sk-abc123

RAG_TEST_DQUOTED="quoted value"
RAG_TEST_SQUOTED='single value'
RAG_TEST_TRAILING=bare # trailing comment
RAG_TEST_EQUALS=a=b=c
malformed line without equals
=novalue
`)

	for _, k := range []string{"RAG_TEST_EXPORTED", "RAG_TEST_DQUOTED", "RAG_TEST_SQUOTED", "RAG_TEST_TRAILING", "RAG_TEST_EQUALS"} {
		os.Unsetenv(k)
		t.Cleanup(func() { os.Unsetenv(k) })
	}

	LoadDotEnv(dir)

	cases := map[string]string{
		"RAG_TEST_EXPORTED": "sk-abc123",
		"RAG_TEST_DQUOTED":  "quoted value",
		"RAG_TEST_SQUOTED":  "single value",
		"RAG_TEST_TRAILING": "bare",
		"RAG_TEST_EQUALS":   "a=b=c",
	}
	for k, want := range cases {
		if got := os.Getenv(k); got != want {
			t.Errorf("%s: expected %q, got %q", k, want, got)
		}
	}
}

func TestLoadDotEnvDoesNotOverrideRealEnvironment(t *testing.T) {
	dir := t.TempDir()
	writeDotEnv(t, dir, "RAG_TEST_PRESET=from_file\n")

	t.Setenv("RAG_TEST_PRESET", "from_shell")

	LoadDotEnv(dir)

	if got := os.Getenv("RAG_TEST_PRESET"); got != "from_shell" {
		t.Errorf("an already-set variable must win over .env, got %q", got)
	}
}

func TestLoadDotEnvMissingFileIsNotAnError(t *testing.T) {
	if loaded := LoadDotEnv(t.TempDir()); len(loaded) != 0 {
		t.Errorf("expected no files loaded, got %v", loaded)
	}
}

func TestLoadDotEnvReadsEachFileOnce(t *testing.T) {
	dir := t.TempDir()
	writeDotEnv(t, dir, "RAG_TEST_ONCE=1\n")
	os.Unsetenv("RAG_TEST_ONCE")
	t.Cleanup(func() { os.Unsetenv("RAG_TEST_ONCE") })

	if loaded := LoadDotEnv(dir, dir); len(loaded) != 1 {
		t.Errorf("the same directory listed twice should load once, got %v", loaded)
	}
}

func TestLoadDotEnvFirstDirectoryWins(t *testing.T) {
	first, second := t.TempDir(), t.TempDir()
	writeDotEnv(t, first, "RAG_TEST_ORDER=first\n")
	writeDotEnv(t, second, "RAG_TEST_ORDER=second\n")

	os.Unsetenv("RAG_TEST_ORDER")
	t.Cleanup(func() { os.Unsetenv("RAG_TEST_ORDER") })

	LoadDotEnv(first, second)

	if got := os.Getenv("RAG_TEST_ORDER"); got != "first" {
		t.Errorf("the first directory should win, got %q", got)
	}
}

func TestLoadDotEnvSearchesParentDirectories(t *testing.T) {
	root := t.TempDir()
	writeDotEnv(t, root, "RAG_TEST_UPWARD=found_in_parent\n")

	nested := filepath.Join(root, "examples", "agentic-rag")
	if err := os.MkdirAll(nested, 0755); err != nil {
		t.Fatal(err)
	}

	os.Unsetenv("RAG_TEST_UPWARD")
	t.Cleanup(func() { os.Unsetenv("RAG_TEST_UPWARD") })

	loaded := LoadDotEnv(nested)
	if len(loaded) != 1 {
		t.Fatalf("expected the parent .env to be found, got %v", loaded)
	}
	if got := os.Getenv("RAG_TEST_UPWARD"); got != "found_in_parent" {
		t.Errorf("expected found_in_parent, got %q", got)
	}
}

func TestLoadDotEnvPrefersNearestDirectory(t *testing.T) {
	root := t.TempDir()
	writeDotEnv(t, root, "RAG_TEST_NEAREST=parent\n")

	nested := filepath.Join(root, "child")
	if err := os.MkdirAll(nested, 0755); err != nil {
		t.Fatal(err)
	}
	writeDotEnv(t, nested, "RAG_TEST_NEAREST=child\n")

	os.Unsetenv("RAG_TEST_NEAREST")
	t.Cleanup(func() { os.Unsetenv("RAG_TEST_NEAREST") })

	LoadDotEnv(nested)

	if got := os.Getenv("RAG_TEST_NEAREST"); got != "child" {
		t.Errorf("the nearest .env should win, got %q", got)
	}
}
