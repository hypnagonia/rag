package fs

import (
	"os"
	"path/filepath"
	"testing"
)

func TestWalkFollowsSymlinkedRoot(t *testing.T) {
	base := t.TempDir()

	real := filepath.Join(base, "real")
	if err := os.Mkdir(real, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(real, "a.txt"), []byte("hello"), 0644); err != nil {
		t.Fatal(err)
	}

	link := filepath.Join(base, "link")
	if err := os.Symlink(real, link); err != nil {
		t.Fatal(err)
	}

	w := NewWalker([]string{"**/*.txt"}, nil)

	viaReal, err := w.Walk(real)
	if err != nil {
		t.Fatalf("walk real dir failed: %v", err)
	}
	if len(viaReal) != 1 {
		t.Fatalf("expected 1 file via the real directory, got %d", len(viaReal))
	}

	viaLink, err := w.Walk(link)
	if err != nil {
		t.Fatalf("walk symlinked dir failed: %v", err)
	}
	if len(viaLink) != 1 {
		t.Errorf("walking a symlinked root found %d files, want 1 (this is why 'rag index /tmp' indexes nothing on macOS, where /tmp -> private/tmp)", len(viaLink))
	}
}

func TestWalkNonexistentRootStillErrors(t *testing.T) {
	w := NewWalker([]string{"**/*.txt"}, nil)
	if _, err := w.Walk(filepath.Join(t.TempDir(), "does-not-exist")); err == nil {
		t.Error("expected walking a missing directory to return an error")
	}
}

func TestWalkResolvesRootToCanonicalPaths(t *testing.T) {
	base := t.TempDir()
	real := filepath.Join(base, "real")
	if err := os.Mkdir(real, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(real, "a.txt"), []byte("x"), 0644); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(base, "link")
	if err := os.Symlink(real, link); err != nil {
		t.Fatal(err)
	}

	files, err := NewWalker([]string{"**/*.txt"}, nil).Walk(link)
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != 1 {
		t.Fatalf("expected 1 file, got %d", len(files))
	}

	resolvedReal, _ := filepath.EvalSymlinks(real)
	want := filepath.Join(resolvedReal, "a.txt")
	if files[0].Path != want {
		t.Errorf("expected the canonical path %q, got %q", want, files[0].Path)
	}
}
