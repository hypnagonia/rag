package usecase

import (
	"path/filepath"
	"strings"

	"rag/internal/adapter/store"
	"rag/internal/domain"
)

// ContextExpander expands search results with related context.
type ContextExpander struct {
	store            *store.BoltStore
	includeImports   bool
	includeTests     bool
	includeInterfaces bool
	maxExpansion     int // Max chunks to add per result
}

// NewContextExpander creates a new context expander.
func NewContextExpander(store *store.BoltStore, includeImports, includeTests, includeInterfaces bool) *ContextExpander {
	return &ContextExpander{
		store:             store,
		includeImports:    includeImports,
		includeTests:      includeTests,
		includeInterfaces: includeInterfaces,
		maxExpansion:      3,
	}
}

// Expand adds related context to search results.
func (e *ContextExpander) Expand(results []domain.ScoredChunk) ([]domain.ScoredChunk, error) {
	if len(results) == 0 {
		return results, nil
	}

	// Track already included chunks
	included := make(map[string]bool)
	for _, r := range results {
		included[r.Chunk.ID] = true
	}

	// Track docs we've seen
	seenDocs := make(map[string]domain.Document)
	for _, r := range results {
		if _, exists := seenDocs[r.Chunk.DocID]; !exists {
			doc, err := e.store.GetDoc(r.Chunk.DocID)
			if err == nil {
				seenDocs[r.Chunk.DocID] = doc
			}
		}
	}

	expanded := make([]domain.ScoredChunk, 0, len(results))
	expanded = append(expanded, results...)

	for _, r := range results {
		doc, exists := seenDocs[r.Chunk.DocID]
		if !exists {
			continue
		}

		// Find related docs
		relatedDocs := e.findRelatedDocs(doc, seenDocs)

		// Add chunks from related docs
		addedForResult := 0
		for _, relDoc := range relatedDocs {
			if addedForResult >= e.maxExpansion {
				break
			}

			chunks, err := e.store.GetChunksByDoc(relDoc.ID)
			if err != nil {
				continue
			}

			for _, chunk := range chunks {
				if included[chunk.ID] {
					continue
				}
				if addedForResult >= e.maxExpansion {
					break
				}

				// Add with reduced score to indicate it's expanded context
				expanded = append(expanded, domain.ScoredChunk{
					Chunk: chunk,
					Score: r.Score * 0.5, // 50% of original score
				})
				included[chunk.ID] = true
				addedForResult++
			}
		}
	}

	return expanded, nil
}

// findRelatedDocs finds documents related to the given document.
func (e *ContextExpander) findRelatedDocs(doc domain.Document, seenDocs map[string]domain.Document) []domain.Document {
	related := make([]domain.Document, 0)

	// Get all indexed docs
	allDocs, err := e.store.ListDocs()
	if err != nil {
		return related
	}

	baseName := filepath.Base(doc.Path)
	baseName = strings.TrimSuffix(baseName, filepath.Ext(baseName))
	dir := filepath.Dir(doc.Path)

	for _, d := range allDocs {
		if d.ID == doc.ID {
			continue
		}
		if _, seen := seenDocs[d.ID]; seen {
			continue
		}

		// Check if this is a related doc
		if e.isRelated(doc, d, baseName, dir) {
			related = append(related, d)
			seenDocs[d.ID] = d
		}
	}

	return related
}

// isRelated determines if two documents are related.
func (e *ContextExpander) isRelated(original, candidate domain.Document, baseName, dir string) bool {
	candidateBase := filepath.Base(candidate.Path)
	candidateBaseNoExt := strings.TrimSuffix(candidateBase, filepath.Ext(candidateBase))
	candidateDir := filepath.Dir(candidate.Path)

	// Include test files for source files
	if e.includeTests {
		if isTestFile(candidate.Path) && !isTestFile(original.Path) {
			// Check if test is for this source file
			if strings.Contains(candidateBaseNoExt, baseName) ||
				(candidateDir == dir && strings.HasPrefix(candidateBaseNoExt, baseName)) {
				return true
			}
		}
		// Include source files for test files
		if !isTestFile(candidate.Path) && isTestFile(original.Path) {
			if strings.Contains(baseName, candidateBaseNoExt) {
				return true
			}
		}
	}

	// Include files in same package/directory
	if e.includeImports {
		if candidateDir == dir {
			// Same directory - likely related
			// Be more selective: only include if names share a prefix
			if len(baseName) >= 3 && len(candidateBaseNoExt) >= 3 {
				minLen := len(baseName)
				if len(candidateBaseNoExt) < minLen {
					minLen = len(candidateBaseNoExt)
				}
				if minLen >= 4 && baseName[:minLen/2] == candidateBaseNoExt[:minLen/2] {
					return true
				}
			}
		}
	}

	// Include interface definitions (for Go)
	if e.includeInterfaces {
		if original.Lang == "go" && candidate.Lang == "go" {
			// Look for files that might contain interface definitions
			if strings.Contains(candidateBase, "interface") ||
				strings.Contains(candidateBase, "types") ||
				strings.Contains(candidateBase, "contract") {
				return true
			}
		}
	}

	return false
}

// isTestFile checks if a path is a test file.
func isTestFile(path string) bool {
	base := filepath.Base(path)
	ext := filepath.Ext(path)
	baseNoExt := strings.TrimSuffix(base, ext)

	// Common test file patterns
	testPatterns := []string{
		"_test",     // Go: foo_test.go
		".test",     // JS/TS: foo.test.js
		".spec",     // JS/TS: foo.spec.js
		"test_",     // Python: test_foo.py
	}

	for _, p := range testPatterns {
		if strings.HasSuffix(baseNoExt, p) || strings.HasPrefix(baseNoExt, p) || strings.Contains(baseNoExt, p) {
			return true
		}
	}

	// Check for test directories
	return strings.Contains(path, "/test/") || strings.Contains(path, "/tests/") ||
		strings.Contains(path, "/__tests__/")
}

// ExpandWithImports expands results by following import statements.
// This is a more sophisticated expansion that parses imports from chunks.
func (e *ContextExpander) ExpandWithImports(results []domain.ScoredChunk) ([]domain.ScoredChunk, error) {
	// First do basic expansion
	expanded, err := e.Expand(results)
	if err != nil {
		return results, err
	}

	// Then look for imports in the chunks
	imports := extractImports(results)
	if len(imports) == 0 {
		return expanded, nil
	}

	// Track what we've included
	included := make(map[string]bool)
	for _, r := range expanded {
		included[r.Chunk.ID] = true
	}

	// Find docs matching imports
	allDocs, err := e.store.ListDocs()
	if err != nil {
		return expanded, nil
	}

	for _, doc := range allDocs {
		if included[doc.ID] {
			continue
		}

		// Check if this doc matches any import
		for _, imp := range imports {
			if matchesImport(doc.Path, imp) {
				chunks, err := e.store.GetChunksByDoc(doc.ID)
				if err != nil {
					continue
				}

				// Add first chunk from imported file
				if len(chunks) > 0 && !included[chunks[0].ID] {
					expanded = append(expanded, domain.ScoredChunk{
						Chunk: chunks[0],
						Score: 0.3, // Lower score for import-expanded context
					})
					included[chunks[0].ID] = true
				}
				break
			}
		}
	}

	return expanded, nil
}

// extractImports extracts import paths from chunk texts.
func extractImports(chunks []domain.ScoredChunk) []string {
	imports := make([]string, 0)
	seen := make(map[string]bool)

	for _, c := range chunks {
		// Simple import extraction for Go
		lines := strings.Split(c.Chunk.Text, "\n")
		inImportBlock := false
		for _, line := range lines {
			line = strings.TrimSpace(line)

			if strings.HasPrefix(line, "import (") {
				inImportBlock = true
				continue
			}
			if inImportBlock && line == ")" {
				inImportBlock = false
				continue
			}
			if inImportBlock || strings.HasPrefix(line, "import ") {
				// Extract import path
				imp := extractImportPath(line)
				if imp != "" && !seen[imp] {
					imports = append(imports, imp)
					seen[imp] = true
				}
			}
		}
	}

	return imports
}

// extractImportPath extracts the import path from an import line.
func extractImportPath(line string) string {
	// Remove "import" keyword if present
	line = strings.TrimPrefix(line, "import ")
	line = strings.TrimSpace(line)

	// Remove alias if present
	parts := strings.Fields(line)
	if len(parts) == 0 {
		return ""
	}
	importPart := parts[len(parts)-1]

	// Remove quotes
	importPart = strings.Trim(importPart, "\"'`")

	// Skip standard library
	if !strings.Contains(importPart, "/") {
		return ""
	}
	if strings.HasPrefix(importPart, "fmt") || strings.HasPrefix(importPart, "os") ||
		strings.HasPrefix(importPart, "io") || strings.HasPrefix(importPart, "net") ||
		strings.HasPrefix(importPart, "encoding") || strings.HasPrefix(importPart, "sync") {
		return ""
	}

	return importPart
}

// matchesImport checks if a file path matches an import path.
func matchesImport(filePath, importPath string) bool {
	// Normalize paths
	filePath = strings.ReplaceAll(filePath, "\\", "/")
	importPath = strings.ReplaceAll(importPath, "\\", "/")

	// Check if file path ends with import path
	return strings.HasSuffix(filePath, importPath) ||
		strings.Contains(filePath, importPath+"/")
}
