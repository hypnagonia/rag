package usecase

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"path/filepath"
	"time"

	"rag/internal/adapter/analyzer"
	"rag/internal/adapter/chunker"
	"rag/internal/adapter/fs"
	"rag/internal/adapter/store"
	"rag/internal/domain"
)

// IndexUseCase handles file indexing operations.
type IndexUseCase struct {
	store     *store.BoltStore
	walker    *fs.Walker
	chunker   *chunker.LineChunker
	tokenizer *analyzer.Tokenizer
}

// NewIndexUseCase creates a new index use case.
func NewIndexUseCase(
	store *store.BoltStore,
	walker *fs.Walker,
	chunker *chunker.LineChunker,
	tokenizer *analyzer.Tokenizer,
) *IndexUseCase {
	return &IndexUseCase{
		store:     store,
		walker:    walker,
		chunker:   chunker,
		tokenizer: tokenizer,
	}
}

// IndexResult contains the results of an indexing operation.
type IndexResult struct {
	FilesIndexed   int
	FilesSkipped   int
	FilesDeleted   int
	ChunksCreated  int
	Errors         []string
}

// Index indexes files in the given directory.
func (u *IndexUseCase) Index(root string) (*IndexResult, error) {
	result := &IndexResult{}

	// Walk the directory
	files, err := u.walker.Walk(root)
	if err != nil {
		return nil, fmt.Errorf("failed to walk directory: %w", err)
	}

	// Get existing documents for incremental update
	existingDocs, err := u.store.ListDocs()
	if err != nil {
		return nil, fmt.Errorf("failed to list existing docs: %w", err)
	}

	existingMap := make(map[string]domain.Document)
	for _, doc := range existingDocs {
		existingMap[doc.Path] = doc
	}

	// Track which files still exist
	seenPaths := make(map[string]bool)

	// Process each file
	totalChunkLen := 0
	totalChunks := 0

	for _, file := range files {
		seenPaths[file.Path] = true

		// Check if file needs re-indexing
		if existing, ok := existingMap[file.Path]; ok {
			if existing.ModTime.Unix() >= file.ModTime {
				result.FilesSkipped++
				// Count existing chunks for stats
				chunks, _ := u.store.GetChunksByDoc(existing.ID)
				for _, c := range chunks {
					totalChunks++
					totalChunkLen += len(c.Tokens)
				}
				continue
			}
			// File modified, delete old data
			if err := u.deleteDocument(existing.ID); err != nil {
				result.Errors = append(result.Errors, fmt.Sprintf("failed to delete old data for %s: %v", file.Path, err))
			}
		}

		// Index the file
		if err := u.indexFile(file, &totalChunks, &totalChunkLen, result); err != nil {
			result.Errors = append(result.Errors, fmt.Sprintf("failed to index %s: %v", file.Path, err))
			continue
		}
		result.FilesIndexed++
	}

	// Delete documents for files that no longer exist
	for path, doc := range existingMap {
		if !seenPaths[path] {
			if err := u.deleteDocument(doc.ID); err != nil {
				result.Errors = append(result.Errors, fmt.Sprintf("failed to delete %s: %v", path, err))
			} else {
				result.FilesDeleted++
			}
		}
	}

	// Update stats
	avgChunkLen := 0.0
	if totalChunks > 0 {
		avgChunkLen = float64(totalChunkLen) / float64(totalChunks)
	}

	stats := domain.Stats{
		TotalDocs:   result.FilesIndexed + result.FilesSkipped,
		TotalChunks: totalChunks,
		AvgChunkLen: avgChunkLen,
	}
	if err := u.store.UpdateStats(stats); err != nil {
		return nil, fmt.Errorf("failed to update stats: %w", err)
	}

	result.ChunksCreated = totalChunks

	return result, nil
}

// indexFile indexes a single file.
func (u *IndexUseCase) indexFile(file fs.FileInfo, totalChunks, totalChunkLen *int, result *IndexResult) error {
	// Read file content
	content, err := fs.ReadFile(file.Path)
	if err != nil {
		return fmt.Errorf("failed to read file: %w", err)
	}

	// Create document
	docID := generateDocID(file.Path)
	doc := domain.Document{
		ID:      docID,
		Path:    file.Path,
		ModTime: time.Unix(file.ModTime, 0),
		Lang:    detectLanguage(file.Path),
	}

	// Store document
	if err := u.store.PutDoc(doc); err != nil {
		return fmt.Errorf("failed to store document: %w", err)
	}

	// Chunk the content
	chunks, err := u.chunker.Chunk(doc, content)
	if err != nil {
		return fmt.Errorf("failed to chunk content: %w", err)
	}

	// Store chunks and build postings
	for _, chunk := range chunks {
		if err := u.store.PutChunk(chunk); err != nil {
			return fmt.Errorf("failed to store chunk: %w", err)
		}

		// Build term frequency map and store postings
		tf := make(map[string]int)
		for _, token := range chunk.Tokens {
			tf[token]++
		}

		for term, count := range tf {
			if err := u.store.PutPosting(term, chunk.ID, count); err != nil {
				return fmt.Errorf("failed to store posting: %w", err)
			}
		}

		*totalChunks++
		*totalChunkLen += len(chunk.Tokens)
	}

	return nil
}

// deleteDocument deletes a document and all its associated data.
func (u *IndexUseCase) deleteDocument(docID string) error {
	// Get chunks for this document
	chunks, err := u.store.GetChunksByDoc(docID)
	if err != nil {
		return err
	}

	// Delete postings for each chunk
	for _, chunk := range chunks {
		uniqueTerms := make(map[string]struct{})
		for _, token := range chunk.Tokens {
			uniqueTerms[token] = struct{}{}
		}
		terms := make([]string, 0, len(uniqueTerms))
		for term := range uniqueTerms {
			terms = append(terms, term)
		}
		if err := u.store.DeletePostings(chunk.ID, terms); err != nil {
			return err
		}
	}

	// Delete chunks
	if err := u.store.DeleteChunksByDoc(docID); err != nil {
		return err
	}

	// Delete document
	return u.store.DeleteDoc(docID)
}

// generateDocID creates a unique ID for a document based on its path.
func generateDocID(path string) string {
	hash := sha256.Sum256([]byte(path))
	return hex.EncodeToString(hash[:8])
}

// detectLanguage detects the programming language based on file extension.
func detectLanguage(path string) string {
	ext := filepath.Ext(path)
	switch ext {
	case ".go":
		return "go"
	case ".py":
		return "python"
	case ".js":
		return "javascript"
	case ".ts":
		return "typescript"
	case ".java":
		return "java"
	case ".c", ".h":
		return "c"
	case ".cpp", ".cc", ".hpp":
		return "cpp"
	case ".rs":
		return "rust"
	case ".rb":
		return "ruby"
	case ".php":
		return "php"
	case ".md":
		return "markdown"
	case ".txt":
		return "text"
	case ".json":
		return "json"
	case ".yaml", ".yml":
		return "yaml"
	case ".xml":
		return "xml"
	case ".html":
		return "html"
	case ".css":
		return "css"
	case ".sql":
		return "sql"
	case ".sh", ".bash":
		return "shell"
	default:
		return "unknown"
	}
}
