package usecase

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"rag/internal/adapter/analyzer"
	"rag/internal/adapter/fs"
	"rag/internal/adapter/store"
	"rag/internal/domain"
	"rag/internal/port"
)

// IndexUseCase handles file indexing operations.
type IndexUseCase struct {
	store     *store.BoltStore
	walker    *fs.Walker
	chunkSvc  port.Chunker
	tokenizer *analyzer.Tokenizer
	workers   int
}

// NewIndexUseCase creates a new index use case.
func NewIndexUseCase(
	store *store.BoltStore,
	walker *fs.Walker,
	chunkSvc port.Chunker,
	tokenizer *analyzer.Tokenizer,
) *IndexUseCase {
	workers := runtime.NumCPU()
	if workers < 2 {
		workers = 2
	}
	return &IndexUseCase{
		store:     store,
		walker:    walker,
		chunkSvc:  chunkSvc,
		tokenizer: tokenizer,
		workers:   workers,
	}
}

// IndexResult contains the results of an indexing operation.
type IndexResult struct {
	FilesIndexed  int
	FilesSkipped  int
	FilesDeleted  int
	ChunksCreated int
	Errors        []string
}

// ProgressCallback is called to report indexing progress.
type ProgressCallback func(processed, total int, currentFile string)

// Index indexes files in the given directory.
func (u *IndexUseCase) Index(root string, progress ProgressCallback) (*IndexResult, error) {
	result := &IndexResult{}

	files, err := u.walker.Walk(root)
	if err != nil {
		return nil, fmt.Errorf("failed to walk directory: %w", err)
	}

	existingDocs, err := u.store.ListDocs()
	if err != nil {
		return nil, fmt.Errorf("failed to list existing docs: %w", err)
	}

	existingMap := make(map[string]domain.Document)
	for _, doc := range existingDocs {
		existingMap[doc.Path] = doc
	}

	seenPaths := make(map[string]bool)

	// Separate files into those needing indexing and those to skip
	var filesToIndex []fs.FileInfo
	var skippedDocs []domain.Document

	for _, file := range files {
		seenPaths[file.Path] = true

		if existing, ok := existingMap[file.Path]; ok {
			if existing.ModTime.Unix() >= file.ModTime {
				result.FilesSkipped++
				skippedDocs = append(skippedDocs, existing)
				continue
			}

			if err := u.deleteDocument(existing.ID); err != nil {
				result.Errors = append(result.Errors, fmt.Sprintf("failed to delete old data for %s: %v", file.Path, err))
			}
		}
		filesToIndex = append(filesToIndex, file)
	}

	for path, doc := range existingMap {
		if !seenPaths[path] {
			if err := u.deleteDocument(doc.ID); err != nil {
				result.Errors = append(result.Errors, fmt.Sprintf("failed to delete %s: %v", path, err))
			} else {
				result.FilesDeleted++
			}
		}
	}

	// Count existing chunks for stats
	var existingChunkLen int64
	var existingChunkCount int64
	for _, doc := range skippedDocs {
		chunks, _ := u.store.GetChunksByDoc(doc.ID)
		for _, c := range chunks {
			atomic.AddInt64(&existingChunkCount, 1)
			atomic.AddInt64(&existingChunkLen, int64(len(c.Tokens)))
		}
	}

	if len(filesToIndex) > 0 {
		indexed, chunkCount, chunkLen, errors := u.indexFilesParallel(filesToIndex, progress)
		result.FilesIndexed = indexed
		result.Errors = append(result.Errors, errors...)
		existingChunkCount += int64(chunkCount)
		existingChunkLen += int64(chunkLen)
	}

	totalChunks := int(existingChunkCount)
	avgChunkLen := 0.0
	if totalChunks > 0 {
		avgChunkLen = float64(existingChunkLen) / float64(totalChunks)
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

// processedFile holds the result of processing a single file.
type processedFile struct {
	file     store.IndexedFile
	err      error
	path     string
	chunkLen int
}

// indexFilesParallel indexes files using parallel workers and batch writes.
func (u *IndexUseCase) indexFilesParallel(files []fs.FileInfo, progress ProgressCallback) (indexed, chunkCount, chunkLen int, errors []string) {
	totalFiles := len(files)
	var processed int64

	jobs := make(chan fs.FileInfo, len(files))
	results := make(chan processedFile, len(files))

	// Start workers
	var wg sync.WaitGroup
	for i := 0; i < u.workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for file := range jobs {
				result := u.processFile(file)
				results <- result

				p := int(atomic.AddInt64(&processed, 1))
				if progress != nil {
					progress(p, totalFiles, file.Path)
				}
			}
		}()
	}

	go func() {
		for _, file := range files {
			jobs <- file
		}
		close(jobs)
	}()

	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect results and batch write
	const batchSize = 50
	batch := make([]store.IndexedFile, 0, batchSize)

	for result := range results {
		if result.err != nil {
			errors = append(errors, fmt.Sprintf("failed to index %s: %v", result.path, result.err))
			continue
		}

		batch = append(batch, result.file)
		indexed++
		chunkCount += len(result.file.Chunks)
		chunkLen += result.chunkLen

		if len(batch) >= batchSize {
			if err := u.store.BatchIndex(batch); err != nil {
				errors = append(errors, fmt.Sprintf("batch write failed: %v", err))
			}
			batch = batch[:0]
		}
	}

	if len(batch) > 0 {
		if err := u.store.BatchIndex(batch); err != nil {
			errors = append(errors, fmt.Sprintf("batch write failed: %v", err))
		}
	}

	return
}

// processFile processes a single file and returns the indexed data.
func (u *IndexUseCase) processFile(file fs.FileInfo) processedFile {
	result := processedFile{path: file.Path}

	content, err := fs.ReadFile(file.Path)
	if err != nil {
		result.err = fmt.Errorf("failed to read file: %w", err)
		return result
	}

	docID := generateDocID(file.Path)
	doc := domain.Document{
		ID:      docID,
		Path:    file.Path,
		ModTime: time.Unix(file.ModTime, 0),
		Lang:    detectLanguage(file.Path),
	}

	chunks, err := u.chunkSvc.Chunk(doc, content)
	if err != nil {
		result.err = fmt.Errorf("failed to chunk content: %w", err)
		return result
	}

	postings := make(map[string]map[string]int)
	chunkLen := 0

	for _, chunk := range chunks {
		tf := make(map[string]int)
		for _, token := range chunk.Tokens {
			tf[token]++
		}
		for term, count := range tf {
			if postings[term] == nil {
				postings[term] = make(map[string]int)
			}
			postings[term][chunk.ID] = count
		}
		chunkLen += len(chunk.Tokens)
	}

	result.file = store.IndexedFile{
		Doc:      doc,
		Chunks:   chunks,
		Postings: postings,
	}
	result.chunkLen = chunkLen

	return result
}

// deleteDocument deletes a document and all its associated data.
func (u *IndexUseCase) deleteDocument(docID string) error {

	chunks, err := u.store.GetChunksByDoc(docID)
	if err != nil {
		return err
	}

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

	if err := u.store.DeleteChunksByDoc(docID); err != nil {
		return err
	}

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
