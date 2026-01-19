package chunker

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"

	"rag/internal/adapter/analyzer"
	"rag/internal/domain"
)

// CompositeChunker routes to language-specific parsers or falls back to line-based chunking.
type CompositeChunker struct {
	parsers     map[string]LanguageParser
	fallback    *LineChunker
	tokenizer   *analyzer.Tokenizer
	maxTokens   int
	overlap     int
	useAST      bool // Can be disabled via config
}

// NewCompositeChunker creates a new composite chunker.
func NewCompositeChunker(maxTokens, overlap int, tokenizer *analyzer.Tokenizer, useAST bool) *CompositeChunker {
	parsers := make(map[string]LanguageParser)

	// Register language parsers
	goParser := NewGoParser()
	parsers[goParser.Language()] = goParser

	return &CompositeChunker{
		parsers:   parsers,
		fallback:  NewLineChunker(maxTokens, overlap, tokenizer),
		tokenizer: tokenizer,
		maxTokens: maxTokens,
		overlap:   overlap,
		useAST:    useAST,
	}
}

// Chunk splits a document's content into chunks.
func (c *CompositeChunker) Chunk(doc domain.Document, content string) ([]domain.Chunk, error) {
	// If AST chunking is disabled or no parser available, use fallback
	if !c.useAST {
		return c.fallback.Chunk(doc, content)
	}

	parser, hasParser := c.parsers[doc.Lang]
	if !hasParser {
		return c.fallback.Chunk(doc, content)
	}

	// Try AST parsing
	units, err := parser.Parse(content)
	if err != nil {
		// Parse failed, fall back to line-based chunking
		return c.fallback.Chunk(doc, content)
	}

	if len(units) == 0 {
		return c.fallback.Chunk(doc, content)
	}

	// Convert code units to chunks
	return c.unitsToChunks(doc, units, content)
}

// unitsToChunks converts code units to domain chunks.
func (c *CompositeChunker) unitsToChunks(doc domain.Document, units []CodeUnit, content string) ([]domain.Chunk, error) {
	var chunks []domain.Chunk

	for _, unit := range units {
		// Check if unit fits in a single chunk
		tokens := c.tokenizer.CountTokens(unit.Content)

		if tokens <= c.maxTokens {
			// Unit fits in a single chunk
			chunk := c.createChunk(doc, unit)
			chunks = append(chunks, chunk)
		} else {
			// Unit is too large, split it
			subChunks := c.splitLargeUnit(doc, unit)
			chunks = append(chunks, subChunks...)
		}
	}

	return chunks, nil
}

// createChunk creates a domain.Chunk from a CodeUnit.
func (c *CompositeChunker) createChunk(doc domain.Document, unit CodeUnit) domain.Chunk {
	tokens := c.tokenizer.Tokenize(unit.Content)

	// Include metadata in the chunk for better retrieval
	// Prepend signature/doc to help with matching
	text := unit.Content
	if unit.DocString != "" && len(unit.DocString) < 500 {
		// Doc string is often valuable context
		text = "// " + unit.DocString + "\n" + text
	}

	return domain.Chunk{
		ID:        generateASTChunkID(doc.ID, unit.Type, unit.Name, unit.StartLine),
		DocID:     doc.ID,
		StartLine: unit.StartLine,
		EndLine:   unit.EndLine,
		Tokens:    tokens,
		Text:      text,
	}
}

// splitLargeUnit splits a large code unit into smaller chunks.
func (c *CompositeChunker) splitLargeUnit(doc domain.Document, unit CodeUnit) []domain.Chunk {
	var chunks []domain.Chunk

	// Strategy: Keep the signature/header in each chunk for context
	header := ""

	if unit.Signature != "" {
		header = unit.Signature
		// Add opening brace if it's a function/method/struct/interface
		if unit.Type == "function" || unit.Type == "method" || unit.Type == "struct" || unit.Type == "interface" {
			header += " {"
		}
	}

	// Use line-based splitting for the content
	lines := splitIntoLines(unit.Content)

	// Skip the first few lines that contain the signature (we'll add it back)
	startIdx := 0
	if unit.Signature != "" {
		// Find where the body starts (after the opening brace)
		for i, line := range lines {
			if containsBrace(line) {
				startIdx = i + 1
				break
			}
		}
	}

	if startIdx >= len(lines) {
		// Signature-only, no body to split
		return []domain.Chunk{c.createChunk(doc, unit)}
	}

	bodyLines := lines[startIdx:]
	currentStart := 0
	chunkNum := 0

	for currentStart < len(bodyLines) {
		// Calculate how much space we have for body content
		headerTokens := c.tokenizer.CountTokens(header)
		availableTokens := c.maxTokens - headerTokens - 10 // Buffer for closing brace etc.
		if availableTokens < 50 {
			availableTokens = 50 // Minimum body size
		}

		// Find end of this chunk
		currentEnd := currentStart
		currentTokens := 0

		for currentEnd < len(bodyLines) {
			lineTokens := c.tokenizer.CountTokens(bodyLines[currentEnd])
			if currentTokens > 0 && currentTokens+lineTokens > availableTokens {
				break
			}
			currentTokens += lineTokens
			currentEnd++
		}

		// Ensure progress
		if currentEnd == currentStart {
			currentEnd = currentStart + 1
		}

		// Build chunk content
		var chunkContent string
		if header != "" {
			chunkContent = header + "\n"
		}
		for i := currentStart; i < currentEnd && i < len(bodyLines); i++ {
			chunkContent += bodyLines[i] + "\n"
		}
		// Add continuation marker if not the last chunk
		if currentEnd < len(bodyLines) {
			chunkContent += "// ... continued"
		} else if unit.Type == "function" || unit.Type == "method" {
			chunkContent += "}"
		}

		tokens := c.tokenizer.Tokenize(chunkContent)

		// Calculate line numbers
		actualStartLine := unit.StartLine + startIdx + currentStart
		actualEndLine := unit.StartLine + startIdx + currentEnd - 1
		if actualEndLine >= unit.EndLine {
			actualEndLine = unit.EndLine
		}

		chunk := domain.Chunk{
			ID:        generateASTChunkID(doc.ID, unit.Type, unit.Name, actualStartLine) + fmt.Sprintf("_%d", chunkNum),
			DocID:     doc.ID,
			StartLine: actualStartLine,
			EndLine:   actualEndLine,
			Tokens:    tokens,
			Text:      chunkContent,
		}
		chunks = append(chunks, chunk)

		// Move to next chunk with overlap
		overlapLines := c.calculateOverlapForBody(bodyLines, currentStart, currentEnd)
		nextStart := currentEnd - overlapLines
		if nextStart <= currentStart {
			nextStart = currentStart + 1
		}
		currentStart = nextStart
		chunkNum++
	}

	return chunks
}

// calculateOverlapForBody calculates overlap lines for body content.
func (c *CompositeChunker) calculateOverlapForBody(lines []string, start, end int) int {
	if c.overlap == 0 {
		return 0
	}

	overlapLines := 0
	tokens := 0

	for i := end - 1; i >= start && tokens < c.overlap; i-- {
		tokens += c.tokenizer.CountTokens(lines[i])
		overlapLines++
	}

	return overlapLines
}

// generateASTChunkID creates a unique ID for an AST-based chunk.
func generateASTChunkID(docID, unitType, unitName string, startLine int) string {
	data := fmt.Sprintf("%s:%s:%s:%d", docID, unitType, unitName, startLine)
	hash := sha256.Sum256([]byte(data))
	return hex.EncodeToString(hash[:8])
}

// splitIntoLines splits content into lines.
func splitIntoLines(content string) []string {
	var lines []string
	start := 0
	for i := 0; i < len(content); i++ {
		if content[i] == '\n' {
			lines = append(lines, content[start:i])
			start = i + 1
		}
	}
	if start < len(content) {
		lines = append(lines, content[start:])
	}
	return lines
}

// containsBrace checks if a line contains an opening brace.
func containsBrace(line string) bool {
	for _, c := range line {
		if c == '{' {
			return true
		}
	}
	return false
}
