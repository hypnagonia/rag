package chunker

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"

	"rag/internal/adapter/analyzer"
	"rag/internal/domain"
)

type CompositeChunker struct {
	parsers   map[string]LanguageParser
	fallback  *LineChunker
	tokenizer *analyzer.Tokenizer
	maxTokens int
	overlap   int
	useAST    bool
}

func NewCompositeChunker(maxTokens, overlap int, tokenizer *analyzer.Tokenizer, useAST bool) *CompositeChunker {
	parsers := make(map[string]LanguageParser)

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

func (c *CompositeChunker) Chunk(doc domain.Document, content string) ([]domain.Chunk, error) {

	if !c.useAST {
		return c.fallback.Chunk(doc, content)
	}

	parser, hasParser := c.parsers[doc.Lang]
	if !hasParser {
		return c.fallback.Chunk(doc, content)
	}

	units, err := parser.Parse(content)
	if err != nil {

		return c.fallback.Chunk(doc, content)
	}

	if len(units) == 0 {
		return c.fallback.Chunk(doc, content)
	}

	return c.unitsToChunks(doc, units, content)
}

func (c *CompositeChunker) unitsToChunks(doc domain.Document, units []CodeUnit, content string) ([]domain.Chunk, error) {
	var chunks []domain.Chunk

	for _, unit := range units {

		tokens := c.tokenizer.CountTokens(unit.Content)

		if tokens <= c.maxTokens {

			chunk := c.createChunk(doc, unit)
			chunks = append(chunks, chunk)
		} else {

			subChunks := c.splitLargeUnit(doc, unit)
			chunks = append(chunks, subChunks...)
		}
	}

	return chunks, nil
}

func (c *CompositeChunker) createChunk(doc domain.Document, unit CodeUnit) domain.Chunk {
	tokens := c.tokenizer.Tokenize(unit.Content)

	text := unit.Content
	if unit.DocString != "" && len(unit.DocString) < 500 {

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

func (c *CompositeChunker) splitLargeUnit(doc domain.Document, unit CodeUnit) []domain.Chunk {
	var chunks []domain.Chunk

	header := ""

	if unit.Signature != "" {
		header = unit.Signature

		if unit.Type == "function" || unit.Type == "method" || unit.Type == "struct" || unit.Type == "interface" {
			header += " {"
		}
	}

	lines := splitIntoLines(unit.Content)

	startIdx := 0
	if unit.Signature != "" {

		for i, line := range lines {
			if containsBrace(line) {
				startIdx = i + 1
				break
			}
		}
	}

	if startIdx >= len(lines) {

		return []domain.Chunk{c.createChunk(doc, unit)}
	}

	bodyLines := lines[startIdx:]
	currentStart := 0
	chunkNum := 0

	for currentStart < len(bodyLines) {

		headerTokens := c.tokenizer.CountTokens(header)
		availableTokens := c.maxTokens - headerTokens - 10
		if availableTokens < 50 {
			availableTokens = 50
		}

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

		if currentEnd == currentStart {
			currentEnd = currentStart + 1
		}

		var chunkContent string
		if header != "" {
			chunkContent = header + "\n"
		}
		for i := currentStart; i < currentEnd && i < len(bodyLines); i++ {
			chunkContent += bodyLines[i] + "\n"
		}

		if currentEnd < len(bodyLines) {
			chunkContent += "// ... continued"
		} else if unit.Type == "function" || unit.Type == "method" {
			chunkContent += "}"
		}

		tokens := c.tokenizer.Tokenize(chunkContent)

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

func generateASTChunkID(docID, unitType, unitName string, startLine int) string {
	data := fmt.Sprintf("%s:%s:%s:%d", docID, unitType, unitName, startLine)
	hash := sha256.Sum256([]byte(data))
	return hex.EncodeToString(hash[:8])
}

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

func containsBrace(line string) bool {
	for _, c := range line {
		if c == '{' {
			return true
		}
	}
	return false
}
