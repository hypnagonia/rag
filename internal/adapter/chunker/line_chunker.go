package chunker

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"

	"rag/internal/adapter/analyzer"
	"rag/internal/domain"
)

type LineChunker struct {
	maxTokens int
	overlap   int
	tokenizer *analyzer.Tokenizer
}

func NewLineChunker(maxTokens, overlap int, tokenizer *analyzer.Tokenizer) *LineChunker {
	return &LineChunker{
		maxTokens: maxTokens,
		overlap:   overlap,
		tokenizer: tokenizer,
	}
}

func (c *LineChunker) Chunk(doc domain.Document, content string) ([]domain.Chunk, error) {
	lines := strings.Split(content, "\n")
	if len(lines) == 0 {
		return nil, nil
	}

	var chunks []domain.Chunk
	startLine := 0

	for startLine < len(lines) {

		endLine := startLine
		currentTokens := 0
		var chunkText strings.Builder

		for endLine < len(lines) {
			lineText := lines[endLine]
			lineTokens := c.tokenizer.CountTokens(lineText)

			if currentTokens > 0 && currentTokens+lineTokens > c.maxTokens {
				break
			}

			if chunkText.Len() > 0 {
				chunkText.WriteString("\n")
			}
			chunkText.WriteString(lineText)
			currentTokens += lineTokens
			endLine++
		}

		if endLine == startLine {
			if chunkText.Len() > 0 {
				chunkText.WriteString("\n")
			}
			chunkText.WriteString(lines[endLine])
			endLine++
		}

		text := chunkText.String()
		tokens := c.tokenizer.Tokenize(text)

		chunk := domain.Chunk{
			ID:        generateChunkID(doc.ID, startLine, endLine),
			DocID:     doc.ID,
			StartLine: startLine + 1,
			EndLine:   endLine,
			Tokens:    tokens,
			Text:      text,
		}
		chunks = append(chunks, chunk)

		overlapLines := c.calculateOverlapLines(lines, startLine, endLine)
		newStart := endLine - overlapLines

		if newStart <= startLine {
			newStart = startLine + 1
		}
		if newStart >= endLine {
			newStart = endLine
		}
		startLine = newStart
	}

	return chunks, nil
}

func (c *LineChunker) calculateOverlapLines(lines []string, start, end int) int {
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

func generateChunkID(docID string, startLine, endLine int) string {
	data := fmt.Sprintf("%s:%d-%d", docID, startLine, endLine)
	hash := sha256.Sum256([]byte(data))
	return hex.EncodeToString(hash[:8])
}
