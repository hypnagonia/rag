package analyzer

import (
	"regexp"
	"strings"
)

// CommentBlock represents an extracted comment from source code.
type CommentBlock struct {
	Text      string
	StartLine int
	EndLine   int
	Type      string // "line", "block", "doc"
}

// CommentExtractor extracts comments from source code based on language.
type CommentExtractor struct {
	patterns map[string]languageCommentPatterns
}

type languageCommentPatterns struct {
	lineComment  *regexp.Regexp
	blockStart   *regexp.Regexp
	blockEnd     *regexp.Regexp
	docComment   *regexp.Regexp
}

// NewCommentExtractor creates a new comment extractor.
func NewCommentExtractor() *CommentExtractor {
	return &CommentExtractor{
		patterns: map[string]languageCommentPatterns{
			"go": {
				lineComment: regexp.MustCompile(`^\s*//(.*)$`),
				blockStart:  regexp.MustCompile(`/\*`),
				blockEnd:    regexp.MustCompile(`\*/`),
				docComment:  regexp.MustCompile(`^\s*//\s*(TODO|FIXME|NOTE|BUG|HACK|XXX|REVIEW|OPTIMIZE):?\s*(.*)$`),
			},
			"python": {
				lineComment: regexp.MustCompile(`^\s*#(.*)$`),
				blockStart:  regexp.MustCompile(`^\s*(?:'''|""")(.*)$`),
				blockEnd:    regexp.MustCompile(`(?:'''|""")$`),
				docComment:  regexp.MustCompile(`^\s*#\s*(TODO|FIXME|NOTE|BUG|HACK|XXX|REVIEW|OPTIMIZE):?\s*(.*)$`),
			},
			"javascript": {
				lineComment: regexp.MustCompile(`^\s*//(.*)$`),
				blockStart:  regexp.MustCompile(`/\*`),
				blockEnd:    regexp.MustCompile(`\*/`),
				docComment:  regexp.MustCompile(`^\s*//\s*(TODO|FIXME|NOTE|BUG|HACK|XXX|REVIEW|OPTIMIZE):?\s*(.*)$`),
			},
			"typescript": {
				lineComment: regexp.MustCompile(`^\s*//(.*)$`),
				blockStart:  regexp.MustCompile(`/\*`),
				blockEnd:    regexp.MustCompile(`\*/`),
				docComment:  regexp.MustCompile(`^\s*//\s*(TODO|FIXME|NOTE|BUG|HACK|XXX|REVIEW|OPTIMIZE):?\s*(.*)$`),
			},
			"java": {
				lineComment: regexp.MustCompile(`^\s*//(.*)$`),
				blockStart:  regexp.MustCompile(`/\*`),
				blockEnd:    regexp.MustCompile(`\*/`),
				docComment:  regexp.MustCompile(`^\s*//\s*(TODO|FIXME|NOTE|BUG|HACK|XXX|REVIEW|OPTIMIZE):?\s*(.*)$`),
			},
			"c": {
				lineComment: regexp.MustCompile(`^\s*//(.*)$`),
				blockStart:  regexp.MustCompile(`/\*`),
				blockEnd:    regexp.MustCompile(`\*/`),
				docComment:  regexp.MustCompile(`^\s*//\s*(TODO|FIXME|NOTE|BUG|HACK|XXX|REVIEW|OPTIMIZE):?\s*(.*)$`),
			},
			"cpp": {
				lineComment: regexp.MustCompile(`^\s*//(.*)$`),
				blockStart:  regexp.MustCompile(`/\*`),
				blockEnd:    regexp.MustCompile(`\*/`),
				docComment:  regexp.MustCompile(`^\s*//\s*(TODO|FIXME|NOTE|BUG|HACK|XXX|REVIEW|OPTIMIZE):?\s*(.*)$`),
			},
			"rust": {
				lineComment: regexp.MustCompile(`^\s*//(.*)$`),
				blockStart:  regexp.MustCompile(`/\*`),
				blockEnd:    regexp.MustCompile(`\*/`),
				docComment:  regexp.MustCompile(`^\s*///?\s*(.*)$`), // Rust doc comments
			},
			"ruby": {
				lineComment: regexp.MustCompile(`^\s*#(.*)$`),
				blockStart:  regexp.MustCompile(`^=begin`),
				blockEnd:    regexp.MustCompile(`^=end`),
				docComment:  regexp.MustCompile(`^\s*#\s*(TODO|FIXME|NOTE|BUG|HACK|XXX|REVIEW|OPTIMIZE):?\s*(.*)$`),
			},
			"shell": {
				lineComment: regexp.MustCompile(`^\s*#(.*)$`),
				blockStart:  nil,
				blockEnd:    nil,
				docComment:  regexp.MustCompile(`^\s*#\s*(TODO|FIXME|NOTE|BUG|HACK|XXX|REVIEW|OPTIMIZE):?\s*(.*)$`),
			},
		},
	}
}

// Extract extracts comments from source code.
func (e *CommentExtractor) Extract(content string, lang string) []CommentBlock {
	patterns, ok := e.patterns[lang]
	if !ok {
		// Default to C-style comments
		patterns = e.patterns["go"]
	}

	lines := strings.Split(content, "\n")
	var comments []CommentBlock

	inBlockComment := false
	blockStartLine := 0
	var blockContent strings.Builder

	for lineNum, line := range lines {
		lineNumber := lineNum + 1 // 1-indexed

		// Handle block comments
		if inBlockComment {
			blockContent.WriteString(line)
			blockContent.WriteString("\n")
			if patterns.blockEnd != nil && patterns.blockEnd.MatchString(line) {
				comments = append(comments, CommentBlock{
					Text:      strings.TrimSpace(blockContent.String()),
					StartLine: blockStartLine,
					EndLine:   lineNumber,
					Type:      "block",
				})
				inBlockComment = false
				blockContent.Reset()
			}
			continue
		}

		// Check for block comment start
		if patterns.blockStart != nil && patterns.blockStart.MatchString(line) {
			// Check if it's a single-line block comment
			if patterns.blockEnd != nil && patterns.blockEnd.MatchString(line) {
				// Single line block comment like /* comment */
				text := extractBetween(line, patterns.blockStart, patterns.blockEnd)
				if text != "" {
					comments = append(comments, CommentBlock{
						Text:      text,
						StartLine: lineNumber,
						EndLine:   lineNumber,
						Type:      "block",
					})
				}
			} else {
				// Start of multi-line block comment
				inBlockComment = true
				blockStartLine = lineNumber
				blockContent.WriteString(line)
				blockContent.WriteString("\n")
			}
			continue
		}

		// Check for doc comments (special annotations like TODO, FIXME)
		if patterns.docComment != nil {
			if matches := patterns.docComment.FindStringSubmatch(line); len(matches) > 0 {
				comments = append(comments, CommentBlock{
					Text:      strings.TrimSpace(line),
					StartLine: lineNumber,
					EndLine:   lineNumber,
					Type:      "doc",
				})
				continue
			}
		}

		// Check for line comments
		if patterns.lineComment != nil {
			if matches := patterns.lineComment.FindStringSubmatch(line); len(matches) > 1 {
				text := strings.TrimSpace(matches[1])
				if text != "" {
					comments = append(comments, CommentBlock{
						Text:      text,
						StartLine: lineNumber,
						EndLine:   lineNumber,
						Type:      "line",
					})
				}
			}
		}
	}

	// Merge consecutive line comments into single blocks
	return mergeConsecutiveComments(comments)
}

// extractBetween extracts text between start and end patterns.
func extractBetween(line string, start, end *regexp.Regexp) string {
	startIdx := start.FindStringIndex(line)
	endIdx := end.FindStringIndex(line)
	if startIdx == nil || endIdx == nil || startIdx[1] >= endIdx[0] {
		return ""
	}
	return strings.TrimSpace(line[startIdx[1]:endIdx[0]])
}

// mergeConsecutiveComments merges consecutive line comments into single blocks.
func mergeConsecutiveComments(comments []CommentBlock) []CommentBlock {
	if len(comments) <= 1 {
		return comments
	}

	var merged []CommentBlock
	i := 0

	for i < len(comments) {
		current := comments[i]

		// Don't merge block or doc comments
		if current.Type != "line" {
			merged = append(merged, current)
			i++
			continue
		}

		// Try to merge consecutive line comments
		var textBuilder strings.Builder
		textBuilder.WriteString(current.Text)
		endLine := current.EndLine

		j := i + 1
		for j < len(comments) {
			next := comments[j]
			// Only merge line comments that are on consecutive lines
			if next.Type == "line" && next.StartLine == endLine+1 {
				textBuilder.WriteString("\n")
				textBuilder.WriteString(next.Text)
				endLine = next.EndLine
				j++
			} else {
				break
			}
		}

		merged = append(merged, CommentBlock{
			Text:      textBuilder.String(),
			StartLine: current.StartLine,
			EndLine:   endLine,
			Type:      "line",
		})
		i = j
	}

	return merged
}

// ExtractCommentTokens extracts and tokenizes comments from source code.
// Returns a map of token -> frequency for comment-specific terms.
func (e *CommentExtractor) ExtractCommentTokens(content string, lang string, tokenizer *Tokenizer) map[string]int {
	comments := e.Extract(content, lang)
	tokenFreq := make(map[string]int)

	for _, comment := range comments {
		tokens := tokenizer.Tokenize(comment.Text)
		for _, token := range tokens {
			tokenFreq[token]++
		}
	}

	return tokenFreq
}
