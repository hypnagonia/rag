package analyzer

import (
	"go/ast"
	"go/parser"
	"go/token"
	"strings"

	"rag/internal/domain"
)

// CallGraphBuilder builds call relationships between functions.
type CallGraphBuilder struct {
	symbols     map[string]domain.Symbol // name -> symbol mapping for resolution
	symbolsByID map[string]domain.Symbol // id -> symbol
}

// NewCallGraphBuilder creates a new call graph builder.
func NewCallGraphBuilder() *CallGraphBuilder {
	return &CallGraphBuilder{
		symbols:     make(map[string]domain.Symbol),
		symbolsByID: make(map[string]domain.Symbol),
	}
}

// RegisterSymbols registers symbols for call resolution.
func (b *CallGraphBuilder) RegisterSymbols(symbols []domain.Symbol) {
	for _, sym := range symbols {
		b.symbolsByID[sym.ID] = sym
		// Register by name for resolution (may have conflicts, but that's okay)
		if sym.Type == "function" || sym.Type == "method" {
			b.symbols[sym.Name] = sym
		}
	}
}

// BuildCallGraph builds the call graph for a Go file.
func (b *CallGraphBuilder) BuildCallGraph(docID, content string) ([]domain.CallGraphEntry, error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "", content, 0)
	if err != nil {
		return nil, err
	}

	var entries []domain.CallGraphEntry

	// Map to track current function being analyzed
	var currentFunc *domain.Symbol

	ast.Inspect(file, func(n ast.Node) bool {
		switch node := n.(type) {
		case *ast.FuncDecl:
			// Find the symbol for this function
			funcName := node.Name.Name
			if node.Recv != nil && len(node.Recv.List) > 0 {
				recvType := formatReceiver(node.Recv.List[0].Type)
				funcName = recvType + "." + funcName
			}

			sym, exists := b.symbols[funcName]
			if exists && sym.DocID == docID {
				currentFunc = &sym
			} else {
				// Create a temporary symbol
				line := fset.Position(node.Pos()).Line
				currentFunc = &domain.Symbol{
					ID:    generateSymbolID(docID, funcName, line),
					Name:  funcName,
					DocID: docID,
					Line:  line,
					Type:  "function",
				}
			}
			return true

		case *ast.CallExpr:
			if currentFunc == nil {
				return true
			}

			calleeName := extractCalleeName(node.Fun)
			if calleeName == "" {
				return true
			}

			// Try to resolve the callee
			calleeID := calleeName
			if sym, exists := b.symbols[calleeName]; exists {
				calleeID = sym.ID
			}

			entries = append(entries, domain.CallGraphEntry{
				CallerID: currentFunc.ID,
				CalleeID: calleeID,
				Line:     fset.Position(node.Pos()).Line,
			})
			return true

		case *ast.FuncLit:
			// Skip analyzing anonymous functions for now
			return false
		}
		return true
	})

	return entries, nil
}

// extractCalleeName extracts the name of the called function/method.
func extractCalleeName(expr ast.Expr) string {
	switch e := expr.(type) {
	case *ast.Ident:
		return e.Name
	case *ast.SelectorExpr:
		// Method call or package function
		if ident, ok := e.X.(*ast.Ident); ok {
			return ident.Name + "." + e.Sel.Name
		}
		// Could be a more complex expression
		return e.Sel.Name
	default:
		return ""
	}
}

// GetCallers returns all functions that call the given symbol.
func (b *CallGraphBuilder) GetCallers(entries []domain.CallGraphEntry, symbolID string) []domain.Symbol {
	var callers []domain.Symbol
	seen := make(map[string]bool)

	for _, entry := range entries {
		if entry.CalleeID == symbolID {
			if !seen[entry.CallerID] {
				if sym, exists := b.symbolsByID[entry.CallerID]; exists {
					callers = append(callers, sym)
					seen[entry.CallerID] = true
				}
			}
		}
	}

	return callers
}

// GetCallees returns all functions called by the given symbol.
func (b *CallGraphBuilder) GetCallees(entries []domain.CallGraphEntry, symbolID string) []domain.Symbol {
	var callees []domain.Symbol
	seen := make(map[string]bool)

	for _, entry := range entries {
		if entry.CallerID == symbolID {
			if !seen[entry.CalleeID] {
				if sym, exists := b.symbolsByID[entry.CalleeID]; exists {
					callees = append(callees, sym)
					seen[entry.CalleeID] = true
				}
			}
		}
	}

	return callees
}

// BuildChunkMetadata enriches chunk metadata with call graph information.
func (b *CallGraphBuilder) BuildChunkMetadata(
	chunk domain.Chunk,
	symbols []domain.Symbol,
	entries []domain.CallGraphEntry,
) domain.ChunkMetadata {
	meta := domain.ChunkMetadata{}

	// Find symbols in this chunk
	var chunkSymbols []domain.Symbol
	for _, sym := range symbols {
		if sym.DocID == chunk.DocID && sym.Line >= chunk.StartLine && sym.Line <= chunk.EndLine {
			chunkSymbols = append(chunkSymbols, sym)
			meta.Symbols = append(meta.Symbols, sym.ID)
		}
	}

	if len(chunkSymbols) == 0 {
		meta.Type = "mixed"
		return meta
	}

	// Determine chunk type based on primary symbol
	primarySym := chunkSymbols[0]
	meta.Name = primarySym.Name
	meta.Signature = primarySym.Signature

	switch primarySym.Type {
	case "function", "method":
		meta.Type = "function"
		// Find calls made by this function
		for _, entry := range entries {
			if entry.CallerID == primarySym.ID {
				meta.Calls = append(meta.Calls, entry.CalleeID)
			}
			if entry.CalleeID == primarySym.ID {
				meta.CalledBy = append(meta.CalledBy, entry.CallerID)
			}
		}
	case "interface", "struct", "type":
		meta.Type = primarySym.Type
	case "class":
		meta.Type = "class"
	default:
		meta.Type = "mixed"
	}

	// Extract imports if present in chunk text
	if strings.Contains(chunk.Text, "import") {
		meta.Imports = extractImportPaths(chunk.Text)
	}

	return meta
}

// extractImportPaths extracts import paths from text.
func extractImportPaths(text string) []string {
	var imports []string
	lines := strings.Split(text, "\n")
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
			imp := extractSingleImport(line)
			if imp != "" {
				imports = append(imports, imp)
			}
		}
	}

	return imports
}

// extractSingleImport extracts an import path from a single line.
func extractSingleImport(line string) string {
	line = strings.TrimPrefix(line, "import ")
	line = strings.TrimSpace(line)

	// Remove alias
	parts := strings.Fields(line)
	if len(parts) == 0 {
		return ""
	}
	importPart := parts[len(parts)-1]

	// Remove quotes
	importPart = strings.Trim(importPart, "\"'`")

	return importPart
}
