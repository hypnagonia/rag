package analyzer

import (
	"go/ast"
	"go/parser"
	"go/token"
	"strings"

	"rag/internal/domain"
)

type CallGraphBuilder struct {
	symbols     map[string]domain.Symbol
	symbolsByID map[string]domain.Symbol
}

func NewCallGraphBuilder() *CallGraphBuilder {
	return &CallGraphBuilder{
		symbols:     make(map[string]domain.Symbol),
		symbolsByID: make(map[string]domain.Symbol),
	}
}

func (b *CallGraphBuilder) RegisterSymbols(symbols []domain.Symbol) {
	for _, sym := range symbols {
		b.symbolsByID[sym.ID] = sym

		if sym.Type == "function" || sym.Type == "method" {
			b.symbols[sym.Name] = sym
		}
	}
}

func (b *CallGraphBuilder) BuildCallGraph(docID, content string) ([]domain.CallGraphEntry, error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "", content, 0)
	if err != nil {
		return nil, err
	}

	var entries []domain.CallGraphEntry

	var currentFunc *domain.Symbol

	ast.Inspect(file, func(n ast.Node) bool {
		switch node := n.(type) {
		case *ast.FuncDecl:

			funcName := node.Name.Name
			if node.Recv != nil && len(node.Recv.List) > 0 {
				recvType := formatReceiver(node.Recv.List[0].Type)
				funcName = recvType + "." + funcName
			}

			sym, exists := b.symbols[funcName]
			if exists && sym.DocID == docID {
				currentFunc = &sym
			} else {

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

			return false
		}
		return true
	})

	return entries, nil
}

func extractCalleeName(expr ast.Expr) string {
	switch e := expr.(type) {
	case *ast.Ident:
		return e.Name
	case *ast.SelectorExpr:

		if ident, ok := e.X.(*ast.Ident); ok {
			return ident.Name + "." + e.Sel.Name
		}

		return e.Sel.Name
	default:
		return ""
	}
}

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

func (b *CallGraphBuilder) BuildChunkMetadata(
	chunk domain.Chunk,
	symbols []domain.Symbol,
	entries []domain.CallGraphEntry,
) domain.ChunkMetadata {
	meta := domain.ChunkMetadata{}

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

	primarySym := chunkSymbols[0]
	meta.Name = primarySym.Name
	meta.Signature = primarySym.Signature

	switch primarySym.Type {
	case "function", "method":
		meta.Type = "function"

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

	if strings.Contains(chunk.Text, "import") {
		meta.Imports = extractImportPaths(chunk.Text)
	}

	return meta
}

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

func extractSingleImport(line string) string {
	line = strings.TrimPrefix(line, "import ")
	line = strings.TrimSpace(line)

	parts := strings.Fields(line)
	if len(parts) == 0 {
		return ""
	}
	importPart := parts[len(parts)-1]

	importPart = strings.Trim(importPart, "\"'`")

	return importPart
}
