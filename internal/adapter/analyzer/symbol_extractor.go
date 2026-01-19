package analyzer

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"strings"

	"rag/internal/domain"
)

// SymbolExtractor extracts symbols from source code.
type SymbolExtractor struct{}

// NewSymbolExtractor creates a new symbol extractor.
func NewSymbolExtractor() *SymbolExtractor {
	return &SymbolExtractor{}
}

// ExtractSymbols extracts symbols from source code based on language.
func (e *SymbolExtractor) ExtractSymbols(docID, content, lang string) ([]domain.Symbol, error) {
	switch lang {
	case "go":
		return e.extractGoSymbols(docID, content)
	default:
		// Fallback to simple pattern-based extraction
		return e.extractSimpleSymbols(docID, content, lang)
	}
}

// extractGoSymbols extracts symbols from Go source code using AST.
func (e *SymbolExtractor) extractGoSymbols(docID, content string) ([]domain.Symbol, error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "", content, parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("failed to parse Go file: %w", err)
	}

	var symbols []domain.Symbol

	ast.Inspect(file, func(n ast.Node) bool {
		switch node := n.(type) {
		case *ast.FuncDecl:
			sym := domain.Symbol{
				DocID: docID,
				Line:  fset.Position(node.Pos()).Line,
			}

			if node.Recv != nil && len(node.Recv.List) > 0 {
				// Method
				sym.Type = "method"
				recvType := formatReceiver(node.Recv.List[0].Type)
				sym.Name = recvType + "." + node.Name.Name
				sym.Signature = formatFuncSignature(node, recvType)
			} else {
				// Function
				sym.Type = "function"
				sym.Name = node.Name.Name
				sym.Signature = formatFuncSignature(node, "")
			}
			sym.ID = generateSymbolID(docID, sym.Name, sym.Line)
			symbols = append(symbols, sym)

		case *ast.TypeSpec:
			sym := domain.Symbol{
				ID:    generateSymbolID(docID, node.Name.Name, fset.Position(node.Pos()).Line),
				Name:  node.Name.Name,
				DocID: docID,
				Line:  fset.Position(node.Pos()).Line,
			}

			switch node.Type.(type) {
			case *ast.InterfaceType:
				sym.Type = "interface"
				sym.Signature = "interface " + node.Name.Name
			case *ast.StructType:
				sym.Type = "struct"
				sym.Signature = "struct " + node.Name.Name
			default:
				sym.Type = "type"
				sym.Signature = "type " + node.Name.Name
			}
			symbols = append(symbols, sym)

		case *ast.ValueSpec:
			for _, name := range node.Names {
				if name.Name == "_" {
					continue
				}
				sym := domain.Symbol{
					ID:    generateSymbolID(docID, name.Name, fset.Position(name.Pos()).Line),
					Name:  name.Name,
					DocID: docID,
					Line:  fset.Position(name.Pos()).Line,
				}
				// Determine if constant or variable
				if node.Values != nil {
					sym.Type = "variable"
				} else {
					sym.Type = "variable"
				}
				symbols = append(symbols, sym)
			}
		}
		return true
	})

	return symbols, nil
}

// extractSimpleSymbols extracts symbols using simple pattern matching.
func (e *SymbolExtractor) extractSimpleSymbols(docID, content, lang string) ([]domain.Symbol, error) {
	var symbols []domain.Symbol
	lines := strings.Split(content, "\n")

	patterns := getLanguagePatterns(lang)

	for i, line := range lines {
		lineNum := i + 1
		trimmed := strings.TrimSpace(line)

		for _, p := range patterns {
			if name := p.match(trimmed); name != "" {
				symbols = append(symbols, domain.Symbol{
					ID:        generateSymbolID(docID, name, lineNum),
					Name:      name,
					Type:      p.symType,
					DocID:     docID,
					Line:      lineNum,
					Signature: trimmed,
				})
				break
			}
		}
	}

	return symbols, nil
}

// symbolPattern defines a pattern for matching symbols.
type symbolPattern struct {
	prefix  string
	symType string
	extract func(string) string
}

func (p *symbolPattern) match(line string) string {
	if !strings.HasPrefix(line, p.prefix) {
		return ""
	}
	return p.extract(line)
}

// getLanguagePatterns returns symbol patterns for a language.
func getLanguagePatterns(lang string) []symbolPattern {
	switch lang {
	case "python":
		return []symbolPattern{
			{"def ", "function", extractPythonFunc},
			{"class ", "class", extractPythonClass},
		}
	case "javascript", "typescript":
		return []symbolPattern{
			{"function ", "function", extractJSFunc},
			{"class ", "class", extractJSClass},
			{"const ", "constant", extractJSConst},
			{"let ", "variable", extractJSLet},
		}
	case "java":
		return []symbolPattern{
			{"public class ", "class", extractJavaClass},
			{"private class ", "class", extractJavaClass},
			{"class ", "class", extractJavaClass},
			{"public interface ", "interface", extractJavaInterface},
			{"interface ", "interface", extractJavaInterface},
		}
	default:
		return nil
	}
}

func extractPythonFunc(line string) string {
	line = strings.TrimPrefix(line, "def ")
	if idx := strings.Index(line, "("); idx > 0 {
		return strings.TrimSpace(line[:idx])
	}
	return ""
}

func extractPythonClass(line string) string {
	line = strings.TrimPrefix(line, "class ")
	for _, sep := range []string{"(", ":"} {
		if idx := strings.Index(line, sep); idx > 0 {
			return strings.TrimSpace(line[:idx])
		}
	}
	return ""
}

func extractJSFunc(line string) string {
	line = strings.TrimPrefix(line, "function ")
	if idx := strings.Index(line, "("); idx > 0 {
		return strings.TrimSpace(line[:idx])
	}
	return ""
}

func extractJSClass(line string) string {
	line = strings.TrimPrefix(line, "class ")
	for _, sep := range []string{" ", "{"} {
		if idx := strings.Index(line, sep); idx > 0 {
			return strings.TrimSpace(line[:idx])
		}
	}
	return strings.TrimSpace(line)
}

func extractJSConst(line string) string {
	line = strings.TrimPrefix(line, "const ")
	for _, sep := range []string{" ", "="} {
		if idx := strings.Index(line, sep); idx > 0 {
			return strings.TrimSpace(line[:idx])
		}
	}
	return ""
}

func extractJSLet(line string) string {
	line = strings.TrimPrefix(line, "let ")
	for _, sep := range []string{" ", "="} {
		if idx := strings.Index(line, sep); idx > 0 {
			return strings.TrimSpace(line[:idx])
		}
	}
	return ""
}

func extractJavaClass(line string) string {
	// Remove access modifier
	for _, mod := range []string{"public ", "private ", "protected "} {
		line = strings.TrimPrefix(line, mod)
	}
	line = strings.TrimPrefix(line, "class ")
	for _, sep := range []string{" ", "{", "<"} {
		if idx := strings.Index(line, sep); idx > 0 {
			return strings.TrimSpace(line[:idx])
		}
	}
	return strings.TrimSpace(line)
}

func extractJavaInterface(line string) string {
	for _, mod := range []string{"public ", "private ", "protected "} {
		line = strings.TrimPrefix(line, mod)
	}
	line = strings.TrimPrefix(line, "interface ")
	for _, sep := range []string{" ", "{", "<"} {
		if idx := strings.Index(line, sep); idx > 0 {
			return strings.TrimSpace(line[:idx])
		}
	}
	return strings.TrimSpace(line)
}

// formatReceiver formats the receiver type for a method.
func formatReceiver(expr ast.Expr) string {
	switch t := expr.(type) {
	case *ast.StarExpr:
		if ident, ok := t.X.(*ast.Ident); ok {
			return "*" + ident.Name
		}
	case *ast.Ident:
		return t.Name
	}
	return ""
}

// formatFuncSignature formats a function signature.
func formatFuncSignature(fn *ast.FuncDecl, recvType string) string {
	var sig strings.Builder
	sig.WriteString("func ")

	if recvType != "" {
		sig.WriteString("(")
		sig.WriteString(recvType)
		sig.WriteString(") ")
	}

	sig.WriteString(fn.Name.Name)
	sig.WriteString("(")

	// Parameters
	if fn.Type.Params != nil {
		params := make([]string, 0)
		for _, field := range fn.Type.Params.List {
			paramType := formatType(field.Type)
			for _, name := range field.Names {
				params = append(params, name.Name+" "+paramType)
			}
			if len(field.Names) == 0 {
				params = append(params, paramType)
			}
		}
		sig.WriteString(strings.Join(params, ", "))
	}
	sig.WriteString(")")

	// Return types
	if fn.Type.Results != nil && len(fn.Type.Results.List) > 0 {
		sig.WriteString(" ")
		if len(fn.Type.Results.List) > 1 {
			sig.WriteString("(")
		}
		results := make([]string, 0)
		for _, field := range fn.Type.Results.List {
			results = append(results, formatType(field.Type))
		}
		sig.WriteString(strings.Join(results, ", "))
		if len(fn.Type.Results.List) > 1 {
			sig.WriteString(")")
		}
	}

	return sig.String()
}

// formatType formats a type expression as a string.
func formatType(expr ast.Expr) string {
	switch t := expr.(type) {
	case *ast.Ident:
		return t.Name
	case *ast.StarExpr:
		return "*" + formatType(t.X)
	case *ast.ArrayType:
		if t.Len == nil {
			return "[]" + formatType(t.Elt)
		}
		return "[...]" + formatType(t.Elt)
	case *ast.MapType:
		return "map[" + formatType(t.Key) + "]" + formatType(t.Value)
	case *ast.SelectorExpr:
		return formatType(t.X) + "." + t.Sel.Name
	case *ast.InterfaceType:
		return "interface{}"
	case *ast.FuncType:
		return "func(...)"
	case *ast.ChanType:
		return "chan " + formatType(t.Value)
	default:
		return "interface{}"
	}
}

// generateSymbolID generates a unique ID for a symbol.
func generateSymbolID(docID, name string, line int) string {
	data := fmt.Sprintf("%s:%s:%d", docID, name, line)
	hash := sha256.Sum256([]byte(data))
	return hex.EncodeToString(hash[:8])
}
