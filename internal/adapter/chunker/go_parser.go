package chunker

import (
	"bytes"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"strings"
)

// GoParser parses Go source code into CodeUnits.
type GoParser struct{}

// NewGoParser creates a new Go parser.
func NewGoParser() *GoParser {
	return &GoParser{}
}

// Language returns the language this parser handles.
func (p *GoParser) Language() string {
	return "go"
}

// Parse parses Go source code and returns code units.
func (p *GoParser) Parse(content string) ([]CodeUnit, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", content, parser.ParseComments)
	if err != nil {
		return nil, err
	}

	lines := strings.Split(content, "\n")
	var units []CodeUnit

	for _, decl := range f.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			unit := p.extractFunction(fset, d, lines)
			units = append(units, unit)

		case *ast.GenDecl:

			genUnits := p.extractGenDecl(fset, d, lines)
			units = append(units, genUnits...)
		}
	}

	return units, nil
}

// extractFunction extracts a function/method declaration.
func (p *GoParser) extractFunction(fset *token.FileSet, fn *ast.FuncDecl, lines []string) CodeUnit {
	startPos := fset.Position(fn.Pos())
	endPos := fset.Position(fn.End())

	// Build signature
	var sig strings.Builder
	sig.WriteString("func ")
	if fn.Recv != nil && len(fn.Recv.List) > 0 {
		sig.WriteString("(")
		sig.WriteString(p.formatFieldList(fn.Recv))
		sig.WriteString(") ")
	}
	sig.WriteString(fn.Name.Name)
	sig.WriteString("(")
	if fn.Type.Params != nil {
		sig.WriteString(p.formatFieldList(fn.Type.Params))
	}
	sig.WriteString(")")
	if fn.Type.Results != nil && len(fn.Type.Results.List) > 0 {
		sig.WriteString(" ")
		if len(fn.Type.Results.List) > 1 || (len(fn.Type.Results.List) == 1 && fn.Type.Results.List[0].Names != nil) {
			sig.WriteString("(")
			sig.WriteString(p.formatFieldList(fn.Type.Results))
			sig.WriteString(")")
		} else {
			sig.WriteString(p.formatFieldList(fn.Type.Results))
		}
	}

	content := extractLines(lines, startPos.Line, endPos.Line)

	unitType := "function"
	if fn.Recv != nil {
		unitType = "method"
	}

	docString := ""
	if fn.Doc != nil {
		docString = fn.Doc.Text()
	}

	calls := p.extractCalls(fn.Body)

	return CodeUnit{
		Type:      unitType,
		Name:      fn.Name.Name,
		Signature: sig.String(),
		StartLine: startPos.Line,
		EndLine:   endPos.Line,
		Content:   content,
		DocString: docString,
		Calls:     calls,
	}
}

// extractGenDecl extracts type, const, var, or import declarations.
func (p *GoParser) extractGenDecl(fset *token.FileSet, decl *ast.GenDecl, lines []string) []CodeUnit {
	var units []CodeUnit

	startPos := fset.Position(decl.Pos())
	endPos := fset.Position(decl.End())

	switch decl.Tok {
	case token.TYPE:
		for _, spec := range decl.Specs {
			ts := spec.(*ast.TypeSpec)
			specStart := fset.Position(ts.Pos())
			specEnd := fset.Position(ts.End())

			unitType := "type"
			switch ts.Type.(type) {
			case *ast.StructType:
				unitType = "struct"
			case *ast.InterfaceType:
				unitType = "interface"
			}

			start := specStart.Line
			end := specEnd.Line
			if decl.Lparen == 0 {
				start = startPos.Line
				end = endPos.Line
			}

			content := extractLines(lines, start, end)
			docString := ""
			if ts.Doc != nil {
				docString = ts.Doc.Text()
			} else if decl.Doc != nil {
				docString = decl.Doc.Text()
			}

			unit := CodeUnit{
				Type:      unitType,
				Name:      ts.Name.Name,
				Signature: p.formatTypeSignature(ts),
				StartLine: start,
				EndLine:   end,
				Content:   content,
				DocString: docString,
			}

			if st, ok := ts.Type.(*ast.StructType); ok {
				unit.Children = p.extractStructFields(st)
			}
			if it, ok := ts.Type.(*ast.InterfaceType); ok {
				unit.Children = p.extractInterfaceMethods(it)
			}

			units = append(units, unit)
		}

	case token.CONST, token.VAR:

		content := extractLines(lines, startPos.Line, endPos.Line)
		unitType := "const"
		if decl.Tok == token.VAR {
			unitType = "var"
		}

		var names []string
		for _, spec := range decl.Specs {
			vs := spec.(*ast.ValueSpec)
			for _, name := range vs.Names {
				names = append(names, name.Name)
			}
		}

		docString := ""
		if decl.Doc != nil {
			docString = decl.Doc.Text()
		}

		units = append(units, CodeUnit{
			Type:      unitType,
			Name:      strings.Join(names, ", "),
			StartLine: startPos.Line,
			EndLine:   endPos.Line,
			Content:   content,
			DocString: docString,
		})

	case token.IMPORT:

		content := extractLines(lines, startPos.Line, endPos.Line)
		var imports []string
		for _, spec := range decl.Specs {
			is := spec.(*ast.ImportSpec)
			imports = append(imports, strings.Trim(is.Path.Value, `"`))
		}

		units = append(units, CodeUnit{
			Type:      "import",
			Name:      "imports",
			StartLine: startPos.Line,
			EndLine:   endPos.Line,
			Content:   content,
			Imports:   imports,
		})
	}

	return units
}

// formatFieldList formats a field list (parameters, results, receiver).
func (p *GoParser) formatFieldList(fl *ast.FieldList) string {
	if fl == nil || len(fl.List) == 0 {
		return ""
	}

	var parts []string
	for _, field := range fl.List {
		typeStr := p.formatExpr(field.Type)
		if len(field.Names) == 0 {
			parts = append(parts, typeStr)
		} else {
			var names []string
			for _, name := range field.Names {
				names = append(names, name.Name)
			}
			parts = append(parts, strings.Join(names, ", ")+" "+typeStr)
		}
	}
	return strings.Join(parts, ", ")
}

// formatExpr formats an expression to string.
func (p *GoParser) formatExpr(expr ast.Expr) string {
	var buf bytes.Buffer
	format.Node(&buf, token.NewFileSet(), expr)
	return buf.String()
}

// formatTypeSignature creates a signature for a type declaration.
func (p *GoParser) formatTypeSignature(ts *ast.TypeSpec) string {
	var sig strings.Builder
	sig.WriteString("type ")
	sig.WriteString(ts.Name.Name)
	sig.WriteString(" ")

	switch t := ts.Type.(type) {
	case *ast.StructType:
		sig.WriteString("struct")
	case *ast.InterfaceType:
		sig.WriteString("interface")
	case *ast.Ident:
		sig.WriteString(t.Name)
	default:
		sig.WriteString(p.formatExpr(ts.Type))
	}

	return sig.String()
}

// extractStructFields extracts field info from a struct.
func (p *GoParser) extractStructFields(st *ast.StructType) []CodeUnit {
	var children []CodeUnit
	if st.Fields == nil {
		return children
	}

	for _, field := range st.Fields.List {
		typeStr := p.formatExpr(field.Type)
		var names []string
		for _, name := range field.Names {
			names = append(names, name.Name)
		}
		if len(names) == 0 {

			names = []string{typeStr}
		}

		children = append(children, CodeUnit{
			Type: "field",
			Name: strings.Join(names, ", "),
		})
	}

	return children
}

// extractInterfaceMethods extracts method signatures from an interface.
func (p *GoParser) extractInterfaceMethods(it *ast.InterfaceType) []CodeUnit {
	var children []CodeUnit
	if it.Methods == nil {
		return children
	}

	for _, method := range it.Methods.List {
		if len(method.Names) == 0 {

			children = append(children, CodeUnit{
				Type: "embedded",
				Name: p.formatExpr(method.Type),
			})
		} else {
			for _, name := range method.Names {
				sig := name.Name
				if ft, ok := method.Type.(*ast.FuncType); ok {
					sig = "func " + name.Name + "(" + p.formatFieldList(ft.Params) + ")"
					if ft.Results != nil && len(ft.Results.List) > 0 {
						sig += " " + p.formatFieldList(ft.Results)
					}
				}
				children = append(children, CodeUnit{
					Type:      "method",
					Name:      name.Name,
					Signature: sig,
				})
			}
		}
	}

	return children
}

// extractCalls extracts function call names from a function body.
func (p *GoParser) extractCalls(body *ast.BlockStmt) []string {
	if body == nil {
		return nil
	}

	var calls []string
	seen := make(map[string]bool)

	ast.Inspect(body, func(n ast.Node) bool {
		if call, ok := n.(*ast.CallExpr); ok {
			name := p.extractCallName(call.Fun)
			if name != "" && !seen[name] {
				calls = append(calls, name)
				seen[name] = true
			}
		}
		return true
	})

	return calls
}

// extractCallName extracts the function name from a call expression.
func (p *GoParser) extractCallName(expr ast.Expr) string {
	switch e := expr.(type) {
	case *ast.Ident:
		return e.Name
	case *ast.SelectorExpr:

		if x, ok := e.X.(*ast.Ident); ok {
			return x.Name + "." + e.Sel.Name
		}
		return e.Sel.Name
	default:
		return ""
	}
}

// extractLines extracts lines from a slice (1-indexed, inclusive).
func extractLines(lines []string, startLine, endLine int) string {
	if startLine < 1 {
		startLine = 1
	}
	if endLine > len(lines) {
		endLine = len(lines)
	}
	if startLine > len(lines) {
		return ""
	}

	selected := lines[startLine-1 : endLine]
	return strings.Join(selected, "\n")
}
