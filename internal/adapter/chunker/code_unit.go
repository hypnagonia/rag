package chunker

// CodeUnit represents a parsed code structure (function, class, method, etc.).
type CodeUnit struct {
	Type      string     // "function", "method", "class", "interface", "struct", "const", "var", "import"
	Name      string     // Symbol name
	Signature string     // Full signature for functions/methods
	StartLine int        // 1-indexed
	EndLine   int        // 1-indexed, inclusive
	Content   string     // Raw source code
	Children  []CodeUnit // Nested definitions (e.g., methods inside a class)
	Imports   []string   // Imports referenced in this unit
	Calls     []string   // Functions/methods called within
	DocString string     // Documentation comment if present
}

// LanguageParser parses source code into CodeUnits.
type LanguageParser interface {
	// Parse parses source code and returns code units.
	Parse(content string) ([]CodeUnit, error)
	// Language returns the language this parser handles.
	Language() string
}
