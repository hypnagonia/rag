package chunker

type CodeUnit struct {
	Type      string
	Name      string
	Signature string
	StartLine int
	EndLine   int
	Content   string
	Children  []CodeUnit
	Imports   []string
	Calls     []string
	DocString string
}

type LanguageParser interface {
	Parse(content string) ([]CodeUnit, error)

	Language() string
}
