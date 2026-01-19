package cli

import (
	"bytes"
	"embed"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"text/template"

	"github.com/spf13/cobra"
	"rag/internal/domain"
)

//go:embed templates/*.txt
var promptTemplates embed.FS

var (
	runpromptRuntime bool
	runpromptBuilder bool
	runpromptCtx     string
	runpromptQuery   string
)

var runpromptCmd = &cobra.Command{
	Use:   "runprompt",
	Short: "Generate prompts for LLM orchestration",
	Long: `Generate formatted prompts from templates for manual LLM orchestration.

Use --builder for the context compression prompt (to feed to an LLM for summarization).
Use --runtime for the question-answering prompt (to feed to an LLM with the context).

Examples:
  rag runprompt --builder --ctx context.json
  rag runprompt --runtime --ctx context.json -q "How does auth work?"`,
	RunE: runPrompt,
}

func init() {
	rootCmd.AddCommand(runpromptCmd)
	runpromptCmd.Flags().BoolVar(&runpromptRuntime, "runtime", false, "use runtime (answering) prompt template")
	runpromptCmd.Flags().BoolVar(&runpromptBuilder, "builder", false, "use builder (compression) prompt template")
	runpromptCmd.Flags().StringVar(&runpromptCtx, "ctx", "", "path to packed context JSON file (required)")
	runpromptCmd.Flags().StringVarP(&runpromptQuery, "query", "q", "", "additional query for runtime prompt")
	runpromptCmd.MarkFlagRequired("ctx")
}

func runPrompt(cmd *cobra.Command, args []string) error {

	if !runpromptRuntime && !runpromptBuilder {
		return fmt.Errorf("must specify either --runtime or --builder")
	}
	if runpromptRuntime && runpromptBuilder {
		return fmt.Errorf("cannot specify both --runtime and --builder")
	}

	ctxData, err := os.ReadFile(runpromptCtx)
	if err != nil {
		return fmt.Errorf("failed to read context file: %w", err)
	}

	var packed domain.PackedContext
	if err := json.Unmarshal(ctxData, &packed); err != nil {
		return fmt.Errorf("failed to parse context file: %w", err)
	}

	data := PromptData{
		Query:    packed.Query,
		Snippets: packed.Snippets,
	}

	if runpromptQuery != "" {
		data.Query = runpromptQuery
	}

	var templateName string
	if runpromptBuilder {
		templateName = "templates/builder_prompt.txt"
	} else {
		templateName = "templates/runtime_prompt.txt"
	}

	tmplContent, err := promptTemplates.ReadFile(templateName)
	if err != nil {

		return fmt.Errorf("template not found: %w", err)
	}

	tmpl, err := template.New("prompt").Funcs(templateFuncs()).Parse(string(tmplContent))
	if err != nil {
		return fmt.Errorf("failed to parse template: %w", err)
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		return fmt.Errorf("failed to render template: %w", err)
	}

	fmt.Println(buf.String())
	return nil
}

type PromptData struct {
	Query    string
	Snippets []domain.Snippet
}

func templateFuncs() template.FuncMap {
	return template.FuncMap{
		"join": strings.Join,
		"formatSnippets": func(snippets []domain.Snippet) string {
			var sb strings.Builder
			for i, s := range snippets {
				sb.WriteString(fmt.Sprintf("### [%d] %s (%s)\n", i+1, s.Path, s.Range))
				sb.WriteString(fmt.Sprintf("Relevance: %s\n\n", s.Why))
				sb.WriteString("```\n")
				sb.WriteString(s.Text)
				sb.WriteString("\n```\n\n")
			}
			return sb.String()
		},
	}
}
