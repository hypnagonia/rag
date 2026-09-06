package usecase

import (
	"encoding/json"
	"fmt"
	"strings"

	"rag/internal/domain"
	"rag/internal/port"
)

const (
	promptExpandQuery = `Generate 3 search queries to find relevant passages.
Rules:
- Use different word forms and common synonyms.
- A query must not be a direct quotation from the source material.
- A query must not contain the answer to the question.
- Output one query per line, no numbering.`

	promptEvaluateContext = `Given text passages and a question, respond in JSON only:
{"sufficient":bool,"reason":"brief","queries":["search"]}
- "sufficient": true only if the passages actually contain the answer.
- "queries": new search terms to try when the passages are insufficient.`

	promptGenerateAnswer = `Answer the question using ONLY the provided context. Be concise.
ALWAYS cite sources as [filename:lines] for every fact you state.
If the context does not contain the answer, say so plainly instead of guessing.`
)

type ChunkRetriever interface {
	Retrieve(query string, topK int) ([]domain.ScoredChunk, error)
}

type ContextPacker interface {
	Pack(query string, chunks []domain.ScoredChunk, budget int) (domain.PackedContext, error)
}

type AskUseCase struct {
	retriever ChunkRetriever
	packer    ContextPacker
	llm       port.LLM
	maxIters  int
	topK      int
	budget    int
	fast      bool
	expand    bool
}

type AskBuilder struct {
	retriever ChunkRetriever
	packer    ContextPacker
	llm       port.LLM
	maxIters  int
	topK      int
	budget    int
	fast      bool
	expand    bool
}

func NewAskBuilder() *AskBuilder {
	return &AskBuilder{maxIters: 2, topK: 10, budget: 4000}
}

func (b *AskBuilder) Retriever(r ChunkRetriever) *AskBuilder {
	b.retriever = r
	return b
}

func (b *AskBuilder) Packer(p ContextPacker) *AskBuilder {
	b.packer = p
	return b
}

func (b *AskBuilder) LLM(l port.LLM) *AskBuilder {
	b.llm = l
	return b
}

func (b *AskBuilder) MaxIterations(n int) *AskBuilder {
	if n > 0 {
		b.maxIters = n
	}
	return b
}

func (b *AskBuilder) TopK(k int) *AskBuilder {
	if k > 0 {
		b.topK = k
	}
	return b
}

func (b *AskBuilder) Budget(tokens int) *AskBuilder {
	if tokens > 0 {
		b.budget = tokens
	}
	return b
}

func (b *AskBuilder) Fast(fast bool) *AskBuilder {
	b.fast = fast
	return b
}

func (b *AskBuilder) ExpandQuery(expand bool) *AskBuilder {
	b.expand = expand
	return b
}

func (b *AskBuilder) Build() (*AskUseCase, error) {
	if b.retriever == nil {
		return nil, fmt.Errorf("ask requires a retriever")
	}
	if b.packer == nil {
		return nil, fmt.Errorf("ask requires a context packer")
	}
	if b.llm == nil {
		return nil, fmt.Errorf("ask requires an LLM")
	}

	return &AskUseCase{
		retriever: b.retriever,
		packer:    b.packer,
		llm:       b.llm,
		maxIters:  b.maxIters,
		topK:      b.topK,
		budget:    b.budget,
		fast:      b.fast,
		expand:    b.expand,
	}, nil
}

type AskResult struct {
	Query          string
	Answer         string
	Chunks         []domain.ScoredChunk
	QueriesUsed    []string
	IterationsUsed int
	LLMCalls       int
	Sufficient     bool
	Context        domain.PackedContext
}

type AskStage string

const (
	StageExpanding  AskStage = "expanding query"
	StageSearching  AskStage = "searching"
	StageEvaluating AskStage = "evaluating context"
	StageAnswering  AskStage = "answering"
)

type AskProgress func(stage AskStage, iteration int, detail string)

type contextDecision struct {
	Sufficient bool     `json:"sufficient"`
	Reason     string   `json:"reason"`
	Queries    []string `json:"queries"`
}

func (u *AskUseCase) Ask(query string, progress AskProgress) (*AskResult, error) {
	result := &AskResult{Query: query}

	queries := []string{query}
	if u.expand {
		u.report(progress, StageExpanding, 0, "")
		expanded, err := u.expandQuery(query)
		result.LLMCalls++
		if err == nil && len(expanded) > 0 {
			queries = expanded
		}
	}
	result.QueriesUsed = queries

	collected := make(map[string]domain.ScoredChunk)
	maxIters := u.maxIters
	if u.fast {
		maxIters = 1
	}

	for iter := 0; iter < maxIters; iter++ {
		u.report(progress, StageSearching, iter+1, strings.Join(queries, " | "))
		u.collect(collected, queries)

		if len(collected) == 0 {
			return nil, fmt.Errorf("no results found for: %s", query)
		}

		packed, err := u.packer.Pack(query, sortByScore(collected), u.budget)
		if err != nil {
			return nil, fmt.Errorf("failed to pack context: %w", err)
		}
		result.Context = packed
		result.IterationsUsed = iter + 1

		if u.fast || iter == maxIters-1 {
			break
		}

		u.report(progress, StageEvaluating, iter+1, "")
		decision := u.evaluate(query, packed)
		result.LLMCalls++
		result.Sufficient = decision.Sufficient

		if decision.Sufficient {
			break
		}

		next := dedupeQueries(decision.Queries, queries)
		if len(next) == 0 {
			break
		}
		queries = next
		result.QueriesUsed = append(result.QueriesUsed, next...)
	}

	u.report(progress, StageAnswering, result.IterationsUsed, "")
	answer, err := u.answer(query, result.Context)
	result.LLMCalls++
	if err != nil {
		return nil, fmt.Errorf("failed to generate answer: %w", err)
	}

	result.Answer = answer
	result.Chunks = sortByScore(collected)

	return result, nil
}

func (u *AskUseCase) collect(into map[string]domain.ScoredChunk, queries []string) {
	for _, q := range queries {
		chunks, err := u.retriever.Retrieve(q, u.topK)
		if err != nil {
			continue
		}
		for _, c := range chunks {
			if existing, ok := into[c.Chunk.ID]; !ok || c.Score > existing.Score {
				into[c.Chunk.ID] = c
			}
		}
	}
}

func (u *AskUseCase) expandQuery(query string) ([]string, error) {
	response, err := u.llm.GenerateWithSystem(promptExpandQuery, "Expand this search query: "+query)
	if err != nil {
		return nil, err
	}

	queries := []string{query}
	for _, line := range strings.Split(response, "\n") {
		line = strings.TrimSpace(line)
		line = strings.TrimLeft(line, "0123456789.-* ")
		if line != "" && line != query {
			queries = append(queries, line)
		}
	}

	if len(queries) > 4 {
		queries = queries[:4]
	}

	return queries, nil
}

func (u *AskUseCase) evaluate(query string, packed domain.PackedContext) contextDecision {
	userPrompt := fmt.Sprintf("Q: %s\n\nContext:\n%s\n\nSufficient?", query, renderContext(packed))

	response, err := u.llm.GenerateWithSystem(promptEvaluateContext, userPrompt)
	if err != nil {
		return contextDecision{Sufficient: true, Reason: "evaluation failed"}
	}

	var decision contextDecision
	if err := json.Unmarshal([]byte(extractJSONObject(response)), &decision); err != nil {
		return contextDecision{Sufficient: true, Reason: "unparseable evaluation"}
	}

	return decision
}

func (u *AskUseCase) answer(query string, packed domain.PackedContext) (string, error) {
	userPrompt := fmt.Sprintf("Q: %s\n\nContext:\n%s", query, renderContext(packed))
	return u.llm.GenerateWithSystem(promptGenerateAnswer, userPrompt)
}

func (u *AskUseCase) report(progress AskProgress, stage AskStage, iteration int, detail string) {
	if progress != nil {
		progress(stage, iteration, detail)
	}
}

func renderContext(packed domain.PackedContext) string {
	var b strings.Builder
	for _, s := range packed.Snippets {
		b.WriteString(fmt.Sprintf("[%s:%s]\n", s.Path, s.Range))
		b.WriteString(s.Text)
		b.WriteString("\n\n")
	}
	return b.String()
}

func extractJSONObject(s string) string {
	start := strings.Index(s, "{")
	end := strings.LastIndex(s, "}")
	if start < 0 || end <= start {
		return s
	}
	return s[start : end+1]
}

func dedupeQueries(candidates, seen []string) []string {
	existing := make(map[string]struct{}, len(seen))
	for _, q := range seen {
		existing[strings.ToLower(strings.TrimSpace(q))] = struct{}{}
	}

	var out []string
	for _, q := range candidates {
		q = strings.TrimSpace(q)
		key := strings.ToLower(q)
		if q == "" {
			continue
		}
		if _, dup := existing[key]; dup {
			continue
		}
		existing[key] = struct{}{}
		out = append(out, q)
	}
	return out
}

func sortByScore(chunks map[string]domain.ScoredChunk) []domain.ScoredChunk {
	out := make([]domain.ScoredChunk, 0, len(chunks))
	for _, c := range chunks {
		out = append(out, c)
	}
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j].Score > out[j-1].Score; j-- {
			out[j], out[j-1] = out[j-1], out[j]
		}
	}
	return out
}
