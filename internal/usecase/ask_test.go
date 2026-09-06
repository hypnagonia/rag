package usecase

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"rag/internal/domain"
)

type scriptedLLM struct {
	responses []string
	calls     int
	prompts   []string
	err       error
}

func (l *scriptedLLM) Generate(prompt string) (string, error) {
	return l.GenerateWithSystem("", prompt)
}

func (l *scriptedLLM) GenerateWithSystem(system, user string) (string, error) {
	l.calls++
	l.prompts = append(l.prompts, user)
	if l.err != nil {
		return "", l.err
	}
	if l.calls-1 < len(l.responses) {
		return l.responses[l.calls-1], nil
	}
	return "final answer", nil
}

func (l *scriptedLLM) ModelName() string { return "scripted" }

type fakeRetriever struct {
	queries []string
	byQuery map[string][]domain.ScoredChunk
	fixed   []domain.ScoredChunk
}

func (r *fakeRetriever) Retrieve(query string, topK int) ([]domain.ScoredChunk, error) {
	r.queries = append(r.queries, query)
	if r.byQuery != nil {
		return r.byQuery[query], nil
	}
	return r.fixed, nil
}

type fakePacker struct{}

func (p *fakePacker) Pack(query string, chunks []domain.ScoredChunk, budget int) (domain.PackedContext, error) {
	snippets := make([]domain.Snippet, 0, len(chunks))
	for _, c := range chunks {
		snippets = append(snippets, domain.Snippet{Path: "f.txt", Range: "L1-2", Text: c.Chunk.Text})
	}
	return domain.PackedContext{Snippets: snippets, BudgetTokens: budget, UsedTokens: len(snippets) * 10}, nil
}

func chunkAt(id string, score float64) domain.ScoredChunk {
	return domain.ScoredChunk{Chunk: domain.Chunk{ID: id, Text: "text " + id}, Score: score}
}

func buildAsk(t *testing.T, r ChunkRetriever, l *scriptedLLM, tune func(*AskBuilder)) *AskUseCase {
	t.Helper()
	b := NewAskBuilder().Retriever(r).Packer(&fakePacker{}).LLM(l)
	if tune != nil {
		tune(b)
	}
	uc, err := b.Build()
	if err != nil {
		t.Fatalf("build failed: %v", err)
	}
	return uc
}

func TestAskFastModeMakesExactlyOneLLMCall(t *testing.T) {
	llm := &scriptedLLM{responses: []string{"the answer"}}
	uc := buildAsk(t, &fakeRetriever{fixed: []domain.ScoredChunk{chunkAt("a", 1)}}, llm, func(b *AskBuilder) {
		b.Fast(true)
	})

	result, err := uc.Ask("q", nil)
	if err != nil {
		t.Fatalf("ask failed: %v", err)
	}

	if llm.calls != 1 {
		t.Errorf("fast mode must make exactly 1 LLM call, got %d", llm.calls)
	}
	if result.LLMCalls != 1 {
		t.Errorf("result should report 1 call, got %d", result.LLMCalls)
	}
	if result.IterationsUsed != 1 {
		t.Errorf("expected 1 iteration, got %d", result.IterationsUsed)
	}
	if result.Answer != "the answer" {
		t.Errorf("unexpected answer %q", result.Answer)
	}
}

func TestAskStopsWhenContextIsSufficient(t *testing.T) {
	llm := &scriptedLLM{responses: []string{
		`{"sufficient":true,"reason":"found it"}`,
		"the answer",
	}}
	uc := buildAsk(t, &fakeRetriever{fixed: []domain.ScoredChunk{chunkAt("a", 1)}}, llm, nil)

	result, err := uc.Ask("q", nil)
	if err != nil {
		t.Fatalf("ask failed: %v", err)
	}

	if result.IterationsUsed != 1 {
		t.Errorf("expected to stop after 1 round, got %d", result.IterationsUsed)
	}
	if llm.calls != 2 {
		t.Errorf("expected evaluate + answer = 2 calls, got %d", llm.calls)
	}
	if !result.Sufficient {
		t.Error("result should record that context was judged sufficient")
	}
}

func TestAskRunsSecondRoundWithSuggestedQueries(t *testing.T) {
	retr := &fakeRetriever{byQuery: map[string][]domain.ScoredChunk{
		"q":            {chunkAt("a", 1)},
		"better query": {chunkAt("b", 2)},
	}}
	llm := &scriptedLLM{responses: []string{
		`{"sufficient":false,"reason":"missing","queries":["better query"]}`,
		"the answer",
	}}
	uc := buildAsk(t, retr, llm, func(b *AskBuilder) { b.MaxIterations(2) })

	result, err := uc.Ask("q", nil)
	if err != nil {
		t.Fatalf("ask failed: %v", err)
	}

	if result.IterationsUsed != 2 {
		t.Errorf("expected 2 rounds, got %d", result.IterationsUsed)
	}
	if len(retr.queries) != 2 || retr.queries[1] != "better query" {
		t.Errorf("second round should use the suggested query, got %v", retr.queries)
	}
	if len(result.Chunks) != 2 {
		t.Errorf("chunks from both rounds should accumulate, got %d", len(result.Chunks))
	}
}

func TestAskStopsEarlyWhenNoNewQueriesSuggested(t *testing.T) {
	llm := &scriptedLLM{responses: []string{
		`{"sufficient":false,"reason":"missing","queries":["q"]}`,
		"the answer",
	}}
	retr := &fakeRetriever{fixed: []domain.ScoredChunk{chunkAt("a", 1)}}
	uc := buildAsk(t, retr, llm, func(b *AskBuilder) { b.MaxIterations(5) })

	result, err := uc.Ask("q", nil)
	if err != nil {
		t.Fatalf("ask failed: %v", err)
	}

	if result.IterationsUsed != 1 {
		t.Errorf("a repeated query offers nothing new, expected to stop at 1 round, got %d", result.IterationsUsed)
	}
	if llm.calls != 2 {
		t.Errorf("expected 2 calls, got %d", llm.calls)
	}
}

func TestAskTreatsUnparseableEvaluationAsSufficient(t *testing.T) {
	llm := &scriptedLLM{responses: []string{"not json at all", "the answer"}}
	uc := buildAsk(t, &fakeRetriever{fixed: []domain.ScoredChunk{chunkAt("a", 1)}}, llm, nil)

	result, err := uc.Ask("q", nil)
	if err != nil {
		t.Fatalf("ask should not fail on a bad evaluation: %v", err)
	}
	if result.Answer != "the answer" {
		t.Errorf("expected an answer despite the bad evaluation, got %q", result.Answer)
	}
}

func TestAskParsesJSONWrappedInProse(t *testing.T) {
	llm := &scriptedLLM{responses: []string{
		"Sure! ```json\n{\"sufficient\":true,\"reason\":\"ok\"}\n``` hope that helps",
		"the answer",
	}}
	uc := buildAsk(t, &fakeRetriever{fixed: []domain.ScoredChunk{chunkAt("a", 1)}}, llm, nil)

	result, err := uc.Ask("q", nil)
	if err != nil {
		t.Fatalf("ask failed: %v", err)
	}
	if !result.Sufficient {
		t.Error("JSON embedded in prose should still parse")
	}
}

func TestAskExpandQueryAddsOneCallAndUsesResults(t *testing.T) {
	retr := &fakeRetriever{byQuery: map[string][]domain.ScoredChunk{
		"q":     {chunkAt("a", 1)},
		"alt 1": {chunkAt("b", 2)},
	}}
	llm := &scriptedLLM{responses: []string{"alt 1", "the answer"}}
	uc := buildAsk(t, retr, llm, func(b *AskBuilder) { b.Fast(true).ExpandQuery(true) })

	result, err := uc.Ask("q", nil)
	if err != nil {
		t.Fatalf("ask failed: %v", err)
	}

	if result.LLMCalls != 2 {
		t.Errorf("expand + answer should be 2 calls, got %d", result.LLMCalls)
	}
	if len(retr.queries) != 2 {
		t.Errorf("expected both the original and expanded query to be searched, got %v", retr.queries)
	}
}

func TestAskErrorsWhenNothingRetrieved(t *testing.T) {
	llm := &scriptedLLM{}
	uc := buildAsk(t, &fakeRetriever{}, llm, func(b *AskBuilder) { b.Fast(true) })

	if _, err := uc.Ask("q", nil); err == nil {
		t.Error("expected an error when retrieval returns nothing")
	}
	if llm.calls != 0 {
		t.Errorf("must not call the LLM when there is no context, got %d calls", llm.calls)
	}
}

func TestAskPropagatesAnswerFailure(t *testing.T) {
	llm := &scriptedLLM{err: errors.New("402 insufficient balance")}
	uc := buildAsk(t, &fakeRetriever{fixed: []domain.ScoredChunk{chunkAt("a", 1)}}, llm, func(b *AskBuilder) {
		b.Fast(true)
	})

	_, err := uc.Ask("q", nil)
	if err == nil || !strings.Contains(err.Error(), "insufficient balance") {
		t.Errorf("expected the API failure to surface, got %v", err)
	}
}

func TestAskIncludesCitableContextInAnswerPrompt(t *testing.T) {
	llm := &scriptedLLM{responses: []string{"the answer"}}
	uc := buildAsk(t, &fakeRetriever{fixed: []domain.ScoredChunk{chunkAt("a", 1)}}, llm, func(b *AskBuilder) {
		b.Fast(true)
	})

	if _, err := uc.Ask("how does it work", nil); err != nil {
		t.Fatalf("ask failed: %v", err)
	}

	prompt := llm.prompts[len(llm.prompts)-1]
	if !strings.Contains(prompt, "how does it work") {
		t.Error("answer prompt should contain the question")
	}
	if !strings.Contains(prompt, "[f.txt:L1-2]") {
		t.Errorf("answer prompt should carry citable ranges, got %q", prompt)
	}
}

func TestAskReportsProgressStages(t *testing.T) {
	llm := &scriptedLLM{responses: []string{"the answer"}}
	uc := buildAsk(t, &fakeRetriever{fixed: []domain.ScoredChunk{chunkAt("a", 1)}}, llm, func(b *AskBuilder) {
		b.Fast(true)
	})

	var stages []AskStage
	if _, err := uc.Ask("q", func(s AskStage, iter int, detail string) {
		stages = append(stages, s)
	}); err != nil {
		t.Fatalf("ask failed: %v", err)
	}

	want := []AskStage{StageSearching, StageAnswering}
	if fmt.Sprint(stages) != fmt.Sprint(want) {
		t.Errorf("expected stages %v, got %v", want, stages)
	}
}

func TestAskBuilderRequiresDependencies(t *testing.T) {
	if _, err := NewAskBuilder().Packer(&fakePacker{}).LLM(&scriptedLLM{}).Build(); err == nil {
		t.Error("expected a missing retriever to be rejected")
	}
	if _, err := NewAskBuilder().Retriever(&fakeRetriever{}).LLM(&scriptedLLM{}).Build(); err == nil {
		t.Error("expected a missing packer to be rejected")
	}
	if _, err := NewAskBuilder().Retriever(&fakeRetriever{}).Packer(&fakePacker{}).Build(); err == nil {
		t.Error("expected a missing LLM to be rejected")
	}
}
