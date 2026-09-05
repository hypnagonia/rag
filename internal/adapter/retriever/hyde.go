package retriever

import (
	"fmt"
	"strings"
	"sync"

	"rag/internal/domain"
	"rag/internal/port"
)

const hydeSystemPrompt = `You write a short passage that could plausibly be the answer to a question,
in the style and vocabulary of the source material being searched.
Write 2-4 sentences of plain prose. Use concrete nouns, names and terminology that would
appear in the source text itself, not in the question.
Do not hedge, do not explain your reasoning, do not answer "I don't know".
If you are unsure of the facts, invent plausible specifics - this text is used only as a
search probe, never shown to a user.`

type HyDERetriever struct {
	inner port.Retriever
	llm   port.LLM
	cache port.HypotheticalCache

	mu    sync.Mutex
	stats HyDEStats
}

type HyDEStats struct {
	Hypothetical string
	CacheHit     bool
	LLMCalls     int
	Err          error
}

type HyDEBuilder struct {
	inner port.Retriever
	llm   port.LLM
	cache port.HypotheticalCache
}

func NewHyDEBuilder() *HyDEBuilder {
	return &HyDEBuilder{}
}

func (b *HyDEBuilder) Inner(r port.Retriever) *HyDEBuilder {
	b.inner = r
	return b
}

func (b *HyDEBuilder) LLM(l port.LLM) *HyDEBuilder {
	b.llm = l
	return b
}

func (b *HyDEBuilder) Cache(c port.HypotheticalCache) *HyDEBuilder {
	b.cache = c
	return b
}

func (b *HyDEBuilder) Build() (*HyDERetriever, error) {
	if b.inner == nil {
		return nil, fmt.Errorf("HyDE requires an inner retriever")
	}
	if b.llm == nil {
		return nil, fmt.Errorf("HyDE requires an LLM")
	}
	return &HyDERetriever{inner: b.inner, llm: b.llm, cache: b.cache}, nil
}

func (r *HyDERetriever) Stats() HyDEStats {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.stats
}

func (r *HyDERetriever) Search(query string, k int) ([]domain.ScoredChunk, error) {
	hypothetical, cacheHit, calls, err := r.hypothetical(query)

	r.mu.Lock()
	r.stats = HyDEStats{Hypothetical: hypothetical, CacheHit: cacheHit, LLMCalls: calls, Err: err}
	r.mu.Unlock()

	if err != nil || strings.TrimSpace(hypothetical) == "" {
		return r.inner.Search(query, k)
	}

	return r.inner.Search(query+"\n\n"+hypothetical, k)
}

func (r *HyDERetriever) hypothetical(query string) (text string, cacheHit bool, calls int, err error) {
	if r.cache != nil {
		if cached, ok := r.cache.Get(query); ok {
			return cached, true, 0, nil
		}
	}

	userPrompt := fmt.Sprintf("Question: %s\n\nWrite the passage:", query)
	text, err = r.llm.GenerateWithSystem(hydeSystemPrompt, userPrompt)
	if err != nil {
		return "", false, 1, err
	}

	text = strings.TrimSpace(text)
	if r.cache != nil && text != "" {
		if putErr := r.cache.Put(query, text); putErr != nil {
			return text, false, 1, nil
		}
	}

	return text, false, 1, nil
}
