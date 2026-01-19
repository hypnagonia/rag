package analyzer

import (
	"strings"
	"unicode"
)

type Tokenizer struct {
	stemmer   *PorterStemmer
	stopwords map[string]struct{}
	useStem   bool
}

func NewTokenizer(useStemming bool) *Tokenizer {
	var stemmer *PorterStemmer
	if useStemming {
		stemmer = NewPorterStemmer()
	}
	return &Tokenizer{
		stemmer:   stemmer,
		stopwords: defaultStopwords(),
		useStem:   useStemming,
	}
}

func (t *Tokenizer) Tokenize(text string) []string {
	words := splitWords(text)
	tokens := make([]string, 0, len(words))

	for _, word := range words {
		word = strings.ToLower(word)
		if len(word) < 2 {
			continue
		}
		if _, isStop := t.stopwords[word]; isStop {
			continue
		}
		if t.useStem && t.stemmer != nil {
			word = t.stemmer.Stem(word)
		}
		tokens = append(tokens, word)
	}

	return tokens
}

func (t *Tokenizer) CountTokens(text string) int {

	words := splitWords(text)
	if len(words) == 0 {
		return 0
	}

	return int(float64(len(words)) * 1.3)
}

func splitWords(text string) []string {
	var words []string
	var current strings.Builder

	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' {
			current.WriteRune(r)
		} else {
			if current.Len() > 0 {
				words = append(words, current.String())
				current.Reset()
			}
		}
	}
	if current.Len() > 0 {
		words = append(words, current.String())
	}

	return words
}

func defaultStopwords() map[string]struct{} {
	stops := []string{
		"a", "an", "and", "are", "as", "at", "be", "by", "for",
		"from", "has", "he", "in", "is", "it", "its", "of", "on",
		"that", "the", "to", "was", "were", "will", "with", "this",
		"have", "had", "but", "not", "you", "your", "we", "our",
		"they", "their", "she", "her", "his", "if", "or", "so",
		"no", "can", "do", "does", "did", "been", "being", "would",
		"could", "should", "may", "might", "must", "shall", "which",
		"who", "whom", "what", "when", "where", "why", "how", "all",
		"each", "every", "both", "few", "more", "most", "other",
		"some", "such", "than", "too", "very", "just", "also",
	}
	m := make(map[string]struct{}, len(stops))
	for _, s := range stops {
		m[s] = struct{}{}
	}
	return m
}
