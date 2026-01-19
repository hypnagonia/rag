package analyzer

import (
	"strings"
)

// PorterStemmer implements the Porter stemming algorithm.
type PorterStemmer struct{}

// NewPorterStemmer creates a new Porter stemmer.
func NewPorterStemmer() *PorterStemmer {
	return &PorterStemmer{}
}

// Stem returns the stem of a word using the Porter algorithm.
func (p *PorterStemmer) Stem(word string) string {
	if len(word) < 3 {
		return word
	}

	word = strings.ToLower(word)
	word = step1a(word)
	word = step1b(word)
	word = step1c(word)
	word = step2(word)
	word = step3(word)
	word = step4(word)
	word = step5a(word)
	word = step5b(word)

	return word
}

func isConsonant(word string, i int) bool {
	switch word[i] {
	case 'a', 'e', 'i', 'o', 'u':
		return false
	case 'y':
		if i == 0 {
			return true
		}
		return !isConsonant(word, i-1)
	}
	return true
}

func measure(word string) int {
	n := len(word)
	m := 0
	i := 0

	// Skip initial consonants
	for i < n && isConsonant(word, i) {
		i++
	}

	for i < n {
		// Count vowel sequence
		for i < n && !isConsonant(word, i) {
			i++
		}
		if i >= n {
			break
		}
		m++
		// Count consonant sequence
		for i < n && isConsonant(word, i) {
			i++
		}
	}

	return m
}

func hasVowel(word string) bool {
	for i := 0; i < len(word); i++ {
		if !isConsonant(word, i) {
			return true
		}
	}
	return false
}

func endsDoubleConsonant(word string) bool {
	n := len(word)
	if n < 2 {
		return false
	}
	return word[n-1] == word[n-2] && isConsonant(word, n-1)
}

func endsCVC(word string) bool {
	n := len(word)
	if n < 3 {
		return false
	}
	if !isConsonant(word, n-3) || isConsonant(word, n-2) || !isConsonant(word, n-1) {
		return false
	}
	c := word[n-1]
	return c != 'w' && c != 'x' && c != 'y'
}

func step1a(word string) string {
	if strings.HasSuffix(word, "sses") {
		return word[:len(word)-2]
	}
	if strings.HasSuffix(word, "ies") {
		return word[:len(word)-2]
	}
	if strings.HasSuffix(word, "ss") {
		return word
	}
	if strings.HasSuffix(word, "s") {
		return word[:len(word)-1]
	}
	return word
}

func step1b(word string) string {
	if strings.HasSuffix(word, "eed") {
		stem := word[:len(word)-3]
		if measure(stem) > 0 {
			return word[:len(word)-1]
		}
		return word
	}

	var stem string
	modified := false

	if strings.HasSuffix(word, "ed") {
		stem = word[:len(word)-2]
		if hasVowel(stem) {
			word = stem
			modified = true
		}
	} else if strings.HasSuffix(word, "ing") {
		stem = word[:len(word)-3]
		if hasVowel(stem) {
			word = stem
			modified = true
		}
	}

	if modified {
		if strings.HasSuffix(word, "at") || strings.HasSuffix(word, "bl") || strings.HasSuffix(word, "iz") {
			return word + "e"
		}
		if endsDoubleConsonant(word) {
			c := word[len(word)-1]
			if c != 'l' && c != 's' && c != 'z' {
				return word[:len(word)-1]
			}
		}
		if measure(word) == 1 && endsCVC(word) {
			return word + "e"
		}
	}

	return word
}

func step1c(word string) string {
	if strings.HasSuffix(word, "y") {
		stem := word[:len(word)-1]
		if hasVowel(stem) {
			return stem + "i"
		}
	}
	return word
}

func step2(word string) string {
	suffixes := map[string]string{
		"ational": "ate", "tional": "tion", "enci": "ence", "anci": "ance",
		"izer": "ize", "abli": "able", "alli": "al", "entli": "ent",
		"eli": "e", "ousli": "ous", "ization": "ize", "ation": "ate",
		"ator": "ate", "alism": "al", "iveness": "ive", "fulness": "ful",
		"ousness": "ous", "aliti": "al", "iviti": "ive", "biliti": "ble",
	}

	for suffix, replacement := range suffixes {
		if strings.HasSuffix(word, suffix) {
			stem := word[:len(word)-len(suffix)]
			if measure(stem) > 0 {
				return stem + replacement
			}
			return word
		}
	}
	return word
}

func step3(word string) string {
	suffixes := map[string]string{
		"icate": "ic", "ative": "", "alize": "al", "iciti": "ic",
		"ical": "ic", "ful": "", "ness": "",
	}

	for suffix, replacement := range suffixes {
		if strings.HasSuffix(word, suffix) {
			stem := word[:len(word)-len(suffix)]
			if measure(stem) > 0 {
				return stem + replacement
			}
			return word
		}
	}
	return word
}

func step4(word string) string {
	suffixes := []string{
		"al", "ance", "ence", "er", "ic", "able", "ible", "ant",
		"ement", "ment", "ent", "ion", "ou", "ism", "ate", "iti",
		"ous", "ive", "ize",
	}

	for _, suffix := range suffixes {
		if strings.HasSuffix(word, suffix) {
			stem := word[:len(word)-len(suffix)]
			if measure(stem) > 1 {
				if suffix == "ion" {
					n := len(stem)
					if n > 0 && (stem[n-1] == 's' || stem[n-1] == 't') {
						return stem
					}
				} else {
					return stem
				}
			}
		}
	}
	return word
}

func step5a(word string) string {
	if strings.HasSuffix(word, "e") {
		stem := word[:len(word)-1]
		if measure(stem) > 1 {
			return stem
		}
		if measure(stem) == 1 && !endsCVC(stem) {
			return stem
		}
	}
	return word
}

func step5b(word string) string {
	if measure(word) > 1 && endsDoubleConsonant(word) && word[len(word)-1] == 'l' {
		return word[:len(word)-1]
	}
	return word
}
