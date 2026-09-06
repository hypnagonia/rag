package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

type evalSet struct {
	Corpus    string     `json:"corpus"`
	Questions []evalCase `json:"questions"`
}

type evalCase struct {
	Question string `json:"question"`
	Anchor   string `json:"anchor"`
}

type searchResult struct {
	Path      string  `json:"path"`
	StartLine int     `json:"start_line"`
	EndLine   int     `json:"end_line"`
	Score     float64 `json:"score"`
	Text      string  `json:"text"`
}

type modeSpec struct {
	name  string
	flags []string
}

func main() {
	binary := flag.String("bin", "./rag", "Path to the rag binary")
	corpus := flag.String("corpus", "", "Indexed directory to evaluate against (required)")
	setPath := flag.String("set", "cmd/rageval/questions.json", "Path to the question set")
	k := flag.Int("k", 10, "Cutoff for recall@k")
	modes := flag.String("modes", "lexical,semantic,hybrid", "Comma-separated: lexical, semantic, hybrid, hyde")
	verbose := flag.Bool("v", false, "Print the rank of every question")
	flag.Parse()

	if *corpus == "" {
		fmt.Fprintln(os.Stderr, "Usage: rageval -corpus /path/to/indexed [-k 10] [-modes lexical,semantic,hybrid]")
		flag.PrintDefaults()
		os.Exit(1)
	}

	set, err := loadSet(*setPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("%s\n%d questions, recall@%d, MMR disabled\n\n", set.Corpus, len(set.Questions), *k)
	fmt.Printf("%-12s %10s %8s   %s\n", "MODE", "RECALL", "MRR", "RANKS")

	for _, spec := range parseModes(*modes) {
		hits, mrr, ranks := evaluate(*binary, *corpus, set, spec, *k)
		fmt.Printf("%-12s %6d/%-3d %8.3f   %s\n", spec.name, hits, len(set.Questions), mrr, formatRanks(ranks))

		if *verbose {
			for i, c := range set.Questions {
				fmt.Printf("    %-56s %s\n", truncate(c.Question, 56), rankLabel(ranks[i]))
			}
		}
	}
}

func loadSet(path string) (*evalSet, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var set evalSet
	if err := json.Unmarshal(data, &set); err != nil {
		return nil, err
	}
	if len(set.Questions) == 0 {
		return nil, fmt.Errorf("question set is empty")
	}
	return &set, nil
}

func parseModes(spec string) []modeSpec {
	available := map[string][]string{
		"lexical":  {"--lexical"},
		"semantic": {"--semantic"},
		"hybrid":   {},
		"hyde":     {"--hyde"},
	}

	var out []modeSpec
	for _, name := range strings.Split(spec, ",") {
		name = strings.TrimSpace(name)
		if flags, ok := available[name]; ok {
			out = append(out, modeSpec{name: name, flags: flags})
		}
	}
	return out
}

func evaluate(binary, corpus string, set *evalSet, spec modeSpec, k int) (hits int, mrr float64, ranks []int) {
	ranks = make([]int, len(set.Questions))

	for i, c := range set.Questions {
		rank := rankOfAnchor(binary, corpus, c, spec, k)
		ranks[i] = rank
		if rank > 0 {
			mrr += 1.0 / float64(rank)
			if rank <= k {
				hits++
			}
		}
	}

	return hits, mrr / float64(len(set.Questions)), ranks
}

func rankOfAnchor(binary, corpus string, c evalCase, spec modeSpec, k int) int {
	args := []string{"-d", corpus, "query", "-q", c.Question, "-k", strconv.Itoa(k), "--json", "--no-mmr", "--no-auto-index"}
	args = append(args, spec.flags...)

	out, err := exec.Command(binary, args...).Output()
	if err != nil {
		return 0
	}

	var results []searchResult
	if err := json.Unmarshal(out, &results); err != nil {
		return 0
	}

	needle := strings.ToLower(c.Anchor)
	for i, r := range results {
		if strings.Contains(strings.ToLower(r.Text), needle) {
			return i + 1
		}
	}

	return 0
}

func formatRanks(ranks []int) string {
	parts := make([]string, len(ranks))
	for i, r := range ranks {
		parts[i] = rankLabel(r)
	}
	return "[" + strings.Join(parts, " ") + "]"
}

func rankLabel(rank int) string {
	if rank == 0 {
		return "-"
	}
	return strconv.Itoa(rank)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "…"
}
