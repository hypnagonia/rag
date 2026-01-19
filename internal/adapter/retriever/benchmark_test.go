package retriever

import (
	"math"
	"testing"
)

func TestPrecisionAtK(t *testing.T) {
	cases := []struct {
		name      string
		retrieved []string
		relevant  []string
		wantP     float64
	}{
		{"perfect", []string{"a", "b", "c"}, []string{"a", "b", "c"}, 1.0},
		{"partial", []string{"a", "b", "x"}, []string{"a", "b", "c"}, 0.666},
		{"none", []string{"x", "y", "z"}, []string{"a", "b", "c"}, 0.0},
		{"empty_retrieved", []string{}, []string{"a", "b"}, 0.0},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := PrecisionAtK(tc.retrieved, tc.relevant)
			if diff := p - tc.wantP; diff > 0.01 || diff < -0.01 {
				t.Errorf("precision = %.3f, want %.3f", p, tc.wantP)
			}
		})
	}
}

func TestRecallAtK(t *testing.T) {
	cases := []struct {
		name      string
		retrieved []string
		relevant  []string
		wantR     float64
	}{
		{"perfect", []string{"a", "b", "c"}, []string{"a", "b", "c"}, 1.0},
		{"partial", []string{"a", "b", "x"}, []string{"a", "b", "c"}, 0.666},
		{"none", []string{"x", "y", "z"}, []string{"a", "b", "c"}, 0.0},
		{"empty_relevant", []string{"a", "b"}, []string{}, 0.0},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := RecallAtK(tc.retrieved, tc.relevant)
			if diff := r - tc.wantR; diff > 0.01 || diff < -0.01 {
				t.Errorf("recall = %.3f, want %.3f", r, tc.wantR)
			}
		})
	}
}

func TestMRR(t *testing.T) {
	cases := []struct {
		name      string
		retrieved []string
		relevant  string
		wantMRR   float64
	}{
		{"first", []string{"a", "b", "c"}, "a", 1.0},
		{"second", []string{"x", "a", "c"}, "a", 0.5},
		{"third", []string{"x", "y", "a"}, "a", 0.333},
		{"missing", []string{"x", "y", "z"}, "a", 0.0},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			mrr := ReciprocalRank(tc.retrieved, tc.relevant)
			if diff := mrr - tc.wantMRR; diff > 0.01 || diff < -0.01 {
				t.Errorf("MRR = %.3f, want %.3f", mrr, tc.wantMRR)
			}
		})
	}
}

func TestNDCG(t *testing.T) {
	cases := []struct {
		name     string
		scores   []float64
		ideal    []float64
		wantNDCG float64
	}{
		{"perfect", []float64{3, 2, 1}, []float64{3, 2, 1}, 1.0},
		{"reversed", []float64{1, 2, 3}, []float64{3, 2, 1}, 0.790},
		{"zeros", []float64{0, 0, 0}, []float64{3, 2, 1}, 0.0},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ndcg := NDCG(tc.scores, tc.ideal)
			if diff := ndcg - tc.wantNDCG; diff > 0.01 || diff < -0.01 {
				t.Errorf("NDCG = %.3f, want %.3f", ndcg, tc.wantNDCG)
			}
		})
	}
}

func PrecisionAtK(retrieved, relevant []string) float64 {
	if len(retrieved) == 0 {
		return 0
	}
	relevantSet := make(map[string]bool)
	for _, r := range relevant {
		relevantSet[r] = true
	}
	hits := 0
	for _, r := range retrieved {
		if relevantSet[r] {
			hits++
		}
	}
	return float64(hits) / float64(len(retrieved))
}

func RecallAtK(retrieved, relevant []string) float64 {
	if len(relevant) == 0 {
		return 0
	}
	relevantSet := make(map[string]bool)
	for _, r := range relevant {
		relevantSet[r] = true
	}
	hits := 0
	for _, r := range retrieved {
		if relevantSet[r] {
			hits++
		}
	}
	return float64(hits) / float64(len(relevant))
}

func ReciprocalRank(retrieved []string, relevant string) float64 {
	for i, r := range retrieved {
		if r == relevant {
			return 1.0 / float64(i+1)
		}
	}
	return 0
}

func NDCG(scores, ideal []float64) float64 {
	dcg := calculateDCG(scores)
	idcg := calculateDCG(ideal)
	if idcg == 0 {
		return 0
	}
	return dcg / idcg
}

func calculateDCG(scores []float64) float64 {
	dcg := 0.0
	for i, score := range scores {
		dcg += score / math.Log2(float64(i+2))
	}
	return dcg
}
