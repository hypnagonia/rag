package store

import (
	"math"
	"math/rand"
	"testing"
)

func TestFloat16AccurateInNormalRange(t *testing.T) {
	rng := rand.New(rand.NewSource(1))

	var worst float64
	for i := 0; i < 20000; i++ {
		v := float32(rng.NormFloat64() * 0.1)
		if math.Abs(float64(v)) < 1e-4 {
			continue
		}
		back := float16ToFloat32(float32ToFloat16(v))
		rel := math.Abs(float64(back-v) / float64(v))
		if rel > worst {
			worst = rel
		}
	}

	if worst > 0.001 {
		t.Errorf("float16 relative error %.5f exceeds 0.1%% in the normal range", worst)
	}
	t.Logf("float16 worst relative error above 1e-4: %.6f", worst)
}

func TestFloat16PreservesCosineSimilarity(t *testing.T) {
	rng := rand.New(rand.NewSource(3))

	vector := make([]float32, 1024)
	for i := range vector {
		vector[i] = float32(rng.NormFloat64() * 0.05)
	}

	back := make([]float32, len(vector))
	for i, v := range vector {
		back[i] = float16ToFloat32(float32ToFloat16(v))
	}

	var dot, na, nb float64
	for i := range vector {
		dot += float64(vector[i]) * float64(back[i])
		na += float64(vector[i]) * float64(vector[i])
		nb += float64(back[i]) * float64(back[i])
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nb))

	if cos < 0.99999 {
		t.Errorf("float16 self-cosine %.7f is too low", cos)
	}
	t.Logf("float16 self-cosine after round trip: %.7f", cos)
}

func TestFloat16HandlesSpecialValues(t *testing.T) {
	cases := map[string]float32{
		"zero":     0,
		"negzero":  float32(math.Copysign(0, -1)),
		"one":      1,
		"negone":   -1,
		"tiny":     1e-8,
		"inf":      float32(math.Inf(1)),
		"neginf":   float32(math.Inf(-1)),
		"overflow": 1e30,
	}

	for name, v := range cases {
		back := float16ToFloat32(float32ToFloat16(v))
		switch name {
		case "inf", "overflow":
			if !math.IsInf(float64(back), 1) {
				t.Errorf("%s: expected +Inf, got %v", name, back)
			}
		case "neginf":
			if !math.IsInf(float64(back), -1) {
				t.Errorf("%s: expected -Inf, got %v", name, back)
			}
		case "tiny":
			if back != 0 && math.Abs(float64(back)) > 1e-6 {
				t.Errorf("%s: expected flush to ~0, got %v", name, back)
			}
		default:
			if math.Abs(float64(back-v)) > 1e-3 {
				t.Errorf("%s: expected ~%v, got %v", name, v, back)
			}
		}
	}

	if !math.IsNaN(float64(float16ToFloat32(float32ToFloat16(float32(math.NaN()))))) {
		t.Error("NaN should survive the round trip")
	}
}

func TestInt8RoundTripPreservesDirection(t *testing.T) {
	rng := rand.New(rand.NewSource(2))

	vector := make([]float32, 1024)
	for i := range vector {
		vector[i] = float32(rng.NormFloat64())
	}

	scale, q := quantizeInt8(vector)
	back := dequantizeInt8(scale, q)

	var dot, na, nb float64
	for i := range vector {
		dot += float64(vector[i]) * float64(back[i])
		na += float64(vector[i]) * float64(vector[i])
		nb += float64(back[i]) * float64(back[i])
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nb))

	if cos < 0.9999 {
		t.Errorf("int8 round trip cosine %.6f is too low", cos)
	}
	t.Logf("int8 self-cosine after quantization: %.6f", cos)
}

func TestInt8HandlesZeroVector(t *testing.T) {
	scale, q := quantizeInt8(make([]float32, 8))
	if scale != 0 {
		t.Errorf("zero vector should give zero scale, got %v", scale)
	}
	back := dequantizeInt8(scale, q)
	for i, v := range back {
		if v != 0 {
			t.Errorf("index %d: expected 0, got %v", i, v)
		}
	}
}

func TestEncodingSizesMatchExpectations(t *testing.T) {
	vector := make([]float32, 1024)
	for i := range vector {
		vector[i] = float32(i) * 0.001
	}

	cases := []struct {
		format byte
		want   int
		name   string
	}{
		{vectorFormatFloat32, vectorHeaderSize + 1024*4, "float32"},
		{vectorFormatFloat16, vectorHeaderSize + 1024*2, "float16"},
		{vectorFormatInt8, vectorHeaderSize + 4 + 1024, "int8"},
	}

	for _, c := range cases {
		data, err := encodeVectorAs(vector, nil, c.format)
		if err != nil {
			t.Fatalf("%s: encode failed: %v", c.name, err)
		}
		if len(data) != c.want {
			t.Errorf("%s: expected %d bytes, got %d", c.name, c.want, len(data))
		}

		decoded, _, err := decodeVector(data)
		if err != nil {
			t.Fatalf("%s: decode failed: %v", c.name, err)
		}
		if len(decoded) != len(vector) {
			t.Errorf("%s: expected %d floats, got %d", c.name, len(vector), len(decoded))
		}
	}
}
