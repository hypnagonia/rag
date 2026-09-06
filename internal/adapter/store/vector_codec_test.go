package store

import (
	"encoding/json"
	"math"
	"testing"

	"go.etcd.io/bbolt"
	"rag/internal/port"
)

func TestVectorCodecRoundTripsExactly(t *testing.T) {
	vector := []float32{0, 1, -1, 0.5, -0.25, 3.4028235e38, 1.1754944e-38, 0.1}

	encoded, err := encodeVector(vector, nil)
	if err != nil {
		t.Fatalf("encode failed: %v", err)
	}

	decoded, meta, err := decodeVector(encoded)
	if err != nil {
		t.Fatalf("decode failed: %v", err)
	}
	if meta != nil {
		t.Errorf("expected no metadata, got %v", meta)
	}
	if len(decoded) != len(vector) {
		t.Fatalf("expected %d floats, got %d", len(vector), len(decoded))
	}
	for i := range vector {
		if decoded[i] != vector[i] {
			t.Errorf("index %d: expected %v, got %v (encoding must be lossless)", i, vector[i], decoded[i])
		}
	}
}

func TestVectorCodecIsSmallerThanJSON(t *testing.T) {
	vector := make([]float32, 1024)
	for i := range vector {
		vector[i] = float32(i) * 0.0013
	}

	binary, err := encodeVector(vector, nil)
	if err != nil {
		t.Fatalf("encode failed: %v", err)
	}
	asJSON, _ := json.Marshal(storedVector{Vector: vector})

	if len(binary) >= len(asJSON) {
		t.Errorf("binary (%d B) should be smaller than JSON (%d B)", len(binary), len(asJSON))
	}
	if want := vectorHeaderSize + 1024*4; len(binary) != want {
		t.Errorf("expected exactly %d bytes, got %d", want, len(binary))
	}
	t.Logf("1024 floats: binary %d B vs JSON %d B (%.1fx)", len(binary), len(asJSON), float64(len(asJSON))/float64(len(binary)))
}

func TestVectorCodecRoundTripsMetadata(t *testing.T) {
	encoded, err := encodeVector([]float32{1, 2}, map[string]string{"path": "a.go"})
	if err != nil {
		t.Fatalf("encode failed: %v", err)
	}

	_, meta, err := decodeVector(encoded)
	if err != nil {
		t.Fatalf("decode failed: %v", err)
	}
	if meta["path"] != "a.go" {
		t.Errorf("expected metadata to survive, got %v", meta)
	}
}

func TestVectorCodecReadsLegacyJSON(t *testing.T) {
	legacy, _ := json.Marshal(storedVector{Vector: []float32{1, 2, 3}, Metadata: map[string]string{"k": "v"}})

	if !isLegacyVectorRecord(legacy) {
		t.Error("a JSON record should be detected as legacy")
	}

	vector, meta, err := decodeVector(legacy)
	if err != nil {
		t.Fatalf("decode failed: %v", err)
	}
	if len(vector) != 3 || vector[2] != 3 {
		t.Errorf("legacy vector not decoded: %v", vector)
	}
	if meta["k"] != "v" {
		t.Errorf("legacy metadata not decoded: %v", meta)
	}
}

func TestVectorCodecRejectsCorruptRecords(t *testing.T) {
	if _, _, err := decodeVector(nil); err == nil {
		t.Error("expected an error for an empty record")
	}
	if _, _, err := decodeVector([]byte{9, 0, 0, 0, 0}); err == nil {
		t.Error("expected an error for an unknown format byte")
	}
	truncated := []byte{vectorFormatFloat32, 100, 0, 0, 0, 1, 2}
	if _, _, err := decodeVector(truncated); err == nil {
		t.Error("expected an error for a truncated body")
	}
}

func TestVectorCodecHandlesNaNAndInf(t *testing.T) {
	vector := []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))}
	encoded, _ := encodeVector(vector, nil)
	decoded, _, err := decodeVector(encoded)
	if err != nil {
		t.Fatalf("decode failed: %v", err)
	}
	if !math.IsNaN(float64(decoded[0])) || !math.IsInf(float64(decoded[1]), 1) || !math.IsInf(float64(decoded[2]), -1) {
		t.Errorf("special values not preserved: %v", decoded)
	}
}

func TestVectorStoreMigratesLegacyRecordsInPlace(t *testing.T) {
	db, err := bbolt.Open(t.TempDir()+"/legacy.db", 0600, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	want := []float32{0.5, -0.5, 0.25}
	err = db.Update(func(tx *bbolt.Tx) error {
		b, err := tx.CreateBucketIfNotExists(bucketVectors)
		if err != nil {
			return err
		}
		data, _ := json.Marshal(storedVector{Vector: want})
		return b.Put([]byte("c1"), data)
	})
	if err != nil {
		t.Fatal(err)
	}

	vs, err := NewBoltVectorStore(db, 3)
	if err != nil {
		t.Fatalf("open failed: %v", err)
	}
	if vs.LegacyRecordCount() != 1 {
		t.Fatalf("expected 1 legacy record, got %d", vs.LegacyRecordCount())
	}

	rewritten, err := vs.RewriteLegacyRecords()
	if err != nil {
		t.Fatalf("rewrite failed: %v", err)
	}
	if rewritten != 1 {
		t.Errorf("expected 1 rewritten, got %d", rewritten)
	}

	reopened, err := NewBoltVectorStore(db, 3)
	if err != nil {
		t.Fatal(err)
	}
	if reopened.LegacyRecordCount() != 0 {
		t.Error("no legacy records should remain after migration")
	}

	results, err := reopened.Search(want, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].Score < 0.999 {
		t.Errorf("migrated vector should still match itself, got %+v", results)
	}
}

func TestVectorStoreWritesBinaryRecords(t *testing.T) {
	vs, db := newTestVectorStore(t, 3)

	if err := vs.Upsert([]port.VectorItem{{ID: "a", Vector: []float32{1, 2, 3}}}); err != nil {
		t.Fatal(err)
	}

	db.View(func(tx *bbolt.Tx) error {
		raw := tx.Bucket(bucketVectors).Get([]byte("a"))
		if isLegacyVectorRecord(raw) {
			t.Error("new writes must use the binary format")
		}
		if want := vectorHeaderSize + 3*4; len(raw) != want {
			t.Errorf("expected %d bytes on disk, got %d", want, len(raw))
		}
		return nil
	})
}

func TestVectorStoreHonoursConfiguredEncoding(t *testing.T) {
	for _, tc := range []struct {
		encoding string
		format   byte
		size     int
	}{
		{EncodingFloat32, vectorFormatFloat32, vectorHeaderSize + 3*4},
		{EncodingFloat16, vectorFormatFloat16, vectorHeaderSize + 3*2},
		{EncodingInt8, vectorFormatInt8, vectorHeaderSize + 4 + 3},
	} {
		vs, db := newTestVectorStore(t, 3)
		vs.SetEncoding(tc.encoding)

		if err := vs.Upsert([]port.VectorItem{{ID: "a", Vector: []float32{0.5, -0.5, 0.25}}}); err != nil {
			t.Fatalf("%s: upsert failed: %v", tc.encoding, err)
		}

		db.View(func(tx *bbolt.Tx) error {
			raw := tx.Bucket(bucketVectors).Get([]byte("a"))
			if raw[0] != tc.format {
				t.Errorf("%s: expected format byte %d, got %d", tc.encoding, tc.format, raw[0])
			}
			if len(raw) != tc.size {
				t.Errorf("%s: expected %d bytes, got %d", tc.encoding, tc.size, len(raw))
			}
			return nil
		})
	}
}

func TestReencodeConvertsEveryRecord(t *testing.T) {
	vs, _ := newTestVectorStore(t, 4)

	if err := vs.Upsert([]port.VectorItem{
		{ID: "a", Vector: []float32{0.1, 0.2, 0.3, 0.4}},
		{ID: "b", Vector: []float32{-0.4, 0.3, -0.2, 0.1}},
	}); err != nil {
		t.Fatal(err)
	}

	converted, err := vs.ReencodeAs(EncodingInt8)
	if err != nil {
		t.Fatalf("reencode failed: %v", err)
	}
	if converted != 2 {
		t.Errorf("expected 2 conversions, got %d", converted)
	}

	breakdown := vs.EncodingBreakdown()
	if breakdown[EncodingInt8] != 2 {
		t.Errorf("expected 2 int8 records, got %v", breakdown)
	}

	again, err := vs.ReencodeAs(EncodingInt8)
	if err != nil {
		t.Fatalf("second reencode failed: %v", err)
	}
	if again != 0 {
		t.Errorf("re-encoding to the same format should be a no-op, converted %d", again)
	}

	results, err := vs.Search([]float32{0.1, 0.2, 0.3, 0.4}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].ID != "a" || results[0].Score < 0.999 {
		t.Errorf("quantized vector should still match itself closely, got %+v", results)
	}
}
