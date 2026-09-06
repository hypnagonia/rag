package store

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
)

const (
	vectorFormatBinaryV1 byte = 1
	vectorHeaderSize          = 5
)

func encodeVector(vector []float32, metadata map[string]string) ([]byte, error) {
	var meta []byte
	if len(metadata) > 0 {
		var err error
		meta, err = json.Marshal(metadata)
		if err != nil {
			return nil, err
		}
	}

	out := make([]byte, vectorHeaderSize+len(vector)*4+len(meta))
	out[0] = vectorFormatBinaryV1
	binary.LittleEndian.PutUint32(out[1:5], uint32(len(vector)))

	for i, v := range vector {
		binary.LittleEndian.PutUint32(out[vectorHeaderSize+i*4:], math.Float32bits(v))
	}

	copy(out[vectorHeaderSize+len(vector)*4:], meta)

	return out, nil
}

func decodeVector(data []byte) ([]float32, map[string]string, error) {
	if len(data) == 0 {
		return nil, nil, fmt.Errorf("empty vector record")
	}

	if data[0] == '{' {
		var legacy storedVector
		if err := json.Unmarshal(data, &legacy); err != nil {
			return nil, nil, err
		}
		return legacy.Vector, legacy.Metadata, nil
	}

	if data[0] != vectorFormatBinaryV1 {
		return nil, nil, fmt.Errorf("unknown vector record format %d", data[0])
	}
	if len(data) < vectorHeaderSize {
		return nil, nil, fmt.Errorf("truncated vector header")
	}

	count := int(binary.LittleEndian.Uint32(data[1:5]))
	end := vectorHeaderSize + count*4
	if count < 0 || end > len(data) {
		return nil, nil, fmt.Errorf("truncated vector body: want %d floats", count)
	}

	vector := make([]float32, count)
	for i := range vector {
		vector[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[vectorHeaderSize+i*4:]))
	}

	var metadata map[string]string
	if end < len(data) {
		if err := json.Unmarshal(data[end:], &metadata); err != nil {
			return nil, nil, err
		}
	}

	return vector, metadata, nil
}

func isLegacyVectorRecord(data []byte) bool {
	return len(data) > 0 && data[0] == '{'
}
