package store

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
)

const (
	vectorFormatFloat32 byte = 1
	vectorFormatFloat16 byte = 2
	vectorFormatInt8    byte = 3
	vectorHeaderSize         = 5
)

const (
	EncodingFloat32 = "float32"
	EncodingFloat16 = "float16"
	EncodingInt8    = "int8"
)

func formatForEncoding(name string) byte {
	switch name {
	case EncodingFloat16:
		return vectorFormatFloat16
	case EncodingInt8:
		return vectorFormatInt8
	default:
		return vectorFormatFloat32
	}
}

func encodingName(format byte) string {
	switch format {
	case vectorFormatFloat16:
		return EncodingFloat16
	case vectorFormatInt8:
		return EncodingInt8
	default:
		return EncodingFloat32
	}
}

func encodeVector(vector []float32, metadata map[string]string) ([]byte, error) {
	return encodeVectorAs(vector, metadata, vectorFormatFloat32)
}

func encodeVectorAs(vector []float32, metadata map[string]string, format byte) ([]byte, error) {
	var meta []byte
	if len(metadata) > 0 {
		var err error
		meta, err = json.Marshal(metadata)
		if err != nil {
			return nil, err
		}
	}

	var body []byte
	switch format {
	case vectorFormatFloat16:
		body = make([]byte, len(vector)*2)
		for i, v := range vector {
			binary.LittleEndian.PutUint16(body[i*2:], float32ToFloat16(v))
		}
	case vectorFormatInt8:
		scale, q := quantizeInt8(vector)
		body = make([]byte, 4+len(q))
		binary.LittleEndian.PutUint32(body[0:4], math.Float32bits(scale))
		for i, v := range q {
			body[4+i] = byte(v)
		}
	default:
		format = vectorFormatFloat32
		body = make([]byte, len(vector)*4)
		for i, v := range vector {
			binary.LittleEndian.PutUint32(body[i*4:], math.Float32bits(v))
		}
	}

	out := make([]byte, vectorHeaderSize+len(body)+len(meta))
	out[0] = format
	binary.LittleEndian.PutUint32(out[1:5], uint32(len(vector)))
	copy(out[vectorHeaderSize:], body)
	copy(out[vectorHeaderSize+len(body):], meta)

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

	format := data[0]
	if format != vectorFormatFloat32 && format != vectorFormatFloat16 && format != vectorFormatInt8 {
		return nil, nil, fmt.Errorf("unknown vector record format %d", format)
	}
	if len(data) < vectorHeaderSize {
		return nil, nil, fmt.Errorf("truncated vector header")
	}

	count := int(binary.LittleEndian.Uint32(data[1:5]))
	if count < 0 {
		return nil, nil, fmt.Errorf("negative vector length")
	}

	var bodyLen int
	switch format {
	case vectorFormatFloat16:
		bodyLen = count * 2
	case vectorFormatInt8:
		bodyLen = 4 + count
	default:
		bodyLen = count * 4
	}

	end := vectorHeaderSize + bodyLen
	if end > len(data) {
		return nil, nil, fmt.Errorf("truncated vector body: want %d floats", count)
	}

	body := data[vectorHeaderSize:end]
	vector := make([]float32, count)

	switch format {
	case vectorFormatFloat16:
		for i := range vector {
			vector[i] = float16ToFloat32(binary.LittleEndian.Uint16(body[i*2:]))
		}
	case vectorFormatInt8:
		scale := math.Float32frombits(binary.LittleEndian.Uint32(body[0:4]))
		q := make([]int8, count)
		for i := 0; i < count; i++ {
			q[i] = int8(body[4+i])
		}
		vector = dequantizeInt8(scale, q)
	default:
		for i := range vector {
			vector[i] = math.Float32frombits(binary.LittleEndian.Uint32(body[i*4:]))
		}
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
