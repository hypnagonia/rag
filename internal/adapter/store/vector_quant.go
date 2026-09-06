package store

import "math"

func float32ToFloat16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int32((bits>>23)&0xff) - 127
	mant := bits & 0x7fffff

	if exp == 128 {
		if mant != 0 {
			return sign | 0x7e00
		}
		return sign | 0x7c00
	}

	if exp > 15 {
		return sign | 0x7c00
	}

	if exp < -14 {
		if exp < -25 {
			return sign
		}
		significand := uint32(mant | 0x800000)
		shift := uint(-exp - 1)
		half := (significand + (1 << (shift - 1))) >> shift
		return sign | uint16(half)
	}

	half := uint32(exp+15)<<10 | mant>>13
	if mant&0x1000 != 0 && (mant&0xfff != 0 || half&1 != 0) {
		half++
	}

	return sign | uint16(half)
}

func float16ToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)

	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		shift := uint32(0)
		for mant&0x400 == 0 {
			mant <<= 1
			shift++
		}
		mant &= 0x3ff
		return math.Float32frombits(sign | (127-15-shift)<<23 | mant<<13)
	case 0x1f:
		return math.Float32frombits(sign | 0xff<<23 | mant<<13)
	default:
		return math.Float32frombits(sign | (exp-15+127)<<23 | mant<<13)
	}
}

func quantizeInt8(vector []float32) (scale float32, out []int8) {
	var maxAbs float64
	for _, v := range vector {
		if a := math.Abs(float64(v)); a > maxAbs {
			maxAbs = a
		}
	}

	out = make([]int8, len(vector))
	if maxAbs == 0 {
		return 0, out
	}

	scale = float32(maxAbs / 127)
	inv := 127 / maxAbs

	for i, v := range vector {
		q := math.Round(float64(v) * inv)
		if q > 127 {
			q = 127
		} else if q < -127 {
			q = -127
		}
		out[i] = int8(q)
	}

	return scale, out
}

func dequantizeInt8(scale float32, values []int8) []float32 {
	out := make([]float32, len(values))
	for i, q := range values {
		out[i] = float32(q) * scale
	}
	return out
}
