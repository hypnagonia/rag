package store

import (
	"fmt"
	"os"
	"path/filepath"

	"go.etcd.io/bbolt"
)

func CompactFile(path string) (before, after int64, err error) {
	info, err := os.Stat(path)
	if err != nil {
		return 0, 0, err
	}
	before = info.Size()

	tmpPath := filepath.Join(filepath.Dir(path), fmt.Sprintf(".%s.compact", filepath.Base(path)))
	os.Remove(tmpPath)

	src, err := bbolt.Open(path, 0600, &bbolt.Options{ReadOnly: true})
	if err != nil {
		return before, 0, err
	}

	dst, err := bbolt.Open(tmpPath, 0600, nil)
	if err != nil {
		src.Close()
		return before, 0, err
	}

	if err := bbolt.Compact(dst, src, 0); err != nil {
		dst.Close()
		src.Close()
		os.Remove(tmpPath)
		return before, 0, err
	}

	dst.Close()
	src.Close()

	if err := os.Rename(tmpPath, path); err != nil {
		os.Remove(tmpPath)
		return before, 0, err
	}

	info, err = os.Stat(path)
	if err != nil {
		return before, 0, err
	}

	return before, info.Size(), nil
}
