package fs

import (
	"os"
	"path/filepath"

	"github.com/bmatcuk/doublestar/v4"
)

type Walker struct {
	includes []string
	excludes []string
}

func NewWalker(includes, excludes []string) *Walker {
	if len(includes) == 0 {
		includes = []string{"**/*"}
	}
	return &Walker{
		includes: includes,
		excludes: excludes,
	}
}

type FileInfo struct {
	Path    string
	ModTime int64
	Size    int64
}

func (w *Walker) Walk(root string) ([]FileInfo, error) {
	var files []FileInfo

	root, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}

	err = filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() {

			relPath, err := filepath.Rel(root, path)
			if err != nil {
				return err
			}
			if w.shouldExclude(relPath + "/") {
				return filepath.SkipDir
			}
			return nil
		}

		relPath, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}

		if w.shouldInclude(relPath) && !w.shouldExclude(relPath) {
			files = append(files, FileInfo{
				Path:    path,
				ModTime: info.ModTime().Unix(),
				Size:    info.Size(),
			})
		}

		return nil
	})

	return files, err
}

func (w *Walker) shouldInclude(path string) bool {
	for _, pattern := range w.includes {
		matched, err := doublestar.Match(pattern, path)
		if err == nil && matched {
			return true
		}
	}
	return false
}

func (w *Walker) shouldExclude(path string) bool {
	for _, pattern := range w.excludes {
		matched, err := doublestar.Match(pattern, path)
		if err == nil && matched {
			return true
		}
	}
	return false
}

func ReadFile(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(data), nil
}
