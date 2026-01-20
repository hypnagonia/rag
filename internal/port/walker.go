package port

type FileWalker interface {
	Walk(root string) ([]FileInfo, error)
}

type FileInfo struct {
	Path    string
	ModTime int64
	Size    int64
}

type FileReader interface {
	ReadFile(path string) (string, error)
}
