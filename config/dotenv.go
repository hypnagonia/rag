package config

import (
	"bufio"
	"os"
	"path/filepath"
	"strings"
)

const DotEnvFileName = ".env"

func LoadDotEnv(dirs ...string) []string {
	var loaded []string
	seen := make(map[string]struct{})

	for _, dir := range dirs {
		if dir == "" {
			continue
		}

		abs, found := findDotEnv(dir)
		if !found {
			continue
		}
		if _, done := seen[abs]; done {
			continue
		}
		seen[abs] = struct{}{}

		if applyDotEnv(abs) {
			loaded = append(loaded, abs)
		}
	}

	return loaded
}

func findDotEnv(dir string) (string, bool) {
	current, err := filepath.Abs(dir)
	if err != nil {
		return "", false
	}

	for {
		candidate := filepath.Join(current, DotEnvFileName)
		if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
			return candidate, true
		}

		parent := filepath.Dir(current)
		if parent == current {
			return "", false
		}
		current = parent
	}
}

func applyDotEnv(path string) bool {
	file, err := os.Open(path)
	if err != nil {
		return false
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		key, value, ok := parseDotEnvLine(scanner.Text())
		if !ok {
			continue
		}
		if _, exists := os.LookupEnv(key); exists {
			continue
		}
		os.Setenv(key, value)
	}

	return scanner.Err() == nil
}

func parseDotEnvLine(line string) (key, value string, ok bool) {
	line = strings.TrimSpace(line)
	if line == "" || strings.HasPrefix(line, "#") {
		return "", "", false
	}

	line = strings.TrimPrefix(line, "export ")
	line = strings.TrimSpace(line)

	eq := strings.Index(line, "=")
	if eq <= 0 {
		return "", "", false
	}

	key = strings.TrimSpace(line[:eq])
	value = strings.TrimSpace(line[eq+1:])

	if len(value) >= 2 {
		first, last := value[0], value[len(value)-1]
		if (first == '"' && last == '"') || (first == '\'' && last == '\'') {
			return key, value[1 : len(value)-1], true
		}
	}

	if hash := strings.Index(value, " #"); hash >= 0 {
		value = strings.TrimSpace(value[:hash])
	}

	return key, value, true
}
