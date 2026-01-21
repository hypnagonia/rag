.PHONY: build build-wasm test clean

build:
	go build -o rag ./cmd/rag

build-wasm:
	GOOS=js GOARCH=wasm go build -o examples/wasm/rag.wasm ./cmd/wasm

test:
	go test ./...

clean:
	rm -f rag examples/wasm/rag.wasm
