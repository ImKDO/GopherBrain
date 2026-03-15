.PHONY: proto build test lint run-server run-worker clean setup-onnx

PROTO_DIR := api/proto/v1
PROTO_OUT := api/proto/v1
GOPATH_BIN := $(shell go env GOPATH)/bin
export PATH := $(GOPATH_BIN):$(PATH)

ONNX_VERSION := 1.22.0
ONNX_LIB_DIR := $(CURDIR)/third_party/onnxruntime

proto:
	PATH=$(GOPATH_BIN):$$PATH protoc \
		--go_out=$(PROTO_OUT) --go_opt=paths=source_relative \
		--go-grpc_out=$(PROTO_OUT) --go-grpc_opt=paths=source_relative \
		-I $(PROTO_DIR) \
		$(PROTO_DIR)/*.proto

setup-onnx:
	@mkdir -p $(ONNX_LIB_DIR)
	@if [ ! -f $(ONNX_LIB_DIR)/lib/libonnxruntime.dylib ]; then \
		echo "Downloading ONNX Runtime $(ONNX_VERSION) for macOS arm64..."; \
		curl -fSL https://github.com/microsoft/onnxruntime/releases/download/v$(ONNX_VERSION)/onnxruntime-osx-arm64-$(ONNX_VERSION).tgz -o /tmp/ort.tgz && \
		tar xzf /tmp/ort.tgz --strip-components=1 -C $(ONNX_LIB_DIR) && \
		rm -f /tmp/ort.tgz && \
		echo "ONNX Runtime installed to $(ONNX_LIB_DIR)"; \
	else \
		echo "ONNX Runtime already present"; \
	fi

build:
	go build -o bin/server ./cmd/server
	go build -o bin/worker ./cmd/worker

test:
	go test -v -race ./...

lint:
	golangci-lint run ./...

run-server: setup-onnx
	DYLD_LIBRARY_PATH=$(ONNX_LIB_DIR)/lib go run ./cmd/server --config configs/config.yaml

run-worker: setup-onnx
	DYLD_LIBRARY_PATH=$(ONNX_LIB_DIR)/lib go run ./cmd/worker --config configs/config.yaml

clean:
	rm -rf bin/
