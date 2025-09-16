# Dolphin Aggregation Service Makefile

# Go parameters
GOCMD=go
GOBUILD=$(GOCMD) build
GOCLEAN=$(GOCMD) clean
GOTEST=$(GOCMD) test
GOGET=$(GOCMD) get
GOMOD=$(GOCMD) mod
GOFMT=gofmt
GOLINT=golangci-lint

# Build parameters
BINARY_NAME=connector
BINARY_UNIX=$(BINARY_NAME)_unix
MAIN_PATH=./cmd/server

# Docker parameters
DOCKER_IMAGE=connector
DOCKER_TAG=latest

# Default target
.PHONY: all
all: deps fmt lint test build

# Install dependencies
.PHONY: deps
deps:
	$(GOMOD) download
	$(GOMOD) tidy

# Format code
.PHONY: fmt
fmt:
	$(GOFMT) -s -w .

# Lint code
.PHONY: lint
lint:
	$(GOLINT) run ./...

# Run tests
.PHONY: test
test:
	$(GOTEST) -v -race -coverprofile=coverage.out ./...

# View test coverage
.PHONY: coverage
coverage: test
	$(GOCMD) tool cover -html=coverage.out

# Build for current platform
.PHONY: build
build:
	$(GOBUILD) -o $(BINARY_NAME) -v $(MAIN_PATH)

# Build for Linux (for Docker)
.PHONY: build-linux
build-linux:
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 $(GOBUILD) -o $(BINARY_UNIX) -v $(MAIN_PATH)

# Build Docker image
.PHONY: docker-build
docker-build:
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) -f deployments/Dockerfile .

# Run Docker Compose (development)
.PHONY: docker-up
docker-up:
	docker-compose -f deployments/docker-compose.yml up -d

# Stop Docker Compose
.PHONY: docker-down
docker-down:
	docker-compose -f deployments/docker-compose.yml down

# View Docker logs
.PHONY: docker-logs
docker-logs:
	docker-compose -f deployments/docker-compose.yml logs -f connector

# Run locally (requires Redis)
.PHONY: run-local
run-local: build
	CONFIG_PATH=configs/local.yaml ./$(BINARY_NAME)

# Run with production config
.PHONY: run-prod
run-prod: build
	CONFIG_PATH=configs/production.yaml ./$(BINARY_NAME)

# Deploy to Kubernetes
.PHONY: k8s-deploy
k8s-deploy:
	kubectl apply -f deployments/k8s/

# Remove from Kubernetes
.PHONY: k8s-remove
k8s-remove:
	kubectl delete -f deployments/k8s/

# View Kubernetes logs
.PHONY: k8s-logs
k8s-logs:
	kubectl logs -f deployment/connector

# Health check
.PHONY: health
health:
	curl -f http://localhost:8080/health || exit 1

# Queue status
.PHONY: queue-status
queue-status:
	curl -s http://localhost:8080/queue/status | jq

# Test single request
.PHONY: test-request
test-request:
	@echo "Testing single parse request..."
	@curl -X POST http://localhost:8080/parse \
		-H "Content-Type: application/json" \
		-d '{"image_data":"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==","filename":"test.png"}' \
		| jq || echo "Request failed"

# Load test (requires hey)
.PHONY: load-test
load-test:
	@echo "Running load test with 10 concurrent requests..."
	hey -n 100 -c 10 -m POST \
		-H "Content-Type: application/json" \
		-d '{"image_data":"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==","filename":"load_test.png"}' \
		http://localhost:8080/parse

# Clean build artifacts
.PHONY: clean
clean:
	$(GOCLEAN)
	rm -f $(BINARY_NAME)
	rm -f $(BINARY_UNIX)
	rm -f coverage.out

# Install development tools
.PHONY: install-tools
install-tools:
	go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
	go install github.com/rakyll/hey@latest

# Help
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all          - Run deps, fmt, lint, test, build"
	@echo "  deps         - Download dependencies"
	@echo "  fmt          - Format code"
	@echo "  lint         - Lint code"
	@echo "  test         - Run tests"
	@echo "  coverage     - View test coverage"
	@echo "  build        - Build binary"
	@echo "  build-linux  - Build Linux binary"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-up    - Start with Docker Compose"
	@echo "  docker-down  - Stop Docker Compose"
	@echo "  docker-logs  - View Docker logs"
	@echo "  run-local    - Run locally with local config"
	@echo "  run-prod     - Run locally with production config"
	@echo "  k8s-deploy   - Deploy to Kubernetes"
	@echo "  k8s-remove   - Remove from Kubernetes"
	@echo "  k8s-logs     - View Kubernetes logs"
	@echo "  health       - Check service health"
	@echo "  queue-status - Check queue status"
	@echo "  test-request - Send test request"
	@echo "  load-test    - Run load test"
	@echo "  clean        - Clean build artifacts"
	@echo "  install-tools - Install development tools"
	@echo "  help         - Show this help"