# Dolphin Connector Service

A high-performance Go-based connector service that acts as intelligent middleware between individual document parsing requests and Modal's batch API. The service aggregates requests into optimal batches for 10-50x throughput improvement while maintaining synchronous operation.

## 🚀 Key Features

- **Synchronous Operation**: All clients wait for actual results, no async job IDs
- **Intelligent Batching**: Adaptive batch sizing based on queue depth and performance
- **High Reliability**: Redis-backed persistence with atomic operations and circuit breakers
- **Auto-Recovery**: Automatic recovery of stuck batches and graceful error handling
- **Production Ready**: Comprehensive monitoring, health checks, and deployment configs

## 📊 Performance Characteristics

- **Throughput**: 256 images/minute (4 containers × 32 images × 2 batches/min)
- **Latency**: 30-60s empty queue, 1-30 minutes under load
- **Queue Capacity**: 10,000 requests (6-hour buffer)
- **Burst Handling**: 2000+ concurrent requests safely queued

## 🏗️ Architecture

```
2000+ Individual        ┌─────────────────┐         Modal Auto-Scaling
Requests               │                 │
     │                 │ Aggregation     │  Batch 1 ──> Container 1 (32 images)
     └─────────────────>│ Service         │  Batch 2 ──> Container 2 (32 images)
                        │ (Single Thread) │  Batch 3 ──> Container 3 (32 images)
     ┌─────────────────>│                 │  Batch 4 ──> Container 4 (32 images)
     │                 │ - Queue Mgmt    │
Individual              │ - Batch Form    │  ┌─────Response Routing─────┐
Responses               │ - Response Route│  │                          │
     │                 └─────────────────┘  │                          │
     └────────────────────────────────────────┘                          │
```

### Core Components

1. **PersistentQueue**: Redis-backed queue with Lua scripts for atomic operations
2. **AdaptiveBatchFormer**: Smart batch sizing based on load and performance
3. **ModalClient**: HTTP client with circuit breaker and exponential backoff
4. **ResponseRouter**: Request ID-based correlation for accurate response routing
5. **WorkerPool**: Bounded concurrency control matching Modal's container limits

## 🚀 Quick Start

### Prerequisites

- Go 1.21+
- Redis 6.0+
- Docker (optional)
- Kubernetes (optional)

### Local Development

```bash
# Clone and setup
git clone <repository>
cd connector

# Install dependencies
make deps

# Start Redis
docker run -d -p 6379:6379 redis:7-alpine

# Run locally
make run-local
```

### Docker Deployment

```bash
# Build and start services
make docker-build
make docker-up

# View logs
make docker-logs

# Test the service
make test-request
```

### Kubernetes Deployment

```bash
# Deploy to cluster
make k8s-deploy

# Check status
kubectl get pods
make k8s-logs
```

## 📋 API Endpoints

### Parse Document

```http
POST /parse
Content-Type: application/json

{
  "image_data": "base64_encoded_image_string",
  "filename": "document.jpg"
}
```

**Response:**
```json
{
  "request_id": "abc12345",
  "filename": "document.jpg",
  "processing_time_seconds": 45.67,
  "timestamp": "2024-01-15T10:30:00Z",
  "elements": [
    {
      "label": "title",
      "bbox": [271, 188, 1194, 221],
      "text": "Document Title",
      "reading_order": 0
    }
  ],
  "element_count": 15,
  "batch_id": "batch_xyz789",
  "success": true
}
```

### Health Check

```http
GET /health
```

### Queue Status

```http
GET /queue/status
```

## ⚙️ Configuration

Configure via YAML files in `configs/`:

```yaml
server:
  port: 8080
  max_connections: 3000
  backpressure_threshold: 2500

queue:
  redis_url: "redis://localhost:6379"
  max_size: 10000
  ttl: 6h

modal:
  base_url: "https://abhishekgautam011--dolphin-parser-dolphinparser"
  timeout: 120s
  max_concurrent_batches: 4
```

Environment variables:
- `REDIS_URL`: Redis connection URL
- `MODAL_API_ENDPOINT`: Modal API base URL
- `LOG_LEVEL`: Logging level (debug, info, warn, error)
- `CONFIG_PATH`: Path to configuration file

## 🔧 Development

### Available Make Targets

```bash
make help                 # Show all available targets
make all                  # Run deps, fmt, lint, test, build
make test                 # Run tests with coverage
make lint                 # Run linters
make docker-up            # Start development environment
make health               # Check service health
make load-test            # Run performance test
```

### Project Structure

```
connector/
├── cmd/server/           # Application entry point
├── internal/
│   ├── server/          # HTTP server and handlers
│   ├── queue/           # Redis queue and batch formation
│   ├── modal/           # Modal API client
│   ├── correlation/     # Response routing
│   └── config/          # Configuration management
├── pkg/worker/          # Worker pool
├── deployments/         # Docker and Kubernetes configs
└── configs/             # YAML configuration files
```

## 📊 Monitoring

### Health Endpoints

- `GET /health`: Service health status
- `GET /queue/status`: Queue depth and utilization
- `GET /metrics`: Prometheus metrics (planned)

### Key Metrics

- Queue depth and utilization
- Batch formation rate and size
- Modal API response times and success rate
- Worker pool utilization
- Request latency histogram

### Logging

Structured JSON logging with configurable levels:
- Request/response correlation IDs
- Batch processing lifecycle
- Error details and recovery actions
- Performance metrics

## 🚨 Error Handling

### Circuit Breaker

Modal API failures trigger circuit breaker:
- Opens after 5 consecutive failures
- 30-second cooldown period
- Automatic health checks for recovery

### Queue Management

- Automatic batch recovery every 5 minutes
- Exponential backoff for retries (max 3 attempts)
- Graceful degradation during overload

### Backpressure

- HTTP 503 when queue >83% full
- Rate limiting: 10 requests/minute per IP
- Request size limits: 50MB per image

## 🔒 Security

- Non-root container execution
- Input validation and sanitization
- Request size and rate limiting
- No sensitive data in logs
- Secure error messages

## 🐳 Deployment Options

### Docker Compose (Development)

```bash
make docker-up
```

Includes Redis and optional Redis Insight for monitoring.

### Kubernetes (Production)

```bash
make k8s-deploy
```

Features:
- High availability (2 replicas)
- Resource limits and requests
- Health checks and rolling updates
- ConfigMaps for configuration
- Ingress with rate limiting

## 📈 Performance Tuning

### Batch Size Optimization

The service automatically adjusts batch sizes based on:
- Queue depth (1-32 images per batch)
- Historical processing times
- Modal API performance

### Resource Limits

- **Memory**: 512MB-1GB per instance
- **CPU**: 0.5-1.0 cores per instance
- **Concurrent batches**: 4 (matching Modal containers)

### Redis Configuration

For production, consider:
- Redis cluster for high availability
- Persistent storage for queue durability
- Memory optimization for large queues

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`make test`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: See design document for architecture details
- **Issues**: Report bugs and feature requests via GitHub Issues
- **Monitoring**: Use health endpoints and logs for troubleshooting

---

**Production Grade Architecture**: This implementation follows the comprehensive design document with atomic operations, panic recovery, circuit breakers, and robust error handling suitable for 2000+ concurrent users.