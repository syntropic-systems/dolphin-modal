package server

import (
	"context"
	"fmt"
	"log"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"

	"connector/docs"
	"connector/internal/config"
	"connector/internal/correlation"
	"connector/internal/queue"

	_ "connector/docs"
)

// HTTPServer represents the HTTP API server
type HTTPServer struct {
	aggregationService AggregationService
	responseRouter     *correlation.ResponseRouter
	router            *gin.Engine
	config            *config.ServiceConfig
	rateLimiter       *RateLimiter
	server            *http.Server
}

// AggregationService interface for dependency injection
type AggregationService interface {
	GetHealthStatus() map[string]interface{}
	GetQueueStatus() map[string]interface{}
	GetQueue() Queue
}

// Queue interface for queue operations
type Queue interface {
	Size() int
	Enqueue(*queue.ParseRequest) error
	StartBatchProcessing(batchID string, maxSize int) (*queue.BatchState, error)
	CompleteBatch(batchID string) error
	FailBatch(batchID string, errorMsg string) error
	RecoverStuckBatches() error
	Close() error
}

// NewHTTPServer creates a new HTTP server
func NewHTTPServer(
	config *config.ServiceConfig,
	aggregationService AggregationService,
	responseRouter *correlation.ResponseRouter,
) *HTTPServer {
	// Set Gin mode based on log level
	if config.Monitoring.LogLevel == "debug" {
		gin.SetMode(gin.DebugMode)
	} else {
		gin.SetMode(gin.ReleaseMode)
	}

	router := gin.New()

	// Add middleware
	router.Use(gin.Logger())
	router.Use(gin.Recovery())

	hs := &HTTPServer{
		aggregationService: aggregationService,
		responseRouter:     responseRouter,
		router:            router,
		config:            config,
		rateLimiter:       NewRateLimiter(config.Server.RateLimitPerClient),
	}

	// Setup routes
	hs.setupRoutes()

	// Create HTTP server
	hs.server = &http.Server{
		Addr:           fmt.Sprintf(":%d", config.Server.Port),
		Handler:        router,
		ReadTimeout:    config.Server.ReadTimeout,
		WriteTimeout:   config.Server.WriteTimeout,
		MaxHeaderBytes: int(config.Server.MaxRequestSize),
	}

	return hs
}

// setupRoutes configures API routes
func (hs *HTTPServer) setupRoutes() {
	// API routes
	hs.router.POST("/parse", hs.handleParseRequest)
	hs.router.GET("/health", hs.handleHealthCheck)
	hs.router.GET("/metrics", hs.handleMetrics)
	hs.router.GET("/queue/status", hs.handleQueueStatus)
	hs.router.POST("/demo", hs.handleDemoRequest)

	// Swagger documentation
	docs.SwaggerInfo.Host = fmt.Sprintf("100.123.49.70:%d", hs.config.Server.Port)
	hs.router.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))
}

// Start starts the HTTP server
func (hs *HTTPServer) Start() error {
	log.Printf("Starting HTTP server on port %d", hs.config.Server.Port)
	return hs.server.ListenAndServe()
}

// Shutdown gracefully shuts down the HTTP server
func (hs *HTTPServer) Shutdown(ctx context.Context) error {
	log.Printf("Shutting down HTTP server...")
	return hs.server.Shutdown(ctx)
}