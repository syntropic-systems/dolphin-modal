package server

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"

	"connector/internal/config"
	"connector/internal/correlation"
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
	hs.router.POST("/parse", hs.handleParseRequest)
	hs.router.GET("/health", hs.handleHealthCheck)
	hs.router.GET("/metrics", hs.handleMetrics)
	hs.router.GET("/queue/status", hs.handleQueueStatus)
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