package server

import (
	"encoding/base64"
	"fmt"
	"log"
	"time"
	"bytes"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"

	"connector/internal/queue"
)

// @title Dolphin Document Parser API
// @version 1.0
// @description A high-performance document parsing service with batch processing and intelligent queueing
// @termsOfService http://swagger.io/terms/

// @contact.name API Support
// @contact.email support@example.com

// @license.name MIT
// @license.url https://opensource.org/licenses/MIT

// @host 100.123.49.70:8199
// @BasePath /

// DocumentRequest represents the incoming API request
type DocumentRequest struct {
	ImageData string `json:"image_data" binding:"required" example:"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==" format:"base64"` // Base64 encoded image data (JPEG/PNG)
	Filename  string `json:"filename" example:"document.jpg"` // Optional filename for the document
}

// ParseResponse represents the API response
type ParseResponse struct {
	RequestID      string    `json:"request_id" example:"550e8400-e29b-41d4-a716-446655440000"`
	Filename       string    `json:"filename" example:"document.jpg"`
	Elements       []Element `json:"elements"`
	ElementCount   int       `json:"element_count" example:"5"`
	ProcessingTime float64   `json:"processing_time_seconds" example:"45.2"`
	Success        bool      `json:"success" example:"true"`
	Error          string    `json:"error,omitempty" example:""`
	BatchID        string    `json:"batch_id" example:"batch_123"`
	Timestamp      string    `json:"timestamp" example:"2025-09-17T14:30:00Z"`
	BatchIndex     int       `json:"batch_index,omitempty" example:"0"`
}

// Element represents a document element
type Element struct {
	Label        string `json:"label" example:"title" enums:"title,para,table,figure,formula"`        // Element type
	Bbox         []int  `json:"bbox" example:"100,200,400,300"`         // Bounding box [x1, y1, x2, y2]
	Text         string `json:"text" example:"Document Title"`         // Extracted text content
	ReadingOrder int    `json:"reading_order" example:"1"` // Sequential reading order
}

// HealthResponse represents health check response
type HealthResponse struct {
	Status           string `json:"status" example:"healthy" enums:"healthy,degraded,unhealthy"`
	ModalAPIError    string `json:"modal_api_error,omitempty" example:""`
	QueueDepth       int    `json:"queue_depth" example:"0"`
	Timestamp        string `json:"timestamp" example:"2025-09-17T14:30:00Z"`
	WorkerPoolStats  map[string]interface{} `json:"worker_pool"`
}

// QueueStatusResponse represents queue status response
type QueueStatusResponse struct {
	QueueDepth              int     `json:"queue_depth" example:"5"`
	MaxQueueSize            int     `json:"max_queue_size" example:"1000"`
	UtilizationPercent      float64 `json:"utilization_percent" example:"0.5"`
	EstimatedWaitTimeMinutes int     `json:"estimated_wait_time_minutes" example:"2"`
}

// ErrorResponse represents error response
type ErrorResponse struct {
	Error string `json:"error" example:"Invalid request format"`
}

// handleParseRequest handles individual document parsing requests with long-polling
// @Summary Parse document
// @Description Parse a document image using OCR and layout analysis. Supports JPEG and PNG formats up to 50MB.
// @Tags document
// @Accept json
// @Produce json
// @Param request body DocumentRequest true "Document parsing request"
// @Success 200 {object} ParseResponse "Successfully parsed document"
// @Failure 400 {object} ErrorResponse "Invalid request format or unsupported image"
// @Failure 413 {object} ErrorResponse "Image too large (>50MB)"
// @Failure 503 {object} ErrorResponse "Service at capacity"
// @Failure 500 {object} ErrorResponse "Internal processing error"
// @Failure 408 {object} ErrorResponse "Processing timeout"
// @Router /parse [post]
func (hs *HTTPServer) handleParseRequest(c *gin.Context) {

	var req DocumentRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": "invalid request format"})
		return
	}

	// Comprehensive request validation
	if err := hs.validateRequest(&req); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}


	// Generate unique filename if not provided
	filename := req.Filename
	if filename == "" {
		filename = fmt.Sprintf("upload_%s.jpg", uuid.New().String()[:8])
	}

	// Create internal request
	parseReq := &queue.ParseRequest{
		ID:           uuid.New().String(),
		ImageData:    req.ImageData,
		Filename:     filename,
		Priority:     queue.NORMAL_PRIORITY,
		EnqueuedAt:   time.Now(),
		ResponseChan: make(chan *queue.ParseResponse, 1),
		Context:      c.Request.Context(),
	}

	// Register for response routing
	hs.responseRouter.RegisterRequest(parseReq)

	// Add to queue with error handling
	if err := hs.aggregationService.GetQueue().Enqueue(parseReq); err != nil {
		// Clean up response router registration
		hs.responseRouter.CleanupRequest(parseReq.ID)

		if err == queue.ErrQueueFull {
			c.JSON(503, gin.H{"error": "service at capacity"})
		} else if err == queue.ErrImageTooLarge {
			c.JSON(413, gin.H{"error": "image too large - max 50MB"})
		} else {
			log.Printf("Queue error for request %s: %v", parseReq.ID, err)
			c.JSON(500, gin.H{"error": "internal error"})
		}
		return
	}

	log.Printf("Request %s queued", parseReq.ID)

	// Wait for response (long-polling) - synchronous operation
	select {
	case response := <-parseReq.ResponseChan:
		if response.Success {
			c.JSON(200, response)
		} else {
			c.JSON(500, gin.H{"error": response.Error})
		}

	case <-c.Request.Context().Done():
		log.Printf("Client disconnected for request %s", parseReq.ID)
		c.JSON(408, gin.H{"error": "request timeout"})

	case <-time.After(15 * time.Minute):
		log.Printf("Processing timeout for request %s", parseReq.ID)
		c.JSON(408, gin.H{"error": "processing timeout"})
	}
}

// validateRequest performs comprehensive request validation
func (hs *HTTPServer) validateRequest(req *DocumentRequest) error {
	if len(req.ImageData) == 0 {
		return fmt.Errorf("image_data is required")
	}

	// Decode and validate base64
	decoded, err := base64.StdEncoding.DecodeString(req.ImageData)
	if err != nil {
		return fmt.Errorf("invalid base64 encoding")
	}

	// Check file size (50MB limit)
	if len(decoded) > 50*1024*1024 {
		return queue.ErrImageTooLarge
	}

	// Validate image format (basic check)
	if !hs.isValidImageFormat(decoded) {
		return fmt.Errorf("unsupported image format - only JPEG/PNG supported")
	}

	return nil
}

// isValidImageFormat checks if the data represents a valid image
func (hs *HTTPServer) isValidImageFormat(data []byte) bool {
	if len(data) < 8 {
		return false
	}

	// Check for JPEG magic bytes
	if bytes.HasPrefix(data, []byte{0xFF, 0xD8, 0xFF}) {
		return true
	}

	// Check for PNG magic bytes
	if bytes.HasPrefix(data, []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}) {
		return true
	}

	return false
}

// estimateWaitTime provides wait time estimate based on queue depth
func (hs *HTTPServer) estimateWaitTime(queueDepth int) int {
	// Rough estimate: 32 images per batch, 45s per batch, 4 concurrent containers
	batchesInQueue := (queueDepth + 31) / 32  // Round up
	parallelBatches := min(batchesInQueue, 4) // Max 4 containers
	timePerBatch := 45 // seconds

	if parallelBatches == 0 {
		return 0
	}

	return (batchesInQueue * timePerBatch) / parallelBatches / 60 // Convert to minutes
}

// handleHealthCheck returns service health status
// @Summary Health check
// @Description Check the health status of the service including Modal API connectivity and queue status
// @Tags monitoring
// @Produce json
// @Success 200 {object} HealthResponse "Service is healthy"
// @Success 503 {object} HealthResponse "Service is degraded or unhealthy"
// @Router /health [get]
func (hs *HTTPServer) handleHealthCheck(c *gin.Context) {
	status := hs.aggregationService.GetHealthStatus()

	if status["status"] == "healthy" {
		c.JSON(200, status)
	} else {
		c.JSON(503, status)
	}
}

// handleQueueStatus returns detailed queue status
// @Summary Queue status
// @Description Get detailed information about the current queue status and estimated wait times
// @Tags monitoring
// @Produce json
// @Success 200 {object} QueueStatusResponse "Queue status information"
// @Router /queue/status [get]
func (hs *HTTPServer) handleQueueStatus(c *gin.Context) {
	queueDepth := hs.aggregationService.GetQueue().Size()
	maxQueueSize := hs.config.Queue.MaxSize

	status := gin.H{
		"queue_depth": queueDepth,
		"max_queue_size": maxQueueSize,
		"utilization_percent": float64(queueDepth) / float64(maxQueueSize) * 100,
		"estimated_wait_time_minutes": hs.estimateWaitTime(queueDepth),
	}

	c.JSON(200, status)
}

// handleMetrics returns Prometheus metrics (placeholder)
// @Summary Metrics
// @Description Get Prometheus-compatible metrics (placeholder implementation)
// @Tags monitoring
// @Produce plain
// @Success 200 {string} string "Prometheus metrics format"
// @Router /metrics [get]
func (hs *HTTPServer) handleMetrics(c *gin.Context) {
	// TODO: Implement Prometheus metrics
	c.String(200, "# Metrics endpoint - implement Prometheus integration\n")
}

// handleDemoRequest handles single demo document parsing requests
// @Summary Demo endpoint
// @Description Demo endpoint for testing single document parsing with example response
// @Tags demo
// @Accept json
// @Produce json
// @Param request body DocumentRequest true "Document parsing request (same as /parse)"
// @Success 200 {object} ParseResponse "Successfully parsed document (demo)"
// @Failure 400 {object} ErrorResponse "Invalid request format"
// @Router /demo [post]
func (hs *HTTPServer) handleDemoRequest(c *gin.Context) {
	// This is the same as parse request - just a different endpoint for demo purposes
	hs.handleParseRequest(c)
}

// Helper function
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}