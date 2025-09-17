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

// DocumentRequest represents the incoming API request
type DocumentRequest struct {
	ImageData string `json:"image_data" binding:"required"`
	Filename  string `json:"filename"`
}

// handleParseRequest handles individual document parsing requests with long-polling
func (hs *HTTPServer) handleParseRequest(c *gin.Context) {
	// Rate limiting check
	clientIP := c.ClientIP()
	if !hs.rateLimiter.Allow(clientIP) {
		c.JSON(429, gin.H{
			"error": "rate limit exceeded",
			"retry_after_seconds": 60,
		})
		return
	}

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

	// Check backpressure before accepting request
	queueDepth := hs.aggregationService.GetQueue().Size()
	if queueDepth >= hs.config.Server.BackpressureThreshold {
		c.JSON(503, gin.H{
			"error": "service at capacity",
			"queue_depth": queueDepth,
			"estimated_wait_minutes": hs.estimateWaitTime(queueDepth),
			"retry_after_seconds": 60,
		})
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

	log.Printf("Request %s queued (queue depth: %d)", parseReq.ID, queueDepth+1)

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
func (hs *HTTPServer) handleHealthCheck(c *gin.Context) {
	status := hs.aggregationService.GetHealthStatus()

	if status["status"] == "healthy" {
		c.JSON(200, status)
	} else {
		c.JSON(503, status)
	}
}

// handleQueueStatus returns detailed queue status
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
func (hs *HTTPServer) handleMetrics(c *gin.Context) {
	// TODO: Implement Prometheus metrics
	c.String(200, "# Metrics endpoint - implement Prometheus integration\n")
}

// Helper function
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}