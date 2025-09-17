package correlation

import (
	"fmt"
	"log"
	"sync"
	"time"
	"context"
	"strings"
	"os"
	"encoding/json"
	"path/filepath"

	"connector/internal/queue"
	"connector/internal/modal"
)

// ResponseRouter handles mapping batch responses back to individual requests
type ResponseRouter struct {
	pendingResponses sync.Map // RequestID -> ResponseContext
	timeout         time.Duration
	cleanupTicker   *time.Ticker
	mu              sync.RWMutex
}

// ResponseContext holds the context for a pending response
type ResponseContext struct {
	ResponseChan chan *queue.ParseResponse
	Context      context.Context
	EnqueuedAt   time.Time
	Filename     string // For correlation
}

// NewResponseRouter creates a new response router
func NewResponseRouter(timeout time.Duration) *ResponseRouter {
	rr := &ResponseRouter{
		timeout:       timeout,
		cleanupTicker: time.NewTicker(5 * time.Minute),
	}

	// Start cleanup goroutine to prevent memory leaks
	go rr.cleanupExpiredResponses()

	return rr
}

// RegisterRequest registers a request for response routing
func (rr *ResponseRouter) RegisterRequest(req *queue.ParseRequest) {
	rr.pendingResponses.Store(req.ID, ResponseContext{
		ResponseChan: req.ResponseChan,
		Context:      req.Context,
		EnqueuedAt:   req.EnqueuedAt,
		Filename:     req.Filename,
	})
}

// RouteResponses routes Modal batch response back to individual clients
func (rr *ResponseRouter) RouteResponses(batch *queue.BatchState, modalResp *modal.ModalBatchResponse) {
	log.Printf("Routing responses for batch %s: %d results for %d requests",
		batch.ID, len(modalResp.Results), len(batch.Requests))

	// Debug: log all filenames returned by Modal
	log.Printf("Modal returned filenames:")
	for i, result := range modalResp.Results {
		log.Printf("  [%d] %s", i, result.Filename)
	}

	// Debug: log all request IDs we're looking for
	log.Printf("Looking for request IDs:")
	for i, req := range batch.Requests {
		log.Printf("  [%d] %s (first 12: %s)", i, req.ID, req.ID[:12])
	}

	// Create correlation map: extract request ID from Modal's correlation filename
	resultMap := make(map[string]modal.ModalImageResult)
	for _, result := range modalResp.Results {
		// Extract request ID from correlation filename
		// Format: req_{requestID}_{batchIndex}_{originalFilename}
		if extractedID := extractRequestIDFromFilename(result.Filename); extractedID != "" {
			resultMap[extractedID] = result
			log.Printf("Mapped filename '%s' to extracted ID '%s'", result.Filename, extractedID)
		} else {
			log.Printf("Warning: Could not extract request ID from filename: %s", result.Filename)
		}
	}

	// Route responses back to individual requests
	for _, req := range batch.Requests {
		var response *queue.ParseResponse

		// Use first 12 characters for correlation since Modal client truncates IDs
		shortID := req.ID[:12]
		if result, exists := resultMap[shortID]; exists {
			// Successful result found by request ID correlation
			response = &queue.ParseResponse{
				RequestID:       req.ID,
				Filename:        req.Filename, // Use original filename, not correlation filename
				Elements:        result.Elements,
				ElementCount:    result.ElementCount,
				ProcessingTime:  modalResp.ProcessingTime,
				Success:         true,
				BatchID:         modalResp.BatchID,
				Timestamp:       time.Now(),
			}
		} else {
			// Missing result for this request ID
			response = &queue.ParseResponse{
				RequestID: req.ID,
				Filename:  req.Filename,
				Success:   false,
				Error:     fmt.Sprintf("no result found for request ID: %s", req.ID),
				BatchID:   modalResp.BatchID,
				Timestamp: time.Now(),
			}
		}

		// Send response to waiting client
		rr.deliverResponse(req.ID, response)
	}
}

// extractRequestIDFromFilename extracts the request ID from Modal's correlation filename
// Format: req_{requestID}_{batchIndex}_{originalFilename}
func extractRequestIDFromFilename(filename string) string {
	if !strings.HasPrefix(filename, "req_") {
		return ""
	}

	parts := strings.Split(filename, "_")
	if len(parts) < 4 {
		return ""
	}

	// Request ID is the second part (parts[1])
	return parts[1]
}

// writeResponseToFile writes the response to a debug file
func (rr *ResponseRouter) writeResponseToFile(response *queue.ParseResponse) {
	// Create debug directory if it doesn't exist
	debugDir := "./debug_responses"
	if err := os.MkdirAll(debugDir, 0755); err != nil {
		log.Printf("Failed to create debug directory: %v", err)
		return
	}

	// Create filename with timestamp
	filename := fmt.Sprintf("response_%s_%d.json", response.RequestID[:12], time.Now().Unix())
	filepath := filepath.Join(debugDir, filename)

	// Write response as JSON
	data, err := json.MarshalIndent(response, "", "  ")
	if err != nil {
		log.Printf("Failed to marshal response: %v", err)
		return
	}

	if err := os.WriteFile(filepath, data, 0644); err != nil {
		log.Printf("Failed to write response file: %v", err)
		return
	}

	log.Printf("Response written to file: %s", filepath)
}

// deliverResponse delivers a response to the waiting client
func (rr *ResponseRouter) deliverResponse(requestID string, response *queue.ParseResponse) {
	// Write response to debug file
	rr.writeResponseToFile(response)

	if ctx, exists := rr.pendingResponses.LoadAndDelete(requestID); exists {
		respCtx := ctx.(ResponseContext)

		select {
		case respCtx.ResponseChan <- response:
			// Response delivered successfully
			log.Printf("Response delivered for request %s", requestID)

		case <-respCtx.Context.Done():
			// Client disconnected
			log.Printf("Client disconnected for request %s", requestID)

		case <-time.After(5 * time.Second):
			// Response channel blocked/full
			log.Printf("Failed to deliver response for request %s - channel blocked", requestID)

			// Try once more with shorter timeout
			select {
			case respCtx.ResponseChan <- response:
				log.Printf("Response delivered on retry for request %s", requestID)
			case <-time.After(1 * time.Second):
				log.Printf("Abandoned response delivery for request %s", requestID)
			}
		}
	} else {
		log.Printf("No pending response context found for request %s", requestID)
	}
}

// HandleBatchFailure handles batch failures by notifying all clients
func (rr *ResponseRouter) HandleBatchFailure(batch *queue.BatchState, errorMsg string) {
	log.Printf("Handling batch failure for batch %s: %s", batch.ID, errorMsg)

	for _, req := range batch.Requests {
		response := &queue.ParseResponse{
			RequestID: req.ID,
			Filename:  req.Filename,
			Success:   false,
			Error:     fmt.Sprintf("batch processing failed: %s", errorMsg),
			BatchID:   batch.ID,
			Timestamp: time.Now(),
		}

		rr.deliverResponse(req.ID, response)
	}
}

// cleanupExpiredResponses removes expired responses to prevent memory leaks
func (rr *ResponseRouter) cleanupExpiredResponses() {
	for range rr.cleanupTicker.C {
		now := time.Now()

		rr.pendingResponses.Range(func(key, value interface{}) bool {
			respCtx := value.(ResponseContext)

			if now.Sub(respCtx.EnqueuedAt) > rr.timeout {
				// Response expired - remove and notify client
				rr.pendingResponses.Delete(key)

				response := &queue.ParseResponse{
					RequestID: key.(string),
					Success:   false,
					Error:     "request timeout - processing took too long",
					Timestamp: now,
				}

				select {
				case respCtx.ResponseChan <- response:
					log.Printf("Sent timeout response for expired request %s", key)
				case <-time.After(1 * time.Second):
					log.Printf("Failed to send timeout response for expired request %s", key)
				}
			}

			return true // Continue iteration
		})
	}
}

// CleanupRequest removes a request from pending responses (for error handling)
func (rr *ResponseRouter) CleanupRequest(requestID string) {
	rr.pendingResponses.Delete(requestID)
}

// Shutdown gracefully shuts down the response router
func (rr *ResponseRouter) Shutdown() {
	if rr.cleanupTicker != nil {
		rr.cleanupTicker.Stop()
	}

	// Notify all pending requests of shutdown
	rr.pendingResponses.Range(func(key, value interface{}) bool {
		respCtx := value.(ResponseContext)

		response := &queue.ParseResponse{
			RequestID: key.(string),
			Success:   false,
			Error:     "service shutting down",
			Timestamp: time.Now(),
		}

		select {
		case respCtx.ResponseChan <- response:
		case <-time.After(1 * time.Second):
		}

		return true
	})
}