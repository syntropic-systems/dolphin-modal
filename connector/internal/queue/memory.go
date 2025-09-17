package queue

import (
	"context"
	"fmt"
	"time"
)

// Request priorities
type RequestPriority int

const (
	LOW_PRIORITY    RequestPriority = 0
	NORMAL_PRIORITY RequestPriority = 1
	HIGH_PRIORITY   RequestPriority = 2
)

// Batch status values
type BatchStatus string

const (
	BatchPending    BatchStatus = "PENDING"
	BatchProcessing BatchStatus = "PROCESSING"
	BatchCompleted  BatchStatus = "COMPLETED"
	BatchFailed     BatchStatus = "FAILED"
)

// Errors
var (
	ErrQueueFull      = fmt.Errorf("queue is full")
	ErrImageTooLarge  = fmt.Errorf("image too large")
	ErrBatchNotFound  = fmt.Errorf("batch not found")
)

// ParseRequest represents a single document parsing request
type ParseRequest struct {
	ID           string             `json:"id"`
	ImageData    string             `json:"image_data"`
	Filename     string             `json:"filename"`
	Priority     RequestPriority    `json:"priority"`
	EnqueuedAt   time.Time         `json:"enqueued_at"`
	ResponseChan chan *ParseResponse `json:"-"` // Not serialized
	Context      context.Context    `json:"-"` // Not serialized
}

// ParseResponse represents the response sent back to the client
type ParseResponse struct {
	RequestID      string    `json:"request_id"`
	Filename       string    `json:"filename"`
	Elements       []Element `json:"elements"`
	ElementCount   int       `json:"element_count"`
	ProcessingTime float64   `json:"processing_time_seconds"`
	Success        bool      `json:"success"`
	Error          string    `json:"error,omitempty"`
	BatchID        string    `json:"batch_id"`
	Timestamp      time.Time `json:"timestamp"`
	BatchIndex     int       `json:"batch_index,omitempty"`
}

// Element represents a parsed document element
type Element struct {
	Label        string  `json:"label"`        // "title", "para", "table", etc.
	Bbox         []int   `json:"bbox"`         // [x1, y1, x2, y2]
	Text         string  `json:"text"`         // Extracted text content
	ReadingOrder int     `json:"reading_order"` // Sequential order on page
}

// BatchState represents a batch being processed
type BatchState struct {
	ID           string          `json:"id"`
	Status       BatchStatus     `json:"status"`
	Requests     []*ParseRequest `json:"requests"`
	SentAt       time.Time       `json:"sent_at"`
	RetryCount   int            `json:"retry_count"`
	LastError    string         `json:"last_error,omitempty"`
}

// InMemoryQueue implements a simple in-memory queue using Go channels
type InMemoryQueue struct {
	requestChan chan *ParseRequest
	maxSize     int
}

// NewInMemoryQueue creates a new in-memory queue
func NewInMemoryQueue(maxSize int) *InMemoryQueue {
	return &InMemoryQueue{
		requestChan: make(chan *ParseRequest, maxSize),
		maxSize:     maxSize,
	}
}

// Enqueue adds a request to the queue
func (q *InMemoryQueue) Enqueue(req *ParseRequest) error {
	// Validate request size (prevent memory exhaustion)
	if len(req.ImageData) > 50*1024*1024 { // 50MB limit
		return ErrImageTooLarge
	}

	// Try to add to channel (non-blocking)
	select {
	case q.requestChan <- req:
		return nil
	default:
		return ErrQueueFull
	}
}

// StartBatchProcessing gets requests from queue for batch processing
func (q *InMemoryQueue) StartBatchProcessing(batchID string, maxSize int) (*BatchState, error) {
	requests := []*ParseRequest{}

	// Collect up to maxSize requests
	for i := 0; i < maxSize; i++ {
		select {
		case req := <-q.requestChan:
			requests = append(requests, req)
		default:
			// No more requests available
			break
		}
	}

	if len(requests) == 0 {
		return nil, nil // No requests available
	}

	return &BatchState{
		ID:       batchID,
		Status:   BatchProcessing,
		Requests: requests,
		SentAt:   time.Now(),
	}, nil
}

// CompleteBatch marks a batch as completed (no-op for in-memory queue)
func (q *InMemoryQueue) CompleteBatch(batchID string) error {
	// No state to clean up in in-memory queue
	return nil
}

// FailBatch handles batch failure (no-op for in-memory queue)
func (q *InMemoryQueue) FailBatch(batchID string, errorMsg string) error {
	// In-memory queue doesn't support retry - requests are lost
	return nil
}

// Size returns the current number of requests in the queue
func (q *InMemoryQueue) Size() int {
	return len(q.requestChan)
}

// RecoverStuckBatches no-op for in-memory queue
func (q *InMemoryQueue) RecoverStuckBatches() error {
	// No stuck batches to recover in in-memory queue
	return nil
}

// Close closes the queue
func (q *InMemoryQueue) Close() error {
	close(q.requestChan)
	return nil
}

// NewPersistentQueue creates a new in-memory queue (for compatibility)
func NewPersistentQueue(redisURL, queueKey, processingKey string, maxSize int, ttl time.Duration) (*InMemoryQueue, error) {
	// Ignore Redis parameters, create in-memory queue
	return NewInMemoryQueue(maxSize), nil
}