package queue

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"strconv"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/google/uuid"
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

// PersistentQueue implements Redis-backed queue with atomic operations
type PersistentQueue struct {
	redis        *redis.Client
	queueKey     string
	processingKey string
	maxSize      int
	ttl          time.Duration
}

// NewPersistentQueue creates a new Redis-backed persistent queue
func NewPersistentQueue(redisURL, queueKey, processingKey string, maxSize int, ttl time.Duration) (*PersistentQueue, error) {
	// Parse Redis URL and create client
	opts, err := redis.ParseURL(redisURL)
	if err != nil {
		return nil, fmt.Errorf("invalid Redis URL: %v", err)
	}

	client := redis.NewClient(opts)

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := client.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to Redis: %v", err)
	}

	log.Printf("Connected to Redis at %s", redisURL)

	return &PersistentQueue{
		redis:         client,
		queueKey:      queueKey,
		processingKey: processingKey,
		maxSize:       maxSize,
		ttl:           ttl,
	}, nil
}

// Enqueue adds a request to the queue with priority ordering
func (pq *PersistentQueue) Enqueue(req *ParseRequest) error {
	// Check queue capacity
	if pq.Size() >= pq.maxSize {
		return ErrQueueFull
	}

	// Validate request size (prevent memory exhaustion)
	if len(req.ImageData) > 50*1024*1024 { // 50MB limit
		return ErrImageTooLarge
	}

	// Serialize request to JSON
	data, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("failed to serialize request: %v", err)
	}

	// Add to Redis sorted set (sorted by enqueue time + priority)
	// Lower scores = higher priority (negative priority values)
	score := float64(req.EnqueuedAt.Unix()) - float64(req.Priority)*1000000

	ctx := context.Background()
	return pq.redis.ZAdd(ctx, pq.queueKey, &redis.Z{
		Score:  score,
		Member: data,
	}).Err()
}

// Lua script for atomic batch processing - ensures no race conditions
const batchProcessingScript = `
local queue_key = KEYS[1]
local processing_key = KEYS[2]
local batch_id = ARGV[1]
local max_size = tonumber(ARGV[2])
local current_time = ARGV[3]

-- Atomically pop from queue
local items = redis.call('ZPOPMIN', queue_key, max_size)
if #items == 0 then
    return nil
end

-- Build batch data
local batch_data = {
    id = batch_id,
    status = 'PROCESSING',
    requests = {},
    sent_at = current_time,
    retry_count = 0
}

-- Extract request data (items come as [member1, score1, member2, score2, ...])
for i = 1, #items, 2 do
    local request_json = items[i]
    table.insert(batch_data.requests, request_json)
end

-- Store in processing queue with 30-minute TTL
local batch_json = cjson.encode(batch_data)
redis.call('SET', processing_key .. ':' .. batch_id, batch_json, 'EX', 1800)

return batch_json
`

// StartBatchProcessing atomically creates a batch for processing
func (pq *PersistentQueue) StartBatchProcessing(batchID string, maxSize int) (*BatchState, error) {
	ctx := context.Background()

	// Execute atomic Lua script
	result := pq.redis.Eval(ctx, batchProcessingScript, []string{
		pq.queueKey,
		pq.processingKey,
	}, batchID, maxSize, time.Now().Unix())

	if result.Err() != nil {
		return nil, fmt.Errorf("atomic batch processing failed: %v", result.Err())
	}

	batchJSON := result.Val()
	if batchJSON == nil {
		return nil, nil // No requests available
	}

	// Parse the batch data returned by Lua script
	var batchData struct {
		ID       string   `json:"id"`
		Status   string   `json:"status"`
		Requests []string `json:"requests"`
		SentAt   int64    `json:"sent_at"`
	}

	if err := json.Unmarshal([]byte(batchJSON.(string)), &batchData); err != nil {
		return nil, fmt.Errorf("failed to parse batch data: %v", err)
	}

	// Convert JSON strings back to ParseRequest objects
	var requests []*ParseRequest
	for _, reqJSON := range batchData.Requests {
		var req ParseRequest
		if err := json.Unmarshal([]byte(reqJSON), &req); err != nil {
			log.Printf("Warning: skipping malformed request: %v", err)
			continue
		}
		requests = append(requests, &req)
	}

	if len(requests) == 0 {
		// Clean up empty batch
		pq.redis.Del(ctx, pq.processingKey+":"+batchID)
		return nil, nil
	}

	return &BatchState{
		ID:       batchData.ID,
		Status:   BatchProcessing,
		Requests: requests,
		SentAt:   time.Unix(batchData.SentAt, 0),
	}, nil
}

// CompleteBatch marks a batch as completed and removes it from processing
func (pq *PersistentQueue) CompleteBatch(batchID string) error {
	ctx := context.Background()
	return pq.redis.Del(ctx, pq.processingKey+":"+batchID).Err()
}

// FailBatch handles batch failure - requeue for retry or mark as permanently failed
func (pq *PersistentQueue) FailBatch(batchID string, errorMsg string) error {
	ctx := context.Background()
	batchKey := pq.processingKey + ":" + batchID

	// Get batch data
	batchData := pq.redis.Get(ctx, batchKey).Val()
	if batchData == "" {
		return nil // Batch already cleaned up
	}

	var batch BatchState
	if err := json.Unmarshal([]byte(batchData), &batch); err != nil {
		return err
	}

	// Increment retry count
	batch.RetryCount++
	batch.LastError = errorMsg

	// Requeue requests for retry (with exponential backoff)
	if batch.RetryCount < 3 {
		return pq.requeueBatchWithDelay(&batch, errorMsg)
	}

	// Max retries exceeded - clean up and log
	log.Printf("Batch %s permanently failed after %d retries: %s", batchID, batch.RetryCount, errorMsg)
	pq.redis.Del(ctx, batchKey)
	return fmt.Errorf("batch permanently failed: %s", errorMsg)
}

// requeueBatchWithDelay requeues batch requests with exponential backoff
func (pq *PersistentQueue) requeueBatchWithDelay(batch *BatchState, errorMsg string) error {
	ctx := context.Background()

	// Exponential backoff delay
	delaySeconds := int(math.Pow(2, float64(batch.RetryCount))) * 30 // 30s, 60s, 120s
	retryTime := time.Now().Add(time.Duration(delaySeconds) * time.Second)

	log.Printf("Requeuing batch %s (attempt %d) with %ds delay", batch.ID, batch.RetryCount, delaySeconds)

	// Requeue all requests with delayed score
	pipe := pq.redis.TxPipeline()
	for _, req := range batch.Requests {
		data, _ := json.Marshal(req)
		score := float64(retryTime.Unix()) - float64(req.Priority)*1000000
		pipe.ZAdd(ctx, pq.queueKey, &redis.Z{Score: score, Member: data})
	}

	// Clean up processing entry
	pipe.Del(ctx, pq.processingKey+":"+batch.ID)

	_, err := pipe.Exec(ctx)
	return err
}

// Size returns the current number of requests in the queue
func (pq *PersistentQueue) Size() int {
	ctx := context.Background()
	count := pq.redis.ZCard(ctx, pq.queueKey)
	return int(count.Val())
}

// RecoverStuckBatches finds and recovers batches that have been processing too long
func (pq *PersistentQueue) RecoverStuckBatches() error {
	ctx := context.Background()
	pattern := pq.processingKey + ":*"
	cursor := uint64(0)

	for {
		// Use SCAN instead of KEYS to avoid blocking Redis
		keys, newCursor, err := pq.redis.Scan(ctx, cursor, pattern, 100).Result()
		if err != nil {
			return fmt.Errorf("scan failed: %v", err)
		}

		// Process keys in batches to avoid memory spikes
		for _, key := range keys {
			if err := pq.recoverSingleBatch(ctx, key); err != nil {
				log.Printf("Failed to recover batch %s: %v", key, err)
			}
		}

		cursor = newCursor
		if cursor == 0 {
			break // Scan complete
		}

		// Small delay to avoid overwhelming Redis
		time.Sleep(10 * time.Millisecond)
	}

	return nil
}

// recoverSingleBatch recovers a single stuck batch
func (pq *PersistentQueue) recoverSingleBatch(ctx context.Context, key string) error {
	batchData := pq.redis.Get(ctx, key).Val()
	if batchData == "" {
		return nil // Batch already cleaned up
	}

	var batch BatchState
	if err := json.Unmarshal([]byte(batchData), &batch); err != nil {
		// Clean up corrupted batch data
		pq.redis.Del(ctx, key)
		return fmt.Errorf("corrupted batch data: %v", err)
	}

	// If batch stuck in processing > 15 minutes, recover
	if batch.Status == BatchProcessing && time.Since(batch.SentAt) > 15*time.Minute {
		log.Printf("Recovering stuck batch %s after %v", batch.ID, time.Since(batch.SentAt))
		return pq.FailBatch(batch.ID, "batch timeout - recovered by system")
	}

	return nil
}

// GetBatchStatus returns the status of a specific batch
func (pq *PersistentQueue) GetBatchStatus(batchID string) (*BatchState, error) {
	ctx := context.Background()
	batchKey := pq.processingKey + ":" + batchID

	batchData := pq.redis.Get(ctx, batchKey).Val()
	if batchData == "" {
		return nil, ErrBatchNotFound
	}

	var batch BatchState
	if err := json.Unmarshal([]byte(batchData), &batch); err != nil {
		return nil, fmt.Errorf("failed to parse batch data: %v", err)
	}

	return &batch, nil
}

// Close closes the Redis connection
func (pq *PersistentQueue) Close() error {
	return pq.redis.Close()
}