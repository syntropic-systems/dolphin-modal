package modal

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"sync"
	"time"

	"connector/internal/queue"
)

// Circuit breaker states
type CBState int

const (
	CBClosed CBState = iota
	CBOpen
	CBHalfOpen
)

// ModalClient handles HTTP communication with Modal API
type ModalClient struct {
	baseURL       string
	httpClient    *http.Client
	timeout       time.Duration
	retries       int
	circuitBreaker *CircuitBreaker
	semaphore     chan struct{} // Limit concurrent requests
}

// CircuitBreaker prevents cascade failures to Modal API
type CircuitBreaker struct {
	mu              sync.RWMutex
	failureCount    int
	lastFailureTime time.Time
	state          CBState
	threshold      int
	cooldown       time.Duration
}

// Modal API request/response types matching modal_app.py
type ModalBatchRequest struct {
	Images []ModalImage `json:"images"`
}

type ModalImage struct {
	ImageData string `json:"image_data"`
	Filename  string `json:"filename"`
}

type ModalBatchResponse struct {
	BatchID         string              `json:"batch_id"`
	ProcessingTime  float64            `json:"processing_time_seconds"`
	ImagesProcessed int                `json:"images_processed"`
	TotalElements   int                `json:"total_elements"`
	Results         []ModalImageResult `json:"results"`
	Metadata        ModalMetadata      `json:"metadata"`
}

type ModalImageResult struct {
	Filename     string          `json:"filename"`
	Elements     []queue.Element `json:"elements"`
	ElementCount int             `json:"element_count"`
}

type ModalMetadata struct {
	ModelDevice        string  `json:"model_device"`
	ContainerID        string  `json:"container_id"`
	ContainerRequests  int     `json:"container_requests"`
	BatchThroughput    float64 `json:"batch_throughput"`
	ElementsPerSecond  float64 `json:"elements_per_second"`
}

// NewModalClient creates a new Modal API client with circuit breaker
func NewModalClient(baseURL string, timeout time.Duration, maxConcurrent int) *ModalClient {
	return &ModalClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: timeout,
			Transport: &http.Transport{
				MaxIdleConns:        10,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     90 * time.Second,
			},
		},
		timeout: timeout,
		retries: 3,
		circuitBreaker: &CircuitBreaker{
			threshold: 5,
			cooldown:  30 * time.Second,
			state:     CBClosed,
		},
		semaphore: make(chan struct{}, maxConcurrent),
	}
}

// ProcessBatch sends a batch to Modal API with retry and circuit breaker
func (mc *ModalClient) ProcessBatch(batch *queue.BatchState) (*ModalBatchResponse, error) {
	// Check circuit breaker
	if mc.circuitBreaker.IsOpen() {
		return nil, fmt.Errorf("circuit breaker open - Modal API unavailable")
	}

	// Acquire semaphore (limit concurrent requests)
	select {
	case mc.semaphore <- struct{}{}:
		defer func() { <-mc.semaphore }()
	case <-time.After(5 * time.Second):
		return nil, fmt.Errorf("too many concurrent requests to Modal")
	}

	// Convert internal batch to Modal format with REQUEST ID correlation
	modalReq := &ModalBatchRequest{
		Images: make([]ModalImage, len(batch.Requests)),
	}

	for i, req := range batch.Requests {
		// ROBUST CORRELATION: Embed request ID in filename for bulletproof routing
		// Format: req_{requestID}_{batchIndex}_{originalFilename}
		correlationFilename := fmt.Sprintf("req_%s_idx_%d_%s", req.ID[:12], i, req.Filename)

		modalReq.Images[i] = ModalImage{
			ImageData: req.ImageData,
			Filename:  correlationFilename,
		}
	}

	// Implement exponential backoff retry
	var lastErr error
	for attempt := 0; attempt < mc.retries; attempt++ {
		resp, err := mc.attemptRequest(modalReq)
		if err == nil {
			mc.circuitBreaker.RecordSuccess()
			return resp, nil
		}

		lastErr = err
		mc.circuitBreaker.RecordFailure()

		if attempt < mc.retries-1 {
			backoff := time.Duration(math.Pow(2, float64(attempt))) * time.Second
			log.Printf("Modal request failed (attempt %d/%d), retrying in %v: %v",
				attempt+1, mc.retries, backoff, err)
			time.Sleep(backoff)
		}
	}

	return nil, fmt.Errorf("max retries exceeded: %v", lastErr)
}

// attemptRequest makes a single HTTP request to Modal
func (mc *ModalClient) attemptRequest(modalReq *ModalBatchRequest) (*ModalBatchResponse, error) {
	// Send HTTP request to Modal
	body, err := json.Marshal(modalReq)
	if err != nil {
		return nil, fmt.Errorf("request serialization failed: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), mc.timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "POST", mc.baseURL+"-parse-batch.modal.run", bytes.NewBuffer(body))
	if err != nil {
		return nil, fmt.Errorf("request creation failed: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := mc.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 500 {
		return nil, fmt.Errorf("Modal server error: %d", resp.StatusCode)
	}
	if resp.StatusCode >= 400 {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Modal client error %d: %s", resp.StatusCode, string(bodyBytes))
	}

	// Parse Modal response
	var modalResp ModalBatchResponse
	if err := json.NewDecoder(resp.Body).Decode(&modalResp); err != nil {
		return nil, fmt.Errorf("response parsing failed: %v", err)
	}

	// Validate response completeness
	if len(modalResp.Results) == 0 {
		return nil, fmt.Errorf("Modal returned empty results")
	}

	return &modalResp, nil
}

// HealthCheck tests Modal API availability
func (mc *ModalClient) HealthCheck() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "GET", mc.baseURL+"-health.modal.run", nil)
	if err != nil {
		return err
	}

	resp, err := mc.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return fmt.Errorf("Modal health check failed: %d", resp.StatusCode)
	}

	return nil
}

// Circuit breaker implementation
func (cb *CircuitBreaker) IsOpen() bool {
	cb.mu.RLock()
	defer cb.mu.RUnlock()

	if cb.state == CBOpen {
		if time.Since(cb.lastFailureTime) > cb.cooldown {
			cb.state = CBHalfOpen
			return false
		}
		return true
	}
	return false
}

func (cb *CircuitBreaker) RecordSuccess() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.failureCount = 0
	cb.state = CBClosed
}

func (cb *CircuitBreaker) RecordFailure() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.failureCount++
	cb.lastFailureTime = time.Now()

	if cb.failureCount >= cb.threshold {
		cb.state = CBOpen
		log.Printf("Circuit breaker opened after %d failures", cb.failureCount)
	}
}

// GetState returns current circuit breaker state (for monitoring)
func (cb *CircuitBreaker) GetState() (CBState, int, time.Time) {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.state, cb.failureCount, cb.lastFailureTime
}