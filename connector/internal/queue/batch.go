package queue

import (
	"log"
	"sync"
	"time"
)

// AdaptiveBatchFormer implements smart batch formation based on queue depth and performance
type AdaptiveBatchFormer struct {
	queue        *PersistentQueue
	maxBatchSize int
	minBatchSize int
	maxWaitTime  time.Duration
	metrics      *BatchMetrics
}

// BatchMetrics tracks performance for adaptive batch sizing
type BatchMetrics struct {
	mu                sync.RWMutex
	avgProcessingTime time.Duration
	recentBatchSizes  []int
	lastOptimization  time.Time
}

// NewAdaptiveBatchFormer creates a new adaptive batch former
func NewAdaptiveBatchFormer(queue *PersistentQueue, maxBatchSize, minBatchSize int, maxWaitTime time.Duration) *AdaptiveBatchFormer {
	return &AdaptiveBatchFormer{
		queue:        queue,
		maxBatchSize: maxBatchSize,
		minBatchSize: minBatchSize,
		maxWaitTime:  maxWaitTime,
		metrics:      &BatchMetrics{},
	}
}

// GetOptimalBatchSize returns the optimal batch size based on current conditions
func (abf *AdaptiveBatchFormer) GetOptimalBatchSize() int {
	queueDepth := abf.queue.Size()

	// Adaptive strategy based on queue depth and recent performance
	var batchSize int
	switch {
	case queueDepth >= 1000:
		// High load: maximize throughput with full batches
		batchSize = 32

	case queueDepth >= 200:
		// Medium-high load: balance throughput and latency
		batchSize = 24

	case queueDepth >= 50:
		// Medium load: favor slightly smaller batches for better latency
		batchSize = 16

	case queueDepth >= 10:
		// Low load: small batches for low latency
		batchSize = 8

	case queueDepth > 0:
		// Very low load: process immediately, any size
		batchSize = min(queueDepth, 4)

	default:
		// No requests
		return 0
	}

	// Performance-based adjustment
	if abf.metrics != nil {
		abf.metrics.mu.RLock()
		avgTime := abf.metrics.avgProcessingTime
		abf.metrics.mu.RUnlock()

		// If processing is slow, reduce batch size for better latency
		if avgTime > 60*time.Second && batchSize > 8 {
			batchSize = batchSize / 2
			log.Printf("Reducing batch size to %d due to slow processing (%v avg)", batchSize, avgTime)
		}
	}

	// Ensure within bounds
	if batchSize > abf.maxBatchSize {
		batchSize = abf.maxBatchSize
	}
	if batchSize < abf.minBatchSize && queueDepth > 0 {
		batchSize = abf.minBatchSize
	}

	return batchSize
}

// UpdateMetrics updates performance metrics for adaptive optimization
func (abf *AdaptiveBatchFormer) UpdateMetrics(batchSize int, processingTime time.Duration) {
	if abf.metrics == nil {
		abf.metrics = &BatchMetrics{}
	}

	abf.metrics.mu.Lock()
	defer abf.metrics.mu.Unlock()

	// Update average processing time (exponential moving average)
	if abf.metrics.avgProcessingTime == 0 {
		abf.metrics.avgProcessingTime = processingTime
	} else {
		// 0.1 weight for new sample, 0.9 for historical average
		abf.metrics.avgProcessingTime = time.Duration(
			float64(abf.metrics.avgProcessingTime)*0.9 + float64(processingTime)*0.1,
		)
	}

	// Track recent batch sizes
	abf.metrics.recentBatchSizes = append(abf.metrics.recentBatchSizes, batchSize)
	if len(abf.metrics.recentBatchSizes) > 10 {
		abf.metrics.recentBatchSizes = abf.metrics.recentBatchSizes[1:]
	}

	abf.metrics.lastOptimization = time.Now()
}

// GetMetrics returns current batch formation metrics
func (abf *AdaptiveBatchFormer) GetMetrics() BatchMetrics {
	if abf.metrics == nil {
		return BatchMetrics{}
	}

	abf.metrics.mu.RLock()
	defer abf.metrics.mu.RUnlock()

	// Create copy to avoid data races
	return BatchMetrics{
		avgProcessingTime: abf.metrics.avgProcessingTime,
		recentBatchSizes:  append([]int(nil), abf.metrics.recentBatchSizes...),
		lastOptimization:  abf.metrics.lastOptimization,
	}
}

// Helper function
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}