package worker

import (
	"fmt"
	"log"
	"sync"
	"sync/atomic"
	"time"
)

// WorkerPool manages concurrent batch processing with bounded resources
type WorkerPool struct {
	semaphore     chan struct{}
	wg            sync.WaitGroup
	shutdown      chan struct{}
	activeWorkers int64 // Atomic counter
	maxWorkers    int
	metrics       *WorkerPoolMetrics
}

// WorkerPoolMetrics tracks worker pool performance
type WorkerPoolMetrics struct {
	TotalTasks    int64 // Total tasks submitted
	CompletedTasks int64 // Successfully completed
	FailedTasks   int64 // Failed/panicked tasks
	RejectedTasks int64 // Rejected due to capacity
}

// NewWorkerPool creates a new worker pool with the specified capacity
func NewWorkerPool(maxWorkers int) *WorkerPool {
	return &WorkerPool{
		semaphore:  make(chan struct{}, maxWorkers),
		shutdown:   make(chan struct{}),
		maxWorkers: maxWorkers,
		metrics:    &WorkerPoolMetrics{},
	}
}

// Submit submits a task to the worker pool
func (wp *WorkerPool) Submit(task func()) error {
	atomic.AddInt64(&wp.metrics.TotalTasks, 1)

	select {
	case wp.semaphore <- struct{}{}: // Acquire semaphore
		atomic.AddInt64(&wp.activeWorkers, 1)
		wp.wg.Add(1)

		go func() {
			defer wp.wg.Done()
			defer func() {
				<-wp.semaphore // Release semaphore
				atomic.AddInt64(&wp.activeWorkers, -1)
			}()

			// Execute task with comprehensive error handling
			success := false
			defer func() {
				if r := recover(); r != nil {
					log.Printf("Worker panic recovered: %v", r)
					atomic.AddInt64(&wp.metrics.FailedTasks, 1)
				} else if success {
					atomic.AddInt64(&wp.metrics.CompletedTasks, 1)
				} else {
					atomic.AddInt64(&wp.metrics.FailedTasks, 1)
				}
			}()

			task()
			success = true
		}()

		return nil

	case <-wp.shutdown:
		atomic.AddInt64(&wp.metrics.RejectedTasks, 1)
		return fmt.Errorf("worker pool shutting down")

	case <-time.After(30 * time.Second):
		atomic.AddInt64(&wp.metrics.RejectedTasks, 1)
		return fmt.Errorf("worker pool full - timeout after 30s")
	}
}

// GetMetrics returns current worker pool metrics
func (wp *WorkerPool) GetMetrics() WorkerPoolMetrics {
	return WorkerPoolMetrics{
		TotalTasks:    atomic.LoadInt64(&wp.metrics.TotalTasks),
		CompletedTasks: atomic.LoadInt64(&wp.metrics.CompletedTasks),
		FailedTasks:   atomic.LoadInt64(&wp.metrics.FailedTasks),
		RejectedTasks: atomic.LoadInt64(&wp.metrics.RejectedTasks),
	}
}

// ActiveWorkers returns the number of currently active workers
func (wp *WorkerPool) ActiveWorkers() int64 {
	return atomic.LoadInt64(&wp.activeWorkers)
}

// Shutdown gracefully shuts down the worker pool
func (wp *WorkerPool) Shutdown() {
	log.Printf("Shutting down worker pool - waiting for %d active workers", wp.ActiveWorkers())
	close(wp.shutdown)
	wp.wg.Wait() // Wait for all workers to finish

	metrics := wp.GetMetrics()
	log.Printf("Worker pool shutdown complete - Total: %d, Completed: %d, Failed: %d, Rejected: %d",
		metrics.TotalTasks, metrics.CompletedTasks, metrics.FailedTasks, metrics.RejectedTasks)
}