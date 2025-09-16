package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/google/uuid"

	"connector/internal/config"
	"connector/internal/correlation"
	"connector/internal/modal"
	"connector/internal/queue"
	"connector/internal/server"
	"connector/pkg/worker"
)

// AggregationService is the main service that orchestrates batch processing
type AggregationService struct {
	queue          *queue.PersistentQueue
	batchFormer    *queue.AdaptiveBatchFormer
	modalClient    *modal.ModalClient
	responseRouter *correlation.ResponseRouter
	config         *config.ServiceConfig
	workerPool     *worker.WorkerPool
	httpServer     *server.HTTPServer

	shutdown chan struct{}
	wg       sync.WaitGroup
}

// NewAggregationService creates a new aggregation service
func NewAggregationService(cfg *config.ServiceConfig) (*AggregationService, error) {
	// Initialize Redis queue
	queue, err := queue.NewPersistentQueue(
		cfg.Queue.RedisURL,
		cfg.Queue.QueueKey,
		cfg.Queue.ProcessingKey,
		cfg.Queue.MaxSize,
		cfg.Queue.TTL,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create queue: %v", err)
	}

	// Initialize batch former
	batchFormer := queue.NewAdaptiveBatchFormer(
		queue,
		cfg.BatchFormation.MaxBatchSize,
		cfg.BatchFormation.MinBatchSize,
		200*time.Millisecond, // max wait time
	)

	// Initialize Modal client
	modalClient := modal.NewModalClient(
		cfg.Modal.BaseURL,
		cfg.Modal.Timeout,
		cfg.Modal.MaxConcurrentBatches,
	)

	// Initialize response router
	responseRouter := correlation.NewResponseRouter(cfg.Response.Timeout)

	// Initialize worker pool
	workerPool := worker.NewWorkerPool(cfg.Modal.MaxConcurrentBatches)

	as := &AggregationService{
		queue:          queue,
		batchFormer:    batchFormer,
		modalClient:    modalClient,
		responseRouter: responseRouter,
		config:         cfg,
		workerPool:     workerPool,
		shutdown:       make(chan struct{}),
	}

	// Initialize HTTP server
	as.httpServer = server.NewHTTPServer(cfg, as, responseRouter)

	return as, nil
}

// Start starts all service components
func (as *AggregationService) Start() error {
	log.Printf("Starting Dolphin Aggregation Service...")

	// Start the batch processing loop
	as.wg.Add(1)
	go as.processBatchesContinuously()

	// Start recovery mechanism for stuck batches
	as.wg.Add(1)
	go as.startRecoveryLoop()

	// Start HTTP server
	return as.httpServer.Start()
}

// processBatchesContinuously is the main batch processing loop
func (as *AggregationService) processBatchesContinuously() {
	defer as.wg.Done()

	ticker := time.NewTicker(as.config.BatchFormation.CheckInterval)
	defer ticker.Stop()

	// Mutex to prevent race conditions in batch formation
	var batchFormationMutex sync.Mutex

	for {
		select {
		case <-as.shutdown:
			log.Printf("Batch processing loop shutting down")
			return

		case <-ticker.C:
			// Lock batch formation to prevent concurrent dequeuing
			batchFormationMutex.Lock()

			batchID := uuid.New().String()
			batch, err := as.queue.StartBatchProcessing(batchID, as.batchFormer.GetOptimalBatchSize())

			batchFormationMutex.Unlock()

			if err != nil {
				log.Printf("Error starting batch processing: %v", err)
				continue
			}

			if batch == nil || len(batch.Requests) == 0 {
				continue // No requests available
			}

			// Submit to worker pool with panic recovery
			if err := as.workerPool.Submit(func() {
				as.sendBatchToModalWithRecovery(batch)
			}); err != nil {
				log.Printf("Failed to submit batch to worker pool: %v", err)
				// Restore batch to queue
				as.queue.FailBatch(batch.ID, fmt.Sprintf("worker pool error: %v", err))
			}
		}
	}
}

// sendBatchToModalWithRecovery wraps batch processing with panic recovery
func (as *AggregationService) sendBatchToModalWithRecovery(batch *queue.BatchState) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("PANIC in batch processing for %s: %v", batch.ID, r)

			// Restore batch to queue to prevent data loss
			if err := as.queue.FailBatch(batch.ID, fmt.Sprintf("worker panic: %v", r)); err != nil {
				log.Printf("CRITICAL: Failed to restore panicked batch %s: %v", batch.ID, err)

				// Last resort: notify all clients of failure
				as.responseRouter.HandleBatchFailure(batch, fmt.Sprintf("unrecoverable panic: %v", r))
			}
		}
	}()

	// Call the actual processing function
	as.sendBatchToModal(batch)
}

// sendBatchToModal processes a batch through Modal API
func (as *AggregationService) sendBatchToModal(batch *queue.BatchState) {
	log.Printf("Sending batch %s with %d requests to Modal", batch.ID, len(batch.Requests))

	startTime := time.Now()

	// Send to Modal with retry and circuit breaker
	response, err := as.modalClient.ProcessBatch(batch)
	if err != nil {
		log.Printf("Modal API error for batch %s: %v", batch.ID, err)

		// Mark batch as failed and handle retry/requeue
		as.queue.FailBatch(batch.ID, err.Error())
		as.responseRouter.HandleBatchFailure(batch, err.Error())
		return
	}

	// Processing successful - mark batch as completed
	err = as.queue.CompleteBatch(batch.ID)
	if err != nil {
		log.Printf("Warning: failed to mark batch %s as completed: %v", batch.ID, err)
	}

	// Update metrics
	processingTime := time.Since(startTime)
	as.batchFormer.UpdateMetrics(len(batch.Requests), processingTime)

	// Route individual responses back to waiting clients
	as.responseRouter.RouteResponses(batch, response)

	log.Printf("Successfully processed batch %s with %d images in %v",
		batch.ID, len(batch.Requests), processingTime)
}

// startRecoveryLoop periodically recovers stuck batches
func (as *AggregationService) startRecoveryLoop() {
	defer as.wg.Done()

	ticker := time.NewTicker(as.config.Queue.RecoveryInterval)
	defer ticker.Stop()

	for {
		select {
		case <-as.shutdown:
			log.Printf("Recovery loop shutting down")
			return

		case <-ticker.C:
			if err := as.queue.RecoverStuckBatches(); err != nil {
				log.Printf("Error during batch recovery: %v", err)
			}
		}
	}
}

// GetHealthStatus returns service health information
func (as *AggregationService) GetHealthStatus() map[string]interface{} {
	status := map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now(),
	}

	// Check Modal API health
	if err := as.modalClient.HealthCheck(); err != nil {
		status["status"] = "degraded"
		status["modal_api_error"] = err.Error()
	}

	// Check queue depth
	queueDepth := as.queue.Size()
	status["queue_depth"] = queueDepth

	if queueDepth > as.config.Queue.MaxSize*8/10 { // 80% of max capacity
		status["status"] = "degraded"
		status["queue_warning"] = fmt.Sprintf("Queue nearly full: %d/%d", queueDepth, as.config.Queue.MaxSize)
	}

	// Worker pool metrics
	metrics := as.workerPool.GetMetrics()
	status["worker_pool"] = map[string]interface{}{
		"active_workers": as.workerPool.ActiveWorkers(),
		"total_tasks":    metrics.TotalTasks,
		"completed_tasks": metrics.CompletedTasks,
		"failed_tasks":   metrics.FailedTasks,
	}

	return status
}

// GetQueueStatus returns detailed queue status
func (as *AggregationService) GetQueueStatus() map[string]interface{} {
	queueDepth := as.queue.Size()

	return map[string]interface{}{
		"queue_depth":     queueDepth,
		"max_queue_size":  as.config.Queue.MaxSize,
		"utilization_percent": float64(queueDepth) / float64(as.config.Queue.MaxSize) * 100,
	}
}

// Shutdown gracefully shuts down the service
func (as *AggregationService) Shutdown(ctx context.Context) error {
	log.Printf("Starting graceful shutdown...")

	// Signal shutdown to background goroutines
	close(as.shutdown)

	// Shutdown HTTP server
	if err := as.httpServer.Shutdown(ctx); err != nil {
		log.Printf("HTTP server shutdown error: %v", err)
	}

	// Wait for background goroutines
	as.wg.Wait()

	// Shutdown worker pool
	as.workerPool.Shutdown()

	// Shutdown response router
	as.responseRouter.Shutdown()

	// Recover any stuck batches back to queue
	as.queue.RecoverStuckBatches()

	// Close queue
	as.queue.Close()

	log.Printf("Graceful shutdown complete")
	return nil
}

func main() {
	// Load configuration
	configPath := os.Getenv("CONFIG_PATH")
	if configPath == "" {
		configPath = "configs/production.yaml"
	}

	cfg, err := config.LoadConfig(configPath)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Create aggregation service
	service, err := NewAggregationService(cfg)
	if err != nil {
		log.Fatalf("Failed to create service: %v", err)
	}

	// Setup graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle shutdown signals
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigChan
		log.Printf("Shutdown signal received")
		cancel()

		// Force shutdown after timeout
		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), cfg.Server.GracefulShutdownTimeout)
		defer shutdownCancel()

		if err := service.Shutdown(shutdownCtx); err != nil {
			log.Printf("Forced shutdown: %v", err)
			os.Exit(1)
		}
		os.Exit(0)
	}()

	// Start service
	log.Printf("Dolphin Aggregation Service starting on port %d", cfg.Server.Port)
	if err := service.Start(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("Service failed: %v", err)
	}
}