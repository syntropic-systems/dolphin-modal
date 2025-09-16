package server

import (
	"sync"
	"time"

	"golang.org/x/time/rate"
)

// RateLimiter implements per-IP rate limiting
type RateLimiter struct {
	clients sync.Map // IP -> *rate.Limiter
	limit   int      // Requests per minute
}

// NewRateLimiter creates a new rate limiter
func NewRateLimiter(requestsPerMinute int) *RateLimiter {
	rl := &RateLimiter{
		limit: requestsPerMinute,
	}

	// Cleanup expired rate limiters every 10 minutes
	go rl.cleanup()

	return rl
}

// Allow checks if a request from the given IP should be allowed
func (rl *RateLimiter) Allow(clientIP string) bool {
	limiter, _ := rl.clients.LoadOrStore(clientIP,
		rate.NewLimiter(rate.Every(time.Minute/time.Duration(rl.limit)), rl.limit))

	return limiter.(*rate.Limiter).Allow()
}

// cleanup removes inactive rate limiters to prevent memory leaks
func (rl *RateLimiter) cleanup() {
	ticker := time.NewTicker(10 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		rl.clients.Range(func(key, value interface{}) bool {
			limiter := value.(*rate.Limiter)

			// Remove limiter if no recent activity
			if limiter.Tokens() == float64(rl.limit) {
				rl.clients.Delete(key)
			}

			return true
		})
	}
}