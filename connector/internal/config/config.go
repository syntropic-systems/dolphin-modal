package config

import (
	"fmt"
	"os"
	"time"

	"gopkg.in/yaml.v3"
)

// ServiceConfig holds all configuration for the aggregation service
type ServiceConfig struct {
	Server         ServerConfig         `yaml:"server"`
	Queue          QueueConfig          `yaml:"queue"`
	BatchFormation BatchFormationConfig `yaml:"batch_formation"`
	Modal          ModalConfig          `yaml:"modal"`
	Response       ResponseConfig       `yaml:"response"`
	Monitoring     MonitoringConfig     `yaml:"monitoring"`
}

type ServerConfig struct {
	Port                     int           `yaml:"port"`
	MaxConnections           int           `yaml:"max_connections"`
	ReadTimeout              time.Duration `yaml:"read_timeout"`
	WriteTimeout             time.Duration `yaml:"write_timeout"`
	MaxRequestSize           int64         `yaml:"max_request_size"`
	RateLimitPerClient       int           `yaml:"rate_limit_per_client"`
	GracefulShutdownTimeout  time.Duration `yaml:"graceful_shutdown_timeout"`
	BackpressureThreshold    int           `yaml:"backpressure_threshold"`
}

type QueueConfig struct {
	RedisURL             string        `yaml:"redis_url"`
	RedisClusterURLs     []string      `yaml:"redis_cluster_urls"`
	MaxSize              int           `yaml:"max_size"`
	TTL                  time.Duration `yaml:"ttl"`
	QueueKey             string        `yaml:"queue_key"`
	ProcessingKey        string        `yaml:"processing_key"`
	MaxRetryCount        int           `yaml:"max_retry_count"`
	RecoveryInterval     time.Duration `yaml:"recovery_interval"`
}

type BatchFormationConfig struct {
	MaxBatchSize    int           `yaml:"max_batch_size"`
	MinBatchSize    int           `yaml:"min_batch_size"`
	CheckInterval   time.Duration `yaml:"check_interval"`
}

type ModalConfig struct {
	BaseURL                  string                 `yaml:"base_url"`
	Timeout                  time.Duration          `yaml:"timeout"`
	MaxRetries               int                    `yaml:"max_retries"`
	RetryDelay               time.Duration          `yaml:"retry_delay"`
	MaxConcurrentBatches     int                    `yaml:"max_concurrent_batches"`
	CircuitBreaker           CircuitBreakerConfig   `yaml:"circuit_breaker"`
	ConnectionPool           ConnectionPoolConfig   `yaml:"connection_pool"`
}

type CircuitBreakerConfig struct {
	FailureThreshold      int           `yaml:"failure_threshold"`
	CooldownPeriod        time.Duration `yaml:"cooldown_period"`
	HealthCheckInterval   time.Duration `yaml:"health_check_interval"`
}

type ConnectionPoolConfig struct {
	MaxIdleConnections       int           `yaml:"max_idle_connections"`
	IdleConnectionTimeout    time.Duration `yaml:"idle_connection_timeout"`
}

type ResponseConfig struct {
	Timeout         time.Duration `yaml:"timeout"`
	CleanupInterval time.Duration `yaml:"cleanup_interval"`
}

type MonitoringConfig struct {
	EnableMetrics    bool   `yaml:"enable_metrics"`
	PrometheusPort   int    `yaml:"prometheus_port"`
	LogLevel         string `yaml:"log_level"`
}

// LoadConfig loads configuration from YAML file with environment variable substitution
func LoadConfig(configPath string) (*ServiceConfig, error) {
	// Read config file
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %v", err)
	}

	// Expand environment variables in config
	expanded := os.ExpandEnv(string(data))

	// Parse YAML
	var config ServiceConfig
	if err := yaml.Unmarshal([]byte(expanded), &config); err != nil {
		return nil, fmt.Errorf("failed to parse config YAML: %v", err)
	}

	// Set defaults
	setDefaults(&config)

	// Validate configuration
	if err := validateConfig(&config); err != nil {
		return nil, fmt.Errorf("invalid configuration: %v", err)
	}

	return &config, nil
}

func setDefaults(config *ServiceConfig) {
	// Server defaults
	if config.Server.Port == 0 {
		config.Server.Port = 8080
	}
	if config.Server.MaxConnections == 0 {
		config.Server.MaxConnections = 3000
	}
	if config.Server.ReadTimeout == 0 {
		config.Server.ReadTimeout = 300 * time.Second
	}
	if config.Server.WriteTimeout == 0 {
		config.Server.WriteTimeout = 300 * time.Second
	}
	if config.Server.MaxRequestSize == 0 {
		config.Server.MaxRequestSize = 52428800 // 50MB
	}
	if config.Server.RateLimitPerClient == 0 {
		config.Server.RateLimitPerClient = 10
	}
	if config.Server.GracefulShutdownTimeout == 0 {
		config.Server.GracefulShutdownTimeout = 30 * time.Second
	}
	if config.Server.BackpressureThreshold == 0 {
		config.Server.BackpressureThreshold = 2500
	}

	// Queue defaults
	if config.Queue.RedisURL == "" {
		config.Queue.RedisURL = "redis://localhost:6379"
	}
	if config.Queue.MaxSize == 0 {
		config.Queue.MaxSize = 10000
	}
	if config.Queue.TTL == 0 {
		config.Queue.TTL = 6 * time.Hour
	}
	if config.Queue.QueueKey == "" {
		config.Queue.QueueKey = "dolphin:requests"
	}
	if config.Queue.ProcessingKey == "" {
		config.Queue.ProcessingKey = "dolphin:processing"
	}
	if config.Queue.MaxRetryCount == 0 {
		config.Queue.MaxRetryCount = 3
	}
	if config.Queue.RecoveryInterval == 0 {
		config.Queue.RecoveryInterval = 5 * time.Minute
	}

	// Batch formation defaults
	if config.BatchFormation.MaxBatchSize == 0 {
		config.BatchFormation.MaxBatchSize = 32
	}
	if config.BatchFormation.MinBatchSize == 0 {
		config.BatchFormation.MinBatchSize = 1
	}
	if config.BatchFormation.CheckInterval == 0 {
		config.BatchFormation.CheckInterval = 50 * time.Millisecond
	}

	// Modal defaults
	if config.Modal.Timeout == 0 {
		config.Modal.Timeout = 120 * time.Second
	}
	if config.Modal.MaxRetries == 0 {
		config.Modal.MaxRetries = 3
	}
	if config.Modal.RetryDelay == 0 {
		config.Modal.RetryDelay = 2 * time.Second
	}
	if config.Modal.MaxConcurrentBatches == 0 {
		config.Modal.MaxConcurrentBatches = 4
	}
	if config.Modal.CircuitBreaker.FailureThreshold == 0 {
		config.Modal.CircuitBreaker.FailureThreshold = 5
	}
	if config.Modal.CircuitBreaker.CooldownPeriod == 0 {
		config.Modal.CircuitBreaker.CooldownPeriod = 30 * time.Second
	}
	if config.Modal.CircuitBreaker.HealthCheckInterval == 0 {
		config.Modal.CircuitBreaker.HealthCheckInterval = 10 * time.Second
	}
	if config.Modal.ConnectionPool.MaxIdleConnections == 0 {
		config.Modal.ConnectionPool.MaxIdleConnections = 10
	}
	if config.Modal.ConnectionPool.IdleConnectionTimeout == 0 {
		config.Modal.ConnectionPool.IdleConnectionTimeout = 90 * time.Second
	}

	// Response defaults
	if config.Response.Timeout == 0 {
		config.Response.Timeout = 15 * time.Minute
	}
	if config.Response.CleanupInterval == 0 {
		config.Response.CleanupInterval = 5 * time.Minute
	}

	// Monitoring defaults
	if config.Monitoring.PrometheusPort == 0 {
		config.Monitoring.PrometheusPort = 9090
	}
	if config.Monitoring.LogLevel == "" {
		config.Monitoring.LogLevel = "info"
	}
}

func validateConfig(config *ServiceConfig) error {
	if config.Modal.BaseURL == "" {
		return fmt.Errorf("modal.base_url is required")
	}

	if config.BatchFormation.MaxBatchSize > 32 {
		return fmt.Errorf("batch_formation.max_batch_size cannot exceed 32 (T4 GPU limit)")
	}

	if config.Server.BackpressureThreshold > config.Queue.MaxSize {
		return fmt.Errorf("server.backpressure_threshold cannot exceed queue.max_size")
	}

	return nil
}