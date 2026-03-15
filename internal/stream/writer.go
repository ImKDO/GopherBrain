package stream

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/ImKDO/GopherBrain/internal/config"
	"github.com/ImKDO/GopherBrain/internal/pipeline"
	"github.com/ImKDO/GopherBrain/pkg/tensor"
)

type RedisWriter struct {
	client *redis.Client
	ttl    time.Duration
	logger *slog.Logger
}

type storedResult struct {
	Shape     []int64   `json:"shape"`
	Data      []float32 `json:"data"`
	LatencyMs float64   `json:"latency_ms"`
	Error     string    `json:"error,omitempty"`
}

func NewRedisWriter(cfg config.RedisConfig, logger *slog.Logger) (*RedisWriter, error) {
	client := redis.NewClient(&redis.Options{
		Addr:     cfg.Addr,
		Password: cfg.Password,
		DB:       cfg.DB,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := client.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("redis ping failed: %w", err)
	}

	return &RedisWriter{
		client: client,
		ttl:    time.Duration(cfg.TTL) * time.Second,
		logger: logger,
	}, nil
}

func (w *RedisWriter) WriteResult(ctx context.Context, requestID string, result *pipeline.Result) error {
	sr := storedResult{
		LatencyMs: result.LatencyMs,
	}

	if result.Err != nil {
		sr.Error = result.Err.Error()
	}

	if result.Output != nil {
		sr.Shape = result.Output.Shape
		sr.Data = result.Output.Data
	}

	data, err := json.Marshal(sr)
	if err != nil {
		return fmt.Errorf("marshaling result: %w", err)
	}

	key := "result:" + requestID
	if err := w.client.Set(ctx, key, data, w.ttl).Err(); err != nil {
		return fmt.Errorf("writing to redis: %w", err)
	}

	return nil
}

func (w *RedisWriter) GetResult(ctx context.Context, requestID string) (*pipeline.Result, error) {
	key := "result:" + requestID
	data, err := w.client.Get(ctx, key).Bytes()
	if err != nil {
		return nil, fmt.Errorf("reading from redis: %w", err)
	}

	var sr storedResult
	if err := json.Unmarshal(data, &sr); err != nil {
		return nil, fmt.Errorf("unmarshaling result: %w", err)
	}

	result := &pipeline.Result{
		LatencyMs: sr.LatencyMs,
	}

	if sr.Error != "" {
		result.Err = fmt.Errorf("%s", sr.Error)
	}

	if len(sr.Data) > 0 {
		t, err := tensor.New[float32](sr.Shape, sr.Data)
		if err != nil {
			return nil, fmt.Errorf("reconstructing tensor: %w", err)
		}
		result.Output = t
	}

	return result, nil
}

func (w *RedisWriter) Close() error {
	return w.client.Close()
}
