package stream

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"sync"

	"github.com/segmentio/kafka-go"

	"github.com/ImKDO/GopherBrain/internal/config"
	"github.com/ImKDO/GopherBrain/internal/pipeline"
	"github.com/ImKDO/GopherBrain/pkg/tensor"
)

type KafkaConsumer struct {
	reader   *kafka.Reader
	pipeline *pipeline.Pipeline
	writer   *RedisWriter
	logger   *slog.Logger
	wg       sync.WaitGroup
}

type kafkaMessage struct {
	RequestID string            `json:"request_id"`
	ModelID   string            `json:"model_id"`
	Shape     []int64           `json:"shape"`
	Data      []float32         `json:"data"`
	Metadata  map[string]string `json:"metadata"`
}

func NewKafkaConsumer(cfg config.KafkaConfig, p *pipeline.Pipeline, w *RedisWriter, logger *slog.Logger) *KafkaConsumer {
	reader := kafka.NewReader(kafka.ReaderConfig{
		Brokers:  cfg.Brokers,
		Topic:    cfg.Topic,
		GroupID:  cfg.GroupID,
		MinBytes: 1,
		MaxBytes: 10e6,
	})

	return &KafkaConsumer{
		reader:   reader,
		pipeline: p,
		writer:   w,
		logger:   logger,
	}
}

func (c *KafkaConsumer) Start(ctx context.Context) {
	c.wg.Add(1)
	go c.consume(ctx)
	c.logger.Info("kafka consumer started")
}

func (c *KafkaConsumer) consume(ctx context.Context) {
	defer c.wg.Done()

	for {
		msg, err := c.reader.ReadMessage(ctx)
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			c.logger.Error("reading kafka message", "error", err)
			continue
		}

		var km kafkaMessage
		if err := json.Unmarshal(msg.Value, &km); err != nil {
			c.logger.Error("unmarshaling kafka message", "error", err, "offset", msg.Offset)
			continue
		}

		input, err := tensor.New[float32](km.Shape, km.Data)
		if err != nil {
			c.logger.Error("creating tensor from kafka message", "error", err, "request_id", km.RequestID)
			continue
		}

		req := &pipeline.Request{
			ID:       km.RequestID,
			Input:    input,
			ModelID:  km.ModelID,
			Metadata: km.Metadata,
			Ctx:      ctx,
		}

		result, err := c.pipeline.Submit(ctx, req)
		if err != nil {
			c.logger.Error("pipeline submit failed", "error", err, "request_id", km.RequestID)
			continue
		}

		if c.writer != nil {
			if err := c.writer.WriteResult(ctx, km.RequestID, &result); err != nil {
				c.logger.Error("writing result to redis", "error", err, "request_id", km.RequestID)
			}
		}

		c.logger.Debug("processed kafka message",
			"request_id", km.RequestID,
			"latency_ms", result.LatencyMs,
		)
	}
}

func (c *KafkaConsumer) Stop() error {
	c.wg.Wait()
	if err := c.reader.Close(); err != nil {
		return fmt.Errorf("closing kafka reader: %w", err)
	}
	c.logger.Info("kafka consumer stopped")
	return nil
}
