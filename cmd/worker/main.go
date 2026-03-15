package main

import (
	"context"
	"flag"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/ImKDO/GopherBrain/internal/config"
	"github.com/ImKDO/GopherBrain/internal/observability"
	"github.com/ImKDO/GopherBrain/internal/pipeline"
	"github.com/ImKDO/GopherBrain/internal/runtime"
	"github.com/ImKDO/GopherBrain/internal/stream"
)

func main() {
	configPath := flag.String("config", "configs/config.yaml", "path to config file")
	flag.Parse()

	cfg, err := config.Load(*configPath)
	if err != nil {
		slog.Error("failed to load config", "error", err)
		os.Exit(1)
	}

	logger := observability.NewLogger(cfg.Observability.LogLevel)

	// Initialize ONNX runtime
	if err := runtime.InitRuntime(cfg.Runtime.ONNXLibPath); err != nil {
		logger.Error("failed to initialize ONNX runtime", "error", err)
		os.Exit(1)
	}

	session, err := runtime.NewONNXSession(cfg.Runtime.ModelPath, cfg.Runtime, logger)
	if err != nil {
		logger.Error("failed to create ONNX session", "error", err)
		os.Exit(1)
	}

	// Build and start pipeline
	p := pipeline.NewPipeline(session, cfg.Pipeline, logger)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	p.Start(ctx)

	// Redis writer
	redisWriter, err := stream.NewRedisWriter(cfg.Redis, logger)
	if err != nil {
		logger.Warn("failed to connect to Redis, results won't be cached", "error", err)
		redisWriter = nil
	}

	// Kafka consumer
	consumer := stream.NewKafkaConsumer(cfg.Kafka, p, redisWriter, logger)
	consumer.Start(ctx)

	// Wait for shutdown signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Info("shutting down worker")

	cancel()

	consumer.Stop()
	p.Stop()

	if redisWriter != nil {
		redisWriter.Close()
	}

	session.Close()
	runtime.DestroyRuntime()

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer shutdownCancel()
	_ = shutdownCtx

	logger.Info("worker shutdown complete")
}
