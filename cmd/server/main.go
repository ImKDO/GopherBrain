package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	_ "net/http/pprof"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"
	"google.golang.org/grpc"

	"github.com/ImKDO/GopherBrain/internal/config"
	"github.com/ImKDO/GopherBrain/internal/observability"
	"github.com/ImKDO/GopherBrain/internal/pipeline"
	"github.com/ImKDO/GopherBrain/internal/runtime"
	"github.com/ImKDO/GopherBrain/internal/server"
)

func main() {
	if err := run(); err != nil {
		slog.Error("fatal", "error", err)
		os.Exit(1)
	}
}

func run() error {
	configPath := flag.String("config", "configs/config.yaml", "path to config file")
	flag.Parse()

	cfg, err := config.Load(*configPath)
	if err != nil {
		return fmt.Errorf("loading config: %w", err)
	}

	logger := observability.NewLogger(cfg.Observability.LogLevel)

	// pprof debug server
	go func() {
		logger.Info("pprof server starting", "port", 6060)
		if err := http.ListenAndServe(":6060", nil); err != nil {
			logger.Warn("pprof server failed", "error", err)
		}
	}()

	// Initialize tracer
	shutdownTracer, err := observability.InitTracer("gopherbrain-server", cfg.Observability.JaegerEndpoint)
	if err != nil {
		logger.Warn("failed to initialize tracer, continuing without tracing", "error", err)
		shutdownTracer = func(ctx context.Context) error { return nil }
	}

	// Initialize metrics
	metrics := observability.NewMetrics(prometheus.DefaultRegisterer)
	_ = metrics
	metricsSrv := observability.StartMetricsServer(cfg.Observability.MetricsPort)
	logger.Info("metrics server started", "port", cfg.Observability.MetricsPort)

	// Initialize ONNX runtime
	if err := runtime.InitRuntime(cfg.Runtime.ONNXLibPath); err != nil {
		return fmt.Errorf("initializing ONNX runtime: %w", err)
	}
	defer runtime.DestroyRuntime()

	session, err := runtime.NewONNXSession(cfg.Runtime.ModelPath, cfg.Runtime, logger)
	if err != nil {
		return fmt.Errorf("creating ONNX session: %w", err)
	}
	defer session.Close()

	// Build and start pipeline
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	p := pipeline.NewPipeline(session, cfg.Pipeline, logger)
	p.Start(ctx)
	defer p.Stop()

	// Health checker
	health := observability.NewHealthChecker()
	health.Register("pipeline", func() error { return nil })

	// gRPC server with tracing interceptors
	grpcSrv, err := server.NewGRPCServer(cfg.Server, logger,
		grpc.StatsHandler(otelgrpc.NewServerHandler()),
	)
	if err != nil {
		return fmt.Errorf("creating gRPC server: %w", err)
	}

	inferHandler := server.NewInferenceHandler(p, logger)
	healthHandler := server.NewHealthGRPCHandler(health)
	grpcSrv.RegisterServices(inferHandler, healthHandler)

	go func() {
		if err := grpcSrv.Start(); err != nil {
			logger.Error("gRPC server failed", "error", err)
		}
	}()
	defer grpcSrv.Stop()

	// REST server
	restSrv := server.NewRESTServer(cfg.Server, p, logger)
	restSrv.RegisterHealthRoutes(health)

	go func() {
		if err := restSrv.Start(); err != nil && err != http.ErrServerClosed {
			logger.Error("REST server failed", "error", err)
		}
	}()

	logger.Info("server started",
		"grpc_port", cfg.Server.GRPCPort,
		"rest_port", cfg.Server.RESTPort,
	)

	// Wait for shutdown signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Info("shutting down")

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	restSrv.Stop(shutdownCtx)
	shutdownTracer(shutdownCtx)
	metricsSrv.Shutdown(shutdownCtx)

	// Defers handle: grpcSrv.Stop(), p.Stop(), session.Close(), runtime.DestroyRuntime()
	logger.Info("shutdown complete")
	return nil
}
