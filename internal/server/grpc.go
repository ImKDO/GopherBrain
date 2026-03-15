package server

import (
	"fmt"
	"log/slog"
	"net"

	"github.com/ImKDO/GopherBrain/internal/config"
	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"

	pbv1 "github.com/ImKDO/GopherBrain/api/proto/v1"
)

type GRPCServer struct {
	server   *grpc.Server
	listener net.Listener
	logger   *slog.Logger
	cfg      config.ServerConfig
}

func NewGRPCServer(cfg config.ServerConfig, logger *slog.Logger, opts ...grpc.ServerOption) (*GRPCServer, error) {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", cfg.GRPCPort))
	if err != nil {
		return nil, fmt.Errorf("failed to listen on port %d: %w", cfg.GRPCPort, err)
	}

	srv := grpc.NewServer(opts...)
	reflection.Register(srv)

	return &GRPCServer{
		server:   srv,
		listener: lis,
		logger:   logger,
		cfg:      cfg,
	}, nil
}

func (s *GRPCServer) RegisterServices(inferSvc pbv1.InferenceServiceServer, healthSvc pbv1.HealthServer) {
	pbv1.RegisterInferenceServiceServer(s.server, inferSvc)
	pbv1.RegisterHealthServer(s.server, healthSvc)
}

func (s *GRPCServer) Start() error {
	s.logger.Info("gRPC server starting", "port", s.cfg.GRPCPort)
	return s.server.Serve(s.listener)
}

func (s *GRPCServer) Stop() {
	s.logger.Info("gRPC server stopping")
	s.server.GracefulStop()
}
