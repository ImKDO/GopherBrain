package server

import (
	"context"
	"log/slog"
	"net/http"

	pbv1 "github.com/ImKDO/GopherBrain/api/proto/v1"
	"github.com/ImKDO/GopherBrain/internal/pipeline"
	"github.com/ImKDO/GopherBrain/pkg/tensor"
	"github.com/google/uuid"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type InferenceHandler struct {
	pbv1.UnimplementedInferenceServiceServer
	pipeline *pipeline.Pipeline
	logger   *slog.Logger
}

func NewInferenceHandler(p *pipeline.Pipeline, logger *slog.Logger) *InferenceHandler {
	return &InferenceHandler{
		pipeline: p,
		logger:   logger,
	}
}

func (h *InferenceHandler) Infer(ctx context.Context, req *pbv1.InferRequest) (*pbv1.InferResponse, error) {
	if h.pipeline == nil {
		return nil, status.Error(codes.Unavailable, "pipeline not ready")
	}

	input, err := tensor.FromProto(req.GetInput())
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "invalid input tensor: %v", err)
	}

	requestID := req.GetRequestId()
	if requestID == "" {
		requestID = uuid.New().String()
	}

	result, err := h.pipeline.Submit(ctx, &pipeline.Request{
		ID:       requestID,
		Input:    input,
		ModelID:  req.GetModelId(),
		Metadata: req.GetMetadata(),
		Ctx:      ctx,
	})
	if err != nil {
		return nil, status.Errorf(codes.Internal, "inference failed: %v", err)
	}

	if result.Err != nil {
		return nil, status.Errorf(codes.Internal, "inference error: %v", result.Err)
	}

	return &pbv1.InferResponse{
		RequestId: requestID,
		Output:    tensor.ToProto(result.Output),
		LatencyMs: result.LatencyMs,
	}, nil
}

func (h *InferenceHandler) InferStream(stream grpc.BidiStreamingServer[pbv1.InferRequest, pbv1.InferResponse]) error {
	for {
		req, err := stream.Recv()
		if err != nil {
			return err
		}

		resp, err := h.Infer(stream.Context(), req)
		if err != nil {
			return err
		}

		if err := stream.Send(resp); err != nil {
			return err
		}
	}
}

// HealthGRPCHandler implements the gRPC Health service.
type HealthGRPCHandler struct {
	pbv1.UnimplementedHealthServer
	checker interface {
		ServeHTTP(http.ResponseWriter, *http.Request)
	}
}

func NewHealthGRPCHandler(checker interface {
	ServeHTTP(http.ResponseWriter, *http.Request)
}) *HealthGRPCHandler {
	return &HealthGRPCHandler{checker: checker}
}

func (h *HealthGRPCHandler) Check(ctx context.Context, req *pbv1.HealthCheckRequest) (*pbv1.HealthCheckResponse, error) {
	return &pbv1.HealthCheckResponse{
		Status: pbv1.HealthCheckResponse_SERVING,
	}, nil
}
