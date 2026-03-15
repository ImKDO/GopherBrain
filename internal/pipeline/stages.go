package pipeline

import (
	"context"
	"log/slog"
	"time"

	"github.com/ImKDO/GopherBrain/internal/runtime"
	"github.com/ImKDO/GopherBrain/pkg/tensor"
)

type PreprocessStage struct {
	logger *slog.Logger
}

func NewPreprocessStage(logger *slog.Logger) *PreprocessStage {
	return &PreprocessStage{logger: logger}
}

func (s *PreprocessStage) Name() string { return "preprocess" }

func (s *PreprocessStage) Process(ctx context.Context, req *Request) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	if req.Input == nil || len(req.Input.Data) == 0 {
		return &StageError{Stage: s.Name(), Msg: "empty input tensor"}
	}

	// Normalize: find min/max and scale to [0,1] if needed
	data := req.Input.Data
	var minVal, maxVal float32
	minVal = data[0]
	maxVal = data[0]
	for _, v := range data[1:] {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	// Only normalize if data is outside [0,1] range
	if minVal < 0 || maxVal > 1 {
		rangeVal := maxVal - minVal
		if rangeVal > 0 {
			for i := range data {
				data[i] = (data[i] - minVal) / rangeVal
			}
		}
	}

	return nil
}

type InferenceStage struct {
	session *runtime.ONNXSession
	logger  *slog.Logger
}

func NewInferenceStage(session *runtime.ONNXSession, logger *slog.Logger) *InferenceStage {
	return &InferenceStage{session: session, logger: logger}
}

func (s *InferenceStage) Name() string { return "inference" }

func (s *InferenceStage) Process(ctx context.Context, req *Request) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	result, err := s.session.RunInference(ctx, req.Input)
	if err != nil {
		return &StageError{Stage: s.Name(), Msg: err.Error()}
	}

	req.Output = result.Output
	req.LatencyMs = float64(result.Latency.Microseconds()) / 1000.0
	return nil
}

type BatchInferenceStage struct {
	session *runtime.ONNXSession
	logger  *slog.Logger
}

func NewBatchInferenceStage(session *runtime.ONNXSession, logger *slog.Logger) *BatchInferenceStage {
	return &BatchInferenceStage{session: session, logger: logger}
}

func (s *BatchInferenceStage) ProcessBatch(ctx context.Context, batch *BatchRequest) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	inputs := make([]*tensor.Tensor[float32], len(batch.Requests))
	for i, req := range batch.Requests {
		inputs[i] = req.Input
	}

	results, err := s.session.RunBatchInference(ctx, inputs)
	if err != nil {
		// Send error to all requests
		for _, req := range batch.Requests {
			req.ResultCh <- Result{Err: &StageError{Stage: "batch_inference", Msg: err.Error()}}
		}
		return err
	}

	for i, req := range batch.Requests {
		req.Output = results[i].Output
		req.LatencyMs = float64(results[i].Latency.Microseconds()) / 1000.0
	}

	return nil
}

type PostprocessStage struct {
	logger *slog.Logger
}

func NewPostprocessStage(logger *slog.Logger) *PostprocessStage {
	return &PostprocessStage{logger: logger}
}

func (s *PostprocessStage) Name() string { return "postprocess" }

func (s *PostprocessStage) Process(ctx context.Context, req *Request) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	if req.Output == nil {
		return &StageError{Stage: s.Name(), Msg: "no output from inference"}
	}

	totalLatency := float64(time.Since(req.ReceivedAt).Microseconds()) / 1000.0
	req.LatencyMs = totalLatency

	return nil
}

type StageError struct {
	Stage string
	Msg   string
}

func (e *StageError) Error() string {
	return e.Stage + ": " + e.Msg
}
