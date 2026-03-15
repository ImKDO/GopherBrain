package pipeline

import (
	"context"
	"time"

	"github.com/ImKDO/GopherBrain/pkg/tensor"
)

type Request struct {
	ID         string
	Input      *tensor.Tensor[float32]
	Output     *tensor.Tensor[float32]
	ModelID    string
	Metadata   map[string]string
	ReceivedAt time.Time
	ResultCh   chan<- Result
	Ctx        context.Context
	LatencyMs  float64
}

type Result struct {
	Output    *tensor.Tensor[float32]
	LatencyMs float64
	Err       error
}

type Stage interface {
	Name() string
	Process(ctx context.Context, req *Request) error
}
