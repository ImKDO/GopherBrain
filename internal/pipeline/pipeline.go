package pipeline

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/ImKDO/GopherBrain/internal/config"
	"github.com/ImKDO/GopherBrain/internal/runtime"
)

type Pipeline struct {
	inCh        chan *Request
	pool        *WorkerPool
	batcher     *Batcher
	batchStage  *BatchInferenceStage
	session     *runtime.ONNXSession
	logger      *slog.Logger
	cfg         config.PipelineConfig
	cancel      context.CancelFunc
	useBatching bool
}

func NewPipeline(session *runtime.ONNXSession, cfg config.PipelineConfig, logger *slog.Logger) *Pipeline {
	inCh := make(chan *Request, cfg.ChannelBuffer)
	useBatching := cfg.BatchSize > 1

	p := &Pipeline{
		inCh:        inCh,
		session:     session,
		logger:      logger,
		cfg:         cfg,
		useBatching: useBatching,
	}

	if useBatching {
		batchCh := make(chan *BatchRequest, cfg.ChannelBuffer)
		p.batcher = NewBatcher(inCh, batchCh, cfg.BatchSize,
			time.Duration(cfg.BatchTimeoutMs)*time.Millisecond, logger)
		p.batchStage = NewBatchInferenceStage(session, logger)

		preprocess := NewPreprocessStage(logger)
		postprocess := NewPostprocessStage(logger)

		p.pool = NewBatchWorkerPool(cfg.NumWorkers, preprocess, p.batchStage,
			postprocess, batchCh, logger)
	} else {
		stages := []Stage{
			NewPreprocessStage(logger),
			NewInferenceStage(session, logger),
			NewPostprocessStage(logger),
		}
		p.pool = NewWorkerPool(cfg.NumWorkers, stages, inCh, logger)
	}

	return p
}

func (p *Pipeline) Start(ctx context.Context) {
	ctx, p.cancel = context.WithCancel(ctx)

	if p.useBatching {
		go p.batcher.Run(ctx)
	}

	p.pool.Start(ctx)
	p.logger.Info("pipeline started",
		"workers", p.cfg.NumWorkers,
		"batching", p.useBatching,
		"batch_size", p.cfg.BatchSize,
	)
}

func (p *Pipeline) Submit(ctx context.Context, req *Request) (Result, error) {
	req.ReceivedAt = time.Now()
	resultCh := make(chan Result, 1)
	req.ResultCh = resultCh

	select {
	case p.inCh <- req:
	case <-ctx.Done():
		return Result{}, fmt.Errorf("context cancelled while submitting: %w", ctx.Err())
	}

	select {
	case result := <-resultCh:
		return result, nil
	case <-ctx.Done():
		return Result{}, fmt.Errorf("context cancelled while waiting: %w", ctx.Err())
	}
}

func (p *Pipeline) Stop() {
	p.logger.Info("pipeline stopping")
	if p.cancel != nil {
		p.cancel()
	}
	close(p.inCh)
	p.pool.Wait()
	p.logger.Info("pipeline stopped")
}
