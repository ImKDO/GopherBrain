package pipeline

import (
	"context"
	"log/slog"
	"sync"
)

type WorkerPool struct {
	numWorkers int
	stages     []Stage
	inCh       <-chan *Request
	wg         sync.WaitGroup
	logger     *slog.Logger
	startBatch func(ctx context.Context) // set by NewBatchWorkerPool
}

func NewWorkerPool(numWorkers int, stages []Stage, inCh <-chan *Request, logger *slog.Logger) *WorkerPool {
	return &WorkerPool{
		numWorkers: numWorkers,
		stages:     stages,
		inCh:       inCh,
		logger:     logger,
	}
}

func (wp *WorkerPool) Start(ctx context.Context) {
	if wp.startBatch != nil {
		wp.startBatch(ctx)
		wp.logger.Info("batch worker pool started", "workers", wp.numWorkers)
		return
	}
	for i := 0; i < wp.numWorkers; i++ {
		wp.wg.Add(1)
		go wp.worker(ctx, i)
	}
	wp.logger.Info("worker pool started", "workers", wp.numWorkers)
}

func (wp *WorkerPool) worker(ctx context.Context, id int) {
	defer wp.wg.Done()

	for {
		select {
		case <-ctx.Done():
			return
		case req, ok := <-wp.inCh:
			if !ok {
				return
			}
			wp.processRequest(ctx, req)
		}
	}
}

func (wp *WorkerPool) processRequest(ctx context.Context, req *Request) {
	for _, stage := range wp.stages {
		if err := stage.Process(ctx, req); err != nil {
			wp.logger.Error("stage failed",
				"stage", stage.Name(),
				"request_id", req.ID,
				"error", err,
			)
			req.ResultCh <- Result{Err: err}
			return
		}
	}

	req.ResultCh <- Result{
		Output:    req.Output,
		LatencyMs: req.LatencyMs,
	}
}

func (wp *WorkerPool) Wait() {
	wp.wg.Wait()
}
