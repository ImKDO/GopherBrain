package pipeline

import (
	"context"
	"log/slog"
	"time"
)

type BatchRequest struct {
	Requests []*Request
}

type Batcher struct {
	inCh         <-chan *Request
	outCh        chan<- *BatchRequest
	maxBatchSize int
	timeout      time.Duration
	logger       *slog.Logger
}

func NewBatcher(inCh <-chan *Request, outCh chan<- *BatchRequest, maxBatch int, timeout time.Duration, logger *slog.Logger) *Batcher {
	return &Batcher{
		inCh:         inCh,
		outCh:        outCh,
		maxBatchSize: maxBatch,
		timeout:      timeout,
		logger:       logger,
	}
}

func (b *Batcher) Run(ctx context.Context) {
	defer close(b.outCh)

	for {
		// Wait for the first request
		select {
		case <-ctx.Done():
			return
		case req, ok := <-b.inCh:
			if !ok {
				return
			}
			b.accumulate(ctx, req)
		}
	}
}

func (b *Batcher) accumulate(ctx context.Context, first *Request) {
	batch := make([]*Request, 0, b.maxBatchSize)
	batch = append(batch, first)

	timer := time.NewTimer(b.timeout)
	defer timer.Stop()

	for len(batch) < b.maxBatchSize {
		select {
		case <-ctx.Done():
			b.flush(batch)
			return
		case req, ok := <-b.inCh:
			if !ok {
				b.flush(batch)
				return
			}
			batch = append(batch, req)
		case <-timer.C:
			b.flush(batch)
			return
		}
	}

	b.flush(batch)
}

func (b *Batcher) flush(batch []*Request) {
	if len(batch) == 0 {
		return
	}
	b.outCh <- &BatchRequest{Requests: batch}
}

// NewBatchWorkerPool creates a worker pool that processes batches.
func NewBatchWorkerPool(
	numWorkers int,
	preprocess *PreprocessStage,
	batchInfer *BatchInferenceStage,
	postprocess *PostprocessStage,
	batchCh <-chan *BatchRequest,
	logger *slog.Logger,
) *WorkerPool {
	// We wrap the batch channel into the worker pool using a custom goroutine approach.
	// The WorkerPool struct is reused but workers are started differently.
	wp := &WorkerPool{
		numWorkers: numWorkers,
		logger:     logger,
	}

	// Override Start to use batch processing
	wp.startBatch = func(ctx context.Context) {
		for i := 0; i < numWorkers; i++ {
			wp.wg.Add(1)
			go func(id int) {
				defer wp.wg.Done()
				for {
					select {
					case <-ctx.Done():
						return
					case batch, ok := <-batchCh:
						if !ok {
							return
						}
						// Preprocess each request
						for _, req := range batch.Requests {
							if err := preprocess.Process(ctx, req); err != nil {
								req.ResultCh <- Result{Err: err}
								continue
							}
						}

						// Batch inference
						if err := batchInfer.ProcessBatch(ctx, batch); err != nil {
							logger.Error("batch inference failed", "error", err)
							continue
						}

						// Postprocess each request and send results
						for _, req := range batch.Requests {
							if req.Output == nil {
								continue // already sent error in ProcessBatch
							}
							if err := postprocess.Process(ctx, req); err != nil {
								req.ResultCh <- Result{Err: err}
								continue
							}
							req.ResultCh <- Result{
								Output:    req.Output,
								LatencyMs: req.LatencyMs,
							}
						}
					}
				}
			}(i)
		}
	}

	return wp
}
