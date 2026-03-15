package runtime

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/ImKDO/GopherBrain/internal/config"
	"github.com/ImKDO/GopherBrain/pkg/tensor"
)

type ONNXSession struct {
	session     *ort.DynamicAdvancedSession
	inputNames  []string
	outputNames []string
	inputSpec   DimensionSpec
	modelPath   string
	mu          sync.RWMutex
	logger      *slog.Logger
}

type InferenceResult struct {
	Output  *tensor.Tensor[float32]
	Latency time.Duration
}

func InitRuntime(libPath string) error {
	if libPath != "" {
		ort.SetSharedLibraryPath(libPath)
	}
	return ort.InitializeEnvironment()
}

func DestroyRuntime() error {
	return ort.DestroyEnvironment()
}

func NewONNXSession(modelPath string, cfg config.RuntimeConfig, logger *slog.Logger) (*ONNXSession, error) {
	// Probe the model for input/output names
	inputInfos, outputInfos, err := ort.GetInputOutputInfo(modelPath)
	if err != nil {
		return nil, fmt.Errorf("getting model info: %w", err)
	}

	inputNames := make([]string, len(inputInfos))
	for i, info := range inputInfos {
		inputNames[i] = info.Name
	}
	outputNames := make([]string, len(outputInfos))
	for i, info := range outputInfos {
		outputNames[i] = info.Name
	}

	// Create the dynamic session (supports variable-shape outputs)
	session, err := ort.NewDynamicAdvancedSession(modelPath, inputNames, outputNames, nil)
	if err != nil {
		return nil, fmt.Errorf("creating ONNX session: %w", err)
	}

	s := &ONNXSession{
		session:     session,
		modelPath:   modelPath,
		inputNames:  inputNames,
		outputNames: outputNames,
		inputSpec: DimensionSpec{
			Name:    "input",
			MinRank: 1,
			MaxRank: 8,
		},
		logger: logger,
	}

	logger.Info("ONNX session created",
		"model", modelPath,
		"inputs", inputNames,
		"outputs", outputNames,
	)
	return s, nil
}

func (s *ONNXSession) RunInference(ctx context.Context, input *tensor.Tensor[float32]) (*InferenceResult, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	start := time.Now()

	if err := ValidateTensor(input, s.inputSpec); err != nil {
		return nil, fmt.Errorf("input validation: %w", err)
	}

	// Create input tensor
	inputShape := ort.NewShape(input.Shape...)
	inputTensor, err := ort.NewTensor(inputShape, input.Data)
	if err != nil {
		return nil, fmt.Errorf("creating input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// nil output → runtime auto-allocates
	inputs := []ort.Value{inputTensor}
	outputs := []ort.Value{nil}

	if err := s.session.Run(inputs, outputs); err != nil {
		return nil, fmt.Errorf("running inference: %w", err)
	}

	// The output was auto-allocated, we need to destroy it after copying data
	outVal := outputs[0]
	defer outVal.Destroy()

	// Extract output data via type assertion to *ort.Tensor[float32]
	outTensor, ok := outVal.(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("unexpected output tensor type: %T", outVal)
	}

	outputData := outTensor.GetData()
	resultData := make([]float32, len(outputData))
	copy(resultData, outputData)

	outputShapeDims := outTensor.GetShape()
	resultShape := make([]int64, len(outputShapeDims))
	copy(resultShape, outputShapeDims)

	result := &InferenceResult{
		Output: &tensor.Tensor[float32]{
			Shape: resultShape,
			Data:  resultData,
		},
		Latency: time.Since(start),
	}

	return result, nil
}

func (s *ONNXSession) RunBatchInference(ctx context.Context, inputs []*tensor.Tensor[float32]) ([]*InferenceResult, error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("no inputs provided")
	}

	if len(inputs) == 1 {
		r, err := s.RunInference(ctx, inputs[0])
		if err != nil {
			return nil, err
		}
		return []*InferenceResult{r}, nil
	}

	batchSizes := make([]int64, len(inputs))
	for i, inp := range inputs {
		if len(inp.Shape) == 0 {
			return nil, fmt.Errorf("input %d has empty shape", i)
		}
		batchSizes[i] = inp.Shape[0]
	}

	batched, err := tensor.ConcatBatch(inputs)
	if err != nil {
		return nil, fmt.Errorf("concatenating batch: %w", err)
	}

	batchResult, err := s.RunInference(ctx, batched)
	if err != nil {
		return nil, fmt.Errorf("batch inference: %w", err)
	}

	splitOutputs, err := tensor.SplitBatch(batchResult.Output, batchSizes)
	if err != nil {
		return nil, fmt.Errorf("splitting batch output: %w", err)
	}

	results := make([]*InferenceResult, len(splitOutputs))
	perItemLatency := batchResult.Latency / time.Duration(len(inputs))
	for i, out := range splitOutputs {
		results[i] = &InferenceResult{
			Output:  out,
			Latency: perItemLatency,
		}
	}

	return results, nil
}

func (s *ONNXSession) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.session != nil {
		err := s.session.Destroy()
		s.session = nil
		s.logger.Info("ONNX session closed")
		return err
	}
	return nil
}
