package tensor

import (
	"fmt"

	pbv1 "github.com/ImKDO/GopherBrain/api/proto/v1"
)

type Numeric interface {
	float32 | float64 | int32 | int64
}

type Tensor[T Numeric] struct {
	Shape []int64
	Data  []T
}

func New[T Numeric](shape []int64, data []T) (*Tensor[T], error) {
	expected := numElements(shape)
	if int64(len(data)) != expected {
		return nil, fmt.Errorf("shape %v expects %d elements, got %d", shape, expected, len(data))
	}
	return &Tensor[T]{Shape: shape, Data: data}, nil
}

func (t *Tensor[T]) NumElements() int64 {
	return numElements(t.Shape)
}

func (t *Tensor[T]) Reshape(newShape []int64) error {
	if numElements(newShape) != t.NumElements() {
		return fmt.Errorf("cannot reshape %v (%d elements) to %v (%d elements)",
			t.Shape, t.NumElements(), newShape, numElements(newShape))
	}
	t.Shape = newShape
	return nil
}

func (t *Tensor[T]) Clone() *Tensor[T] {
	dataCopy := make([]T, len(t.Data))
	copy(dataCopy, t.Data)
	shapeCopy := make([]int64, len(t.Shape))
	copy(shapeCopy, t.Shape)
	return &Tensor[T]{Shape: shapeCopy, Data: dataCopy}
}

func FromProto(pb *pbv1.Tensor) (*Tensor[float32], error) {
	if pb == nil {
		return nil, fmt.Errorf("nil tensor proto")
	}
	return New[float32](pb.GetShape(), pb.GetData())
}

func FromProtoInto(pb *pbv1.Tensor, buf []float32) (*Tensor[float32], error) {
	if pb == nil {
		return nil, fmt.Errorf("nil tensor proto")
	}
	shape := pb.GetShape()
	data := pb.GetData()
	expected := numElements(shape)
	if int64(len(data)) != expected {
		return nil, fmt.Errorf("shape %v expects %d elements, got %d", shape, expected, len(data))
	}
	if cap(buf) >= len(data) {
		buf = buf[:len(data)]
	} else {
		buf = make([]float32, len(data))
	}
	copy(buf, data)
	return &Tensor[float32]{Shape: shape, Data: buf}, nil
}

func ToProto(t *Tensor[float32]) *pbv1.Tensor {
	if t == nil {
		return nil
	}
	return &pbv1.Tensor{
		Shape: t.Shape,
		Data:  t.Data,
	}
}

func ConcatBatch(tensors []*Tensor[float32]) (*Tensor[float32], error) {
	if len(tensors) == 0 {
		return nil, fmt.Errorf("no tensors to concatenate")
	}

	innerShape := tensors[0].Shape[1:]
	innerSize := numElements(innerShape)

	totalBatch := int64(0)
	for _, t := range tensors {
		if len(t.Shape) == 0 {
			return nil, fmt.Errorf("tensor has empty shape")
		}
		totalBatch += t.Shape[0]
		tInner := t.Shape[1:]
		if !shapeEqual(tInner, innerShape) {
			return nil, fmt.Errorf("shape mismatch: expected inner %v, got %v", innerShape, tInner)
		}
	}

	data := make([]float32, 0, totalBatch*innerSize)
	for _, t := range tensors {
		data = append(data, t.Data...)
	}

	batchShape := make([]int64, len(tensors[0].Shape))
	batchShape[0] = totalBatch
	copy(batchShape[1:], innerShape)

	return &Tensor[float32]{Shape: batchShape, Data: data}, nil
}

func SplitBatch(t *Tensor[float32], batchSizes []int64) ([]*Tensor[float32], error) {
	if len(t.Shape) == 0 {
		return nil, fmt.Errorf("tensor has empty shape")
	}

	innerShape := t.Shape[1:]
	innerSize := numElements(innerShape)

	var totalBatch int64
	for _, bs := range batchSizes {
		totalBatch += bs
	}
	if totalBatch != t.Shape[0] {
		return nil, fmt.Errorf("batch sizes sum to %d but tensor has batch dim %d", totalBatch, t.Shape[0])
	}

	result := make([]*Tensor[float32], len(batchSizes))
	offset := int64(0)
	for i, bs := range batchSizes {
		size := bs * innerSize
		shape := make([]int64, len(t.Shape))
		shape[0] = bs
		copy(shape[1:], innerShape)
		result[i] = &Tensor[float32]{
			Shape: shape,
			Data:  t.Data[offset : offset+size],
		}
		offset += size
	}

	return result, nil
}

func numElements(shape []int64) int64 {
	if len(shape) == 0 {
		return 0
	}
	n := int64(1)
	for _, d := range shape {
		n *= d
	}
	return n
}

func shapeEqual(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
