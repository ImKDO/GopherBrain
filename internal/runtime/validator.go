package runtime

import (
	"fmt"

	"github.com/ImKDO/GopherBrain/pkg/tensor"
)

type DimRange struct {
	Min int64 // -1 means dynamic
	Max int64 // -1 means dynamic
}

type DimensionSpec struct {
	Name      string
	MinRank   int
	MaxRank   int
	DimRanges []DimRange
}

func ValidateTensor(t *tensor.Tensor[float32], spec DimensionSpec) error {
	if t == nil {
		return fmt.Errorf("nil tensor for %s", spec.Name)
	}

	rank := len(t.Shape)
	if rank < spec.MinRank {
		return fmt.Errorf("%s: rank %d is below minimum %d", spec.Name, rank, spec.MinRank)
	}
	if spec.MaxRank > 0 && rank > spec.MaxRank {
		return fmt.Errorf("%s: rank %d exceeds maximum %d", spec.Name, rank, spec.MaxRank)
	}

	for i, dr := range spec.DimRanges {
		if i >= rank {
			break
		}
		dim := t.Shape[i]
		if dr.Min >= 0 && dim < dr.Min {
			return fmt.Errorf("%s: dimension %d is %d, below minimum %d", spec.Name, i, dim, dr.Min)
		}
		if dr.Max >= 0 && dim > dr.Max {
			return fmt.Errorf("%s: dimension %d is %d, exceeds maximum %d", spec.Name, i, dim, dr.Max)
		}
	}

	return nil
}
