package pool

import (
	"sync"
)

type TensorPool struct {
	pools   map[int]*sync.Pool
	buckets []int
}

var defaultBuckets = []int{1024, 4096, 16384, 65536, 262144}

func NewTensorPool(buckets []int) *TensorPool {
	if len(buckets) == 0 {
		buckets = defaultBuckets
	}

	pools := make(map[int]*sync.Pool, len(buckets))
	for _, size := range buckets {
		s := size
		pools[s] = &sync.Pool{
			New: func() any {
				buf := make([]float32, 0, s)
				return &buf
			},
		}
	}

	return &TensorPool{
		pools:   pools,
		buckets: buckets,
	}
}

func (tp *TensorPool) Get(minCap int) []float32 {
	bucket := tp.findBucket(minCap)
	if bucket == 0 {
		return make([]float32, 0, minCap)
	}

	bufPtr := tp.pools[bucket].Get().(*[]float32)
	return (*bufPtr)[:0]
}

func (tp *TensorPool) Put(s []float32) {
	bucket := tp.findBucket(cap(s))
	if bucket == 0 {
		return
	}

	s = s[:0]
	tp.pools[bucket].Put(&s)
}

func (tp *TensorPool) findBucket(size int) int {
	for _, b := range tp.buckets {
		if b >= size {
			return b
		}
	}
	return 0
}
