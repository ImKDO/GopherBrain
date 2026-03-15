package observability

import (
	"fmt"
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

type Metrics struct {
	RequestsTotal    *prometheus.CounterVec
	RequestDuration  *prometheus.HistogramVec
	ErrorsTotal      *prometheus.CounterVec
	BatchSize        prometheus.Histogram
	InferenceLatency *prometheus.HistogramVec
	PipelineDepth    prometheus.Gauge
	ActiveWorkers    prometheus.Gauge
}

func NewMetrics(reg prometheus.Registerer) *Metrics {
	m := &Metrics{
		RequestsTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: "gopherbrain",
				Name:      "requests_total",
				Help:      "Total number of inference requests",
			},
			[]string{"method", "status"},
		),
		RequestDuration: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Namespace: "gopherbrain",
				Name:      "request_duration_seconds",
				Help:      "Request duration in seconds",
				Buckets:   []float64{0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0},
			},
			[]string{"method"},
		),
		ErrorsTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: "gopherbrain",
				Name:      "errors_total",
				Help:      "Total number of errors",
			},
			[]string{"stage", "error_type"},
		),
		BatchSize: prometheus.NewHistogram(
			prometheus.HistogramOpts{
				Namespace: "gopherbrain",
				Name:      "batch_size",
				Help:      "Number of requests per inference batch",
				Buckets:   []float64{1, 2, 4, 8, 16, 32, 64},
			},
		),
		InferenceLatency: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Namespace: "gopherbrain",
				Name:      "inference_latency_seconds",
				Help:      "ML model inference latency in seconds",
				Buckets:   []float64{0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0},
			},
			[]string{"model_id"},
		),
		PipelineDepth: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Namespace: "gopherbrain",
				Name:      "pipeline_depth",
				Help:      "Current number of requests in the pipeline queue",
			},
		),
		ActiveWorkers: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Namespace: "gopherbrain",
				Name:      "active_workers",
				Help:      "Number of currently active workers",
			},
		),
	}

	reg.MustRegister(
		m.RequestsTotal,
		m.RequestDuration,
		m.ErrorsTotal,
		m.BatchSize,
		m.InferenceLatency,
		m.PipelineDepth,
		m.ActiveWorkers,
	)

	return m
}

func StartMetricsServer(port int) *http.Server {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.Handler())

	srv := &http.Server{
		Addr:    fmt.Sprintf(":%d", port),
		Handler: mux,
	}

	go srv.ListenAndServe()
	return srv
}
