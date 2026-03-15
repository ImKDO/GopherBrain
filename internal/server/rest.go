package server

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ImKDO/GopherBrain/internal/config"
	"github.com/ImKDO/GopherBrain/internal/pipeline"
	"github.com/ImKDO/GopherBrain/pkg/tensor"
	"github.com/google/uuid"
)

type RESTServer struct {
	engine   *gin.Engine
	server   *http.Server
	logger   *slog.Logger
	cfg      config.ServerConfig
	pipeline *pipeline.Pipeline
}

type inferRequest struct {
	RequestID string            `json:"request_id"`
	ModelID   string            `json:"model_id"`
	Input     tensorPayload     `json:"input"`
	Metadata  map[string]string `json:"metadata"`
}

type tensorPayload struct {
	Shape []int64   `json:"shape"`
	Data  []float32 `json:"data"`
}

type inferResponse struct {
	RequestID string        `json:"request_id"`
	Output    tensorPayload `json:"output"`
	LatencyMs float64       `json:"latency_ms"`
}

func NewRESTServer(cfg config.ServerConfig, p *pipeline.Pipeline, logger *slog.Logger) *RESTServer {
	gin.SetMode(gin.ReleaseMode)
	engine := gin.New()
	engine.Use(gin.Recovery())

	s := &RESTServer{
		engine:   engine,
		logger:   logger,
		cfg:      cfg,
		pipeline: p,
	}

	s.registerRoutes()
	return s
}

func (s *RESTServer) registerRoutes() {
	v1 := s.engine.Group("/v1")
	v1.POST("/infer", s.handleInfer)
}

func (s *RESTServer) handleInfer(c *gin.Context) {
	var req inferRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("invalid request: %v", err)})
		return
	}

	if s.pipeline == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "pipeline not ready"})
		return
	}

	input, err := tensor.New[float32](req.Input.Shape, req.Input.Data)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("invalid tensor: %v", err)})
		return
	}

	requestID := req.RequestID
	if requestID == "" {
		requestID = uuid.New().String()
	}

	result, err := s.pipeline.Submit(c.Request.Context(), &pipeline.Request{
		ID:       requestID,
		Input:    input,
		ModelID:  req.ModelID,
		Metadata: req.Metadata,
		Ctx:      c.Request.Context(),
	})
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	if result.Err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": result.Err.Error()})
		return
	}

	resp := inferResponse{
		RequestID: requestID,
		Output:    tensorPayload{Shape: result.Output.Shape, Data: result.Output.Data},
		LatencyMs: result.LatencyMs,
	}

	c.JSON(http.StatusOK, resp)
}

func (s *RESTServer) Engine() *gin.Engine {
	return s.engine
}

func (s *RESTServer) Start() error {
	s.server = &http.Server{
		Addr:    fmt.Sprintf(":%d", s.cfg.RESTPort),
		Handler: s.engine,
	}
	s.logger.Info("REST server starting", "port", s.cfg.RESTPort)
	return s.server.ListenAndServe()
}

func (s *RESTServer) Stop(ctx context.Context) error {
	s.logger.Info("REST server stopping")
	shutdownCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	return s.server.Shutdown(shutdownCtx)
}

// MarshalJSON is a helper for tests.
func marshalJSON(v any) []byte {
	data, _ := json.Marshal(v)
	return data
}

func (s *RESTServer) RegisterHealthRoutes(hc interface {
	LivenessHandler() gin.HandlerFunc
	ReadinessHandler() gin.HandlerFunc
}) {
	s.engine.GET("/healthz", hc.LivenessHandler())
	s.engine.GET("/readyz", hc.ReadinessHandler())
}
