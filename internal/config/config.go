package config

import (
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Server        ServerConfig        `yaml:"server"`
	Runtime       RuntimeConfig       `yaml:"runtime"`
	Pipeline      PipelineConfig      `yaml:"pipeline"`
	Kafka         KafkaConfig         `yaml:"kafka"`
	Redis         RedisConfig         `yaml:"redis"`
	Observability ObservabilityConfig `yaml:"observability"`
}

type ServerConfig struct {
	GRPCPort int `yaml:"grpc_port"`
	RESTPort int `yaml:"rest_port"`
}

type RuntimeConfig struct {
	ModelPath      string `yaml:"model_path"`
	ONNXLibPath    string `yaml:"onnx_lib_path"`
	InterOpThreads int    `yaml:"inter_op_threads"`
	IntraOpThreads int    `yaml:"intra_op_threads"`
}

type PipelineConfig struct {
	NumWorkers     int `yaml:"num_workers"`
	BatchSize      int `yaml:"batch_size"`
	BatchTimeoutMs int `yaml:"batch_timeout_ms"`
	ChannelBuffer  int `yaml:"channel_buffer"`
}

type KafkaConfig struct {
	Brokers []string `yaml:"brokers"`
	Topic   string   `yaml:"topic"`
	GroupID string   `yaml:"group_id"`
}

type RedisConfig struct {
	Addr     string `yaml:"addr"`
	Password string `yaml:"password"`
	DB       int    `yaml:"db"`
	TTL      int    `yaml:"ttl_seconds"`
}

type ObservabilityConfig struct {
	LogLevel       string `yaml:"log_level"`
	MetricsPort    int    `yaml:"metrics_port"`
	JaegerEndpoint string `yaml:"jaeger_endpoint"`
}

func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading config: %w", err)
	}

	cfg := &Config{
		Server: ServerConfig{
			GRPCPort: 50051,
			RESTPort: 8080,
		},
		Pipeline: PipelineConfig{
			NumWorkers:     4,
			BatchSize:      32,
			BatchTimeoutMs: 10,
			ChannelBuffer:  256,
		},
		Observability: ObservabilityConfig{
			LogLevel:    "info",
			MetricsPort: 9090,
		},
	}

	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("parsing config: %w", err)
	}

	// Resolve relative paths to absolute using config file location
	absConfigPath, err := filepath.Abs(path)
	if err != nil {
		return nil, fmt.Errorf("resolving config path: %w", err)
	}
	// Project root = parent of configs/ directory
	baseDir := filepath.Dir(filepath.Dir(absConfigPath))

	cfg.Runtime.ModelPath = resolvePath(baseDir, cfg.Runtime.ModelPath)
	cfg.Runtime.ONNXLibPath = resolvePath(baseDir, cfg.Runtime.ONNXLibPath)

	return cfg, nil
}

func resolvePath(base, p string) string {
	if p == "" || filepath.IsAbs(p) {
		return p
	}
	return filepath.Join(base, p)
}
