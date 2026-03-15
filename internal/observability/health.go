package observability

import (
	"encoding/json"
	"net/http"
	"sync"

	"github.com/gin-gonic/gin"
)

type HealthChecker struct {
	checks map[string]func() error
	mu     sync.RWMutex
}

func NewHealthChecker() *HealthChecker {
	return &HealthChecker{
		checks: make(map[string]func() error),
	}
}

func (hc *HealthChecker) Register(name string, check func() error) {
	hc.mu.Lock()
	defer hc.mu.Unlock()
	hc.checks[name] = check
}

func (hc *HealthChecker) LivenessHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "alive"})
	}
}

func (hc *HealthChecker) ReadinessHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		hc.mu.RLock()
		defer hc.mu.RUnlock()

		results := make(map[string]string, len(hc.checks))
		allHealthy := true

		for name, check := range hc.checks {
			if err := check(); err != nil {
				results[name] = err.Error()
				allHealthy = false
			} else {
				results[name] = "ok"
			}
		}

		status := http.StatusOK
		if !allHealthy {
			status = http.StatusServiceUnavailable
		}

		c.JSON(status, gin.H{
			"status": map[bool]string{true: "ready", false: "not_ready"}[allHealthy],
			"checks": results,
		})
	}
}

func (hc *HealthChecker) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	hc.mu.RLock()
	defer hc.mu.RUnlock()

	results := make(map[string]string, len(hc.checks))
	allHealthy := true

	for name, check := range hc.checks {
		if err := check(); err != nil {
			results[name] = err.Error()
			allHealthy = false
		} else {
			results[name] = "ok"
		}
	}

	status := http.StatusOK
	if !allHealthy {
		status = http.StatusServiceUnavailable
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]any{
		"status": map[bool]string{true: "ready", false: "not_ready"}[allHealthy],
		"checks": results,
	})
}
