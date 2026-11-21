package translate

import (
	"testing"

	"a2m/internal/anthropic"
	"a2m/internal/mistral"
)

func reasoningModel(name string) mistral.Model {
	return mistral.Model{ID: name, Name: name, Capabilities: mistral.ModelCapabilities{Reasoning: true}}
}

func TestReasoningEffort(t *testing.T) {
	on := &anthropic.Thinking{Type: "adaptive"}
	off := &anthropic.Thinking{Type: anthropic.ThinkingDisabled}
	effort := func(level string) *anthropic.OutputConfig { return &anthropic.OutputConfig{Effort: level} }

	tests := []struct {
		name     string
		model    mistral.Model
		thinking *anthropic.Thinking
		cfg      *anthropic.OutputConfig
		want     string
	}{
		{"no reasoning capability", mistral.Model{Name: "codestral-2508"}, on, effort("high"), ""},
		{"unknown reasoning model gets default effort", reasoningModel("future-model"), on, effort("high"), effortHigh},
		{"unknown reasoning model thinking off", reasoningModel("future-model"), off, nil, effortNone},
		{"matched by id when name differs", mistral.Model{ID: "mistral-large-4-2603", Name: "Mistral Large 4", Capabilities: mistral.ModelCapabilities{Reasoning: true}}, on, effort("medium"), effortMedium},
		{"small low rounds up", reasoningModel("mistral-small-2603"), on, effort("low"), effortHigh},
		{"small thinking off", reasoningModel("mistral-small-2603"), off, nil, effortNone},
		{"small no thinking field", reasoningModel("mistral-small-2603"), nil, nil, effortNone},
		{"large medium", reasoningModel("mistral-large-4"), on, effort("medium"), effortMedium},
		{"large max rounds down", reasoningModel("mistral-large-4"), on, effort("max"), effortHigh},
		{"large off cannot disable", reasoningModel("mistral-large-4"), off, nil, ""},
		{"glm xhigh rounds to max", reasoningModel("zai-glm-5-3-flash"), on, effort("xhigh"), effortMax},
		{"glm medium tie rounds up", reasoningModel("zai-glm-5-3"), on, effort("medium"), effortHigh},
		{"deepseek default effort", reasoningModel("deepseek-v4-1-flash-1017"), on, nil, effortHigh},
		{"unknown effort uses default", reasoningModel("mistral-large-4"), on, effort("weird"), effortHigh},
	}
	for _, tt := range tests {
		if got := reasoningEffort(tt.model, tt.thinking, tt.cfg); got != tt.want {
			t.Errorf("%s: got %q, want %q", tt.name, got, tt.want)
		}
	}
}
