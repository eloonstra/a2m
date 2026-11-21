package launcher

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"testing"

	"a2m/internal/mistral"
)

func chatModel(id, name string, context int) mistral.Model {
	return mistral.Model{
		ID: id, Name: name, MaxContextLength: context,
		Capabilities: mistral.ModelCapabilities{CompletionChat: true, FunctionCalling: true},
	}
}

func TestGroupModels(t *testing.T) {
	deprecated := json.RawMessage(`"2026-01-01"`)
	audio := chatModel("voxtral-small-latest", "voxtral", 32768)
	audio.Capabilities.Audio = true
	old := chatModel("old-model", "old", 32768)
	old.Deprecation = &deprecated
	noTools := chatModel("embed", "embed", 8192)
	noTools.Capabilities.FunctionCalling = false

	models := []mistral.Model{
		chatModel("mistral-small-2603", "mistral-small-2603", 262144),
		chatModel("mistral-small-latest", "mistral-small-2603", 262144),
		chatModel("magistral-small-latest", "mistral-small-2603", 262144),
		chatModel("glm-5-2", "glm-5-2", 1048576),
		chatModel("zai-glm-5-2", "glm-5-2", 1048576),
		audio, old, noTools,
	}
	var ids []string
	for _, g := range groupModels(models) {
		ids = append(ids, g.id)
	}
	want := []string{"mistral-small-latest", "glm-5-2"}
	if !slices.Equal(ids, want) {
		t.Errorf("ids = %v, want %v", ids, want)
	}
}

func TestCompactWindow(t *testing.T) {
	tests := []struct{ context, reserve, want int }{
		{262144, 32768, 229376},
		{131072, 32768, minAutoCompactWindow},
		{1048576, 32768, maxAutoCompactWindow},
	}
	for _, tt := range tests {
		if got := compactWindow(tt.context, tt.reserve); got != tt.want {
			t.Errorf("compactWindow(%d, %d) = %d, want %d", tt.context, tt.reserve, got, tt.want)
		}
	}
}

func TestDescribe(t *testing.T) {
	m := chatModel("a", "a", 1048576)
	m.Description = "Large"
	if got := describe(m); got != "Large · 1M context" {
		t.Errorf("describe = %q", got)
	}
	if got := describe(chatModel("b", "b", 256000)); got != "256k context" {
		t.Errorf("describe = %q", got)
	}
}

func TestBuildSettings(t *testing.T) {
	settings, maxContext, err := buildSettings([]mistral.Model{
		chatModel("a-latest", "a", 262144),
		chatModel("b-latest", "b", 1048576),
	}, 32768)
	if err != nil {
		t.Fatal(err)
	}
	if maxContext != 1048576 {
		t.Errorf("maxContext = %d", maxContext)
	}
	var decoded claudeSettings
	if err := json.Unmarshal([]byte(settings), &decoded); err != nil {
		t.Fatal(err)
	}
	if !decoded.ModelPicker.ReplaceBuiltInOptions || len(decoded.ModelPicker.Options) != 2 {
		t.Errorf("picker = %+v", decoded.ModelPicker)
	}
	if decoded.ModelSettings["a-latest"].AutoCompactWindow != 229376 {
		t.Errorf("modelSettings = %+v", decoded.ModelSettings)
	}
}

func TestBuildSettingsWithoutUsableModels(t *testing.T) {
	if _, _, err := buildSettings(nil, 0); err == nil {
		t.Errorf("expected error for empty model list")
	}
}

func envMap(env []string) map[string]string {
	out := make(map[string]string, len(env))
	for _, kv := range env {
		key, value, _ := strings.Cut(kv, "=")
		out[key] = value
	}
	return out
}

func TestChildEnv(t *testing.T) {
	t.Setenv(envAPIKey, "real-anthropic-key")
	t.Setenv(envBaseURL, "https://api.anthropic.com")
	t.Setenv(envAutoModeServer, "1")

	env := envMap(childEnv("http://127.0.0.1:1", "secret", map[string]string{
		envAutoModeServer: disabledFlag,
		envAttribution:    disabledFlag,
	}))

	if _, ok := env[envAPIKey]; ok {
		t.Errorf("%s was not removed", envAPIKey)
	}
	if env[envBaseURL] != "http://127.0.0.1:1" || env[envAuthToken] != "secret" {
		t.Errorf("overrides not applied: %s=%q %s=%q", envBaseURL, env[envBaseURL], envAuthToken, env[envAuthToken])
	}
	if env[envAutoModeServer] != "1" {
		t.Errorf("user value lost: %s=%q", envAutoModeServer, env[envAutoModeServer])
	}
	if env[envAttribution] != disabledFlag {
		t.Errorf("default not applied: %s=%q", envAttribution, env[envAttribution])
	}
}

type fakeLister struct {
	models []mistral.Model
	err    error
}

func (f fakeLister) ListModels(_ context.Context, _ string) ([]mistral.Model, error) {
	return f.models, f.err
}

func TestWithModelPicker(t *testing.T) {
	opts := Options{OutputReserve: 32768, Models: fakeLister{models: []mistral.Model{chatModel("a-latest", "a", 262144)}}}
	args, maxContext := withModelPicker([]string{"claude", "--continue"}, opts)
	if len(args) != 4 || args[0] != "claude" || args[1] != settingsFlag || args[3] != "--continue" || maxContext != 262144 {
		t.Errorf("args = %v, maxContext = %d", args, maxContext)
	}
}

func TestWithModelPickerKeepsArgsOnFailure(t *testing.T) {
	opts := Options{Models: fakeLister{err: errors.New("down")}}
	args, maxContext := withModelPicker([]string{"claude", "-p", "x"}, opts)
	if !slices.Equal(args, []string{"claude", "-p", "x"}) || maxContext != 0 {
		t.Errorf("args = %v, maxContext = %d", args, maxContext)
	}
}

func TestNewToken(t *testing.T) {
	a, errA := NewToken()
	b, errB := NewToken()
	if errA != nil || errB != nil || len(a) != 2*tokenBytes || a == b {
		t.Errorf("tokens %q %q errors %v %v", a, b, errA, errB)
	}
}

func TestRunGuards(t *testing.T) {
	tests := []struct {
		name string
		opts Options
		args []string
	}{
		{"no command", Options{APIKey: "k", Token: "t"}, nil},
		{"no key", Options{Token: "t"}, []string{"true"}},
		{"no token", Options{APIKey: "k"}, []string{"true"}},
	}
	for _, tt := range tests {
		code, err := Run(nil, tt.opts, tt.args)
		if err == nil || code != exitFailure {
			t.Errorf("%s: code=%d err=%v", tt.name, code, err)
		}
	}
}
