package translate

import (
	"encoding/json"
	"reflect"
	"testing"

	"a2m/internal/anthropic"
	"a2m/internal/mistral"
)

var testOptions = Options{Model: "big", MediumModel: "medium", SmallModel: "small", AutoModel: "auto-model", MaxTokens: 1000}

func decodeRequest(t *testing.T, raw string) *anthropic.MessagesRequest {
	t.Helper()
	var req anthropic.MessagesRequest
	if err := json.Unmarshal([]byte(raw), &req); err != nil {
		t.Fatalf("decode request: %v", err)
	}
	return &req
}

func TestMapModel(t *testing.T) {
	tests := map[string]string{
		"claude-haiku-5-5":     "small",
		"claude-medium-5-5":    "medium",
		"claude-opus-5-5":      "big",
		"claude-sonnet-5[1m]":  "big",
		"auto":                 "auto-model",
		"claude-auto":          "auto-model",
		"mistral-small-latest": "mistral-small-latest",
		"codestral-latest":     "codestral-latest",
	}
	for in, want := range tests {
		if got := MapModel(in, testOptions); got != want {
			t.Errorf("MapModel(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestCapTokens(t *testing.T) {
	tests := []struct{ requested, limit, want int }{
		{500, 1000, 500},
		{64000, 1000, 1000},
		{64000, 0, 64000},
	}
	for _, tt := range tests {
		if got := capTokens(tt.requested, tt.limit); got != tt.want {
			t.Errorf("capTokens(%d, %d) = %d, want %d", tt.requested, tt.limit, got, tt.want)
		}
	}
}

func TestToolCallID(t *testing.T) {
	native := "abcDEF123"
	if got := toolCallID(native); got != native {
		t.Errorf("native id changed: %q", got)
	}
	hashed := toolCallID("toolu_01ABCdef")
	if !isMistralToolCallID(hashed) {
		t.Errorf("hashed id %q is not a Mistral id", hashed)
	}
	if again := toolCallID("toolu_01ABCdef"); again != hashed {
		t.Errorf("hash is not stable: %q != %q", again, hashed)
	}
	if other := toolCallID("toolu_other"); other == hashed {
		t.Errorf("different ids share hash %q", hashed)
	}
}

func TestConvertMessages(t *testing.T) {
	req := decodeRequest(t, `{
		"system": [{"type":"text","text":"one"},{"type":"text","text":"two"}],
		"messages": [
			{"role":"user","content":"hi"},
			{"role":"system","content":[{"type":"text","text":"reminder"}]},
			{"role":"assistant","content":[
				{"type":"thinking","thinking":"plan","signature":"mistral:sig"},
				{"type":"thinking","thinking":"foreign","signature":"claude-sig"},
				{"type":"text","text":"ok"},
				{"type":"tool_use","id":"toolu_1","name":"Read","input":{"path":"/a"}}
			]},
			{"role":"user","content":[
				{"type":"tool_result","tool_use_id":"toolu_1","content":"body"},
				{"type":"text","text":"after"}
			]},
			{"role":"assistant","content":[{"type":"tool_use","id":"toolu_2","name":"Bash"}]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_2"}]}
		]}`)

	got := convertMessages(req.System, req.Messages)

	id1, id2 := toolCallID("toolu_1"), toolCallID("toolu_2")
	want := []mistral.Message{
		{Role: mistral.RoleSystem, Content: mistral.Content{textChunk("one\n\ntwo")}},
		{Role: mistral.RoleUser, Content: mistral.Content{textChunk("hi"), textChunk("reminder")}},
		{
			Role: mistral.RoleAssistant,
			Content: mistral.Content{
				{Type: mistral.ChunkThinking, Thinking: []mistral.Chunk{textChunk("plan")}, Signature: "sig"},
				{Type: mistral.ChunkThinking, Thinking: []mistral.Chunk{textChunk("foreign")}},
				textChunk("ok"),
			},
			ToolCalls: []mistral.ToolCall{{ID: id1, Type: mistral.ToolTypeFunction, Function: mistral.FunctionCall{Name: "Read", Arguments: json.RawMessage(`{"path":"/a"}`)}}},
		},
		{Role: mistral.RoleTool, Content: mistral.Content{textChunk("body")}, ToolCallID: id1, Name: "Read"},
		{Role: mistral.RoleUser, Content: mistral.Content{textChunk("after")}},
		{Role: mistral.RoleAssistant, ToolCalls: []mistral.ToolCall{{ID: id2, Type: mistral.ToolTypeFunction, Function: mistral.FunctionCall{Name: "Bash", Arguments: emptyObject}}}},
		{Role: mistral.RoleTool, Content: mistral.Content{textChunk(emptyToolResult)}, ToolCallID: id2, Name: "Bash"},
	}
	if !reflect.DeepEqual(got, want) {
		gotJSON, _ := json.MarshalIndent(got, "", " ")
		wantJSON, _ := json.MarshalIndent(want, "", " ")
		t.Errorf("messages differ\ngot:  %s\nwant: %s", gotJSON, wantJSON)
	}
}

func TestConvertMessagesPrefill(t *testing.T) {
	req := decodeRequest(t, `{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"Sure"}]}`)
	got := convertMessages(req.System, req.Messages)
	if !got[len(got)-1].Prefix {
		t.Errorf("last assistant message is not marked as prefix")
	}
}

func TestConvertImage(t *testing.T) {
	req := decodeRequest(t, `{"messages":[{"role":"user","content":[
		{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}},
		{"type":"image","source":{"type":"url","url":"https://x/y.png"}}
	]}]}`)
	got := convertMessages(req.System, req.Messages)[0].Content
	want := mistral.Content{
		{Type: mistral.ChunkImageURL, ImageURL: "data:image/png;base64,AAAA"},
		{Type: mistral.ChunkImageURL, ImageURL: "https://x/y.png"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("images = %+v, want %+v", got, want)
	}
}

func TestRequestTools(t *testing.T) {
	req := decodeRequest(t, `{
		"model":"claude-opus-5-5","max_tokens":64000,
		"messages":[{"role":"user","content":"hi"}],
		"tools":[{"name":"Read","description":"read","input_schema":{"type":"object"}},{"name":"Empty"},{"type":"advisor_20260301","name":"advisor"}],
		"tool_choice":{"type":"tool","name":"Read","disable_parallel_tool_use":true}
	}`)
	got := Request(req, testOptions, Target{Model: mistral.Model{ID: "big"}, CacheKey: "session"})

	if got.Model != "big" || got.MaxTokens != 1000 || got.PromptCacheKey != "session" {
		t.Errorf("model=%q max_tokens=%d cache=%q", got.Model, got.MaxTokens, got.PromptCacheKey)
	}
	if len(got.Tools) != 2 || string(got.Tools[1].Function.Parameters) != string(emptySchema) {
		t.Errorf("tools = %+v", got.Tools)
	}
	choice, ok := got.ToolChoice.(mistral.ToolChoiceFunction)
	if !ok || choice.Function.Name != "Read" {
		t.Errorf("tool_choice = %#v", got.ToolChoice)
	}
	if got.ParallelToolCalls == nil || *got.ParallelToolCalls {
		t.Errorf("parallel_tool_calls = %v, want false", got.ParallelToolCalls)
	}
}

func TestRequestWithoutToolsOmitsToolChoice(t *testing.T) {
	req := decodeRequest(t, `{"messages":[{"role":"user","content":"hi"}],"tool_choice":{"type":"any"}}`)
	got := Request(req, testOptions, Target{Model: mistral.Model{ID: "big"}})
	if got.ToolChoice != nil || got.ParallelToolCalls != nil {
		t.Errorf("tool_choice = %v, parallel = %v", got.ToolChoice, got.ParallelToolCalls)
	}
}

func TestConvertToolChoice(t *testing.T) {
	tests := map[string]any{
		anthropic.ToolChoiceAuto: mistral.ToolChoiceAuto,
		anthropic.ToolChoiceAny:  mistral.ToolChoiceAny,
		anthropic.ToolChoiceNone: mistral.ToolChoiceNone,
	}
	for in, want := range tests {
		got, parallel := convertToolChoice(&anthropic.ToolChoice{Type: in})
		if got != want || parallel != nil {
			t.Errorf("convertToolChoice(%q) = %v, %v", in, got, parallel)
		}
	}
}
