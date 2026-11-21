package translate

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"a2m/internal/anthropic"
	"a2m/internal/mistral"
)

type recorder struct {
	events []string
	data   []any
}

func (r *recorder) Emit(event string, data any) error {
	r.events = append(r.events, event)
	r.data = append(r.data, data)
	return nil
}

func (r *recorder) messageDelta(t *testing.T) anthropic.MessageDelta {
	t.Helper()
	for _, d := range r.data {
		if md, ok := d.(anthropic.MessageDelta); ok {
			return md
		}
	}
	t.Fatalf("no message_delta event")
	return anthropic.MessageDelta{}
}

func streamChunk(t *testing.T, raw string) *mistral.StreamChunk {
	t.Helper()
	var c mistral.StreamChunk
	if err := json.Unmarshal([]byte(raw), &c); err != nil {
		t.Fatalf("decode chunk: %v", err)
	}
	return &c
}

func translateChunks(t *testing.T, chunks ...string) *recorder {
	t.Helper()
	rec := &recorder{}
	tr := NewTranslator(rec, "claude-opus-5-5")
	if err := tr.Start(); err != nil {
		t.Fatalf("start: %v", err)
	}
	for _, raw := range chunks {
		if err := tr.Chunk(streamChunk(t, raw)); err != nil {
			t.Fatalf("chunk: %v", err)
		}
	}
	if err := tr.Finish(); err != nil {
		t.Fatalf("finish: %v", err)
	}
	return rec
}

func TestTranslatorEventOrder(t *testing.T) {
	rec := translateChunks(t,
		`{"choices":[{"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"hm"}],"signature":"s1"}]}}]}`,
		`{"choices":[{"delta":{"content":"Hello"}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"id":"abcDEF123","function":{"name":"Read","arguments":"{\"a\":"},"index":0}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"id":"null","function":{"name":"","arguments":"1}"},"index":0}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"id":"xyzXYZ789","function":{"name":"Bash","arguments":{"cmd":"ls"}},"index":1}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":100,"completion_tokens":7,"prompt_tokens_details":{"cached_tokens":60}}}`,
	)

	want := []string{
		anthropic.EventMessageStart, anthropic.EventPing,
		anthropic.EventContentBlockStart, anthropic.EventContentBlockDelta, anthropic.EventContentBlockDelta, anthropic.EventContentBlockStop,
		anthropic.EventContentBlockStart, anthropic.EventContentBlockDelta, anthropic.EventContentBlockStop,
		anthropic.EventContentBlockStart, anthropic.EventContentBlockDelta, anthropic.EventContentBlockDelta, anthropic.EventContentBlockStop,
		anthropic.EventContentBlockStart, anthropic.EventContentBlockDelta, anthropic.EventContentBlockStop,
		anthropic.EventMessageDelta, anthropic.EventMessageStop,
	}
	if !reflect.DeepEqual(rec.events, want) {
		t.Fatalf("events = %v\nwant %v", rec.events, want)
	}

	signature := rec.data[4].(anthropic.ContentBlockDelta).Delta.(anthropic.SignatureDelta)
	if signature.Signature != signaturePrefix+"s1" {
		t.Errorf("signature = %q", signature.Signature)
	}
	objectArgs := rec.data[14].(anthropic.ContentBlockDelta).Delta.(anthropic.InputJSONDelta)
	if objectArgs.PartialJSON != `{"cmd":"ls"}` {
		t.Errorf("object arguments = %q", objectArgs.PartialJSON)
	}

	md := rec.messageDelta(t)
	wantUsage := anthropic.Usage{InputTokens: 40, OutputTokens: 7, CacheReadInputTokens: 60}
	if md.Delta.StopReason != anthropic.StopToolUse || md.Usage != wantUsage {
		t.Errorf("message_delta = %+v", md)
	}
}

func TestStopReason(t *testing.T) {
	tests := map[string]string{
		"stop":         anthropic.StopEndTurn,
		"length":       anthropic.StopMaxTokens,
		"model_length": anthropic.StopMaxTokens,
		"tool_calls":   anthropic.StopToolUse,
	}
	for finish, want := range tests {
		rec := translateChunks(t, `{"choices":[{"delta":{"content":"x"},"finish_reason":"`+finish+`"}]}`)
		if got := rec.messageDelta(t).Delta.StopReason; got != want {
			t.Errorf("finish %q: stop_reason = %q, want %q", finish, got, want)
		}
	}
}

func TestFinishReasonError(t *testing.T) {
	tr := NewTranslator(&recorder{}, "m")
	err := tr.Chunk(streamChunk(t, `{"choices":[{"delta":{"content":"x"},"finish_reason":"error"}]}`))
	if !errors.Is(err, ErrGeneration) {
		t.Errorf("err = %v, want ErrGeneration", err)
	}
}

func TestWebSearch(t *testing.T) {
	results := `{"a":{"url":"https://go.dev/dl/","title":"Downloads","rank":1},"b":{"url":"https://blocked.example/x","title":"Blocked","rank":0},"c":{"url":"https://www.go.dev/doc","title":"Docs","rank":2}}`
	resultJSON, _ := json.Marshal(results)
	raw := `{"outputs":[
		{"type":"tool.execution","name":"web_search","arguments":"{\"query\": \"go\"}","info":{"result":` + string(resultJSON) + `}},
		{"type":"message.output","content":[{"type":"text","text":"Go 1.27"},{"type":"tool_reference","tool":"web_search","title":"t","url":"u"},{"type":"text","text":" is out."}]}
	],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":115}}`
	var resp mistral.ConversationResponse
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}

	rec := &recorder{}
	tr := NewTranslator(rec, "m")
	tool := anthropic.Tool{BlockedDomains: []string{"blocked.example"}}
	if err := tr.Start(); err != nil {
		t.Fatal(err)
	}
	if err := tr.WebSearch(&resp, tool); err != nil {
		t.Fatal(err)
	}
	if err := tr.Finish(); err != nil {
		t.Fatal(err)
	}

	use := rec.data[2].(anthropic.ContentBlockStart).ContentBlock.(anthropic.ToolUseBlock)
	if use.Type != anthropic.BlockServerToolUse || !strings.HasPrefix(use.ID, serverToolIDPrefix) {
		t.Errorf("server tool use = %+v", use)
	}
	result := rec.data[5].(anthropic.ContentBlockStart).ContentBlock.(anthropic.WebSearchToolResultBlock)
	if result.ToolUseID != use.ID || len(result.Content) != 2 || result.Content[0].URL != "https://go.dev/dl/" || result.Content[1].URL != "https://www.go.dev/doc" {
		t.Errorf("search result = %+v", result)
	}
	md := rec.messageDelta(t)
	if md.Usage.InputTokens != 110 || md.Usage.OutputTokens != 5 {
		t.Errorf("usage = %+v", md.Usage)
	}
}

func TestWebSearchRequest(t *testing.T) {
	req := decodeRequest(t, `{
		"system":[{"type":"text","text":"Search once."}],
		"messages":[{"role":"user","content":[{"type":"text","text":"Perform a web search for the query: go"}]}],
		"tools":[{"type":"web_search_20250305","name":"web_search","allowed_domains":["go.dev"]}]
	}`)
	tool, ok := WebSearchTool(req.Tools)
	if !ok {
		t.Fatal("web search tool not found")
	}
	got := WebSearchRequest(req, "big", tool)
	if got.Model != "big" || got.Inputs != "Perform a web search for the query: go" || got.Store {
		t.Errorf("request = %+v", got)
	}
	if !strings.Contains(got.Instructions, "Search once.") || !strings.Contains(got.Instructions, allowedDomainsRule+"go.dev") {
		t.Errorf("instructions = %q", got.Instructions)
	}
	if len(got.Tools) != 1 || got.Tools[0].Type != mistral.ToolWebSearch {
		t.Errorf("tools = %+v", got.Tools)
	}
}

func TestDomainAllowed(t *testing.T) {
	allowOnly := anthropic.Tool{AllowedDomains: []string{"go.dev"}}
	tests := []struct {
		url  string
		tool anthropic.Tool
		want bool
	}{
		{"https://go.dev/x", allowOnly, true},
		{"https://pkg.go.dev/x", allowOnly, true},
		{"https://notgo.dev/x", allowOnly, false},
		{"https://example.com", anthropic.Tool{}, true},
		{"https://ads.example.com", anthropic.Tool{BlockedDomains: []string{"www.example.com"}}, true},
		{"https://www.example.com", anthropic.Tool{BlockedDomains: []string{"example.com"}}, false},
	}
	for _, tt := range tests {
		if got := domainAllowed(tt.url, tt.tool); got != tt.want {
			t.Errorf("domainAllowed(%q) = %v, want %v", tt.url, got, tt.want)
		}
	}
}
