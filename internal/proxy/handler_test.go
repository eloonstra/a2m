package proxy

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"a2m/internal/anthropic"
	"a2m/internal/mistral"
	"a2m/internal/translate"
)

const (
	testToken  = "proxy-token"
	testAPIKey = "mistral-key"
)

type fakeMistral struct {
	mu              sync.Mutex
	chat            mistral.ChatRequest
	conv            mistral.ConversationRequest
	auth            string
	chatBody        string
	status          int
	errBody         string
	chatCalls       int
	rejectReasoning bool
}

func (f *fakeMistral) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.auth = r.Header.Get("Authorization")
	switch r.URL.Path {
	case "/v1/models":
		fmt.Fprint(w, `{"data":[{"id":"mistral-small-latest","name":"mistral-small-2603","capabilities":{"reasoning":true}}]}`)
	case "/v1/conversations":
		_ = json.NewDecoder(r.Body).Decode(&f.conv)
		fmt.Fprint(w, `{"outputs":[{"type":"tool.execution","name":"web_search","arguments":"{}","info":{"result":"{\"x\":{\"url\":\"https://go.dev\",\"title\":\"Go\",\"rank\":0}}"}},{"type":"message.output","content":"Go 1.27"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":10}}`)
	case "/v1/chat/completions":
		var chat mistral.ChatRequest
		_ = json.NewDecoder(r.Body).Decode(&chat)
		f.chat = chat
		f.chatCalls++
		firstCall := f.chatCalls == 1
		if f.rejectReasoning && firstCall {
			w.WriteHeader(http.StatusBadRequest)
			fmt.Fprint(w, `{"message":"reasoning_effort is not supported"}`)
			return
		}
		if f.status != 0 {
			w.WriteHeader(f.status)
			fmt.Fprint(w, f.errBody)
			return
		}
		fmt.Fprint(w, f.chatBody)
	}
}

func newTestServer(t *testing.T, fake *fakeMistral, cfg Config) *httptest.Server {
	t.Helper()
	upstream := httptest.NewServer(fake)
	t.Cleanup(upstream.Close)
	client := mistral.NewClient(upstream.URL)
	cfg.Translate = translate.Options{Model: "mistral-large-latest", MediumModel: "mistral-medium-latest", SmallModel: "mistral-small-latest", MaxTokens: 1000}
	srv := httptest.NewServer(NewHandler(client, mistral.NewCatalog(client), cfg).Routes())
	t.Cleanup(srv.Close)
	return srv
}

func post(t *testing.T, srv *httptest.Server, token, body string, headers map[string]string) (*http.Response, string) {
	t.Helper()
	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/v1/messages?beta=true", strings.NewReader(body))
	req.Header.Set("x-api-key", token)
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	return resp, string(data)
}

const chatStream = "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":1}}\n\ndata: [DONE]\n\n"

func TestRejectsWrongToken(t *testing.T) {
	srv := newTestServer(t, &fakeMistral{chatBody: chatStream}, Config{APIKey: testAPIKey, Token: testToken})
	resp, body := post(t, srv, "wrong", `{"model":"claude-opus-5-5","messages":[{"role":"user","content":"hi"}]}`, nil)
	if resp.StatusCode != http.StatusUnauthorized || !strings.Contains(body, errAuthentication) {
		t.Errorf("status = %d body = %s", resp.StatusCode, body)
	}
}

func TestRejectsInvalidBodyBeforeAuth(t *testing.T) {
	srv := newTestServer(t, &fakeMistral{}, Config{APIKey: testAPIKey, Token: testToken})
	resp, body := post(t, srv, "wrong", `{bad`, nil)
	if resp.StatusCode != http.StatusBadRequest || !strings.Contains(body, errInvalidRequest) {
		t.Errorf("status = %d body = %s", resp.StatusCode, body)
	}
}

func TestTokenModeNeedsServerKey(t *testing.T) {
	srv := newTestServer(t, &fakeMistral{}, Config{Token: testToken})
	resp, body := post(t, srv, testToken, `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, nil)
	if resp.StatusCode != http.StatusUnauthorized || !strings.Contains(body, "MISTRAL_API_KEY") {
		t.Errorf("status = %d body = %s", resp.StatusCode, body)
	}
}

func TestClientKeyPassthroughWithoutToken(t *testing.T) {
	fake := &fakeMistral{chatBody: chatStream}
	srv := newTestServer(t, fake, Config{})
	resp, _ := post(t, srv, "client-key", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, nil)
	if resp.StatusCode != http.StatusOK || fake.auth != "Bearer client-key" {
		t.Errorf("status = %d auth = %q", resp.StatusCode, fake.auth)
	}
}

func TestStreamingChat(t *testing.T) {
	fake := &fakeMistral{chatBody: chatStream}
	srv := newTestServer(t, fake, Config{APIKey: testAPIKey, Token: testToken})
	body := `{"model":"claude-haiku-5-5","stream":true,"thinking":{"type":"adaptive"},"output_config":{"effort":"low"},"messages":[{"role":"user","content":"hi"}]}`
	resp, out := post(t, srv, testToken, body, map[string]string{headerSession: "s1", headerAgent: "a1"})

	if resp.StatusCode != http.StatusOK || resp.Header.Get("Content-Type") != contentTypeSSE {
		t.Fatalf("status = %d type = %q", resp.StatusCode, resp.Header.Get("Content-Type"))
	}
	for _, want := range []string{"event: message_start", `"text":"Hi"`, `"stop_reason":"end_turn"`, "event: message_stop"} {
		if !strings.Contains(out, want) {
			t.Errorf("stream misses %q:\n%s", want, out)
		}
	}
	if fake.auth != "Bearer "+testAPIKey {
		t.Errorf("upstream auth = %q", fake.auth)
	}
	if fake.chat.Model != "mistral-small-latest" || fake.chat.PromptCacheKey != "s1:a1" || fake.chat.ReasoningEffort != "high" {
		t.Errorf("upstream request model=%q cache=%q effort=%q", fake.chat.Model, fake.chat.PromptCacheKey, fake.chat.ReasoningEffort)
	}
}

func TestNonStreamingChat(t *testing.T) {
	srv := newTestServer(t, &fakeMistral{chatBody: chatStream}, Config{APIKey: testAPIKey})
	resp, out := post(t, srv, "", `{"model":"claude-opus-5-5","messages":[{"role":"user","content":"hi"}]}`, nil)
	var msg anthropic.MessageResponse
	if err := json.Unmarshal([]byte(out), &msg); err != nil {
		t.Fatalf("decode %s: %v", out, err)
	}
	if resp.StatusCode != http.StatusOK || len(msg.Content) != 1 || *msg.StopReason != anthropic.StopEndTurn || msg.Usage.InputTokens != 5 {
		t.Errorf("status = %d msg = %+v", resp.StatusCode, msg)
	}
}

func TestUpstreamErrorKeepsStatus(t *testing.T) {
	srv := newTestServer(t, &fakeMistral{status: http.StatusTooManyRequests, errBody: `{"message":"rejected"}`}, Config{APIKey: testAPIKey})
	resp, out := post(t, srv, "", `{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`, nil)
	if resp.StatusCode != http.StatusTooManyRequests || !strings.Contains(out, errRateLimit) || !strings.Contains(out, "rejected") {
		t.Errorf("status = %d body = %s", resp.StatusCode, out)
	}
}

func TestMidStreamErrorEvent(t *testing.T) {
	stream := "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\ndata: {\"object\":\"error\",\"message\":\"busy\",\"raw_status_code\":503}\n\n"
	srv := newTestServer(t, &fakeMistral{chatBody: stream}, Config{APIKey: testAPIKey})
	_, out := post(t, srv, "", `{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`, nil)
	if !strings.Contains(out, "event: error") || !strings.Contains(out, errOverloaded) || strings.Contains(out, "message_stop") {
		t.Errorf("stream = %s", out)
	}
}

func TestTruncatedStreamIsError(t *testing.T) {
	stream := "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n"
	srv := newTestServer(t, &fakeMistral{chatBody: stream}, Config{APIKey: testAPIKey})
	_, out := post(t, srv, "", `{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`, nil)
	if !strings.Contains(out, "event: error") || strings.Contains(out, "message_stop") {
		t.Errorf("stream = %s", out)
	}
}

func TestWebSearchRoute(t *testing.T) {
	fake := &fakeMistral{}
	srv := newTestServer(t, fake, Config{APIKey: testAPIKey})
	body := `{"model":"claude-opus-5-5","tools":[{"type":"web_search_20250305","name":"web_search"}],"messages":[{"role":"user","content":"Perform a web search for the query: go"}]}`
	_, out := post(t, srv, "", body, nil)
	var msg struct {
		Content []map[string]any `json:"content"`
	}
	if err := json.Unmarshal([]byte(out), &msg); err != nil {
		t.Fatalf("decode %s: %v", out, err)
	}
	var types []string
	for _, block := range msg.Content {
		types = append(types, block["type"].(string))
	}
	want := []string{anthropic.BlockServerToolUse, anthropic.BlockWebSearchToolResult, anthropic.BlockText}
	if strings.Join(types, ",") != strings.Join(want, ",") {
		t.Errorf("block types = %v, want %v\n%s", types, want, out)
	}
	if fake.conv.Model != "mistral-small-latest" {
		t.Errorf("web search model = %q, want the small model", fake.conv.Model)
	}
}

func TestReasoningEffortRejectionRetriesWithoutIt(t *testing.T) {
	fake := &fakeMistral{chatBody: chatStream, rejectReasoning: true}
	srv := newTestServer(t, fake, Config{APIKey: testAPIKey})
	body := `{"model":"claude-haiku-5-5","stream":true,"thinking":{"type":"adaptive"},"messages":[{"role":"user","content":"hi"}]}`
	resp, out := post(t, srv, "", body, nil)
	if resp.StatusCode != http.StatusOK || !strings.Contains(out, `"text":"Hi"`) {
		t.Errorf("status = %d body = %s", resp.StatusCode, out)
	}
	if fake.chatCalls != 2 || fake.chat.ReasoningEffort != "" {
		t.Errorf("chat calls = %d, last reasoning_effort = %q, want 2 calls and empty effort", fake.chatCalls, fake.chat.ReasoningEffort)
	}
}

func TestCacheKey(t *testing.T) {
	tests := []struct{ session, agent, want string }{
		{"", "", ""},
		{"s", "", "s"},
		{"s", "a", "s:a"},
		{"", "a", ""},
	}
	for _, tt := range tests {
		r := httptest.NewRequest(http.MethodPost, "/", nil)
		r.Header.Set(headerSession, tt.session)
		r.Header.Set(headerAgent, tt.agent)
		if got := cacheKey(r); got != tt.want {
			t.Errorf("cacheKey(%q, %q) = %q, want %q", tt.session, tt.agent, got, tt.want)
		}
	}
}

func TestPromptTooLongUsesClaudeWording(t *testing.T) {
	errBody := `{"object":"error","message":"Prompt 544383 > 131072 maximum context length","type":"invalid_request_prompt_too_long","code":"3059","raw_status_code":400}`
	srv := newTestServer(t, &fakeMistral{status: http.StatusBadRequest, errBody: errBody}, Config{APIKey: testAPIKey})
	resp, out := post(t, srv, "", `{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`, nil)
	var body anthropic.ErrorBody
	if err := json.Unmarshal([]byte(out), &body); err != nil {
		t.Fatalf("decode %s: %v", out, err)
	}
	want := "prompt is too long: 544383 tokens > 131072 maximum"
	if resp.StatusCode != http.StatusBadRequest || body.Error.Type != errInvalidRequest || body.Error.Message != want {
		t.Errorf("status = %d body = %+v", resp.StatusCode, body)
	}
}
