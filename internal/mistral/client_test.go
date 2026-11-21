package mistral

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

const testKey = "key"

func sseServer(t *testing.T, body string) *Client {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != bearerPrefix+testKey {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		fmt.Fprint(w, body)
	}))
	t.Cleanup(srv.Close)
	return NewClient(srv.URL)
}

func readAll(t *testing.T, c *Client) ([]*StreamChunk, error) {
	t.Helper()
	stream, err := c.Stream(context.Background(), testKey, &ChatRequest{Model: "m"})
	if err != nil {
		return nil, err
	}
	defer stream.Close()
	var chunks []*StreamChunk
	for {
		chunk, err := stream.Next()
		if err != nil {
			return chunks, err
		}
		chunks = append(chunks, chunk)
	}
}

func TestStreamReadsUntilDone(t *testing.T) {
	c := sseServer(t, ": comment\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\ndata:{\"choices\":[]}\n\ndata: [DONE]\n\n")
	chunks, err := readAll(t, c)
	if !errors.Is(err, io.EOF) {
		t.Fatalf("err = %v, want io.EOF", err)
	}
	if len(chunks) != 2 || chunks[0].Choices[0].Delta.Content[0].Text != "a" {
		t.Errorf("chunks = %+v", chunks)
	}
}

func TestStreamWithoutDoneIsUnexpectedEOF(t *testing.T) {
	c := sseServer(t, "data: {\"choices\":[]}\n\n")
	_, err := readAll(t, c)
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Errorf("err = %v, want io.ErrUnexpectedEOF", err)
	}
}

func TestStreamErrorObject(t *testing.T) {
	c := sseServer(t, "data: {\"object\":\"error\",\"message\":\"overloaded\",\"raw_status_code\":503}\n\n")
	_, err := readAll(t, c)
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.Status != http.StatusServiceUnavailable {
		t.Errorf("err = %v, want APIError 503", err)
	}
}

func TestStreamErrorObjectWithoutStatus(t *testing.T) {
	c := sseServer(t, "data: {\"object\":\"error\",\"message\":\"boom\"}\n\n")
	_, err := readAll(t, c)
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.Status != http.StatusInternalServerError {
		t.Errorf("err = %v, want APIError 500", err)
	}
}

func TestAPIErrorKeepsStatusAndRetryAfter(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set(headerRetryAfter, "7")
		w.WriteHeader(http.StatusTooManyRequests)
		fmt.Fprint(w, ` {"message":"slow down"} `)
	}))
	defer srv.Close()
	_, err := NewClient(srv.URL).Stream(context.Background(), testKey, &ChatRequest{})
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("err = %v, want APIError", err)
	}
	if apiErr.Status != http.StatusTooManyRequests || apiErr.RetryAfter != "7" || apiErr.Body != `{"message":"slow down"}` {
		t.Errorf("apiErr = %+v", apiErr)
	}
}

func TestConverse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != conversationsPath {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		fmt.Fprint(w, `{"outputs":[{"type":"message.output","content":"hi"}],"usage":{"prompt_tokens":3}}`)
	}))
	defer srv.Close()
	resp, err := NewClient(srv.URL).Converse(context.Background(), testKey, &ConversationRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Outputs[0].Content[0].Text != "hi" || resp.Usage.PromptTokens != 3 {
		t.Errorf("resp = %+v", resp)
	}
}

func TestCatalogCachesModels(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		fmt.Fprint(w, `{"data":[{"id":"mistral-small-latest","name":"mistral-small-2603","capabilities":{"reasoning":true}}]}`)
	}))
	defer srv.Close()
	catalog := NewCatalog(NewClient(srv.URL))

	m, ok := catalog.Lookup(context.Background(), testKey, "mistral-small-latest")
	if !ok || m.Name != "mistral-small-2603" || !m.Capabilities.Reasoning {
		t.Errorf("lookup = %+v, %v", m, ok)
	}
	if _, ok := catalog.Lookup(context.Background(), testKey, "missing"); ok {
		t.Errorf("missing model found")
	}
	if calls.Load() != 1 {
		t.Errorf("model list fetched %d times, want 1", calls.Load())
	}
}

func TestCatalogRetriesAfterFailure(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			w.WriteHeader(http.StatusBadGateway)
			return
		}
		fmt.Fprint(w, `{"data":[{"id":"a"}]}`)
	}))
	defer srv.Close()
	catalog := NewCatalog(NewClient(srv.URL))
	if _, ok := catalog.Lookup(context.Background(), testKey, "a"); ok {
		t.Errorf("lookup succeeded on failed fetch")
	}
	if _, ok := catalog.Lookup(context.Background(), testKey, "a"); !ok {
		t.Errorf("lookup failed after recovery")
	}
}

func TestCatalogExpiresCache(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		fmt.Fprint(w, `{"data":[{"id":"a"}]}`)
	}))
	defer srv.Close()
	catalog := NewCatalog(NewClient(srv.URL))
	catalog.ttl = 0
	if _, ok := catalog.Lookup(context.Background(), testKey, "a"); !ok {
		t.Fatal("first lookup failed")
	}
	if _, ok := catalog.Lookup(context.Background(), testKey, "a"); !ok {
		t.Fatal("second lookup failed")
	}
	if calls.Load() != 2 {
		t.Errorf("model list fetched %d times, want 2 after cache expiry", calls.Load())
	}
}

func TestResponseHeaderTimeout(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(300 * time.Millisecond)
		fmt.Fprint(w, `{"data":[]}`)
	}))
	defer srv.Close()
	c := &Client{baseURL: srv.URL, http: newHTTPClient(20 * time.Millisecond)}
	if _, err := c.ListModels(context.Background(), testKey); err == nil {
		t.Error("expected a timeout error, got nil")
	}
}

func TestRejectsReasoningEffort(t *testing.T) {
	tests := []struct {
		status int
		body   string
		want   bool
	}{
		{http.StatusBadRequest, `{"message":"reasoning_effort is invalid"}`, true},
		{http.StatusBadRequest, `{"message":"unknown model"}`, false},
		{http.StatusTooManyRequests, `{"message":"reasoning_effort"}`, false},
	}
	for _, tt := range tests {
		got := (&APIError{Status: tt.status, Body: tt.body}).RejectsReasoningEffort()
		if got != tt.want {
			t.Errorf("RejectsReasoningEffort(%d, %s) = %v, want %v", tt.status, tt.body, got, tt.want)
		}
	}
}

func TestPromptTooLong(t *testing.T) {
	tests := []struct {
		body          string
		actual, limit int
		ok            bool
	}{
		{`{"object":"error","message":"Prompt 544383 > 131072 maximum context length","type":"invalid_request_prompt_too_long","code":"3059"}`, 544383, 131072, true},
		{`{"message":"Prompt 5 > 3 maximum context length","type":"invalid_request_invalid_args"}`, 0, 0, false},
		{`{"message":"too long","type":"invalid_request_prompt_too_long"}`, 0, 0, false},
		{`not json`, 0, 0, false},
	}
	for _, tt := range tests {
		actual, limit, ok := (&APIError{Body: tt.body}).PromptTooLong()
		if actual != tt.actual || limit != tt.limit || ok != tt.ok {
			t.Errorf("PromptTooLong(%s) = %d, %d, %v", tt.body, actual, limit, ok)
		}
	}
}
