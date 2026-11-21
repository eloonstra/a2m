package mistral

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"
)

const (
	chatPath          = "/v1/chat/completions"
	modelsPath        = "/v1/models"
	conversationsPath = "/v1/conversations"
	contentTypeJSON   = "application/json"
	acceptSSE         = "text/event-stream"
	bearerPrefix      = "Bearer "
	headerRetryAfter  = "Retry-After"
	maxErrorBody      = 1 << 20

	responseHeaderTimeout = 30 * time.Second

	errTypePromptTooLong    = "invalid_request_prompt_too_long"
	errFieldReasoningEffort = "reasoning_effort"
)

var (
	dataPrefix        = []byte("data:")
	doneMarker        = []byte("[DONE]")
	tokenLimitPattern = regexp.MustCompile(`(\d+)\s*>\s*(\d+)`)
)

type APIError struct {
	Status     int
	Body       string
	RetryAfter string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("mistral: status %d: %s", e.Status, e.Body)
}

func (e *APIError) PromptTooLong() (actual, limit int, ok bool) {
	var body struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	}
	if err := json.Unmarshal([]byte(e.Body), &body); err != nil || body.Type != errTypePromptTooLong {
		return 0, 0, false
	}
	match := tokenLimitPattern.FindStringSubmatch(body.Message)
	if match == nil {
		return 0, 0, false
	}
	actual, errActual := strconv.Atoi(match[1])
	limit, errLimit := strconv.Atoi(match[2])
	if errActual != nil || errLimit != nil {
		return 0, 0, false
	}
	return actual, limit, true
}

func (e *APIError) RejectsReasoningEffort() bool {
	isBadRequest := e.Status == http.StatusBadRequest
	return isBadRequest && strings.Contains(e.Body, errFieldReasoningEffort)
}

type Client struct {
	baseURL string
	http    *http.Client
}

func NewClient(baseURL string) *Client {
	return &Client{baseURL: baseURL, http: newHTTPClient(responseHeaderTimeout)}
}

func newHTTPClient(responseTimeout time.Duration) *http.Client {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.ResponseHeaderTimeout = responseTimeout
	return &http.Client{Transport: transport}
}

func (c *Client) Stream(ctx context.Context, apiKey string, req *ChatRequest) (*Stream, error) {
	req.Stream = true
	resp, err := c.postJSON(ctx, chatPath, apiKey, acceptSSE, req)
	if err != nil {
		return nil, err
	}
	return &Stream{body: resp.Body, reader: bufio.NewReader(resp.Body)}, nil
}

func (c *Client) Converse(ctx context.Context, apiKey string, req *ConversationRequest) (*ConversationResponse, error) {
	resp, err := c.postJSON(ctx, conversationsPath, apiKey, contentTypeJSON, req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var out ConversationResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *Client) ListModels(ctx context.Context, apiKey string) ([]Model, error) {
	resp, err := c.do(ctx, http.MethodGet, modelsPath, apiKey, contentTypeJSON, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var list ModelList
	if err := json.NewDecoder(resp.Body).Decode(&list); err != nil {
		return nil, err
	}
	return list.Data, nil
}

func (c *Client) postJSON(ctx context.Context, path, apiKey, accept string, payload any) (*http.Response, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	return c.do(ctx, http.MethodPost, path, apiKey, accept, bytes.NewReader(body))
}

func (c *Client) do(ctx context.Context, method, path, apiKey, accept string, body io.Reader) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, body)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", contentTypeJSON)
	}
	req.Header.Set("Accept", accept)
	req.Header.Set("Authorization", bearerPrefix+apiKey)

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode == http.StatusOK {
		return resp, nil
	}
	defer resp.Body.Close()
	errBody, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBody))
	return nil, &APIError{
		Status:     resp.StatusCode,
		Body:       string(bytes.TrimSpace(errBody)),
		RetryAfter: resp.Header.Get(headerRetryAfter),
	}
}

type Stream struct {
	body   io.Closer
	reader *bufio.Reader
}

func (s *Stream) Next() (*StreamChunk, error) {
	for {
		line, readErr := s.reader.ReadBytes('\n')
		chunk, err := parseLine(line)
		if err != nil || chunk != nil {
			return chunk, err
		}
		if readErr == io.EOF {
			return nil, io.ErrUnexpectedEOF
		}
		if readErr != nil {
			return nil, readErr
		}
	}
}

func (s *Stream) Close() error {
	return s.body.Close()
}

func parseLine(line []byte) (*StreamChunk, error) {
	line = bytes.TrimSpace(line)
	if !bytes.HasPrefix(line, dataPrefix) {
		return nil, nil
	}
	payload := bytes.TrimSpace(line[len(dataPrefix):])
	if bytes.Equal(payload, doneMarker) {
		return nil, io.EOF
	}
	var chunk StreamChunk
	if err := json.Unmarshal(payload, &chunk); err != nil {
		return nil, err
	}
	if chunk.Object == ObjectError {
		return nil, &APIError{Status: streamErrorStatus(chunk.RawStatusCode), Body: string(payload)}
	}
	return &chunk, nil
}

func streamErrorStatus(raw int) int {
	if raw == 0 {
		return http.StatusInternalServerError
	}
	return raw
}
