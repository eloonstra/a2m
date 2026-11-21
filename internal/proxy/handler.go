package proxy

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"

	"a2m/internal/anthropic"
	"a2m/internal/mistral"
	"a2m/internal/translate"
)

const (
	maxRequestBody = 64 << 20
	bearerPrefix   = "Bearer "
	headerAPIKey   = "X-Api-Key"
	headerAuth     = "Authorization"
	headerRetry    = "Retry-After"
	headerSession  = "X-Claude-Code-Session-Id"
	headerAgent    = "X-Claude-Code-Agent-Id"
	cacheKeySep    = ":"
	contentTypeSSE = "text/event-stream"
	contentJSON    = "application/json"

	errInvalidRequest = "invalid_request_error"
	errAuthentication = "authentication_error"
	errPermission     = "permission_error"
	errNotFound       = "not_found_error"
	errTooLarge       = "request_too_large"
	errRateLimit      = "rate_limit_error"
	errOverloaded     = "overloaded_error"
	errAPI            = "api_error"

	statusOverloaded = 529

	msgInvalidJSON  = "invalid request body: "
	msgNoMessages   = "messages: field required"
	msgUnauthorized = "invalid proxy token"
	msgNoAPIKey     = "missing Mistral API key: set MISTRAL_API_KEY on the proxy or ANTHROPIC_AUTH_TOKEN in Claude Code"

	promptTooLongFormat = "prompt is too long: %d tokens > %d maximum"
)

type Config struct {
	APIKey    string
	Token     string
	Translate translate.Options
}

type Handler struct {
	client  *mistral.Client
	catalog *mistral.Catalog
	cfg     Config
}

func NewHandler(client *mistral.Client, catalog *mistral.Catalog, cfg Config) *Handler {
	return &Handler{client: client, catalog: catalog, cfg: cfg}
}

func (h *Handler) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/messages", h.messages)
	return mux
}

type producer func(t *translate.Translator) error

func (h *Handler) messages(w http.ResponseWriter, r *http.Request) {
	var req anthropic.MessagesRequest
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxRequestBody)).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, msgInvalidJSON+err.Error(), "")
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, msgNoMessages, "")
		return
	}
	if !h.authorized(r) {
		writeError(w, http.StatusUnauthorized, msgUnauthorized, "")
		return
	}
	key := h.resolveKey(r)
	if key == "" {
		writeError(w, http.StatusUnauthorized, msgNoAPIKey, "")
		return
	}

	if tool, ok := translate.WebSearchTool(req.Tools); ok {
		h.webSearch(r.Context(), w, key, &req, tool)
		return
	}
	h.chat(w, r, key, &req, translate.MapModel(req.Model, h.cfg.Translate))
}

func (h *Handler) chat(w http.ResponseWriter, r *http.Request, key string, req *anthropic.MessagesRequest, model string) {
	target := translate.Target{Model: h.modelInfo(r.Context(), key, model), CacheKey: cacheKey(r)}
	stream, err := h.client.Stream(r.Context(), key, translate.Request(req, h.cfg.Translate, target))
	if rejectsReasoningEffort(err) {
		target.Model.Capabilities.Reasoning = false
		stream, err = h.client.Stream(r.Context(), key, translate.Request(req, h.cfg.Translate, target))
	}
	if err != nil {
		writeUpstreamError(w, err)
		return
	}
	defer stream.Close()
	respond(w, req, func(t *translate.Translator) error { return pump(stream, t) })
}

func rejectsReasoningEffort(err error) bool {
	var apiErr *mistral.APIError
	return errors.As(err, &apiErr) && apiErr.RejectsReasoningEffort()
}

func (h *Handler) webSearch(ctx context.Context, w http.ResponseWriter, key string, req *anthropic.MessagesRequest, tool anthropic.Tool) {
	resp, err := h.client.Converse(ctx, key, translate.WebSearchRequest(req, h.cfg.Translate.SmallModel, tool))
	if err != nil {
		writeUpstreamError(w, err)
		return
	}
	respond(w, req, func(t *translate.Translator) error { return t.WebSearch(resp, tool) })
}

func (h *Handler) modelInfo(ctx context.Context, key, id string) mistral.Model {
	info, ok := h.catalog.Lookup(ctx, key, id)
	if !ok {
		return mistral.Model{ID: id}
	}
	return info
}

func (h *Handler) authorized(r *http.Request) bool {
	if h.cfg.Token == "" {
		return true
	}
	given := clientCredential(r)
	return subtle.ConstantTimeCompare([]byte(given), []byte(h.cfg.Token)) == 1
}

func (h *Handler) resolveKey(r *http.Request) string {
	if h.cfg.APIKey != "" {
		return h.cfg.APIKey
	}
	if h.cfg.Token != "" {
		return ""
	}
	return clientCredential(r)
}

func clientCredential(r *http.Request) string {
	if key := r.Header.Get(headerAPIKey); key != "" {
		return key
	}
	return strings.TrimPrefix(r.Header.Get(headerAuth), bearerPrefix)
}

func cacheKey(r *http.Request) string {
	session := r.Header.Get(headerSession)
	agent := r.Header.Get(headerAgent)
	if session == "" || agent == "" {
		return session
	}
	return session + cacheKeySep + agent
}

func respond(w http.ResponseWriter, req *anthropic.MessagesRequest, produce producer) {
	if req.Stream {
		respondSSE(w, req.Model, produce)
		return
	}
	respondJSON(w, req.Model, produce)
}

func respondSSE(w http.ResponseWriter, model string, produce producer) {
	w.Header().Set("Content-Type", contentTypeSSE)
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	sse := newSSEWriter(w)
	err := run(translate.NewTranslator(sse, model), produce)
	if err == nil {
		return
	}
	log.Printf("stream error: %v", err)
	status, message, _ := upstreamError(err)
	_ = sse.Emit(anthropic.EventError, errorBody(errorType(status), message))
}

func respondJSON(w http.ResponseWriter, model string, produce producer) {
	c := &collector{}
	if err := run(translate.NewTranslator(c, model), produce); err != nil {
		writeUpstreamError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, c.result())
}

func run(t *translate.Translator, produce producer) error {
	if err := t.Start(); err != nil {
		return err
	}
	if err := produce(t); err != nil {
		return err
	}
	return t.Finish()
}

func pump(stream *mistral.Stream, t *translate.Translator) error {
	for {
		chunk, err := stream.Next()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		if err := t.Chunk(chunk); err != nil {
			return err
		}
	}
}

func writeUpstreamError(w http.ResponseWriter, err error) {
	log.Printf("upstream error: %v", err)
	status, message, retryAfter := upstreamError(err)
	writeError(w, status, message, retryAfter)
}

func upstreamError(err error) (int, string, string) {
	var apiErr *mistral.APIError
	if !errors.As(err, &apiErr) {
		return http.StatusBadGateway, err.Error(), ""
	}
	actual, limit, tooLong := apiErr.PromptTooLong()
	if tooLong {
		return http.StatusBadRequest, fmt.Sprintf(promptTooLongFormat, actual, limit), ""
	}
	return apiErr.Status, apiErr.Body, apiErr.RetryAfter
}

func writeError(w http.ResponseWriter, status int, message, retryAfter string) {
	if retryAfter != "" {
		w.Header().Set(headerRetry, retryAfter)
	}
	writeJSON(w, status, errorBody(errorType(status), message))
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", contentJSON)
	w.WriteHeader(status)
	encoder := json.NewEncoder(w)
	encoder.SetEscapeHTML(false)
	_ = encoder.Encode(body)
}

func errorBody(errType, message string) anthropic.ErrorBody {
	return anthropic.ErrorBody{
		Type:  anthropic.EventError,
		Error: anthropic.ErrorDetail{Type: errType, Message: message},
	}
}

func errorType(status int) string {
	switch status {
	case http.StatusBadRequest, http.StatusUnprocessableEntity:
		return errInvalidRequest
	case http.StatusUnauthorized:
		return errAuthentication
	case http.StatusForbidden:
		return errPermission
	case http.StatusNotFound:
		return errNotFound
	case http.StatusRequestEntityTooLarge:
		return errTooLarge
	case http.StatusTooManyRequests:
		return errRateLimit
	case statusOverloaded, http.StatusServiceUnavailable:
		return errOverloaded
	}
	return errAPI
}
