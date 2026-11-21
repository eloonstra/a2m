package translate

import (
	"encoding/json"
	"net/url"
	"slices"
	"strings"
	"testing"

	"a2m/internal/anthropic"
	"a2m/internal/mistral"
)

const (
	fuzzTargetModel = "mistral-large-test"
	fuzzCacheKey    = "session"
	extraDomain     = "extra.invalid"
	minEventCount   = 4

	seedModelAuto  = "Claude-AUTO-5"
	seedModelPlain = "codestral-latest"
	seedModelEmpty = ``

	seedToolID   = "toolu_01ABCdef"
	seedNativeID = "abcDEF123"

	seedURLGo       = "https://go.dev/x"
	seedURLNotGo    = "https://notgo.dev/x"
	seedURLEmptyDom = "https://example.com"

	seedSearchJSON  = `{"a":{"url":"https://go.dev","title":"Go","rank":1},"b":{"url":"","title":"empty","rank":0}}`
	seedSearchStr   = `"{\"a\":{\"url\":\"https://go.dev\",\"rank\":0}}"`
	seedSearchBad   = `{broken`
	seedSearchEmpty = ``

	seedArgsObject = `{"cmd":"ls"}`
	seedArgsString = `"{\"a\":1}"`
	seedArgsBroken = `"{\"a\":`
	seedArgsEmpty  = ``

	seedReqMinimal    = `{"model":"claude-opus-5-5","messages":[{"role":"user","content":"hi"}]}`
	seedReqBlocks     = `{"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Read","input":{"p":1}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"ok"}]}]}`
	seedReqBadJSON    = `{bad`
	seedReqBadContent = `{"messages":[{"role":"user","content":123}]}`
	seedReqEmpty      = ``

	seedModelLarge  = "mistral-large-4-2603"
	seedModelNew    = "future-model"
	seedEffortMed   = "medium"
	seedEffortWeird = "weird"
	seedNameEmpty   = ``
	seedEffortEmpty = ``

	seedChunkText  = `{"choices":[{"delta":{"content":"Hi"}}]}`
	seedChunkThink = `{"choices":[{"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"hm"}],"signature":"s1"}]}}]}`
	seedChunkTool  = `{"choices":[{"delta":{"tool_calls":[{"id":"abcDEF123","function":{"name":"Read","arguments":"{\"a\":1}"},"index":0}]},"finish_reason":"tool_calls"],"usage":{"prompt_tokens":10,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":4}}}`
	seedChunkBad   = `{bad`
	seedChunkEmpty = ``
)

func FuzzMapModel(f *testing.F) {
	f.Add(seedModelAuto)
	f.Add(seedModelPlain)
	f.Add(seedModelEmpty)
	f.Fuzz(func(t *testing.T, model string) {
		got := MapModel(model, testOptions)
		mapped := []string{testOptions.Model, testOptions.MediumModel, testOptions.SmallModel, testOptions.AutoModel, model}
		if !slices.Contains(mapped, got) {
			t.Fatalf("MapModel(%q) = %q, not an option and not the input", model, got)
		}
		lowered := MapModel(strings.ToLower(model), testOptions)
		stable := got == lowered || got == model
		if !stable {
			t.Fatalf("MapModel(%q) = %q but MapModel(lowercase) = %q", model, got, lowered)
		}
	})
}

func FuzzToolCallID(f *testing.F) {
	f.Add(seedToolID)
	f.Add(seedNativeID)
	f.Add(seedModelEmpty)
	f.Fuzz(func(t *testing.T, id string) {
		got := toolCallID(id)
		if !isMistralToolCallID(got) {
			t.Fatalf("toolCallID(%q) = %q is not a Mistral ID", id, got)
		}
		if toolCallID(got) != got {
			t.Fatalf("toolCallID(%q) = %q is not stable under a second call", id, got)
		}
		nativeKept := !isMistralToolCallID(id) || got == id
		if !nativeKept {
			t.Fatalf("toolCallID(%q) changed a native ID to %q", id, got)
		}
	})
}

func FuzzDomainAllowed(f *testing.F) {
	f.Add(seedURLGo, "go.dev")
	f.Add(seedURLNotGo, "go.dev")
	f.Add(seedURLEmptyDom, seedModelEmpty)
	f.Fuzz(func(t *testing.T, rawURL, domain string) {
		tool := anthropic.Tool{AllowedDomains: []string{domain}}
		got := domainAllowed(rawURL, tool)
		lowerTool := anthropic.Tool{AllowedDomains: []string{strings.ToLower(domain)}}
		if got != domainAllowed(rawURL, lowerTool) {
			t.Fatalf("domainAllowed(%q, %q) changed when the domain was lowercased", rawURL, domain)
		}
		_, parseErr := url.Parse(rawURL)
		parseable := parseErr == nil
		if domainAllowed(rawURL, anthropic.Tool{}) != parseable {
			t.Fatalf("domainAllowed(%q, empty tool) disagrees with url.Parse error %v", rawURL, parseErr)
		}
		wider := anthropic.Tool{AllowedDomains: []string{domain, extraDomain}}
		if got && !domainAllowed(rawURL, wider) {
			t.Fatalf("domainAllowed(%q, %q) = true but false with an extra allowed domain", rawURL, domain)
		}
	})
}

func FuzzArgumentsText(f *testing.F) {
	f.Add(seedArgsObject)
	f.Add(seedArgsString)
	f.Add(seedArgsBroken)
	f.Add(seedArgsEmpty)
	f.Fuzz(func(t *testing.T, raw string) {
		got := argumentsText(json.RawMessage(raw))
		hasBytes := len(raw) > 0
		isString := hasBytes && raw[0] == '"'
		if !isString {
			if got != raw {
				t.Fatalf("argumentsText(%q) = %q, want the raw value", raw, got)
			}
			return
		}
		var s string
		if err := json.Unmarshal([]byte(raw), &s); err != nil {
			if got != "" {
				t.Fatalf("argumentsText(%q) = %q, want empty for a broken string", raw, got)
			}
			return
		}
		if got != s {
			t.Fatalf("argumentsText(%q) = %q, want %q", raw, got, s)
		}
	})
}

func FuzzDecodeSearchResults(f *testing.F) {
	f.Add(seedSearchJSON)
	f.Add(seedSearchStr)
	f.Add(seedSearchBad)
	f.Add(seedSearchEmpty)
	f.Fuzz(func(t *testing.T, raw string) {
		found := decodeSearchResults(json.RawMessage(raw))
		for _, r := range found {
			if r.URL == "" {
				t.Fatalf("decodeSearchResults(%q) returned a result without a URL", raw)
			}
		}
		var byID map[string]mistral.SearchResult
		if err := json.Unmarshal([]byte(raw), &byID); err != nil {
			return
		}
		want := 0
		for _, r := range byID {
			if r.URL != "" {
				want++
			}
		}
		if len(found) != want {
			t.Fatalf("decodeSearchResults(%q) = %d results, want %d", raw, len(found), want)
		}
	})
}

func FuzzReasoningEffort(f *testing.F) {
	f.Add(seedModelLarge, seedEffortMed)
	f.Add(seedModelNew, seedEffortWeird)
	f.Add(seedNameEmpty, seedEffortEmpty)
	on := &anthropic.Thinking{Type: "adaptive"}
	off := &anthropic.Thinking{Type: anthropic.ThinkingDisabled}
	f.Fuzz(func(t *testing.T, name, effort string) {
		model := mistral.Model{ID: name, Name: name, Capabilities: mistral.ModelCapabilities{Reasoning: true}}
		cfg := &anthropic.OutputConfig{Effort: effort}
		got := reasoningEffort(model, on, cfg)
		if got != "" && !slices.Contains(allEfforts, got) {
			t.Fatalf("reasoningEffort(%q, on, %q) = %q, not a known level", name, effort, got)
		}
		offGot := reasoningEffort(model, off, nil)
		if offGot != "" && offGot != effortNone {
			t.Fatalf("reasoningEffort(%q, off) = %q, want empty or %q", name, offGot, effortNone)
		}
	})
}

func FuzzRequestRoundTrip(f *testing.F) {
	f.Add(seedReqMinimal)
	f.Add(seedReqBlocks)
	f.Add(seedReqBadJSON)
	f.Add(seedReqBadContent)
	f.Add(seedReqEmpty)
	f.Fuzz(func(t *testing.T, raw string) {
		var req anthropic.MessagesRequest
		if err := json.Unmarshal([]byte(raw), &req); err != nil {
			return
		}
		target := Target{Model: mistral.Model{ID: fuzzTargetModel}, CacheKey: fuzzCacheKey}
		out := Request(&req, testOptions, target)
		encoded, err := json.Marshal(out)
		if err != nil {
			t.Fatalf("marshal translated request from %q: %v", raw, err)
		}
		if !json.Valid(encoded) {
			t.Fatalf("translated request from %q is invalid JSON: %s", raw, encoded)
		}
		var back mistral.ChatRequest
		if err := json.Unmarshal(encoded, &back); err != nil {
			t.Fatalf("decode translated request from %q: %v", raw, err)
		}
		if back.Model != fuzzTargetModel {
			t.Fatalf("translated model = %q, want %q", back.Model, fuzzTargetModel)
		}
	})
}

func FuzzTranslatorEvents(f *testing.F) {
	f.Add(seedChunkText, seedChunkThink, seedChunkTool)
	f.Add(seedChunkBad, seedChunkEmpty, seedChunkText)
	f.Add(seedChunkTool, seedChunkTool, seedChunkTool)
	f.Fuzz(func(t *testing.T, first, second, third string) {
		rec := &recorder{}
		tr := NewTranslator(rec, fuzzTargetModel)
		if err := tr.Start(); err != nil {
			t.Fatalf("start: %v", err)
		}
		for _, raw := range []string{first, second, third} {
			var chunk mistral.StreamChunk
			if err := json.Unmarshal([]byte(raw), &chunk); err != nil {
				continue
			}
			if err := tr.Chunk(&chunk); err != nil {
				break
			}
		}
		if err := tr.Finish(); err != nil {
			t.Fatalf("finish: %v", err)
		}
		validateEventLog(t, rec)
	})
}

func validateEventLog(t *testing.T, rec *recorder) {
	t.Helper()
	events := rec.events
	n := len(events)
	if n < minEventCount {
		t.Fatalf("too few events: %v", events)
	}
	framed := events[0] == anthropic.EventMessageStart &&
		events[1] == anthropic.EventPing &&
		events[n-2] == anthropic.EventMessageDelta &&
		events[n-1] == anthropic.EventMessageStop
	if !framed {
		t.Fatalf("bad frame events: %v", events)
	}
	index := -1
	open := false
	for i := 2; i < n-2; i++ {
		switch events[i] {
		case anthropic.EventContentBlockStart:
			if open {
				t.Fatalf("block start at %d while a block is open: %v", i, events)
			}
			index++
			if rec.data[i].(anthropic.ContentBlockStart).Index != index {
				t.Fatalf("block start index at %d: %v", i, events)
			}
			open = true
		case anthropic.EventContentBlockDelta:
			if !open {
				t.Fatalf("delta at %d without an open block: %v", i, events)
			}
			if rec.data[i].(anthropic.ContentBlockDelta).Index != index {
				t.Fatalf("delta index at %d: %v", i, events)
			}
		case anthropic.EventContentBlockStop:
			if !open {
				t.Fatalf("stop at %d without an open block: %v", i, events)
			}
			if rec.data[i].(anthropic.ContentBlockStop).Index != index {
				t.Fatalf("stop index at %d: %v", i, events)
			}
			open = false
		default:
			t.Fatalf("unexpected event %q at %d: %v", events[i], i, events)
		}
	}
	if open {
		t.Fatalf("unclosed block at the end: %v", events)
	}
}
