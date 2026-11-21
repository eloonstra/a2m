package proxy

import (
	"encoding/json"
	"testing"
)

const (
	fallbackObject = `{}`
	seedValidJSON  = `{"a":1}`
	seedInvalid    = `not json`
	seedEmpty      = ``
)

func FuzzToolInput(f *testing.F) {
	f.Add(seedValidJSON)
	f.Add(seedInvalid)
	f.Add(seedEmpty)
	f.Fuzz(func(t *testing.T, raw string) {
		got := toolInput(raw, json.RawMessage(fallbackObject))
		if !json.Valid([]byte(got)) {
			t.Fatalf("toolInput(%q) = %q, not valid JSON", raw, got)
		}
	})
}
