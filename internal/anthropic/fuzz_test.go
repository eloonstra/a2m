package anthropic

import (
	"encoding/json"
	"testing"
)

const (
	seedTextBlocks  = `[{"type":"text","text":"hi"}]`
	seedStringBlock = `"hello"`
	seedInvalidJSON = `not json`
	seedEmpty       = ``
)

func FuzzBlocksUnmarshal(f *testing.F) {
	f.Add(seedTextBlocks)
	f.Add(seedStringBlock)
	f.Add(seedInvalidJSON)
	f.Add(seedEmpty)
	f.Fuzz(func(t *testing.T, raw string) {
		var blocks Blocks
		if err := json.Unmarshal([]byte(raw), &blocks); err != nil {
			return
		}
		encoded, err := json.Marshal(blocks)
		if err != nil {
			t.Fatalf("marshal blocks: %v", err)
		}
		if !json.Valid(encoded) {
			t.Fatalf("blocks marshal to invalid JSON: %s", encoded)
		}
		var value any
		if err := json.Unmarshal([]byte(raw), &value); err != nil {
			t.Fatalf("accepted input %q failed to decode: %v", raw, err)
		}
		text, isString := value.(string)
		if !isString {
			return
		}
		oneBlock := len(blocks) == 1
		isTextBlock := oneBlock && blocks[0].Type == BlockText && blocks[0].Text == text
		if !isTextBlock {
			t.Fatalf("string input %q gave blocks %+v", raw, blocks)
		}
	})
}
