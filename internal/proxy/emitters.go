package proxy

import (
	"bufio"
	"encoding/json"
	"net/http"
	"strings"

	"a2m/internal/anthropic"
)

type sseWriter struct {
	buf     *bufio.Writer
	flusher http.Flusher
}

func newSSEWriter(w http.ResponseWriter) *sseWriter {
	flusher, _ := w.(http.Flusher)
	return &sseWriter{buf: bufio.NewWriter(w), flusher: flusher}
}

func (s *sseWriter) Emit(event string, data any) error {
	payload, err := json.Marshal(data)
	if err != nil {
		return err
	}
	s.buf.WriteString("event: ")
	s.buf.WriteString(event)
	s.buf.WriteString("\ndata: ")
	s.buf.Write(payload)
	s.buf.WriteString("\n\n")
	if err := s.buf.Flush(); err != nil {
		return err
	}
	if s.flusher != nil {
		s.flusher.Flush()
	}
	return nil
}

type collectedBlock struct {
	block any
	text  strings.Builder
	sig   string
}

type collector struct {
	message anthropic.MessageResponse
	blocks  []*collectedBlock
}

func (c *collector) Emit(_ string, data any) error {
	switch e := data.(type) {
	case anthropic.MessageStart:
		c.message = e.Message
	case anthropic.ContentBlockStart:
		c.blocks = append(c.blocks, &collectedBlock{block: e.ContentBlock})
	case anthropic.ContentBlockDelta:
		c.applyDelta(c.blocks[e.Index], e.Delta)
	case anthropic.MessageDelta:
		stopReason := e.Delta.StopReason
		c.message.StopReason = &stopReason
		c.message.Usage = e.Usage
	}
	return nil
}

func (c *collector) applyDelta(b *collectedBlock, delta any) {
	switch d := delta.(type) {
	case anthropic.TextDelta:
		b.text.WriteString(d.Text)
	case anthropic.ThinkingDelta:
		b.text.WriteString(d.Thinking)
	case anthropic.InputJSONDelta:
		b.text.WriteString(d.PartialJSON)
	case anthropic.SignatureDelta:
		b.sig = d.Signature
	}
}

func (c *collector) result() anthropic.MessageResponse {
	content := make([]any, 0, len(c.blocks))
	for _, b := range c.blocks {
		content = append(content, b.finalize())
	}
	c.message.Content = content
	return c.message
}

func (b *collectedBlock) finalize() any {
	switch block := b.block.(type) {
	case anthropic.TextBlock:
		block.Text = b.text.String()
		return block
	case anthropic.ThinkingBlock:
		block.Thinking = b.text.String()
		block.Signature = b.sig
		return block
	case anthropic.ToolUseBlock:
		block.Input = toolInput(b.text.String(), block.Input)
		return block
	}
	return b.block
}

func toolInput(raw string, fallback json.RawMessage) json.RawMessage {
	if !json.Valid([]byte(raw)) {
		return fallback
	}
	return json.RawMessage(raw)
}
