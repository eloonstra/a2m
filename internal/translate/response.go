package translate

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"

	"a2m/internal/anthropic"
	"a2m/internal/mistral"
)

const (
	messageIDPrefix = "msg_"
	toolUseIDPrefix = "toolu_"
	messageType     = "message"
	nullToolCallID  = "null"
	randomIDBytes   = 12
	errGeneration   = "mistral: generation failed"
)

var ErrGeneration = errors.New(errGeneration)

type blockKind int

const (
	blockNone blockKind = iota
	blockText
	blockThinking
	blockTool
	blockSearchResult
)

type Translator struct {
	emit       anthropic.Emitter
	model      string
	open       blockKind
	index      int
	toolIndex  int
	signature  string
	sawToolUse bool
	finish     string
	usage      anthropic.Usage
}

func NewTranslator(emit anthropic.Emitter, model string) *Translator {
	return &Translator{emit: emit, model: model, index: -1}
}

func (t *Translator) Start() error {
	msg := anthropic.MessageResponse{
		ID:      messageIDPrefix + randomID(),
		Type:    messageType,
		Role:    anthropic.RoleAssistant,
		Model:   t.model,
		Content: []any{},
	}
	if err := t.emit.Emit(anthropic.EventMessageStart, anthropic.MessageStart{Type: anthropic.EventMessageStart, Message: msg}); err != nil {
		return err
	}
	return t.emit.Emit(anthropic.EventPing, anthropic.Ping{Type: anthropic.EventPing})
}

func (t *Translator) Chunk(c *mistral.StreamChunk) error {
	if c.Usage != nil {
		t.usage = convertUsage(*c.Usage, c.Usage.PromptTokens)
	}
	for _, choice := range c.Choices {
		if err := t.delta(choice.Delta); err != nil {
			return err
		}
		if choice.FinishReason == nil {
			continue
		}
		if *choice.FinishReason == mistral.FinishError {
			return ErrGeneration
		}
		t.finish = *choice.FinishReason
	}
	return nil
}

func convertUsage(u mistral.Usage, inputTokens int) anthropic.Usage {
	cached := 0
	if u.PromptTokensDetails != nil {
		cached = min(u.PromptTokensDetails.CachedTokens, inputTokens)
	}
	return anthropic.Usage{
		InputTokens:          inputTokens - cached,
		OutputTokens:         u.CompletionTokens,
		CacheReadInputTokens: cached,
	}
}

func (t *Translator) Finish() error {
	if err := t.closeBlock(); err != nil {
		return err
	}
	delta := anthropic.MessageDelta{
		Type:  anthropic.EventMessageDelta,
		Delta: anthropic.MessageDeltaBody{StopReason: t.stopReason()},
		Usage: t.usage,
	}
	if err := t.emit.Emit(anthropic.EventMessageDelta, delta); err != nil {
		return err
	}
	return t.emit.Emit(anthropic.EventMessageStop, anthropic.MessageStop{Type: anthropic.EventMessageStop})
}

func (t *Translator) stopReason() string {
	if t.sawToolUse {
		return anthropic.StopToolUse
	}
	switch t.finish {
	case mistral.FinishToolCalls:
		return anthropic.StopToolUse
	case mistral.FinishLength, mistral.FinishModelLength:
		return anthropic.StopMaxTokens
	}
	return anthropic.StopEndTurn
}

func (t *Translator) delta(d mistral.Delta) error {
	for _, c := range d.Content {
		if err := t.contentChunk(c); err != nil {
			return err
		}
	}
	for _, tc := range d.ToolCalls {
		if err := t.toolCall(tc); err != nil {
			return err
		}
	}
	return nil
}

func (t *Translator) contentChunk(c mistral.Chunk) error {
	switch c.Type {
	case mistral.ChunkText:
		return t.text(c.Text)
	case mistral.ChunkThinking:
		return t.thinking(c)
	}
	return nil
}

func (t *Translator) text(text string) error {
	if text == "" {
		return nil
	}
	if err := t.ensureBlock(blockText, anthropic.TextBlock{Type: anthropic.BlockText}); err != nil {
		return err
	}
	return t.emitDelta(anthropic.TextDelta{Type: anthropic.DeltaText, Text: text})
}

func (t *Translator) thinking(c mistral.Chunk) error {
	if c.Signature != "" {
		t.signature = signaturePrefix + c.Signature
	}
	for _, part := range c.Thinking {
		if err := t.thinkingText(part.Text); err != nil {
			return err
		}
	}
	return nil
}

func (t *Translator) thinkingText(text string) error {
	if text == "" {
		return nil
	}
	if err := t.ensureBlock(blockThinking, anthropic.ThinkingBlock{Type: anthropic.BlockThinking}); err != nil {
		return err
	}
	return t.emitDelta(anthropic.ThinkingDelta{Type: anthropic.DeltaThinking, Thinking: text})
}

func (t *Translator) toolCall(tc mistral.ToolCall) error {
	isNewCall := t.open != blockTool || tc.Index != t.toolIndex
	if isNewCall {
		if err := t.startTool(tc); err != nil {
			return err
		}
	}
	args := argumentsText(tc.Function.Arguments)
	if args == "" {
		return nil
	}
	return t.emitDelta(anthropic.InputJSONDelta{Type: anthropic.DeltaInputJSON, PartialJSON: args})
}

func (t *Translator) startTool(tc mistral.ToolCall) error {
	if err := t.closeBlock(); err != nil {
		return err
	}
	id := tc.ID
	if id == "" || id == nullToolCallID {
		id = toolUseIDPrefix + randomID()
	}
	block := anthropic.ToolUseBlock{Type: anthropic.BlockToolUse, ID: id, Name: tc.Function.Name, Input: emptyObject}
	t.toolIndex = tc.Index
	t.sawToolUse = true
	return t.startBlock(blockTool, block)
}

func argumentsText(raw json.RawMessage) string {
	isString := len(raw) > 0 && raw[0] == '"'
	if !isString {
		return string(raw)
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return ""
	}
	return s
}

func (t *Translator) ensureBlock(kind blockKind, block any) error {
	if t.open == kind {
		return nil
	}
	if err := t.closeBlock(); err != nil {
		return err
	}
	return t.startBlock(kind, block)
}

func (t *Translator) startBlock(kind blockKind, block any) error {
	t.index++
	t.open = kind
	return t.emit.Emit(anthropic.EventContentBlockStart, anthropic.ContentBlockStart{
		Type:         anthropic.EventContentBlockStart,
		Index:        t.index,
		ContentBlock: block,
	})
}

func (t *Translator) closeBlock() error {
	if t.open == blockNone {
		return nil
	}
	if err := t.flushSignature(); err != nil {
		return err
	}
	t.open = blockNone
	return t.emit.Emit(anthropic.EventContentBlockStop, anthropic.ContentBlockStop{Type: anthropic.EventContentBlockStop, Index: t.index})
}

func (t *Translator) flushSignature() error {
	hasSignature := t.open == blockThinking && t.signature != ""
	if !hasSignature {
		return nil
	}
	signature := t.signature
	t.signature = ""
	return t.emitDelta(anthropic.SignatureDelta{Type: anthropic.DeltaSignature, Signature: signature})
}

func (t *Translator) emitDelta(delta any) error {
	return t.emit.Emit(anthropic.EventContentBlockDelta, anthropic.ContentBlockDelta{
		Type:  anthropic.EventContentBlockDelta,
		Index: t.index,
		Delta: delta,
	})
}

func randomID() string {
	b := make([]byte, randomIDBytes)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}
