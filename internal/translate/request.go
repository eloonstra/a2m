package translate

import (
	"encoding/json"
	"hash/fnv"
	"strings"

	"a2m/internal/anthropic"
	"a2m/internal/mistral"
)

const (
	systemSeparator = "\n\n"
	emptyToolResult = "(no output)"
	autoModelHint   = "auto"
	smallModelHint  = "haiku"
	mediumModelHint = "medium"
	claudeModelHint = "claude"
	signaturePrefix = "mistral:"
	toolCallIDLen   = 9
	base62Alphabet  = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)

var (
	emptyObject = json.RawMessage(`{}`)
	emptySchema = json.RawMessage(`{"type":"object","properties":{}}`)
)

type Options struct {
	Model       string
	MediumModel string
	SmallModel  string
	AutoModel   string
	MaxTokens   int
}

type Target struct {
	Model    mistral.Model
	CacheKey string
}

func Request(in *anthropic.MessagesRequest, opts Options, target Target) *mistral.ChatRequest {
	out := &mistral.ChatRequest{
		Model:           target.Model.ID,
		Messages:        convertMessages(in.System, in.Messages),
		MaxTokens:       capTokens(in.MaxTokens, opts.MaxTokens),
		Temperature:     in.Temperature,
		TopP:            in.TopP,
		Stop:            in.StopSequences,
		ReasoningEffort: reasoningEffort(target.Model, in.Thinking, in.OutputConfig),
		PromptCacheKey:  target.CacheKey,
	}
	out.Tools = convertTools(in.Tools)
	if len(out.Tools) == 0 {
		return out
	}
	out.ToolChoice, out.ParallelToolCalls = convertToolChoice(in.ToolChoice)
	return out
}

func MapModel(model string, opts Options) string {
	name := strings.ToLower(model)
	if strings.Contains(name, autoModelHint) {
		return opts.AutoModel
	}
	if strings.Contains(name, smallModelHint) {
		return opts.SmallModel
	}
	if strings.Contains(name, mediumModelHint) {
		return opts.MediumModel
	}
	if strings.Contains(name, claudeModelHint) {
		return opts.Model
	}
	return model
}

func capTokens(requested, limit int) int {
	noLimit := limit <= 0
	if noLimit || requested < limit {
		return requested
	}
	return limit
}

func thinkingEnabled(t *anthropic.Thinking) bool {
	return t != nil && t.Type != "" && t.Type != anthropic.ThinkingDisabled
}

func convertMessages(system anthropic.Blocks, msgs []anthropic.Message) []mistral.Message {
	out := make([]mistral.Message, 0, len(msgs)+1)
	if text := joinText(system); text != "" {
		out = append(out, mistral.Message{Role: mistral.RoleSystem, Content: mistral.Content{textChunk(text)}})
	}
	toolNames := make(map[string]string)
	for _, m := range msgs {
		if m.Role == anthropic.RoleAssistant {
			out = appendAssistant(out, m.Content, toolNames)
			continue
		}
		out = appendUser(out, m.Content, toolNames)
	}
	markPrefix(out)
	return out
}

func joinText(blocks anthropic.Blocks) string {
	parts := make([]string, 0, len(blocks))
	for _, b := range blocks {
		if b.Type == anthropic.BlockText && b.Text != "" {
			parts = append(parts, b.Text)
		}
	}
	return strings.Join(parts, systemSeparator)
}

func appendAssistant(out []mistral.Message, blocks anthropic.Blocks, toolNames map[string]string) []mistral.Message {
	msg := mistral.Message{Role: mistral.RoleAssistant}
	for _, b := range blocks {
		switch b.Type {
		case anthropic.BlockText:
			msg.Content = appendText(msg.Content, b.Text)
		case anthropic.BlockThinking:
			msg.Content = appendThinking(msg.Content, b)
		case anthropic.BlockToolUse:
			toolNames[b.ID] = b.Name
			msg.ToolCalls = append(msg.ToolCalls, toolCall(b))
		}
	}
	isEmpty := len(msg.Content) == 0 && len(msg.ToolCalls) == 0
	if isEmpty {
		return out
	}
	return append(out, msg)
}

func appendThinking(content mistral.Content, b anthropic.ContentBlock) mistral.Content {
	if b.Thinking == "" {
		return content
	}
	return append(content, mistral.Chunk{
		Type:      mistral.ChunkThinking,
		Thinking:  []mistral.Chunk{textChunk(b.Thinking)},
		Signature: mistralSignature(b.Signature),
	})
}

func mistralSignature(signature string) string {
	original, ok := strings.CutPrefix(signature, signaturePrefix)
	if !ok {
		return ""
	}
	return original
}

func toolCall(b anthropic.ContentBlock) mistral.ToolCall {
	args := b.Input
	if len(args) == 0 {
		args = emptyObject
	}
	return mistral.ToolCall{
		ID:       toolCallID(b.ID),
		Type:     mistral.ToolTypeFunction,
		Function: mistral.FunctionCall{Name: b.Name, Arguments: args},
	}
}

func appendUser(out []mistral.Message, blocks anthropic.Blocks, toolNames map[string]string) []mistral.Message {
	var pending mistral.Content
	for _, b := range blocks {
		if b.Type == anthropic.BlockToolResult {
			out = append(out, toolMessage(b, toolNames))
			continue
		}
		pending = appendUserChunk(pending, b)
	}
	if len(pending) == 0 {
		return out
	}
	return mergeUserContent(out, pending)
}

func mergeUserContent(out []mistral.Message, content mistral.Content) []mistral.Message {
	last := len(out) - 1
	canMerge := last >= 0 && out[last].Role == mistral.RoleUser
	if !canMerge {
		return append(out, mistral.Message{Role: mistral.RoleUser, Content: content})
	}
	out[last].Content = append(out[last].Content, content...)
	return out
}

func toolMessage(b anthropic.ContentBlock, toolNames map[string]string) mistral.Message {
	var content mistral.Content
	for _, c := range b.Content {
		content = appendUserChunk(content, c)
	}
	if len(content) == 0 {
		content = mistral.Content{textChunk(emptyToolResult)}
	}
	return mistral.Message{
		Role:       mistral.RoleTool,
		Content:    content,
		ToolCallID: toolCallID(b.ToolUseID),
		Name:       toolNames[b.ToolUseID],
	}
}

func appendUserChunk(content mistral.Content, b anthropic.ContentBlock) mistral.Content {
	switch b.Type {
	case anthropic.BlockText:
		return appendText(content, b.Text)
	case anthropic.BlockImage:
		return appendImage(content, b.Source)
	}
	return content
}

func appendText(content mistral.Content, text string) mistral.Content {
	if text == "" {
		return content
	}
	return append(content, textChunk(text))
}

func appendImage(content mistral.Content, src *anthropic.ImageSource) mistral.Content {
	url := imageURL(src)
	if url == "" {
		return content
	}
	return append(content, mistral.Chunk{Type: mistral.ChunkImageURL, ImageURL: url})
}

func imageURL(src *anthropic.ImageSource) string {
	if src == nil {
		return ""
	}
	switch src.Type {
	case anthropic.SourceBase64:
		return "data:" + src.MediaType + ";base64," + src.Data
	case anthropic.SourceURL:
		return src.URL
	}
	return ""
}

func textChunk(text string) mistral.Chunk {
	return mistral.Chunk{Type: mistral.ChunkText, Text: text}
}

func markPrefix(out []mistral.Message) {
	last := len(out) - 1
	isPrefill := last >= 0 && out[last].Role == mistral.RoleAssistant && len(out[last].ToolCalls) == 0
	if isPrefill {
		out[last].Prefix = true
	}
}

func convertTools(tools []anthropic.Tool) []mistral.Tool {
	out := make([]mistral.Tool, 0, len(tools))
	for _, t := range tools {
		isServerTool := t.Type != "" && t.Type != anthropic.ToolTypeCustom
		if isServerTool {
			continue
		}
		schema := t.InputSchema
		if len(schema) == 0 {
			schema = emptySchema
		}
		out = append(out, mistral.Tool{
			Type:     mistral.ToolTypeFunction,
			Function: mistral.Function{Name: t.Name, Description: t.Description, Parameters: schema},
		})
	}
	return out
}

func convertToolChoice(tc *anthropic.ToolChoice) (any, *bool) {
	if tc == nil {
		return nil, nil
	}
	var parallel *bool
	if tc.DisableParallelToolUse {
		parallel = new(bool)
	}
	switch tc.Type {
	case anthropic.ToolChoiceAny:
		return mistral.ToolChoiceAny, parallel
	case anthropic.ToolChoiceNone:
		return mistral.ToolChoiceNone, parallel
	case anthropic.ToolChoiceTool:
		return mistral.ToolChoiceFunction{
			Type:     mistral.ToolTypeFunction,
			Function: mistral.FunctionName{Name: tc.Name},
		}, parallel
	}
	return mistral.ToolChoiceAuto, parallel
}

func toolCallID(id string) string {
	if isMistralToolCallID(id) {
		return id
	}
	h := fnv.New64a()
	h.Write([]byte(id))
	sum := h.Sum64()
	out := make([]byte, toolCallIDLen)
	for i := range out {
		out[i] = base62Alphabet[sum%uint64(len(base62Alphabet))]
		sum /= uint64(len(base62Alphabet))
	}
	return string(out)
}

func isMistralToolCallID(id string) bool {
	if len(id) != toolCallIDLen {
		return false
	}
	for i := 0; i < len(id); i++ {
		if strings.IndexByte(base62Alphabet, id[i]) < 0 {
			return false
		}
	}
	return true
}
