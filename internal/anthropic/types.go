package anthropic

import "encoding/json"

const (
	RoleUser      = "user"
	RoleAssistant = "assistant"

	BlockText       = "text"
	BlockImage      = "image"
	BlockThinking   = "thinking"
	BlockToolUse    = "tool_use"
	BlockToolResult = "tool_result"

	BlockServerToolUse       = "server_tool_use"
	BlockWebSearchToolResult = "web_search_tool_result"
	WebSearchResultType      = "web_search_result"
	WebSearchToolName        = "web_search"

	SourceBase64 = "base64"
	SourceURL    = "url"

	ToolTypeCustom = "custom"

	ToolChoiceAuto = "auto"
	ToolChoiceAny  = "any"
	ToolChoiceTool = "tool"
	ToolChoiceNone = "none"

	ThinkingDisabled = "disabled"

	StopEndTurn   = "end_turn"
	StopMaxTokens = "max_tokens"
	StopToolUse   = "tool_use"
)

type MessagesRequest struct {
	Model         string        `json:"model"`
	MaxTokens     int           `json:"max_tokens"`
	System        Blocks        `json:"system"`
	Messages      []Message     `json:"messages"`
	Tools         []Tool        `json:"tools"`
	ToolChoice    *ToolChoice   `json:"tool_choice"`
	Temperature   *float64      `json:"temperature"`
	TopP          *float64      `json:"top_p"`
	StopSequences []string      `json:"stop_sequences"`
	Stream        bool          `json:"stream"`
	Thinking      *Thinking     `json:"thinking"`
	OutputConfig  *OutputConfig `json:"output_config"`
}

type OutputConfig struct {
	Effort string `json:"effort"`
}

type Message struct {
	Role    string `json:"role"`
	Content Blocks `json:"content"`
}

type ContentBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text"`
	Thinking  string          `json:"thinking"`
	Signature string          `json:"signature"`
	Source    *ImageSource    `json:"source"`
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Input     json.RawMessage `json:"input"`
	ToolUseID string          `json:"tool_use_id"`
	Content   Blocks          `json:"content"`
}

type Blocks []ContentBlock

func (b *Blocks) UnmarshalJSON(data []byte) error {
	isString := len(data) > 0 && data[0] == '"'
	if !isString {
		return json.Unmarshal(data, (*[]ContentBlock)(b))
	}
	var text string
	if err := json.Unmarshal(data, &text); err != nil {
		return err
	}
	*b = Blocks{{Type: BlockText, Text: text}}
	return nil
}

type ImageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
	URL       string `json:"url"`
}

type Tool struct {
	Type           string          `json:"type"`
	Name           string          `json:"name"`
	Description    string          `json:"description"`
	InputSchema    json.RawMessage `json:"input_schema"`
	AllowedDomains []string        `json:"allowed_domains"`
	BlockedDomains []string        `json:"blocked_domains"`
}

type ToolChoice struct {
	Type                   string `json:"type"`
	Name                   string `json:"name"`
	DisableParallelToolUse bool   `json:"disable_parallel_tool_use"`
}

type Thinking struct {
	Type string `json:"type"`
}

type Usage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
}

type MessageResponse struct {
	ID           string  `json:"id"`
	Type         string  `json:"type"`
	Role         string  `json:"role"`
	Model        string  `json:"model"`
	Content      []any   `json:"content"`
	StopReason   *string `json:"stop_reason"`
	StopSequence *string `json:"stop_sequence"`
	Usage        Usage   `json:"usage"`
}

type TextBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type ThinkingBlock struct {
	Type      string `json:"type"`
	Thinking  string `json:"thinking"`
	Signature string `json:"signature"`
}

type ToolUseBlock struct {
	Type  string          `json:"type"`
	ID    string          `json:"id"`
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input"`
}

type WebSearchToolResultBlock struct {
	Type      string            `json:"type"`
	ToolUseID string            `json:"tool_use_id"`
	Content   []WebSearchResult `json:"content"`
}

type WebSearchResult struct {
	Type             string  `json:"type"`
	URL              string  `json:"url"`
	Title            string  `json:"title"`
	EncryptedContent string  `json:"encrypted_content"`
	PageAge          *string `json:"page_age"`
}

type ErrorBody struct {
	Type  string      `json:"type"`
	Error ErrorDetail `json:"error"`
}

type ErrorDetail struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}
