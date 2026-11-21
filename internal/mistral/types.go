package mistral

import "encoding/json"

const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleTool      = "tool"

	ChunkText     = "text"
	ChunkImageURL = "image_url"
	ChunkThinking = "thinking"

	ToolTypeFunction = "function"

	ToolChoiceAuto = "auto"
	ToolChoiceAny  = "any"
	ToolChoiceNone = "none"

	FinishToolCalls   = "tool_calls"
	FinishLength      = "length"
	FinishModelLength = "model_length"
	FinishError       = "error"

	ObjectError = "error"

	EntryToolExecution = "tool.execution"
	EntryMessageOutput = "message.output"
	ToolWebSearch      = "web_search"
)

type ChatRequest struct {
	Model             string    `json:"model"`
	Messages          []Message `json:"messages"`
	Tools             []Tool    `json:"tools,omitempty"`
	ToolChoice        any       `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool     `json:"parallel_tool_calls,omitempty"`
	MaxTokens         int       `json:"max_tokens,omitempty"`
	Temperature       *float64  `json:"temperature,omitempty"`
	TopP              *float64  `json:"top_p,omitempty"`
	Stop              []string  `json:"stop,omitempty"`
	Stream            bool      `json:"stream"`
	ReasoningEffort   string    `json:"reasoning_effort,omitempty"`
	PromptCacheKey    string    `json:"prompt_cache_key,omitempty"`
}

type Message struct {
	Role       string     `json:"role"`
	Content    Content    `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Name       string     `json:"name,omitempty"`
	Prefix     bool       `json:"prefix,omitempty"`
}

type Chunk struct {
	Type      string  `json:"type"`
	Text      string  `json:"text,omitempty"`
	ImageURL  string  `json:"image_url,omitempty"`
	Thinking  []Chunk `json:"thinking,omitempty"`
	Signature string  `json:"signature,omitempty"`
}

type Content []Chunk

func (c *Content) UnmarshalJSON(data []byte) error {
	isString := len(data) > 0 && data[0] == '"'
	if !isString {
		return json.Unmarshal(data, (*[]Chunk)(c))
	}
	var text string
	if err := json.Unmarshal(data, &text); err != nil {
		return err
	}
	*c = Content{{Type: ChunkText, Text: text}}
	return nil
}

type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

type Function struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
}

type ToolChoiceFunction struct {
	Type     string       `json:"type"`
	Function FunctionName `json:"function"`
}

type FunctionName struct {
	Name string `json:"name"`
}

type ToolCall struct {
	ID       string       `json:"id,omitempty"`
	Type     string       `json:"type,omitempty"`
	Function FunctionCall `json:"function"`
	Index    int          `json:"index,omitempty"`
}

type FunctionCall struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

type StreamChunk struct {
	Object        string         `json:"object"`
	RawStatusCode int            `json:"raw_status_code"`
	Choices       []StreamChoice `json:"choices"`
	Usage         *Usage         `json:"usage"`
}

type StreamChoice struct {
	Delta        Delta   `json:"delta"`
	FinishReason *string `json:"finish_reason"`
}

type Delta struct {
	Content   Content    `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls"`
}

type Usage struct {
	PromptTokens        int                  `json:"prompt_tokens"`
	CompletionTokens    int                  `json:"completion_tokens"`
	TotalTokens         int                  `json:"total_tokens"`
	PromptTokensDetails *PromptTokensDetails `json:"prompt_tokens_details"`
}

type PromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

type ConversationRequest struct {
	Model        string        `json:"model"`
	Inputs       string        `json:"inputs"`
	Instructions string        `json:"instructions,omitempty"`
	Tools        []BuiltInTool `json:"tools"`
	Store        bool          `json:"store"`
}

type BuiltInTool struct {
	Type string `json:"type"`
}

type ConversationResponse struct {
	Outputs []ConversationEntry `json:"outputs"`
	Usage   Usage               `json:"usage"`
}

type ConversationEntry struct {
	Type      string        `json:"type"`
	Name      string        `json:"name"`
	Arguments string        `json:"arguments"`
	Info      ExecutionInfo `json:"info"`
	Content   Content       `json:"content"`
}

type ExecutionInfo struct {
	Result json.RawMessage `json:"result"`
}

type SearchResult struct {
	URL   string `json:"url"`
	Title string `json:"title"`
	Rank  int    `json:"rank"`
}

type ModelList struct {
	Data []Model `json:"data"`
}

type Model struct {
	ID               string            `json:"id"`
	Name             string            `json:"name"`
	Description      string            `json:"description"`
	MaxContextLength int               `json:"max_context_length"`
	Deprecation      *json.RawMessage  `json:"deprecation"`
	Capabilities     ModelCapabilities `json:"capabilities"`
}

type ModelCapabilities struct {
	CompletionChat  bool `json:"completion_chat"`
	FunctionCalling bool `json:"function_calling"`
	Reasoning       bool `json:"reasoning"`
	Audio           bool `json:"audio"`
}
