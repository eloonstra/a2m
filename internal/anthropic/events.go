package anthropic

const (
	EventMessageStart      = "message_start"
	EventPing              = "ping"
	EventContentBlockStart = "content_block_start"
	EventContentBlockDelta = "content_block_delta"
	EventContentBlockStop  = "content_block_stop"
	EventMessageDelta      = "message_delta"
	EventMessageStop       = "message_stop"
	EventError             = "error"

	DeltaText      = "text_delta"
	DeltaThinking  = "thinking_delta"
	DeltaSignature = "signature_delta"
	DeltaInputJSON = "input_json_delta"
)

type Emitter interface {
	Emit(event string, data any) error
}

type MessageStart struct {
	Type    string          `json:"type"`
	Message MessageResponse `json:"message"`
}

type Ping struct {
	Type string `json:"type"`
}

type ContentBlockStart struct {
	Type         string `json:"type"`
	Index        int    `json:"index"`
	ContentBlock any    `json:"content_block"`
}

type ContentBlockDelta struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
	Delta any    `json:"delta"`
}

type ContentBlockStop struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
}

type TextDelta struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type ThinkingDelta struct {
	Type     string `json:"type"`
	Thinking string `json:"thinking"`
}

type SignatureDelta struct {
	Type      string `json:"type"`
	Signature string `json:"signature"`
}

type InputJSONDelta struct {
	Type        string `json:"type"`
	PartialJSON string `json:"partial_json"`
}

type MessageDelta struct {
	Type  string           `json:"type"`
	Delta MessageDeltaBody `json:"delta"`
	Usage Usage            `json:"usage"`
}

type MessageDeltaBody struct {
	StopReason   string  `json:"stop_reason"`
	StopSequence *string `json:"stop_sequence"`
}

type MessageStop struct {
	Type string `json:"type"`
}
