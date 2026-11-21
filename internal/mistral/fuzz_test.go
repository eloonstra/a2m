package mistral

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"strconv"
	"strings"
	"testing"
)

const (
	seedDataChunk   = `data: {"choices":[{"delta":{"content":"a"}}]}`
	seedDone        = "data: [DONE]"
	seedDoneCompact = "data:[DONE]"
	seedComment     = ": comment"
	seedBrokenJSON  = "data: {invalid"
	seedErrorObject = `data: {"object":"error","message":"busy","raw_status_code":503}`
	seedStream      = seedDataChunk + "\n\n" + seedDone + "\n"
	seedTruncated   = seedDataChunk + "\n\n"
	seedEmpty       = ``

	seedTooLongBody = `{"type":"invalid_request_prompt_too_long","message":"Prompt 544383 > 131072 maximum context length"}`
	seedOtherBody   = `{"type":"invalid_request_invalid_args","message":"5 > 3"}`
	seedNoNumbers   = `{"type":"invalid_request_prompt_too_long","message":"no numbers here"}`
	seedPlainText   = `not json at all`

	seedStringJSON = `"hello"`
	seedArrayJSON  = `[{"type":"text","text":"hi"}]`
	seedNullJSON   = `null`
)

func FuzzParseLine(f *testing.F) {
	f.Add([]byte(seedDataChunk))
	f.Add([]byte(seedDone))
	f.Add([]byte(seedDoneCompact))
	f.Add([]byte(seedComment))
	f.Add([]byte(seedBrokenJSON))
	f.Add([]byte(seedErrorObject))
	f.Add([]byte(seedEmpty))
	f.Fuzz(func(t *testing.T, line []byte) {
		chunk, err := parseLine(line)
		trimmed := bytes.TrimSpace(line)
		isDataLine := bytes.HasPrefix(trimmed, dataPrefix)
		if !isDataLine {
			if chunk != nil || err != nil {
				t.Fatalf("non-data line %q gave chunk=%v err=%v", line, chunk, err)
			}
			return
		}
		payload := bytes.TrimSpace(trimmed[len(dataPrefix):])
		isDone := bytes.Equal(payload, doneMarker)
		if isDone {
			if !errors.Is(err, io.EOF) || chunk != nil {
				t.Fatalf("DONE line %q gave chunk=%v err=%v", line, chunk, err)
			}
			return
		}
		if err == nil {
			validChunk := chunk != nil && chunk.Object != ObjectError
			if !validChunk {
				t.Fatalf("line %q gave chunk=%v with nil error", line, chunk)
			}
			return
		}
		if chunk != nil {
			t.Fatalf("line %q gave chunk=%v and err=%v", line, chunk, err)
		}
	})
}

func FuzzStreamNext(f *testing.F) {
	f.Add([]byte(seedStream))
	f.Add([]byte(seedTruncated))
	f.Add([]byte(seedErrorObject + "\n\n"))
	f.Add([]byte(seedEmpty))
	f.Fuzz(func(t *testing.T, data []byte) {
		reader := bytes.NewReader(data)
		stream := &Stream{body: io.NopCloser(reader), reader: bufio.NewReader(reader)}
		for {
			chunk, err := stream.Next()
			if err == nil {
				if chunk == nil {
					t.Fatalf("stream %q returned a nil chunk with nil error", data)
				}
				continue
			}
			reachedDone := errors.Is(err, io.EOF)
			hasDoneMarker := bytes.Contains(data, doneMarker)
			if reachedDone && !hasDoneMarker {
				t.Fatalf("stream %q ended with EOF but has no DONE marker", data)
			}
			return
		}
	})
}

func FuzzContentUnmarshal(f *testing.F) {
	f.Add(seedStringJSON)
	f.Add(seedArrayJSON)
	f.Add(seedPlainText)
	f.Add(seedNullJSON)
	f.Add(seedEmpty)
	f.Fuzz(func(t *testing.T, raw string) {
		var content Content
		if err := json.Unmarshal([]byte(raw), &content); err != nil {
			return
		}
		encoded, err := json.Marshal(content)
		if err != nil {
			t.Fatalf("marshal content: %v", err)
		}
		if !json.Valid(encoded) {
			t.Fatalf("content marshal to invalid JSON: %s", encoded)
		}
		var value any
		if err := json.Unmarshal([]byte(raw), &value); err != nil {
			t.Fatalf("accepted input %q failed to decode: %v", raw, err)
		}
		text, isString := value.(string)
		if !isString {
			return
		}
		oneChunk := len(content) == 1
		isTextChunk := oneChunk && content[0].Type == ChunkText && content[0].Text == text
		if !isTextChunk {
			t.Fatalf("string input %q gave content %+v", raw, content)
		}
	})
}

func FuzzPromptTooLong(f *testing.F) {
	f.Add(seedTooLongBody)
	f.Add(seedOtherBody)
	f.Add(seedNoNumbers)
	f.Add(seedPlainText)
	f.Fuzz(func(t *testing.T, body string) {
		actual, limit, ok := (&APIError{Body: body}).PromptTooLong()
		if !ok {
			return
		}
		nonNegative := actual >= 0 && limit >= 0
		mentionsActual := strings.Contains(body, strconv.Itoa(actual))
		mentionsLimit := strings.Contains(body, strconv.Itoa(limit))
		if !nonNegative || !mentionsActual || !mentionsLimit {
			t.Fatalf("PromptTooLong(%q) = %d, %d, true", body, actual, limit)
		}
	})
}
