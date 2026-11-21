package translate

import (
	"encoding/json"
	"net/url"
	"sort"
	"strings"

	"a2m/internal/anthropic"
	"a2m/internal/mistral"
)

const (
	webSearchToolPrefix = "web_search_"
	serverToolIDPrefix  = "srvtoolu_"
	allowedDomainsRule  = "Only use sources from these domains: "
	blockedDomainsRule  = "Never use sources from these domains: "
	domainSeparator     = ", "
	inputSeparator      = "\n\n"
	subdomainDot        = "."
)

func WebSearchTool(tools []anthropic.Tool) (anthropic.Tool, bool) {
	for _, t := range tools {
		if strings.HasPrefix(t.Type, webSearchToolPrefix) {
			return t, true
		}
	}
	return anthropic.Tool{}, false
}

func WebSearchRequest(in *anthropic.MessagesRequest, model string, tool anthropic.Tool) *mistral.ConversationRequest {
	return &mistral.ConversationRequest{
		Model:        model,
		Inputs:       userText(in.Messages),
		Instructions: searchInstructions(in.System, tool),
		Tools:        []mistral.BuiltInTool{{Type: mistral.ToolWebSearch}},
	}
}

func userText(msgs []anthropic.Message) string {
	parts := make([]string, 0, len(msgs))
	for _, m := range msgs {
		if m.Role != anthropic.RoleUser {
			continue
		}
		if text := joinText(m.Content); text != "" {
			parts = append(parts, text)
		}
	}
	return strings.Join(parts, inputSeparator)
}

func searchInstructions(system anthropic.Blocks, tool anthropic.Tool) string {
	parts := []string{joinText(system)}
	if len(tool.AllowedDomains) > 0 {
		parts = append(parts, allowedDomainsRule+strings.Join(tool.AllowedDomains, domainSeparator))
	}
	if len(tool.BlockedDomains) > 0 {
		parts = append(parts, blockedDomainsRule+strings.Join(tool.BlockedDomains, domainSeparator))
	}
	return strings.TrimSpace(strings.Join(parts, systemSeparator))
}

func (t *Translator) WebSearch(resp *mistral.ConversationResponse, tool anthropic.Tool) error {
	t.usage = convertUsage(resp.Usage, resp.Usage.TotalTokens-resp.Usage.CompletionTokens)
	for _, entry := range resp.Outputs {
		if err := t.searchEntry(entry, tool); err != nil {
			return err
		}
	}
	return nil
}

func (t *Translator) searchEntry(entry mistral.ConversationEntry, tool anthropic.Tool) error {
	switch entry.Type {
	case mistral.EntryToolExecution:
		return t.searchExecution(entry, tool)
	case mistral.EntryMessageOutput:
		return t.delta(mistral.Delta{Content: entry.Content})
	}
	return nil
}

func (t *Translator) searchExecution(entry mistral.ConversationEntry, tool anthropic.Tool) error {
	if entry.Name != mistral.ToolWebSearch {
		return nil
	}
	id := serverToolIDPrefix + randomID()
	use := anthropic.ToolUseBlock{Type: anthropic.BlockServerToolUse, ID: id, Name: anthropic.WebSearchToolName, Input: emptyObject}
	if err := t.closeBlock(); err != nil {
		return err
	}
	if err := t.startBlock(blockTool, use); err != nil {
		return err
	}
	if entry.Arguments != "" {
		if err := t.emitDelta(anthropic.InputJSONDelta{Type: anthropic.DeltaInputJSON, PartialJSON: entry.Arguments}); err != nil {
			return err
		}
	}
	if err := t.closeBlock(); err != nil {
		return err
	}
	result := anthropic.WebSearchToolResultBlock{
		Type:      anthropic.BlockWebSearchToolResult,
		ToolUseID: id,
		Content:   searchResults(entry.Info.Result, tool),
	}
	if err := t.startBlock(blockSearchResult, result); err != nil {
		return err
	}
	return t.closeBlock()
}

func searchResults(raw json.RawMessage, tool anthropic.Tool) []anthropic.WebSearchResult {
	found := decodeSearchResults(raw)
	sort.SliceStable(found, func(i, j int) bool { return found[i].Rank < found[j].Rank })
	out := make([]anthropic.WebSearchResult, 0, len(found))
	for _, r := range found {
		if !domainAllowed(r.URL, tool) {
			continue
		}
		out = append(out, anthropic.WebSearchResult{Type: anthropic.WebSearchResultType, URL: r.URL, Title: r.Title})
	}
	return out
}

func decodeSearchResults(raw json.RawMessage) []mistral.SearchResult {
	payload := []byte(raw)
	isString := len(payload) > 0 && payload[0] == '"'
	if isString {
		var text string
		if err := json.Unmarshal(payload, &text); err != nil {
			return nil
		}
		payload = []byte(text)
	}
	var byID map[string]mistral.SearchResult
	if err := json.Unmarshal(payload, &byID); err != nil {
		return nil
	}
	out := make([]mistral.SearchResult, 0, len(byID))
	for _, r := range byID {
		if r.URL != "" {
			out = append(out, r)
		}
	}
	return out
}

func domainAllowed(rawURL string, tool anthropic.Tool) bool {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return false
	}
	host := strings.ToLower(parsed.Hostname())
	if matchesAny(host, tool.BlockedDomains) {
		return false
	}
	return len(tool.AllowedDomains) == 0 || matchesAny(host, tool.AllowedDomains)
}

func matchesAny(host string, domains []string) bool {
	for _, d := range domains {
		domain := strings.ToLower(d)
		if host == domain || strings.HasSuffix(host, subdomainDot+domain) {
			return true
		}
	}
	return false
}
