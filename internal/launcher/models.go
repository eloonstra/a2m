package launcher

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"a2m/internal/mistral"
)

const (
	latestSuffix         = "-latest"
	minAutoCompactWindow = 100000
	maxAutoCompactWindow = 1000000
	tokensPerK           = 1000
	tokensPerM           = tokensPerK * tokensPerK
	contextFormatM       = "%dM context"
	contextFormatK       = "%dk context"
	descriptionSeparator = " · "
	errNoUsableModels    = "no Mistral model supports chat and tool calls"
)

type claudeSettings struct {
	ModelPicker   modelPicker             `json:"modelPicker"`
	ModelSettings map[string]modelSetting `json:"modelSettings"`
}

type modelPicker struct {
	ReplaceBuiltInOptions bool           `json:"replaceBuiltInOptions"`
	Options               []pickerOption `json:"options"`
}

type pickerOption struct {
	Model       string `json:"model"`
	Label       string `json:"label"`
	Description string `json:"description"`
}

type modelSetting struct {
	AutoCompactWindow int `json:"autoCompactWindow"`
}

type modelGroup struct {
	id    string
	model mistral.Model
}

func buildSettings(models []mistral.Model, outputReserve int) (string, int, error) {
	groups := groupModels(models)
	if len(groups) == 0 {
		return "", 0, errors.New(errNoUsableModels)
	}
	settings := claudeSettings{
		ModelPicker:   modelPicker{ReplaceBuiltInOptions: true, Options: make([]pickerOption, 0, len(groups))},
		ModelSettings: make(map[string]modelSetting, len(groups)),
	}
	maxContext := 0
	for _, g := range groups {
		window := g.model.MaxContextLength
		maxContext = max(maxContext, window)
		settings.ModelPicker.Options = append(settings.ModelPicker.Options, pickerOption{
			Model:       g.id,
			Label:       g.id,
			Description: describe(g.model),
		})
		settings.ModelSettings[g.id] = modelSetting{AutoCompactWindow: compactWindow(window, outputReserve)}
	}
	encoded, err := json.Marshal(settings)
	if err != nil {
		return "", 0, err
	}
	return string(encoded), maxContext, nil
}

func compactWindow(contextWindow, outputReserve int) int {
	return min(max(contextWindow-outputReserve, minAutoCompactWindow), maxAutoCompactWindow)
}

func groupModels(models []mistral.Model) []modelGroup {
	groups := make([]modelGroup, 0, len(models))
	index := make(map[string]int, len(models))
	for _, m := range models {
		if !usable(m) {
			continue
		}
		i, seen := index[m.Name]
		if !seen {
			index[m.Name] = len(groups)
			groups = append(groups, modelGroup{id: m.ID, model: m})
			continue
		}
		preferAlias := !strings.HasSuffix(groups[i].id, latestSuffix) && strings.HasSuffix(m.ID, latestSuffix)
		if preferAlias {
			groups[i].id = m.ID
		}
	}
	return groups
}

func usable(m mistral.Model) bool {
	c := m.Capabilities
	return c.CompletionChat && c.FunctionCalling && !c.Audio && m.Deprecation == nil && m.MaxContextLength > 0
}

func describe(m mistral.Model) string {
	context := fmt.Sprintf(contextFormatK, m.MaxContextLength/tokensPerK)
	if m.MaxContextLength >= tokensPerM {
		context = fmt.Sprintf(contextFormatM, m.MaxContextLength/tokensPerM)
	}
	if m.Description == "" {
		return context
	}
	return m.Description + descriptionSeparator + context
}
