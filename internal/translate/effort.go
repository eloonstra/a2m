package translate

import (
	"strings"

	"a2m/internal/anthropic"
	"a2m/internal/mistral"
)

const (
	effortNone    = "none"
	effortMinimal = "minimal"
	effortLow     = "low"
	effortMedium  = "medium"
	effortHigh    = "high"
	effortXHigh   = "xhigh"
	effortMax     = "max"

	defaultEffort = effortHigh
)

var effortRank = map[string]int{
	effortNone:    0,
	effortMinimal: 1,
	effortLow:     2,
	effortMedium:  3,
	effortHigh:    4,
	effortXHigh:   5,
	effortMax:     6,
}

var allEfforts = []string{effortNone, effortMinimal, effortLow, effortMedium, effortHigh, effortXHigh, effortMax}

var reasoningLevels = []struct {
	namePrefix string
	levels     []string
}{
	{"mistral-small-", []string{effortNone, effortHigh}},
	{"mistral-medium-", []string{effortNone, effortHigh}},
	{"mistral-large-4", []string{effortLow, effortMedium, effortHigh}},
	{"zai-glm-5-3", []string{effortLow, effortHigh, effortMax}},
	{"glm-5-2", allEfforts},
	{"deepseek-v4", []string{effortNone, effortLow, effortHigh, effortXHigh, effortMax}},
}

func reasoningEffort(model mistral.Model, thinking *anthropic.Thinking, cfg *anthropic.OutputConfig) string {
	if !model.Capabilities.Reasoning {
		return ""
	}
	levels := supportedLevels(model)
	unknownModel := len(levels) == 0
	if unknownModel {
		levels = allEfforts
	}
	if !thinkingEnabled(thinking) {
		return lowestOff(levels)
	}
	return nearestLevel(levels, requestedEffort(cfg))
}

func supportedLevels(model mistral.Model) []string {
	names := []string{strings.ToLower(model.ID), strings.ToLower(model.Name)}
	for _, name := range names {
		if levels := levelsForPrefix(name); levels != nil {
			return levels
		}
	}
	return nil
}

func levelsForPrefix(name string) []string {
	for _, entry := range reasoningLevels {
		if strings.HasPrefix(name, entry.namePrefix) {
			return entry.levels
		}
	}
	return nil
}

func lowestOff(levels []string) string {
	if levels[0] == effortNone {
		return effortNone
	}
	return ""
}

func requestedEffort(cfg *anthropic.OutputConfig) string {
	if cfg == nil {
		return defaultEffort
	}
	if _, known := effortRank[cfg.Effort]; !known {
		return defaultEffort
	}
	return cfg.Effort
}

func nearestLevel(levels []string, want string) string {
	target := effortRank[want]
	best, bestDistance := "", -1
	for _, level := range levels {
		if level == effortNone {
			continue
		}
		distance := abs(effortRank[level] - target)
		isCloserOrHigherTie := bestDistance < 0 || distance <= bestDistance
		if isCloserOrHigherTie {
			best, bestDistance = level, distance
		}
	}
	return best
}

func abs(n int) int {
	if n < 0 {
		return -n
	}
	return n
}
