package config

import (
	"fmt"
	"os"
	"strconv"

	"a2m/internal/translate"
)

const (
	envAddr        = "PROXY_ADDR"
	envBaseURL     = "MISTRAL_BASE_URL"
	envAPIKey      = "MISTRAL_API_KEY"
	envModel       = "MISTRAL_MODEL"
	envMediumModel = "MISTRAL_MEDIUM_MODEL"
	envSmallModel  = "MISTRAL_SMALL_MODEL"
	envAutoModel   = "MISTRAL_AUTO_MODEL"
	envMaxTokens   = "MISTRAL_MAX_TOKENS"
	envProxyToken  = "PROXY_TOKEN"

	defaultAddr        = "127.0.0.1:8787"
	defaultBaseURL     = "https://api.mistral.ai"
	defaultModel       = "mistral-large-latest"
	defaultMediumModel = "mistral-medium-latest"
	defaultSmallModel  = "mistral-small-latest"
	defaultAutoModel   = "mistral-small-latest"
	defaultMaxTokens   = 32768
)

type Config struct {
	Addr       string
	BaseURL    string
	APIKey     string
	ProxyToken string
	Translate  translate.Options
}

func Load() (Config, error) {
	maxTokens, err := intEnv(envMaxTokens, defaultMaxTokens)
	if err != nil {
		return Config{}, err
	}
	return Config{
		Addr:       stringEnv(envAddr, defaultAddr),
		BaseURL:    stringEnv(envBaseURL, defaultBaseURL),
		APIKey:     os.Getenv(envAPIKey),
		ProxyToken: os.Getenv(envProxyToken),
		Translate: translate.Options{
			Model:       stringEnv(envModel, defaultModel),
			MediumModel: stringEnv(envMediumModel, defaultMediumModel),
			SmallModel:  stringEnv(envSmallModel, defaultSmallModel),
			AutoModel:   stringEnv(envAutoModel, defaultAutoModel),
			MaxTokens:   maxTokens,
		},
	}, nil
}

func stringEnv(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func intEnv(key string, fallback int) (int, error) {
	value := os.Getenv(key)
	if value == "" {
		return fallback, nil
	}
	n, err := strconv.Atoi(value)
	if err != nil {
		return 0, fmt.Errorf("%s: %w", key, err)
	}
	return n, nil
}
