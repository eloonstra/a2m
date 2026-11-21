package launcher

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"a2m/internal/mistral"
)

const (
	listenAddr        = "127.0.0.1:0"
	readHeaderTimeout = 10 * time.Second
	exitFailure       = 1
	tokenBytes        = 32

	envBaseURL        = "ANTHROPIC_BASE_URL"
	envAuthToken      = "ANTHROPIC_AUTH_TOKEN"
	envAPIKey         = "ANTHROPIC_API_KEY"
	envAutoModeServer = "CLAUDE_CODE_AUTO_MODE_SERVER"
	envAutoModeModel  = "CLAUDE_CODE_AUTO_MODE_MODEL"
	envAttribution    = "CLAUDE_CODE_ATTRIBUTION_HEADER"
	envOpusModel      = "ANTHROPIC_DEFAULT_OPUS_MODEL"
	envSonnetModel    = "ANTHROPIC_DEFAULT_SONNET_MODEL"
	envHaikuModel     = "ANTHROPIC_DEFAULT_HAIKU_MODEL"
	envMaxContext     = "CLAUDE_CODE_MAX_CONTEXT_TOKENS"
	disabledFlag      = "0"

	claudeCommand     = "claude"
	settingsFlag      = "--settings"
	modelListTimeout  = 5 * time.Second
	warnNoModelList   = "a2m: built-in model list kept: %v\n"
	errMissingKey     = "MISTRAL_API_KEY is required to launch a command"
	errMissingCommand = "no command to launch"
	errMissingToken   = "no proxy token for the launched command"
)

type ModelLister interface {
	ListModels(ctx context.Context, apiKey string) ([]mistral.Model, error)
}

type Options struct {
	APIKey        string
	Token         string
	Model         string
	SmallModel    string
	AutoModel     string
	OutputReserve int
	Models        ModelLister
}

func Run(handler http.Handler, opts Options, args []string) (int, error) {
	if len(args) == 0 {
		return exitFailure, errors.New(errMissingCommand)
	}
	if opts.APIKey == "" {
		return exitFailure, errors.New(errMissingKey)
	}
	if opts.Token == "" {
		return exitFailure, errors.New(errMissingToken)
	}
	defaults := map[string]string{
		envAutoModeServer: disabledFlag,
		envAttribution:    disabledFlag,
		envOpusModel:      opts.Model,
		envSonnetModel:    opts.Model,
		envHaikuModel:     opts.SmallModel,
		envAutoModeModel:  opts.AutoModel,
	}
	isClaude := filepath.Base(args[0]) == claudeCommand
	if isClaude {
		var maxContext int
		args, maxContext = withModelPicker(args, opts)
		if maxContext > 0 {
			defaults[envMaxContext] = strconv.Itoa(maxContext)
		}
	}

	listener, err := net.Listen("tcp", listenAddr)
	if err != nil {
		return exitFailure, err
	}
	server := &http.Server{Handler: handler, ReadHeaderTimeout: readHeaderTimeout}
	go server.Serve(listener)
	defer server.Close()

	cmd := exec.Command(args[0], args[1:]...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	cmd.Env = childEnv("http://"+listener.Addr().String(), opts.Token, defaults)

	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM, syscall.SIGHUP)
	defer signal.Stop(signals)

	if err := cmd.Start(); err != nil {
		return exitFailure, err
	}
	go forwardSignals(signals, cmd.Process)
	return exitCode(cmd.Wait())
}

func withModelPicker(args []string, opts Options) ([]string, int) {
	ctx, cancel := context.WithTimeout(context.Background(), modelListTimeout)
	defer cancel()
	models, err := opts.Models.ListModels(ctx, opts.APIKey)
	if err != nil {
		fmt.Fprintf(os.Stderr, warnNoModelList, err)
		return args, 0
	}
	settings, maxContext, err := buildSettings(models, opts.OutputReserve)
	if err != nil {
		fmt.Fprintf(os.Stderr, warnNoModelList, err)
		return args, 0
	}
	return append([]string{args[0], settingsFlag, settings}, args[1:]...), maxContext
}

func NewToken() (string, error) {
	b := make([]byte, tokenBytes)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}

func childEnv(baseURL, token string, defaults map[string]string) []string {
	overrides := map[string]string{envBaseURL: baseURL, envAuthToken: token}
	parent := os.Environ()
	env := make([]string, 0, len(parent)+len(overrides)+len(defaults))
	for _, kv := range parent {
		key, _, _ := strings.Cut(kv, "=")
		_, isOverridden := overrides[key]
		if isOverridden || key == envAPIKey {
			continue
		}
		delete(defaults, key)
		env = append(env, kv)
	}
	for key, value := range overrides {
		env = append(env, key+"="+value)
	}
	for key, value := range defaults {
		env = append(env, key+"="+value)
	}
	return env
}

func forwardSignals(signals <-chan os.Signal, process *os.Process) {
	for sig := range signals {
		sentToGroupByTerminal := sig == syscall.SIGINT
		if sentToGroupByTerminal {
			continue
		}
		_ = process.Signal(sig)
	}
}

func exitCode(err error) (int, error) {
	if err == nil {
		return 0, nil
	}
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		return exitErr.ExitCode(), nil
	}
	return exitFailure, err
}
