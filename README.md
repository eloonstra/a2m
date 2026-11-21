# a2m

a2m lets you use Mistral models as the provider in Claude Code. It is a local proxy that accepts requests in the Anthropic Messages API format and sends them to the Mistral API. Claude Code talks to the proxy; the proxy talks to Mistral. The name a2m means Anthropic API to Mistral API.

The main use case is to run Claude Code with Mistral models. The proxy also works with other clients that speak the Anthropic Messages API.

## Requirements

- Go 1.27 or newer
- A Mistral API key

The project uses only the Go standard library. It has no third-party dependencies.

## Install

If you have Go, install the latest release into your `GOBIN`:

```sh
go install github.com/eloonstra/a2m@latest
```

## Build from source

```sh
go build -o a2m .
```

Or install it into your `GOBIN` from a local copy:

```sh
go install .
```

## Run as a server

Set your Mistral API key, then start the proxy:

```sh
export MISTRAL_API_KEY=your-key
./a2m
```

The proxy listens on `http://127.0.0.1:8787` by default. Point Claude Code at it:

```sh
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
export ANTHROPIC_AUTH_TOKEN=any-value
claude
```

## Run a command directly

You can also start a command through the proxy in one step:

```sh
MISTRAL_API_KEY=your-key ./a2m claude
```

The proxy starts on a random local port. It creates a random token and passes it to the child command. The proxy stops when the child command exits. When the command is `claude`, the proxy also sets `CLAUDE_CODE_AUTO_MODE_MODEL` to `MISTRAL_AUTO_MODEL`, so auto mode uses that model.

## Configuration

The proxy reads these environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `PROXY_ADDR` | `127.0.0.1:8787` | Listen address of the proxy server |
| `MISTRAL_BASE_URL` | `https://api.mistral.ai` | Base URL of the Mistral API |
| `MISTRAL_API_KEY` | (empty) | Mistral API key for upstream calls |
| `PROXY_TOKEN` | (empty) | If set, clients must send this value as `X-Api-Key` or `Authorization: Bearer` |
| `MISTRAL_MODEL` | `mistral-large-latest` | Main model for requests that name a Claude model |
| `MISTRAL_MEDIUM_MODEL` | `mistral-medium-latest` | Model for names that contain `medium` |
| `MISTRAL_SMALL_MODEL` | `mistral-small-latest` | Model for names that contain `haiku`, and for web search |
| `MISTRAL_AUTO_MODEL` | `ministral-8b-latest` | Model for Claude Code auto mode and for names that contain `auto` |
| `MISTRAL_MAX_TOKENS` | `32768` | Upper limit for `max_tokens` |

Rules for API keys:

- If `MISTRAL_API_KEY` is set on the proxy, the proxy uses it for all upstream calls.
- If `MISTRAL_API_KEY` is not set and `PROXY_TOKEN` is not set, the proxy uses the client's `X-Api-Key` or `Bearer` value as the Mistral key.
- If `PROXY_TOKEN` is set, also set `MISTRAL_API_KEY` on the proxy. Otherwise requests fail.

## How it works

### Endpoint

The proxy offers one endpoint:

- `POST /v1/messages`

It supports streaming (SSE) and non-streaming responses.

### Model mapping

The proxy maps the requested model name:

- Names that contain `auto` map to `MISTRAL_AUTO_MODEL`.
- Names that contain `haiku` map to `MISTRAL_SMALL_MODEL`.
- Names that contain `medium` map to `MISTRAL_MEDIUM_MODEL`.
- Names that contain `claude` map to `MISTRAL_MODEL`.
- Other names pass through unchanged.

### Reasoning effort

If the target model supports reasoning, the proxy sends a `reasoning_effort` value:

- Known models use the closest supported level from a built-in table. The proxy matches the model ID or name.
- Unknown reasoning models receive the requested effort. The default is `high`.
- If thinking is disabled, the proxy sends the lowest "off" level.
- If Mistral rejects the `reasoning_effort` value, the proxy retries the request once without it.

### Caching and timeouts

- The model list from `/v1/models` is cached for 5 minutes. After that, the proxy fetches it again.
- An outbound request fails if the Mistral API does not send response headers within 30 seconds. A streaming response can stay open longer.

### Web search

If a request contains a `web_search_*` tool, the proxy sends it to the Mistral conversations API with `MISTRAL_SMALL_MODEL`. It converts the result back to the Anthropic format.

## Development

Run the tests:

```sh
go test ./...
```

Run static checks:

```sh
go vet ./...
gofmt -l .
```
