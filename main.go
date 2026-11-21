package main

import (
	"io"
	"log"
	"net/http"
	"os"
	"time"

	"a2m/internal/config"
	"a2m/internal/launcher"
	"a2m/internal/mistral"
	"a2m/internal/proxy"
)

const (
	readHeaderTimeout = 10 * time.Second
	logPrefix         = "a2m: "
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("config: %v", err)
	}
	client := mistral.NewClient(cfg.BaseURL)
	catalog := mistral.NewCatalog(client)

	args := os.Args[1:]
	if len(args) > 0 {
		launch(cfg, client, catalog, args)
		return
	}
	serve(cfg, newHandler(cfg, client, catalog, cfg.ProxyToken))
}

func newHandler(cfg config.Config, client *mistral.Client, catalog *mistral.Catalog, token string) http.Handler {
	return proxy.NewHandler(client, catalog, proxy.Config{
		APIKey:    cfg.APIKey,
		Token:     token,
		Translate: cfg.Translate,
	}).Routes()
}

func launch(cfg config.Config, client *mistral.Client, catalog *mistral.Catalog, args []string) {
	logger := log.New(os.Stderr, logPrefix, 0)
	log.SetOutput(io.Discard)
	token, err := launcher.NewToken()
	if err != nil {
		logger.Fatal(err)
	}
	opts := launcher.Options{
		APIKey:        cfg.APIKey,
		Token:         token,
		Model:         cfg.Translate.Model,
		SmallModel:    cfg.Translate.SmallModel,
		AutoModel:     cfg.Translate.AutoModel,
		OutputReserve: cfg.Translate.MaxTokens,
		Models:        catalog,
	}
	code, err := launcher.Run(newHandler(cfg, client, catalog, token), opts, args)
	if err != nil {
		logger.Print(err)
	}
	os.Exit(code)
}

func serve(cfg config.Config, handler http.Handler) {
	server := &http.Server{Addr: cfg.Addr, Handler: handler, ReadHeaderTimeout: readHeaderTimeout}
	log.Printf("listening on http://%s (model %s, medium model %s, small model %s)", cfg.Addr, cfg.Translate.Model, cfg.Translate.MediumModel, cfg.Translate.SmallModel)
	log.Fatal(server.ListenAndServe())
}
