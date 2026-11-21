package mistral

import (
	"context"
	"sync"
	"time"
)

const defaultCacheTTL = 5 * time.Minute

type Catalog struct {
	client    *Client
	mu        sync.Mutex
	models    []Model
	byID      map[string]Model
	fetchedAt time.Time
	ttl       time.Duration
}

func NewCatalog(client *Client) *Catalog {
	return &Catalog{client: client, ttl: defaultCacheTTL}
}

func (c *Catalog) ListModels(ctx context.Context, apiKey string) ([]Model, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.fresh(time.Now()) {
		return c.models, nil
	}
	models, err := c.client.ListModels(ctx, apiKey)
	if err != nil {
		return nil, err
	}
	byID := make(map[string]Model, len(models))
	for _, m := range models {
		byID[m.ID] = m
	}
	c.models, c.byID, c.fetchedAt = models, byID, time.Now()
	return models, nil
}

func (c *Catalog) fresh(now time.Time) bool {
	hasCache := c.byID != nil
	return hasCache && now.Sub(c.fetchedAt) < c.ttl
}

func (c *Catalog) Lookup(ctx context.Context, apiKey, id string) (Model, bool) {
	if _, err := c.ListModels(ctx, apiKey); err != nil {
		return Model{}, false
	}
	m, ok := c.byID[id]
	return m, ok
}
