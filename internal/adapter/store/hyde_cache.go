package store

import (
	"crypto/sha256"
	"encoding/hex"

	"go.etcd.io/bbolt"
)

var bucketHyDECache = []byte("hyde_cache")

type HyDECache struct {
	db *bbolt.DB
}

func NewHyDECache(db *bbolt.DB) (*HyDECache, error) {
	err := db.Update(func(tx *bbolt.Tx) error {
		_, err := tx.CreateBucketIfNotExists(bucketHyDECache)
		return err
	})
	if err != nil {
		return nil, err
	}
	return &HyDECache{db: db}, nil
}

func (c *HyDECache) Get(query string) (string, bool) {
	var value string
	found := false

	c.db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket(bucketHyDECache)
		if b == nil {
			return nil
		}
		data := b.Get(hydeKey(query))
		if data == nil {
			return nil
		}
		value = string(data)
		found = true
		return nil
	})

	return value, found
}

func (c *HyDECache) Put(query, hypothetical string) error {
	return c.db.Update(func(tx *bbolt.Tx) error {
		b, err := tx.CreateBucketIfNotExists(bucketHyDECache)
		if err != nil {
			return err
		}
		return b.Put(hydeKey(query), []byte(hypothetical))
	})
}

func hydeKey(query string) []byte {
	sum := sha256.Sum256([]byte(query))
	return []byte(hex.EncodeToString(sum[:16]))
}
