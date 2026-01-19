package store

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"

	"go.etcd.io/bbolt"
	"rag/config"
)

// CurrentSchemaVersion is the current schema version.
// Increment this when making breaking changes to the storage format.
const CurrentSchemaVersion = 2

var (
	keySchemaVersion = []byte("schema_version")
	keyConfigHash    = []byte("config_hash")
)

// SchemaInfo stores schema version and configuration hash.
type SchemaInfo struct {
	Version    int    `json:"version"`
	ConfigHash string `json:"config_hash"`
}

// GetSchemaInfo retrieves the current schema info from the database.
func (s *BoltStore) GetSchemaInfo() (*SchemaInfo, error) {
	var info SchemaInfo
	err := s.db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket(bucketStats)
		if b == nil {
			return nil
		}

		versionData := b.Get(keySchemaVersion)
		if versionData != nil {
			if err := json.Unmarshal(versionData, &info.Version); err != nil {

				info.Version = 1
			}
		}

		hashData := b.Get(keyConfigHash)
		if hashData != nil {
			info.ConfigHash = string(hashData)
		}

		return nil
	})
	return &info, err
}

// SetSchemaInfo stores the schema info in the database.
func (s *BoltStore) SetSchemaInfo(info *SchemaInfo) error {
	return s.db.Update(func(tx *bbolt.Tx) error {
		b := tx.Bucket(bucketStats)

		versionData, err := json.Marshal(info.Version)
		if err != nil {
			return err
		}
		if err := b.Put(keySchemaVersion, versionData); err != nil {
			return err
		}

		return b.Put(keyConfigHash, []byte(info.ConfigHash))
	})
}

// ComputeConfigHash computes a hash of index-relevant configuration.
// Changes to this hash indicate the index should be rebuilt.
func ComputeConfigHash(cfg *config.Config) string {

	relevant := struct {
		Stemming     bool    `json:"stemming"`
		ChunkTokens  int     `json:"chunk_tokens"`
		ChunkOverlap int     `json:"chunk_overlap"`
		K1           float64 `json:"k1"`
		B            float64 `json:"b"`
		ASTChunking  bool    `json:"ast_chunking"`
		EmbEnabled   bool    `json:"emb_enabled"`
		EmbProvider  string  `json:"emb_provider"`
		EmbModel     string  `json:"emb_model"`
	}{
		Stemming:     cfg.Index.Stemming,
		ChunkTokens:  cfg.Index.ChunkTokens,
		ChunkOverlap: cfg.Index.ChunkOverlap,
		K1:           cfg.Index.K1,
		B:            cfg.Index.B,
		ASTChunking:  cfg.Index.ASTChunking,
		EmbEnabled:   cfg.Embedding.Enabled,
		EmbProvider:  cfg.Embedding.Provider,
		EmbModel:     cfg.Embedding.Model,
	}

	data, _ := json.Marshal(relevant)
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:8])
}

// MigrationResult describes the result of a migration check.
type MigrationResult struct {
	NeedsMigration bool
	NeedsRebuild   bool
	OldVersion     int
	NewVersion     int
	Reason         string
}

// CheckMigration checks if migration or rebuild is needed.
func (s *BoltStore) CheckMigration(cfg *config.Config) (*MigrationResult, error) {
	info, err := s.GetSchemaInfo()
	if err != nil {
		return nil, fmt.Errorf("failed to get schema info: %w", err)
	}

	result := &MigrationResult{
		OldVersion: info.Version,
		NewVersion: CurrentSchemaVersion,
	}

	if info.Version == 0 {

		result.NeedsMigration = true
		result.Reason = "initializing schema version"
	} else if info.Version < CurrentSchemaVersion {

		result.NeedsMigration = true
		result.Reason = fmt.Sprintf("schema upgrade from v%d to v%d", info.Version, CurrentSchemaVersion)
	} else if info.Version > CurrentSchemaVersion {

		result.NeedsRebuild = true
		result.Reason = fmt.Sprintf("database created by newer version (v%d > v%d)", info.Version, CurrentSchemaVersion)
		return result, nil
	}

	newHash := ComputeConfigHash(cfg)
	if info.ConfigHash != "" && info.ConfigHash != newHash {
		result.NeedsRebuild = true
		result.Reason = "index configuration changed"
	}

	return result, nil
}

// Migrate performs any necessary schema migrations.
func (s *BoltStore) Migrate(cfg *config.Config) error {
	info, err := s.GetSchemaInfo()
	if err != nil {
		return err
	}

	for v := info.Version; v < CurrentSchemaVersion; v++ {
		if err := s.runMigration(v, v+1); err != nil {
			return fmt.Errorf("migration from v%d to v%d failed: %w", v, v+1, err)
		}
	}

	newInfo := &SchemaInfo{
		Version:    CurrentSchemaVersion,
		ConfigHash: ComputeConfigHash(cfg),
	}
	return s.SetSchemaInfo(newInfo)
}

// runMigration runs a specific version migration.
func (s *BoltStore) runMigration(from, to int) error {
	switch {
	case from == 0 && to == 1:

		return nil
	case from == 1 && to == 2:

		return s.db.Update(func(tx *bbolt.Tx) error {
			_, err := tx.CreateBucketIfNotExists(bucketDocChunks)
			return err
		})
	default:

		return nil
	}
}

// Clear removes all data from the database (for rebuild).
func (s *BoltStore) Clear() error {
	return s.db.Update(func(tx *bbolt.Tx) error {
		buckets := [][]byte{bucketDocs, bucketChunks, bucketBlobs, bucketTerms, bucketDocChunks}
		for _, name := range buckets {
			b := tx.Bucket(name)
			if b == nil {
				continue
			}

			c := b.Cursor()
			for k, _ := c.First(); k != nil; k, _ = c.Next() {
				if err := b.Delete(k); err != nil {
					return err
				}
			}
		}

		statsBucket := tx.Bucket(bucketStats)
		if statsBucket != nil {
			c := statsBucket.Cursor()
			for k, _ := c.First(); k != nil; k, _ = c.Next() {

				if string(k) != string(keySchemaVersion) && string(k) != string(keyConfigHash) {
					if err := statsBucket.Delete(k); err != nil {
						return err
					}
				}
			}
		}

		return nil
	})
}

// NeedsRebuild checks if the index needs a full rebuild due to config changes.
func (s *BoltStore) NeedsRebuild(cfg *config.Config) (bool, string, error) {
	result, err := s.CheckMigration(cfg)
	if err != nil {
		return false, "", err
	}
	return result.NeedsRebuild, result.Reason, nil
}
