package store

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"go.etcd.io/bbolt"
	"rag/internal/domain"
)

var (
	bucketDocs       = []byte("docs")
	bucketChunks     = []byte("chunks")
	bucketBlobs      = []byte("blobs")
	bucketTerms      = []byte("terms")
	bucketStats      = []byte("stats")
	bucketDocChunks  = []byte("doc_chunks")
	bucketSymbols    = []byte("symbols")
	bucketDocSymbols = []byte("doc_symbols")
	bucketCallGraph  = []byte("callgraph")
	keyStats         = []byte("corpus_stats")
)

type BoltStore struct {
	db *bbolt.DB
}

func NewBoltStore(path string) (*BoltStore, error) {
	db, err := bbolt.Open(path, 0600, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to open bolt db: %w", err)
	}

	err = db.Update(func(tx *bbolt.Tx) error {
		buckets := [][]byte{bucketDocs, bucketChunks, bucketBlobs, bucketTerms, bucketStats, bucketDocChunks, bucketSymbols, bucketDocSymbols, bucketCallGraph}
		for _, b := range buckets {
			if _, err := tx.CreateBucketIfNotExists(b); err != nil {
				return fmt.Errorf("failed to create bucket %s: %w", b, err)
			}
		}
		return nil
	})
	if err != nil {
		db.Close()
		return nil, err
	}

	return &BoltStore{db: db}, nil
}

func (s *BoltStore) DB() *bbolt.DB {
	return s.db
}

type docMeta struct {
	Path    string `json:"path"`
	ModTime int64  `json:"mod_time"`
	Lang    string `json:"lang"`
}

type chunkMeta struct {
	DocID     string   `json:"doc_id"`
	StartLine int      `json:"start_line"`
	EndLine   int      `json:"end_line"`
	Tokens    []string `json:"tokens"`
}

func (s *BoltStore) PutDoc(doc domain.Document) error {
	return s.db.Update(func(tx *bbolt.Tx) error {
		meta := docMeta{
			Path:    doc.Path,
			ModTime: doc.ModTime.Unix(),
			Lang:    doc.Lang,
		}
		data, err := json.Marshal(meta)
		if err != nil {
			return err
		}
		return tx.Bucket(bucketDocs).Put([]byte(doc.ID), data)
	})
}

func (s *BoltStore) GetDoc(id string) (domain.Document, error) {
	var doc domain.Document
	err := s.db.View(func(tx *bbolt.Tx) error {
		data := tx.Bucket(bucketDocs).Get([]byte(id))
		if data == nil {
			return fmt.Errorf("document not found: %s", id)
		}
		var meta docMeta
		if err := json.Unmarshal(data, &meta); err != nil {
			return err
		}
		doc = domain.Document{
			ID:      id,
			Path:    meta.Path,
			ModTime: time.Unix(meta.ModTime, 0),
			Lang:    meta.Lang,
		}
		return nil
	})
	return doc, err
}

func (s *BoltStore) DeleteDoc(id string) error {
	return s.db.Update(func(tx *bbolt.Tx) error {
		return tx.Bucket(bucketDocs).Delete([]byte(id))
	})
}

func (s *BoltStore) ListDocs() ([]domain.Document, error) {
	var docs []domain.Document
	err := s.db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket(bucketDocs)
		return b.ForEach(func(k, v []byte) error {
			var meta docMeta
			if err := json.Unmarshal(v, &meta); err != nil {
				return err
			}
			docs = append(docs, domain.Document{
				ID:      string(k),
				Path:    meta.Path,
				ModTime: time.Unix(meta.ModTime, 0),
				Lang:    meta.Lang,
			})
			return nil
		})
	})
	return docs, err
}

func (s *BoltStore) PutChunk(chunk domain.Chunk) error {
	return s.db.Update(func(tx *bbolt.Tx) error {

		meta := chunkMeta{
			DocID:     chunk.DocID,
			StartLine: chunk.StartLine,
			EndLine:   chunk.EndLine,
			Tokens:    chunk.Tokens,
		}
		data, err := json.Marshal(meta)
		if err != nil {
			return err
		}
		if err := tx.Bucket(bucketChunks).Put([]byte(chunk.ID), data); err != nil {
			return err
		}

		if err := tx.Bucket(bucketBlobs).Put([]byte(chunk.ID), []byte(chunk.Text)); err != nil {
			return err
		}

		docChunks := tx.Bucket(bucketDocChunks)
		var chunkIDs []string
		if existing := docChunks.Get([]byte(chunk.DocID)); existing != nil {
			json.Unmarshal(existing, &chunkIDs)
		}
		chunkIDs = append(chunkIDs, chunk.ID)
		chunkIDsData, _ := json.Marshal(chunkIDs)
		return docChunks.Put([]byte(chunk.DocID), chunkIDsData)
	})
}

func (s *BoltStore) GetChunk(id string) (domain.Chunk, error) {
	var chunk domain.Chunk
	err := s.db.View(func(tx *bbolt.Tx) error {
		data := tx.Bucket(bucketChunks).Get([]byte(id))
		if data == nil {
			return fmt.Errorf("chunk not found: %s", id)
		}
		var meta chunkMeta
		if err := json.Unmarshal(data, &meta); err != nil {
			return err
		}
		text := tx.Bucket(bucketBlobs).Get([]byte(id))
		chunk = domain.Chunk{
			ID:        id,
			DocID:     meta.DocID,
			StartLine: meta.StartLine,
			EndLine:   meta.EndLine,
			Tokens:    meta.Tokens,
			Text:      string(text),
		}
		return nil
	})
	return chunk, err
}

func (s *BoltStore) GetChunksByDoc(docID string) ([]domain.Chunk, error) {
	var chunks []domain.Chunk
	err := s.db.View(func(tx *bbolt.Tx) error {
		docChunks := tx.Bucket(bucketDocChunks)
		data := docChunks.Get([]byte(docID))
		if data == nil {
			return nil
		}
		var chunkIDs []string
		if err := json.Unmarshal(data, &chunkIDs); err != nil {
			return err
		}
		chunkBucket := tx.Bucket(bucketChunks)
		blobBucket := tx.Bucket(bucketBlobs)
		for _, id := range chunkIDs {
			data := chunkBucket.Get([]byte(id))
			if data == nil {
				continue
			}
			var meta chunkMeta
			if err := json.Unmarshal(data, &meta); err != nil {
				continue
			}
			text := blobBucket.Get([]byte(id))
			chunks = append(chunks, domain.Chunk{
				ID:        id,
				DocID:     meta.DocID,
				StartLine: meta.StartLine,
				EndLine:   meta.EndLine,
				Tokens:    meta.Tokens,
				Text:      string(text),
			})
		}
		return nil
	})
	return chunks, err
}

func (s *BoltStore) DeleteChunksByDoc(docID string) error {
	return s.db.Update(func(tx *bbolt.Tx) error {
		docChunks := tx.Bucket(bucketDocChunks)
		data := docChunks.Get([]byte(docID))
		if data == nil {
			return nil
		}
		var chunkIDs []string
		if err := json.Unmarshal(data, &chunkIDs); err != nil {
			return err
		}
		chunkBucket := tx.Bucket(bucketChunks)
		blobBucket := tx.Bucket(bucketBlobs)
		for _, id := range chunkIDs {
			chunkBucket.Delete([]byte(id))
			blobBucket.Delete([]byte(id))
		}
		return docChunks.Delete([]byte(docID))
	})
}

func (s *BoltStore) PutPosting(term string, chunkID string, tf int) error {
	return s.db.Update(func(tx *bbolt.Tx) error {
		b := tx.Bucket(bucketTerms)
		var postings []domain.Posting
		if data := b.Get([]byte(term)); data != nil {
			json.Unmarshal(data, &postings)
		}

		found := false
		for i := range postings {
			if postings[i].ChunkID == chunkID {
				postings[i].TF = tf
				found = true
				break
			}
		}
		if !found {
			postings = append(postings, domain.Posting{ChunkID: chunkID, TF: tf})
		}
		data, err := json.Marshal(postings)
		if err != nil {
			return err
		}
		return b.Put([]byte(term), data)
	})
}

func (s *BoltStore) GetPostings(term string) ([]domain.Posting, error) {
	var postings []domain.Posting
	err := s.db.View(func(tx *bbolt.Tx) error {
		data := tx.Bucket(bucketTerms).Get([]byte(term))
		if data == nil {
			return nil
		}
		return json.Unmarshal(data, &postings)
	})
	return postings, err
}

func (s *BoltStore) DeletePostings(chunkID string, terms []string) error {
	return s.db.Update(func(tx *bbolt.Tx) error {
		b := tx.Bucket(bucketTerms)
		for _, term := range terms {
			data := b.Get([]byte(term))
			if data == nil {
				continue
			}
			var postings []domain.Posting
			if err := json.Unmarshal(data, &postings); err != nil {
				continue
			}

			filtered := make([]domain.Posting, 0, len(postings))
			for _, p := range postings {
				if p.ChunkID != chunkID {
					filtered = append(filtered, p)
				}
			}
			if len(filtered) == 0 {
				b.Delete([]byte(term))
			} else {
				data, _ := json.Marshal(filtered)
				b.Put([]byte(term), data)
			}
		}
		return nil
	})
}

func (s *BoltStore) GetStats() (domain.Stats, error) {
	var stats domain.Stats
	err := s.db.View(func(tx *bbolt.Tx) error {
		data := tx.Bucket(bucketStats).Get(keyStats)
		if data == nil {
			return nil
		}
		return json.Unmarshal(data, &stats)
	})
	return stats, err
}

func (s *BoltStore) UpdateStats(stats domain.Stats) error {
	return s.db.Update(func(tx *bbolt.Tx) error {
		data, err := json.Marshal(stats)
		if err != nil {
			return err
		}
		return tx.Bucket(bucketStats).Put(keyStats, data)
	})
}

func (s *BoltStore) Close() error {
	return s.db.Close()
}

func (s *BoltStore) AllTerms() ([]string, error) {
	var terms []string
	err := s.db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket(bucketTerms)
		return b.ForEach(func(k, v []byte) error {
			terms = append(terms, string(k))
			return nil
		})
	})
	return terms, err
}

type IndexedFile struct {
	Doc      domain.Document
	Chunks   []domain.Chunk
	Postings map[string]map[string]int
}

func (s *BoltStore) BatchIndex(files []IndexedFile) error {
	return s.db.Update(func(tx *bbolt.Tx) error {
		docsBucket := tx.Bucket(bucketDocs)
		chunksBucket := tx.Bucket(bucketChunks)
		blobsBucket := tx.Bucket(bucketBlobs)
		docChunksBucket := tx.Bucket(bucketDocChunks)
		termsBucket := tx.Bucket(bucketTerms)

		allPostings := make(map[string][]domain.Posting)

		for _, file := range files {

			meta := docMeta{
				Path:    file.Doc.Path,
				ModTime: file.Doc.ModTime.Unix(),
				Lang:    file.Doc.Lang,
			}
			data, err := json.Marshal(meta)
			if err != nil {
				return err
			}
			if err := docsBucket.Put([]byte(file.Doc.ID), data); err != nil {
				return err
			}

			chunkIDs := make([]string, 0, len(file.Chunks))
			for _, chunk := range file.Chunks {
				chunkMeta := chunkMeta{
					DocID:     chunk.DocID,
					StartLine: chunk.StartLine,
					EndLine:   chunk.EndLine,
					Tokens:    chunk.Tokens,
				}
				data, err := json.Marshal(chunkMeta)
				if err != nil {
					return err
				}
				if err := chunksBucket.Put([]byte(chunk.ID), data); err != nil {
					return err
				}
				if err := blobsBucket.Put([]byte(chunk.ID), []byte(chunk.Text)); err != nil {
					return err
				}
				chunkIDs = append(chunkIDs, chunk.ID)
			}

			chunkIDsData, _ := json.Marshal(chunkIDs)
			if err := docChunksBucket.Put([]byte(file.Doc.ID), chunkIDsData); err != nil {
				return err
			}

			for term, chunkTFs := range file.Postings {
				for chunkID, tf := range chunkTFs {
					allPostings[term] = append(allPostings[term], domain.Posting{
						ChunkID: chunkID,
						TF:      tf,
					})
				}
			}
		}

		for term, newPostings := range allPostings {
			var existing []domain.Posting
			if data := termsBucket.Get([]byte(term)); data != nil {
				json.Unmarshal(data, &existing)
			}
			existing = append(existing, newPostings...)
			data, err := json.Marshal(existing)
			if err != nil {
				return err
			}
			if err := termsBucket.Put([]byte(term), data); err != nil {
				return err
			}
		}

		return nil
	})
}

func (s *BoltStore) PutSymbols(docID string, symbols []domain.Symbol) error {
	return s.db.Update(func(tx *bbolt.Tx) error {
		symbolBucket := tx.Bucket(bucketSymbols)
		docSymbolsBucket := tx.Bucket(bucketDocSymbols)

		symbolIDs := make([]string, 0, len(symbols))
		for _, sym := range symbols {
			data, err := json.Marshal(sym)
			if err != nil {
				return err
			}
			if err := symbolBucket.Put([]byte(sym.ID), data); err != nil {
				return err
			}
			symbolIDs = append(symbolIDs, sym.ID)
		}

		idsData, err := json.Marshal(symbolIDs)
		if err != nil {
			return err
		}
		return docSymbolsBucket.Put([]byte(docID), idsData)
	})
}

func (s *BoltStore) GetSymbol(id string) (domain.Symbol, error) {
	var sym domain.Symbol
	err := s.db.View(func(tx *bbolt.Tx) error {
		data := tx.Bucket(bucketSymbols).Get([]byte(id))
		if data == nil {
			return fmt.Errorf("symbol not found: %s", id)
		}
		return json.Unmarshal(data, &sym)
	})
	return sym, err
}

func (s *BoltStore) GetSymbolsByDoc(docID string) ([]domain.Symbol, error) {
	var symbols []domain.Symbol
	err := s.db.View(func(tx *bbolt.Tx) error {
		data := tx.Bucket(bucketDocSymbols).Get([]byte(docID))
		if data == nil {
			return nil
		}
		var ids []string
		if err := json.Unmarshal(data, &ids); err != nil {
			return err
		}
		symbolBucket := tx.Bucket(bucketSymbols)
		for _, id := range ids {
			symData := symbolBucket.Get([]byte(id))
			if symData != nil {
				var sym domain.Symbol
				if err := json.Unmarshal(symData, &sym); err == nil {
					symbols = append(symbols, sym)
				}
			}
		}
		return nil
	})
	return symbols, err
}

func (s *BoltStore) DeleteSymbolsByDoc(docID string) error {
	return s.db.Update(func(tx *bbolt.Tx) error {
		docSymbolsBucket := tx.Bucket(bucketDocSymbols)
		data := docSymbolsBucket.Get([]byte(docID))
		if data == nil {
			return nil
		}
		var ids []string
		if err := json.Unmarshal(data, &ids); err != nil {
			return err
		}
		symbolBucket := tx.Bucket(bucketSymbols)
		for _, id := range ids {
			symbolBucket.Delete([]byte(id))
		}
		return docSymbolsBucket.Delete([]byte(docID))
	})
}

func (s *BoltStore) PutCallGraph(docID string, entries []domain.CallGraphEntry) error {
	return s.db.Update(func(tx *bbolt.Tx) error {
		b := tx.Bucket(bucketCallGraph)
		data, err := json.Marshal(entries)
		if err != nil {
			return err
		}
		return b.Put([]byte(docID), data)
	})
}

func (s *BoltStore) GetCallGraph(docID string) ([]domain.CallGraphEntry, error) {
	var entries []domain.CallGraphEntry
	err := s.db.View(func(tx *bbolt.Tx) error {
		data := tx.Bucket(bucketCallGraph).Get([]byte(docID))
		if data == nil {
			return nil
		}
		return json.Unmarshal(data, &entries)
	})
	return entries, err
}

func (s *BoltStore) DeleteCallGraph(docID string) error {
	return s.db.Update(func(tx *bbolt.Tx) error {
		return tx.Bucket(bucketCallGraph).Delete([]byte(docID))
	})
}

func (s *BoltStore) GetAllSymbols() ([]domain.Symbol, error) {
	var symbols []domain.Symbol
	err := s.db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket(bucketSymbols)
		return b.ForEach(func(k, v []byte) error {
			var sym domain.Symbol
			if err := json.Unmarshal(v, &sym); err != nil {
				return nil
			}
			symbols = append(symbols, sym)
			return nil
		})
	})
	return symbols, err
}

func (s *BoltStore) SearchSymbols(query string) ([]domain.Symbol, error) {
	all, err := s.GetAllSymbols()
	if err != nil {
		return nil, err
	}

	var matches []domain.Symbol
	queryLower := strings.ToLower(query)
	for _, sym := range all {
		if strings.Contains(strings.ToLower(sym.Name), queryLower) {
			matches = append(matches, sym)
		}
	}
	return matches, nil
}
