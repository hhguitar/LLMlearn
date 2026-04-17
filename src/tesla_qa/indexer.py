from __future__ import annotations

import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from .embeddings import EmbeddingClient
from .utils import ensure_dir, read_jsonl, safe_json_dump, tokenize_for_bm25


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for k, v in metadata.items():
        key = str(k)
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[key] = v
        elif isinstance(v, (list, dict)):
            clean[key] = json.dumps(v, ensure_ascii=False)
        else:
            clean[key] = str(v)
    return clean


def build_stable_unique_id(chunk: dict[str, Any], fallback_index: int) -> str:
    metadata = chunk.get('metadata', {}) or {}
    base_id = str(chunk.get('chunk_id') or f'chunk_{fallback_index}')
    filing = metadata.get('accession_no') or metadata.get('source_file') or metadata.get('filename') or 'unknown_filing'
    form_type = metadata.get('form') or 'unknown_form'
    filing_date = metadata.get('filing_date') or 'unknown_date'
    page = metadata.get('page') or metadata.get('page_start') or 'na'
    filing = str(filing).replace('\\', '_').replace('/', '_').replace(' ', '_')
    form_type = str(form_type).replace(' ', '_')
    return f'{filing_date}__{form_type}__p{page}__{base_id}'


def ensure_unique_ids(chunks: list[dict[str, Any]]) -> list[str]:
    proposed_ids = [build_stable_unique_id(chunk, i) for i, chunk in enumerate(chunks)]
    counts = Counter(proposed_ids)
    if not any(v > 1 for v in counts.values()):
        return proposed_ids
    seen: defaultdict[str, int] = defaultdict(int)
    final_ids: list[str] = []
    for pid in proposed_ids:
        seen[pid] += 1
        if counts[pid] == 1:
            final_ids.append(pid)
        else:
            final_ids.append(f'{pid}__dup{seen[pid]}')
    return final_ids


class HybridIndexer:
    def __init__(self, persist_dir: Path) -> None:
        self.persist_dir = ensure_dir(persist_dir)
        self.embedding_client = EmbeddingClient()
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_dir / 'chroma'),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    def build(self, chunks_path: Path, collection_name: str = 'tesla_filings') -> dict[str, Any]:
        chunks = read_jsonl(chunks_path)
        collection = self.chroma_client.get_or_create_collection(name=collection_name, metadata={'hnsw:space': 'cosine'})
        existing = collection.get(include=[])
        existing_ids = existing.get('ids', []) if existing else []
        if existing_ids:
            collection.delete(ids=existing_ids)

        texts = [str(chunk.get('text', '')) for chunk in chunks]
        ids = ensure_unique_ids(chunks)
        metadatas = [sanitize_metadata(chunk.get('metadata', {}) or {}) for chunk in chunks]
        for i, meta in enumerate(metadatas):
            meta['chroma_id'] = ids[i]
            meta['original_chunk_id'] = str(chunks[i].get('chunk_id'))

        embeddings = self.embedding_client.embed(texts)
        batch_size = 64
        for start in range(0, len(chunks), batch_size):
            end = min(start + batch_size, len(chunks))
            collection.add(
                ids=ids[start:end],
                documents=texts[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )

        tokenized = [tokenize_for_bm25(text) for text in texts]
        bm25 = BM25Okapi(tokenized)
        with open(self.persist_dir / 'bm25.pkl', 'wb') as f:
            pickle.dump({'bm25': bm25, 'chunks': chunks, 'ids': ids}, f)

        duplicate_original_ids = len([x for x, c in Counter([c.get('chunk_id') for c in chunks]).items() if c and c > 1])
        summary = {
            'collection_name': collection_name,
            'chunk_count': len(chunks),
            'unique_chroma_ids': len(set(ids)),
            'duplicate_original_chunk_ids_detected': duplicate_original_ids,
        }
        safe_json_dump(summary, self.persist_dir / 'index_summary.json')
        return summary
