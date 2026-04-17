from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from .embeddings import EmbeddingClient
from .schema import RetrievedChunk
from .utils import expand_query_terms, tokenize_for_bm25


class HybridRetriever:
    def __init__(self, persist_dir: Path, collection_name: str = 'tesla_filings') -> None:
        self.persist_dir = Path(persist_dir)
        self.embedding_client = EmbeddingClient()
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_dir / 'chroma'),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_collection(collection_name)
        with open(self.persist_dir / 'bm25.pkl', 'rb') as f:
            payload = pickle.load(f)
        self.bm25 = payload['bm25']
        self.chunks = payload['chunks']
        self.chunk_map = {str(chunk['chunk_id']): chunk for chunk in self.chunks}
        self.chroma_map: dict[str, dict[str, Any]] = {}
        ids = payload.get('ids')
        if ids:
            for cid, chunk in zip(ids, self.chunks):
                self.chroma_map[str(cid)] = chunk

    def search(self, question: str, top_k: int = 10, form_filter: str | None = None, years: list[int] | None = None, chunk_type: str | None = None) -> list[RetrievedChunk]:
        bm25_hits = self._bm25_search(question, top_k=top_k * 3)
        vector_hits = self._vector_search(question, top_k=top_k * 3)

        combined: dict[str, RetrievedChunk] = {}
        for hit in bm25_hits + vector_hits:
            if not self._passes_filter(hit.metadata, form_filter, years, chunk_type):
                continue
            if hit.chunk_id not in combined:
                combined[hit.chunk_id] = hit
            else:
                cur = combined[hit.chunk_id]
                cur.bm25_score = max(cur.bm25_score, hit.bm25_score)
                cur.vector_score = max(cur.vector_score, hit.vector_score)

        max_bm25 = max((x.bm25_score for x in combined.values()), default=1.0) or 1.0
        max_vec = max((x.vector_score for x in combined.values()), default=1.0) or 1.0
        for hit in combined.values():
            year_bonus = 0.0
            filing_year = str(hit.metadata.get('filing_date', ''))[:4]
            if years and filing_year.isdigit() and int(filing_year) in years:
                year_bonus = 0.05
            table_bonus = 0.05 if hit.metadata.get('chunk_type') == 'table' and any(term in question.lower() for term in ['margin', 'revenue', 'cash flow', 'expense', 'sum', 'total']) else 0.0
            hit.final_score = 0.45 * (hit.bm25_score / max_bm25) + 0.45 * (hit.vector_score / max_vec) + year_bonus + table_bonus

        ranked = sorted(combined.values(), key=lambda x: x.final_score, reverse=True)
        return ranked[:top_k]

    def _passes_filter(self, metadata: dict[str, Any], form_filter: str | None, years: list[int] | None, chunk_type: str | None) -> bool:
        if form_filter and metadata.get('form') != form_filter:
            return False
        if chunk_type and metadata.get('chunk_type') != chunk_type:
            return False
        if years:
            filing_date = str(metadata.get('filing_date', ''))
            try:
                if int(filing_date[:4]) not in years:
                    return False
            except ValueError:
                return False
        return True

    def _bm25_search(self, question: str, top_k: int) -> list[RetrievedChunk]:
        expanded = ' '.join(expand_query_terms(question))
        scores = self.bm25.get_scores(tokenize_for_bm25(expanded))
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            RetrievedChunk(
                chunk_id=str(self.chunks[i]['chunk_id']),
                text=self.chunks[i]['text'],
                metadata=self.chunks[i]['metadata'],
                bm25_score=float(scores[i]),
            )
            for i in ranked_idx if scores[i] > 0
        ]

    def _vector_search(self, question: str, top_k: int) -> list[RetrievedChunk]:
        query = f'Represent this financial filing question for retrieval: {question}'
        query_embedding = self.embedding_client.embed([query])[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances'],
        )
        ids = result.get('ids', [[]])[0]
        docs = result.get('documents', [[]])[0]
        metas = result.get('metadatas', [[]])[0]
        distances = result.get('distances', [[1.0] * len(ids)])[0]
        hits: list[RetrievedChunk] = []
        for chunk_id, doc, meta, distance in zip(ids, docs, metas, distances):
            score = max(0.0, 1.0 - float(distance))
            normalized_meta = self._normalize_metadata(meta or {})
            original_chunk = self.chroma_map.get(str(chunk_id))
            if original_chunk is None:
                original_chunk_id = normalized_meta.get('original_chunk_id')
                if original_chunk_id:
                    original_chunk = self.chunk_map.get(str(original_chunk_id))
            if original_chunk is None:
                original_chunk = self.chunk_map.get(str(chunk_id))
            if original_chunk:
                original_meta = original_chunk.get('metadata', {}) or {}
                merged_meta = {**original_meta, **normalized_meta}
                text = original_chunk.get('text', doc)
            else:
                merged_meta = normalized_meta
                text = doc
            hits.append(RetrievedChunk(chunk_id=str(chunk_id), text=text, metadata=merged_meta, vector_score=score))
        return hits

    def _normalize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in metadata.items():
            if isinstance(v, str):
                vv = v.strip()
                if (vv.startswith('[') and vv.endswith(']')) or (vv.startswith('{') and vv.endswith('}')):
                    try:
                        out[k] = json.loads(vv)
                        continue
                    except Exception:
                        pass
                out[k] = v
            else:
                out[k] = v
        return out
