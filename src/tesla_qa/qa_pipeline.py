from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .llm import LLMClient
from .retriever import HybridRetriever
from .schema import QAResponse, RetrievedChunk


class FilingQASystem:
    def __init__(self, index_dir: Path, collection_name: str = 'tesla_filings') -> None:
        self.retriever = HybridRetriever(index_dir, collection_name=collection_name)
        self.llm = LLMClient()

    def answer(self, question: str, form_filter: str | None = None, years: list[int] | None = None, top_k: int = 10) -> QAResponse:
        reasoning_trace: list[dict[str, Any]] = []
        text_hits = self.retriever.search(question, top_k=max(6, top_k), form_filter=form_filter, years=years, chunk_type='text')
        table_hits = self.retriever.search(question, top_k=max(4, top_k // 2), form_filter=form_filter, years=years, chunk_type='table')
        merged = self._merge_hits(text_hits, table_hits, top_k)

        reasoning_trace.append({'stage': 'retrieve_text', 'count': len(text_hits)})
        reasoning_trace.append({'stage': 'retrieve_table', 'count': len(table_hits)})

        prompt = self._build_prompt(question, merged)
        answer = self.llm.chat(system_prompt=self._system_prompt(), user_prompt=prompt)
        citations = self._build_citations(merged)

        return QAResponse(answer=answer, citations=citations, retrieved=merged, reasoning_trace=reasoning_trace)

    def _merge_hits(self, text_hits: list[RetrievedChunk], table_hits: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        combined: dict[str, RetrievedChunk] = {}
        for hit in text_hits + table_hits:
            if hit.chunk_id not in combined:
                combined[hit.chunk_id] = hit
            else:
                combined[hit.chunk_id].final_score = max(combined[hit.chunk_id].final_score, hit.final_score)
        ranked = sorted(combined.values(), key=lambda x: x.final_score, reverse=True)
        return ranked[:top_k]

    def _build_prompt(self, question: str, retrieved: list[RetrievedChunk]) -> str:
        context_blocks: list[str] = []
        for idx, hit in enumerate(retrieved, start=1):
            meta = hit.metadata
            section_name = meta.get('section_title') or meta.get('table_title') or 'N/A'
            context_blocks.append(
                f"[Evidence {idx}]\n"
                f"Form: {meta.get('form')}\n"
                f"Filing date: {meta.get('filing_date')}\n"
                f"Report date: {meta.get('report_date')}\n"
                f"Section: {section_name}\n"
                f"Chunk type: {meta.get('chunk_type')}\n"
                f"Content:\n{hit.text}\n"
            )
        evidence_text = "\n".join(context_blocks)
        return (
            f"Question:\n{question}\n\n"
            f"Evidence:\n{evidence_text}\n\n"
            "Instructions:\n"
            "1. Answer only from the evidence.\n"
            "2. For cross-document comparison, explicitly align by year/quarter before concluding.\n"
            "3. If numeric calculation is required, show the arithmetic.\n"
            "4. If the evidence is insufficient, say what is missing.\n"
            "5. Cite evidence inline using [Evidence n].\n"
        )

    def _system_prompt(self) -> str:
        return (
            'You are a financial filings QA assistant. '
            'Your job is to answer complex cross-year questions over Tesla 10-K and 10-Q filings. '
            'Be precise about quarters, years, and units. '
            'Prefer exact figures from tables and qualitative explanations from narrative sections. '
            'Never invent data not present in the evidence.'
        )

    def _build_citations(self, retrieved: list[RetrievedChunk]) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        for idx, hit in enumerate(retrieved, start=1):
            meta = hit.metadata
            citations.append(
                {
                    'evidence_id': idx,
                    'form': meta.get('form'),
                    'filing_date': meta.get('filing_date'),
                    'report_date': meta.get('report_date'),
                    'section_title': meta.get('section_title') or meta.get('table_title'),
                    'source_file': meta.get('source_file'),
                    'chunk_type': meta.get('chunk_type'),
                    'chunk_id': hit.chunk_id,
                }
            )
        return citations


def parse_years(question: str) -> list[int]:
    return sorted({int(y) for y in re.findall(r'20\d{2}', question)})
