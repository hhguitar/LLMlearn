from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .schema import ChunkRecord
from .utils import ensure_dir, normalize_text, safe_json_dump, safe_json_load, write_jsonl


class SmartChunker:
    def __init__(self, max_chars: int = 1200, overlap: int = 150) -> None:
        self.max_chars = max_chars
        self.overlap = overlap

    def build_chunks_for_filing(self, parsed: dict[str, Any]) -> list[ChunkRecord]:
        filing = parsed['filing']
        out: list[ChunkRecord] = []

        for section in parsed.get('sections', []):
            section_title = section.get('title', 'Untitled Section')
            text = normalize_text(section.get('text', ''))
            if not text:
                continue
            for i, chunk_text in enumerate(self._split_text(text)):
                chunk_id = self._make_chunk_id(filing['accession_no'], 'text', section_title, i)
                out.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        chunk_type='text',
                        text=chunk_text,
                        metadata={
                            'ticker': filing['ticker'],
                            'company_name': filing['company_name'],
                            'form': filing['form'],
                            'filing_date': filing['filing_date'],
                            'report_date': filing.get('report_date'),
                            'accession_no': filing['accession_no'],
                            'source_file': filing.get('local_html_path') or filing.get('local_pdf_path'),
                            'section_title': section_title,
                            'page_start': section.get('page_start'),
                            'page_end': section.get('page_end'),
                            'chunk_type': 'text',
                        },
                    )
                )

        for table in parsed.get('tables', []):
            table_title = table.get('title', 'Untitled Table')
            table_text = f"{table_title}\n{table.get('markdown', '')}"
            chunk_id = self._make_chunk_id(filing['accession_no'], 'table', table_title, 0)
            out.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    chunk_type='table',
                    text=table_text,
                    metadata={
                        'ticker': filing['ticker'],
                        'company_name': filing['company_name'],
                        'form': filing['form'],
                        'filing_date': filing['filing_date'],
                        'report_date': filing.get('report_date'),
                        'accession_no': filing['accession_no'],
                        'source_file': filing.get('local_html_path') or filing.get('local_pdf_path'),
                        'section_title': table.get('section_title'),
                        'table_id': table.get('table_id'),
                        'table_title': table_title,
                        'page': table.get('page'),
                        'chunk_type': 'table',
                        'table_json_rows': table.get('json_rows', []),
                    },
                )
            )
        return out

    def _split_text(self, text: str) -> list[str]:
        if len(text) <= self.max_chars:
            return [text]
        paragraphs = [p.strip() for p in text.split('. ') if p.strip()]
        chunks: list[str] = []
        current = ''
        for paragraph in paragraphs:
            candidate = f'{current}. {paragraph}'.strip('. ').strip() if current else paragraph
            if len(candidate) <= self.max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(paragraph) <= self.max_chars:
                    current = paragraph
                else:
                    start = 0
                    while start < len(paragraph):
                        end = min(len(paragraph), start + self.max_chars)
                        chunks.append(paragraph[start:end])
                        start = max(end - self.overlap, start + 1)
                    current = ''
        if current:
            chunks.append(current)
        return chunks

    def _make_chunk_id(self, accession_no: str, chunk_type: str, title: str, idx: int) -> str:
        key = f'{accession_no}|{chunk_type}|{title}|{idx}'
        digest = hashlib.md5(key.encode('utf-8')).hexdigest()[:12]
        return f'{chunk_type}_{digest}'


def chunk_all(processed_manifest_path: Path, output_path: Path, max_chars: int = 1200, overlap: int = 150) -> list[dict[str, Any]]:
    parsed_docs = safe_json_load(processed_manifest_path)
    chunker = SmartChunker(max_chars=max_chars, overlap=overlap)
    all_chunks: list[dict[str, Any]] = []
    for doc in parsed_docs:
        all_chunks.extend([c.model_dump() for c in chunker.build_chunks_for_filing(doc)])
    ensure_dir(output_path.parent)
    write_jsonl(all_chunks, output_path)
    safe_json_dump({'count': len(all_chunks)}, output_path.with_suffix('.summary.json'))
    return all_chunks
