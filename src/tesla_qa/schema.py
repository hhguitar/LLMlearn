from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class FilingRecord(BaseModel):
    ticker: str
    company_name: str
    cik: str
    form: Literal['10-K', '10-Q']
    accession_no: str
    filing_date: str
    report_date: str | None = None
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    primary_doc_url: str
    filing_detail_url: str
    local_html_path: str | None = None
    local_pdf_path: str | None = None


class ParsedSection(BaseModel):
    title: str
    level: int = 1
    page_start: int | None = None
    page_end: int | None = None
    text: str


class TableRecord(BaseModel):
    table_id: str
    title: str
    page: int | None = None
    section_title: str | None = None
    markdown: str
    json_rows: list[dict[str, Any]] = Field(default_factory=list)


class ChunkRecord(BaseModel):
    chunk_id: str
    chunk_type: Literal['text', 'table']
    text: str
    metadata: dict[str, Any]


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    bm25_score: float = 0.0
    vector_score: float = 0.0
    final_score: float = 0.0


class QAResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    retrieved: list[RetrievedChunk]
    reasoning_trace: list[dict[str, Any]] = Field(default_factory=list)
