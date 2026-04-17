from __future__ import annotations

import io
import re
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import pdfplumber
from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning
from pypdf import PdfReader

from .schema import FilingRecord, ParsedSection, TableRecord
from .utils import ensure_dir, logger, markdown_table_from_df, normalize_text, safe_json_dump, safe_json_load

warnings.filterwarnings('ignore', category=XMLParsedAsHTMLWarning)

HEADING_RE = re.compile(r"^(item\s+\d+[a-z]?\.?|management[’']?s discussion|liquidity|results of operations|risk factors|financial statements)", re.I)


class FilingParser:
    def __init__(self, processed_dir: Path) -> None:
        self.processed_dir = processed_dir

    def parse(self, filing: FilingRecord) -> dict[str, Any]:
        if filing.local_html_path:
            parsed = self.parse_html(filing)
        elif filing.local_pdf_path:
            parsed = self.parse_pdf(filing)
        else:
            raise ValueError(f'No local source for filing {filing.accession_no}')

        out_dir = ensure_dir(self.processed_dir / filing.filing_date / filing.form / filing.accession_no)
        safe_json_dump(parsed, out_dir / 'parsed.json')
        return parsed

    def parse_html(self, filing: FilingRecord) -> dict[str, Any]:
        html = Path(filing.local_html_path).read_text(encoding='utf-8', errors='ignore')
        soup = BeautifulSoup(html, 'lxml')

        for tag in soup(['script', 'style', 'ix:header', 'header', 'footer']):
            tag.decompose()

        sections = self._extract_sections_from_html(soup)
        tables = self._extract_tables_from_html(soup)

        return {
            'filing': filing.model_dump(),
            'sections': [s.model_dump() for s in sections],
            'tables': [t.model_dump() for t in tables],
        }

    def _extract_sections_from_html(self, soup: BeautifulSoup) -> list[ParsedSection]:
        body = soup.body or soup
        blocks: list[ParsedSection] = []
        current_title = 'Document Overview'
        current_text: list[str] = []

        def flush() -> None:
            nonlocal current_text, current_title
            text = normalize_text(' '.join(current_text))
            if len(text) > 80:
                blocks.append(ParsedSection(title=current_title, text=text, level=1))
            current_text = []

        for node in body.descendants:
            if not isinstance(node, Tag):
                continue
            name = node.name.lower()
            if name in {'h1', 'h2', 'h3', 'b', 'strong'}:
                title = normalize_text(node.get_text(' ', strip=True))
                if title and (len(title) < 200 and HEADING_RE.search(title)):
                    flush()
                    current_title = title
                    continue
            if name in {'p', 'div', 'span'}:
                text = normalize_text(node.get_text(' ', strip=True))
                if text and len(text) > 40:
                    current_text.append(text)
        flush()
        return self._dedupe_sections(blocks)

    def _dedupe_sections(self, sections: list[ParsedSection]) -> list[ParsedSection]:
        out: list[ParsedSection] = []
        seen = set()
        for section in sections:
            key = (section.title[:100], section.text[:200])
            if key in seen:
                continue
            seen.add(key)
            out.append(section)
        return out

    def _extract_tables_from_html(self, soup: BeautifulSoup) -> list[TableRecord]:
        tables: list[TableRecord] = []
        html_tables = soup.find_all('table')
        for idx, table in enumerate(html_tables):
            try:
                dfs = pd.read_html(io.StringIO(str(table)))
            except ValueError:
                continue
            if not dfs:
                continue
            df = dfs[0]
            if df.empty or (df.shape[0] < 2 and df.shape[1] < 2):
                continue
            title = self._infer_table_title(table) or f'Table {idx + 1}'
            df = df.copy()
            df.columns = [normalize_text(str(c)) for c in df.columns]
            markdown = markdown_table_from_df(df)
            json_rows = [{str(k): v for k, v in row.items()} for row in df.fillna('').to_dict(orient='records')]
            tables.append(
                TableRecord(
                    table_id=f'html_table_{idx + 1}',
                    title=title,
                    markdown=markdown,
                    json_rows=json_rows,
                )
            )
        return tables

    def _infer_table_title(self, table: Tag) -> str | None:
        prev = table.find_previous(['p', 'div', 'strong', 'b'])
        if prev:
            title = normalize_text(prev.get_text(' ', strip=True))
            if 3 <= len(title) <= 180:
                return title
        caption = table.find('caption')
        if caption:
            title = normalize_text(caption.get_text(' ', strip=True))
            if title:
                return title
        return None

    def parse_pdf(self, filing: FilingRecord) -> dict[str, Any]:
        pdf_path = Path(filing.local_pdf_path)
        reader = PdfReader(str(pdf_path))
        text_pages = [page.extract_text() or '' for page in reader.pages]
        sections = [
            ParsedSection(title=f'Page {idx + 1}', page_start=idx + 1, page_end=idx + 1, text=normalize_text(text))
            for idx, text in enumerate(text_pages)
            if normalize_text(text)
        ]

        tables: list[TableRecord] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                try:
                    page_tables = page.extract_tables() or []
                except Exception as exc:
                    logger.warning('PDF table extraction failed on page %s: %s', page_idx + 1, exc)
                    page_tables = []
                for tbl_idx, tbl in enumerate(page_tables):
                    df = pd.DataFrame(tbl[1:], columns=tbl[0]) if tbl and tbl[0] else pd.DataFrame(tbl)
                    if df.empty:
                        continue
                    df = df.copy()
                    df.columns = [normalize_text(str(c)) for c in df.columns]
                    tables.append(
                        TableRecord(
                            table_id=f'pdf_p{page_idx + 1}_t{tbl_idx + 1}',
                            title=f'PDF Table page {page_idx + 1}',
                            page=page_idx + 1,
                            markdown=markdown_table_from_df(df),
                            json_rows=[{str(k): v for k, v in row.items()} for row in df.fillna('').to_dict(orient='records')],
                        )
                    )

        return {
            'filing': filing.model_dump(),
            'sections': [s.model_dump() for s in sections],
            'tables': [t.model_dump() for t in tables],
        }


def parse_all(raw_manifest_path: Path, processed_dir: Path) -> list[dict[str, Any]]:
    manifest = safe_json_load(raw_manifest_path)
    parser = FilingParser(processed_dir=processed_dir)
    parsed = []
    for item in manifest:
        filing = FilingRecord(**item)
        parsed.append(parser.parse(filing))
    safe_json_dump(parsed, processed_dir / 'parsed_manifest.json')
    return parsed
