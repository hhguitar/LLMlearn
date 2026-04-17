from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

import jsonlines
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
)
logger = logging.getLogger('tesla_qa')


QUARTER_MONTH_MAP = {3: 'Q1', 6: 'Q2', 9: 'Q3', 12: 'Q4'}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


NON_WORD_RE = re.compile(r'[^a-zA-Z0-9]+')


def normalize_text(text: str) -> str:
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


FINANCIAL_TERM_SYNONYMS = {
    'free cash flow': ['fcf', 'cash provided by operating activities', 'capital expenditures'],
    'automotive gross margin': ['automotive gross profit', 'automotive revenues', 'cost of automotive revenues'],
    'research and development': ['r&d', 'research development'],
    'china market risk': ['china', 'geopolitical', 'international', 'foreign exchange'],
    'supply chain': ['logistics', 'semiconductor', 'parts shortages', 'shortages'],
}


def expand_query_terms(question: str) -> list[str]:
    q = question.lower()
    out = [question]
    for term, synonyms in FINANCIAL_TERM_SYNONYMS.items():
        if term in q or any(s in q for s in synonyms):
            out.extend([term, *synonyms])
    years = re.findall(r'20\d{2}', question)
    out.extend(years)
    quarters = re.findall(r'Q[1-4]', question, flags=re.I)
    out.extend([q.upper() for q in quarters])
    seen = []
    for item in out:
        if item and item not in seen:
            seen.append(item)
    return seen


NUMBER_RE = re.compile(r'\(?\$?[\d,]+(?:\.\d+)?\)?')


def parse_numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s or s in {'—', '-', 'nan', 'None'}:
        return None
    negative = s.startswith('(') and s.endswith(')')
    s = s.replace('$', '').replace(',', '').replace('(', '').replace(')', '').replace('%', '')
    try:
        val = float(s)
    except ValueError:
        return None
    return -val if negative else val


def safe_json_dump(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def safe_json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with jsonlines.open(path, mode='w') as writer:
        for record in records:
            writer.write(record)



def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with jsonlines.open(path, mode='r') as reader:
        return list(reader)



def markdown_table_from_df(df: pd.DataFrame) -> str:
    clean = df.copy()
    clean.columns = [normalize_text(str(c)) for c in clean.columns]
    clean = clean.fillna('')
    return clean.to_markdown(index=False)



def tokenize_for_bm25(text: str) -> list[str]:
    normalized = NON_WORD_RE.sub(' ', text.lower())
    return [tok for tok in normalized.split() if tok]
