from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from .schema import FilingRecord
from .settings import settings
from .utils import ensure_dir, logger, safe_json_dump

TESLA_CIK = '1318605'
TESLA_TICKER = 'TSLA'
TESLA_NAME = 'Tesla, Inc.'
SEC_BASE = 'https://www.sec.gov'
DATA_SEC = 'https://data.sec.gov'


class SecDownloader:
    def __init__(self, raw_dir: Path | None = None, user_agent: str | None = None) -> None:
        self.raw_dir = raw_dir or Path(settings.raw_dir)
        self.session = requests.Session()
        self.session.headers.update(
            {
                'User-Agent': user_agent or settings.sec_user_agent,
                'Accept-Encoding': 'gzip, deflate',
                'Host': 'data.sec.gov',
            }
        )

    def _get_json(self, url: str) -> dict:
        response = self.session.get(url, timeout=60)
        response.raise_for_status()
        time.sleep(0.2)
        return response.json()

    def _get_text(self, url: str, host: str = 'www.sec.gov') -> str:
        headers = dict(self.session.headers)
        headers['Host'] = host
        response = self.session.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        time.sleep(0.2)
        return response.text

    def list_filings(self, start_year: int = 2021, end_year: int = 2025) -> list[FilingRecord]:
        submissions = self._get_json(f'{DATA_SEC}/submissions/CIK{TESLA_CIK.zfill(10)}.json')
        recent = submissions['filings']['recent']
        records: list[FilingRecord] = []
        for i, form in enumerate(recent['form']):
            if form not in {'10-K', '10-Q'}:
                continue
            filing_date = recent['filingDate'][i]
            year = int(filing_date[:4])
            if not start_year <= year <= end_year:
                continue
            accession = recent['accessionNumber'][i]
            primary_doc = recent['primaryDocument'][i]
            accession_no_dashless = accession.replace('-', '')
            filing_detail_url = f'{SEC_BASE}/Archives/edgar/data/{TESLA_CIK}/{accession_no_dashless}/{accession}-index.html'
            primary_doc_url = f'{SEC_BASE}/Archives/edgar/data/{TESLA_CIK}/{accession_no_dashless}/{primary_doc}'
            records.append(
                FilingRecord(
                    ticker=TESLA_TICKER,
                    company_name=TESLA_NAME,
                    cik=TESLA_CIK,
                    form=form,
                    accession_no=accession,
                    filing_date=filing_date,
                    report_date=recent['reportDate'][i] if i < len(recent['reportDate']) else None,
                    primary_doc_url=primary_doc_url,
                    filing_detail_url=filing_detail_url,
                )
            )
        records.sort(key=lambda x: (x.filing_date, x.form))
        return records

    def download_filing(self, filing: FilingRecord) -> FilingRecord:
        filing_dir = ensure_dir(self.raw_dir / filing.filing_date / filing.form / filing.accession_no)
        html_path = filing_dir / 'primary.html'
        detail_path = filing_dir / 'filing_index.html'

        if not detail_path.exists():
            detail_path.write_text(self._get_text(filing.filing_detail_url, host='www.sec.gov'), encoding='utf-8')
        if not html_path.exists():
            html_path.write_text(self._get_text(filing.primary_doc_url, host='www.sec.gov'), encoding='utf-8')

        soup = BeautifulSoup(detail_path.read_text(encoding='utf-8'), 'lxml')
        pdf_path = None
        for link in soup.select('a'):
            href = link.get('href')
            if href and href.lower().endswith('.pdf'):
                pdf_url = urljoin(filing.filing_detail_url, href)
                pdf_path = filing_dir / Path(href).name
                if not pdf_path.exists():
                    logger.info('Downloading PDF exhibit for %s', filing.accession_no)
                    content = self.session.get(pdf_url, headers={'User-Agent': settings.sec_user_agent}, timeout=60)
                    if content.ok:
                        pdf_path.write_bytes(content.content)
                break

        filing.local_html_path = str(html_path)
        filing.local_pdf_path = str(pdf_path) if pdf_path else None
        safe_json_dump(filing.model_dump(), filing_dir / 'metadata.json')
        return filing

    def run(self, start_year: int = 2021, end_year: int = 2025) -> list[FilingRecord]:
        records = self.list_filings(start_year, end_year)
        logger.info('Found %s filings', len(records))
        out = [self.download_filing(record) for record in records]
        manifest = [r.model_dump() for r in out]
        safe_json_dump(manifest, self.raw_dir / 'manifest.json')
        return out
