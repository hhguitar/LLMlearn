from pathlib import Path

from tesla_qa.chunking import chunk_all
from tesla_qa.downloader import SecDownloader
from tesla_qa.indexer import HybridIndexer
from tesla_qa.parser import parse_all


if __name__ == '__main__':
    downloader = SecDownloader()
    downloader.run(start_year=2021, end_year=2025)
    parse_all(Path('data/raw/manifest.json'), Path('data/processed'))
    chunk_all(Path('data/processed/parsed_manifest.json'), Path('data/processed/chunks.jsonl'))
    indexer = HybridIndexer(Path('data/index'))
    print(indexer.build(Path('data/processed/chunks.jsonl')))
