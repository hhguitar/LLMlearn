from pathlib import Path

from tesla_qa.chunking import chunk_all


if __name__ == '__main__':
    chunk_all(Path('data/processed/parsed_manifest.json'), Path('data/processed/chunks.jsonl'))
