from pathlib import Path

from tesla_qa.indexer import HybridIndexer


if __name__ == '__main__':
    indexer = HybridIndexer(Path('data/index'))
    print(indexer.build(Path('data/processed/chunks.jsonl')))
