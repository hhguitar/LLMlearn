from pathlib import Path

from tesla_qa.parser import parse_all


if __name__ == '__main__':
    parse_all(Path('data/raw/manifest.json'), Path('data/processed'))
