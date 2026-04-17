import argparse
from pathlib import Path

from tesla_qa.qa_pipeline import FilingQASystem, parse_years


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('question', type=str)
    parser.add_argument('--form', type=str, default=None, choices=[None, '10-K', '10-Q'])
    args = parser.parse_args()

    qa = FilingQASystem(Path('data/index'))
    response = qa.answer(args.question, form_filter=args.form, years=parse_years(args.question) or None)
    print('\nANSWER\n')
    print(response.answer)
    print('\nCITATIONS\n')
    for c in response.citations:
        print(c)
