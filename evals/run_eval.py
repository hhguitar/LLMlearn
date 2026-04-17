from __future__ import annotations

import json
from pathlib import Path

from tesla_qa.qa_pipeline import FilingQASystem, parse_years


QUESTIONS_PATH = Path('evals/test_questions.json')
OUTPUT_PATH = Path('evals/eval_results.json')


if __name__ == '__main__':
    qa = FilingQASystem(Path('data/index'))
    questions = json.loads(QUESTIONS_PATH.read_text(encoding='utf-8'))
    results = []
    for item in questions:
        response = qa.answer(item['question'], form_filter=item.get('form_filter'), years=parse_years(item['question']) or None)
        results.append(
            {
                'id': item['id'],
                'question': item['question'],
                'gold_hint': item.get('gold_hint'),
                'answer': response.answer,
                'citations': response.citations,
            }
        )
    OUTPUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote {OUTPUT_PATH}')
