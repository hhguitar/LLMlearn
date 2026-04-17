from __future__ import annotations

import argparse
from tesla_qa.evals import run_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="evals/test_questions.jsonl")
    parser.add_argument("--output", default="artifacts/eval_results.jsonl")
    args = parser.parse_args()
    rows = run_eval(args.questions, args.output)
    print(f"Saved {len(rows)} eval results to {args.output}")


if __name__ == "__main__":
    main()
