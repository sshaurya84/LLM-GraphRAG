from __future__ import annotations

import json
from pathlib import Path
from typing import List

from src.ingestion import load_corpus, PageRecord


def _first_sentence(text: str) -> str:
    for sentence in text.split("."):
        sentence = sentence.strip()
        if len(sentence) > 20:
            return sentence + "."
    return text.strip()


def generate_eval_pairs(records: List[PageRecord], limit: int = 12) -> List[dict]:
    pairs: List[dict] = []
    for record in records[:limit]:
        question = f"What is {record.title}?"
        answer = _first_sentence(record.summary or record.clean_text)
        pairs.append(
            {
                "question": question,
                "answer": answer,
                "page_title": record.title,
                "source_seed": record.source_seed,
            }
        )
    return pairs


def save_eval_set(pairs: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main(
    raw_path: Path = Path("data/raw/science_tech_pages.jsonl"),
    output_path: Path = Path("data/eval/qa.jsonl"),
    limit: int = 12,
) -> None:
    records = load_corpus(raw_path)
    pairs = generate_eval_pairs(records, limit=limit)
    save_eval_set(pairs, output_path)


if __name__ == "__main__":
    main()

