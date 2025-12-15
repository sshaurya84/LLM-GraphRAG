"""
Chunking utilities for turning Wikipedia pages into retrieval-sized passages.

Chunks are tokenized with a transformer tokenizer to target ~512 token windows
with configurable overlap. Output is written as JSONL for downstream components.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from transformers import AutoTokenizer

from src.ingestion.wikipedia import PageRecord


@dataclass
class Chunk:
    chunk_id: str
    page_title: str
    source_seed: str
    depth: int
    text: str
    token_start: int
    token_end: int
    outlinks: List[str]
    entities: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "page_title": self.page_title,
            "source_seed": self.source_seed,
            "depth": self.depth,
            "text": self.text,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "outlinks": self.outlinks,
            "entities": self.entities,
        }


def _generate_chunks(
    record: PageRecord,
    tokenizer: AutoTokenizer,
    window_size: int,
    overlap: int,
) -> Iterator[Chunk]:
    tokens = tokenizer(
        record.clean_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False,
    )

    token_count = len(tokens["input_ids"])
    if token_count == 0:
        return

    offsets = tokens["offset_mapping"]
    text = record.clean_text
    chunk_index = 0
    start = 0
    while start < token_count:
        end = min(start + window_size, token_count)
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1] if end - 1 < len(offsets) else len(text)
        snippet = text[char_start:char_end].strip()
        if snippet:
            chunk_id = f"{record.title.replace(' ', '_')}::chunk-{chunk_index}"
            yield Chunk(
                chunk_id=chunk_id,
                page_title=record.title,
                source_seed=record.source_seed,
                depth=record.depth,
                text=snippet,
                token_start=start,
                token_end=end,
                outlinks=record.outlinks,
                entities=[],
            )
            chunk_index += 1
        if end == token_count:
            break
        start = max(end - overlap, start + 1)


def chunk_records(
    records: Iterable[PageRecord],
    output_path: Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    window_size: int = 512,
    overlap: int = 128,
) -> List[Chunk]:
    """
    Chunk the provided records and save them to ``output_path`` (JSONL).
    Returns the generated chunk list.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunks: List[Chunk] = []
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            for chunk in _generate_chunks(record, tokenizer, window_size, overlap):
                chunks.append(chunk)
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    return chunks


def load_chunks(path: Path) -> List[Chunk]:
    """Load chunk metadata from JSONL."""
    chunks: List[Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            payload = json.loads(line)
            chunks.append(Chunk(**payload))
    return chunks


__all__ = ["Chunk", "chunk_records", "load_chunks"]

