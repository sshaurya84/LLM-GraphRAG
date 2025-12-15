from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.processing.chunk import Chunk


@dataclass
class RetrievalHit:
    chunk: Chunk
    score: float
    metadata: Dict[str, float]
    highlight: Optional[str] = None


__all__ = ["RetrievalHit"]

