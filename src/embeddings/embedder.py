"""
Embedding utilities with device-aware SentenceTransformer loading.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.processing.chunk import Chunk
from src.utils.device import get_torch_device


def load_embedder(model_name: str = "BAAI/bge-small-en-v1.5") -> SentenceTransformer:
    device = get_torch_device()
    device_str = str(device)
    if device_str not in {"mps", "cuda"}:
        device_str = "cpu"
    return SentenceTransformer(model_name, device=device_str)


def encode_chunks(
    embedder: SentenceTransformer,
    chunks: Iterable[Chunk],
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return embeddings


__all__ = ["load_embedder", "encode_chunks"]

