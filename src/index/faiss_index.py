"""
FAISS index helpers for dense retrieval.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import faiss
import numpy as np


def build_faiss_index(
    embeddings: np.ndarray,
    output_path: Path,
    metric: str = "ip",
) -> faiss.Index:
    """
    Build and persist a FAISS index for the embeddings.

    Parameters
    ----------
    embeddings:
        2D numpy array (num_chunks, dim).
    output_path:
        File location to save the index.
    metric:
        Either ``"ip"`` (inner product) or ``"l2"``.
    """
    dim = embeddings.shape[1]
    if metric == "ip":
        index = faiss.IndexFlatIP(dim)
    elif metric == "l2":
        index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError(f"Unsupported FAISS metric: {metric}")
    index.add(embeddings.astype(np.float32))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_path))
    return index


def load_faiss_index(path: Path) -> faiss.Index:
    return faiss.read_index(str(path))


def search_index(
    index: faiss.Index,
    query_embeddings: np.ndarray,
    top_k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    distances, indices = index.search(query_embeddings.astype(np.float32), top_k)
    return distances, indices


__all__ = ["build_faiss_index", "load_faiss_index", "search_index"]

