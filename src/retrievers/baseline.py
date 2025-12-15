from __future__ import annotations

from typing import Iterable, List

import numpy as np

from src.embeddings import encode_chunks, load_embedder
from src.index import search_index
from src.processing.chunk import Chunk
from src.retrievers.types import RetrievalHit
from src.utils.device import get_torch_device


class BaselineRetriever:
    def __init__(self, embedder, index, chunks: List[Chunk]) -> None:
        self.embedder = embedder
        self.index = index
        self.chunks = chunks

    @classmethod
    def from_embeddings(
        cls,
        chunks: List[Chunk],
        index,
        embedder=None,
        model_name: str = "BAAI/bge-small-en-v1.5",
    ) -> "BaselineRetriever":
        embedder = embedder or load_embedder(model_name)
        return cls(embedder, index, chunks)

    def encode_query(self, query: str) -> np.ndarray:
        vec = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vec

    def search(self, query: str, top_k: int = 5) -> List[RetrievalHit]:
        query_vec = self.encode_query(query)
        distances, indices = search_index(self.index, query_vec, top_k=top_k)
        hits: List[RetrievalHit] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            hits.append(
                RetrievalHit(
                    chunk=self.chunks[idx],
                    score=float(score),
                    metadata={"similarity": float(score)},
                )
            )
        return hits


__all__ = ["BaselineRetriever"]

