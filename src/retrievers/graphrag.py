"""GraphRAG retriever that combines semantic similarity with graph signals."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import networkx as nx
import numpy as np
import spacy
from rank_bm25 import BM25Okapi

from src.graph.entity_matching import EntityMatcher
from src.graph.query import chunk_relation_types, personalized_pagerank
from src.index import search_index
from src.processing.chunk import Chunk
from src.retrievers.types import RetrievalHit


RELATION_WEIGHTS = {"is_a": 1.0, "part_of": 0.7, "founded_by": 0.6}
DEFAULT_RELATION_BONUS = 0.3


def normalize_ppr_scores(ppr_scores: Dict[str, float]) -> Dict[str, float]:
    """Normalize PPR scores to [0, 1] range so they compete with cosine similarity."""
    if not ppr_scores:
        return {}
    values = list(ppr_scores.values())
    min_val, max_val = min(values), max(values)
    if max_val - min_val < 1e-12:
        return {k: 0.5 for k in ppr_scores}
    return {k: (v - min_val) / (max_val - min_val) for k, v in ppr_scores.items()}


def combine_scores(
    sim: float,
    ppr_norm: float,
    bm25_norm: float,
    relation_types: Iterable[str],
    alpha: float = 0.3,
    beta: float = 0.2,
    gamma: float = 0.15,
) -> float:
    """Combine similarity, normalized PPR, BM25, and relation bonus."""
    rel_bonus = 0.0
    for rel in relation_types:
        rel_bonus += RELATION_WEIGHTS.get(rel, DEFAULT_RELATION_BONUS)
    return float(sim) + alpha * float(ppr_norm) + gamma * float(bm25_norm) + beta * rel_bonus


class GraphRAGRetriever:
    def __init__(
        self,
        embedder,
        index,
        graph: nx.MultiDiGraph,
        chunks: List[Chunk],
        candidate_k: int = 50,
        nlp_model: str = "en_core_web_sm",
    ) -> None:
        self.embedder = embedder
        self.index = index
        self.graph = graph
        self.chunks = chunks
        self.candidate_k = candidate_k
        self._nlp = spacy.load(nlp_model)
        self.matcher = EntityMatcher.build(graph, chunks)
        self._title_to_first_chunk = self._build_title_index(chunks)
        self._bm25, self._bm25_corpus = self._build_bm25_index(chunks)

    @staticmethod
    def _build_bm25_index(chunks: List[Chunk]):
        """Build BM25 index over chunk texts for hybrid retrieval."""
        tokenized = [chunk.text.lower().split() for chunk in chunks]
        return BM25Okapi(tokenized), tokenized

    @staticmethod
    def _build_title_index(chunks: List[Chunk]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for chunk in chunks:
            mapping.setdefault(chunk.page_title, chunk.chunk_id)
        return mapping

    def _collect_candidates(self, doc) -> List[str]:
        candidates: List[str] = []
        seen: set[str] = set()
        for ent in doc.ents:
            text = ent.text.strip()
            if text and text not in seen:
                candidates.append(text)
                seen.add(text)
        for chunk in doc.noun_chunks:
            text = chunk.text.strip()
            if text and text not in seen:
                candidates.append(text)
                seen.add(text)
        tokens = [token.text for token in doc if token.is_alpha]
        for token in tokens:
            if token not in seen:
                candidates.append(token)
                seen.add(token)
        significant = [
            token.lemma_.strip()
            for token in doc
            if token.is_alpha and not token.is_stop
        ]
        for n in (2, 3):
            for i in range(len(significant) - n + 1):
                phrase = " ".join(significant[i : i + n])
                if phrase and phrase not in seen:
                    candidates.append(phrase)
                    seen.add(phrase)
        return candidates

    def encode_query(self, query: str):
        return self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.4,
        beta: float = 0.15,
        gamma: float = 0.2,
    ) -> List[RetrievalHit]:
        query_vec = self.encode_query(query)
        distances, indices = search_index(self.index, query_vec, top_k=self.candidate_k)
        candidate_indices = indices[0]
        candidate_scores = distances[0]

        # BM25 scores for all chunks (for hybrid retrieval)
        query_tokens = query.lower().split()
        bm25_scores_all = self._bm25.get_scores(query_tokens)
        bm25_max = max(bm25_scores_all) if max(bm25_scores_all) > 0 else 1.0
        bm25_norm_all = bm25_scores_all / bm25_max  # normalize to [0, 1]

        # Expand candidate set with top BM25 hits (hybrid retrieval)
        bm25_top_indices = np.argsort(bm25_scores_all)[::-1][: self.candidate_k]
        expanded_indices = set(candidate_indices.tolist()) | set(bm25_top_indices.tolist())

        doc = self._nlp(query)
        candidates = self._collect_candidates(doc)
        entity_seeds = self.matcher.match_entities(candidates)
        doc_titles = self.matcher.match_docs(candidates)
        doc_seed_nodes = [
            self._title_to_first_chunk[title]
            for title in doc_titles
            if title in self._title_to_first_chunk
        ]
        ppr_scores = personalized_pagerank(
            self.graph,
            seed_entities=entity_seeds or None,
            seed_nodes=doc_seed_nodes or None,
        )
        if not ppr_scores:
            fallback_titles: List[str] = []
            for idx in list(expanded_indices)[: min(15, len(expanded_indices))]:
                if idx < 0 or idx >= len(self.chunks):
                    continue
                fallback_titles.append(self.chunks[idx].page_title)
            extra_entities = self.matcher.match_entities(fallback_titles)
            extra_doc_titles = self.matcher.match_docs(fallback_titles)
            extra_nodes = [
                self._title_to_first_chunk[title]
                for title in extra_doc_titles
                if title in self._title_to_first_chunk
            ]
            if extra_entities or extra_nodes:
                ppr_scores = personalized_pagerank(
                    self.graph,
                    seed_entities=extra_entities or None,
                    seed_nodes=extra_nodes or None,
                )

        # Normalize PPR scores to [0, 1] range
        ppr_norm = normalize_ppr_scores(ppr_scores)

        # Build similarity lookup for dense scores
        sim_lookup: Dict[int, float] = dict(zip(candidate_indices.tolist(), candidate_scores.tolist()))

        hits: List[RetrievalHit] = []
        for idx in expanded_indices:
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            sim = sim_lookup.get(idx, 0.0)  # 0 if not in dense top-k
            chunk_ppr = ppr_norm.get(chunk.chunk_id, 0.0)
            bm25_score = float(bm25_norm_all[idx])
            relations = chunk_relation_types(self.graph, chunk.chunk_id)
            combined = combine_scores(
                sim, chunk_ppr, bm25_score, relations, alpha=alpha, beta=beta, gamma=gamma
            )
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    score=combined,
                    metadata={
                        "similarity": float(sim),
                        "ppr": float(chunk_ppr),
                        "ppr_raw": float(ppr_scores.get(chunk.chunk_id, 0.0)),
                        "bm25": float(bm25_score),
                        "combined": float(combined),
                    },
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]


__all__ = ["GraphRAGRetriever", "combine_scores"]
