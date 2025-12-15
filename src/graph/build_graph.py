"""Graph construction for the heterogeneous entity-document graph."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import spacy

from src.processing.chunk import Chunk

ENTITY_LABELS = {
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "NORP",
    "PRODUCT",
    "EVENT",
    "WORK_OF_ART",
    "LAW",
    "FAC",
    "LANGUAGE",
    "DATE",
    "TIME",
    "PERCENT",
    "MONEY",
    "QUANTITY",
}

_DEF_RELATION_TYPES = {
    "is_a": ["is a", "is an", "was a", "was an"],
    "part_of": ["part of", "component of"],
    "founded_by": ["founded by", "created by", "developed by"],
}


class GraphBuilder:
    def __init__(self, model: str = "en_core_web_sm") -> None:
        self.nlp = spacy.load(model)

    def annotate_entities(self, chunks: Iterable[Chunk]) -> None:
        for chunk in chunks:
            doc = self.nlp(chunk.text)
            entities = {
                ent.text.strip()
                for ent in doc.ents
                if ent.label_ in ENTITY_LABELS and len(ent.text.strip()) > 1
            }
            # Include the source page title as an entity anchor.
            entities.add(chunk.page_title)
            chunk.entities = sorted(entities)

    def _detect_relations(self, chunk: Chunk) -> List[Tuple[str, str, str]]:
        relations: List[Tuple[str, str, str]] = []
        text = chunk.text.lower()
        for rel_type, patterns in _DEF_RELATION_TYPES.items():
            for pattern in patterns:
                if pattern in text:
                    relations.append((rel_type, pattern, chunk.chunk_id))
        return relations

    def build_graph(self, chunks: List[Chunk]) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()

        entity_counts = Counter()
        chunk_entities: Dict[str, List[str]] = {}

        for chunk in chunks:
            chunk_entities[chunk.chunk_id] = chunk.entities
            for entity in chunk.entities:
                entity_counts[entity] += 1

        cooccurrence_counts = defaultdict(int)
        for entities in chunk_entities.values():
            unique_entities = sorted(set(entities))
            for i in range(len(unique_entities)):
                for j in range(i + 1, len(unique_entities)):
                    pair = (unique_entities[i], unique_entities[j])
                    cooccurrence_counts[pair] += 1

        total_chunks = float(len(chunks))
        page_to_chunks: Dict[str, List[Chunk]] = defaultdict(list)
        for chunk in chunks:
            page_to_chunks[chunk.page_title].append(chunk)

        for chunk in chunks:
            graph.add_node(
                chunk.chunk_id,
                type="doc",
                title=chunk.page_title,
                source_seed=chunk.source_seed,
                depth=chunk.depth,
            )
            for entity in chunk.entities:
                entity_node = f"entity::{entity}"
                graph.add_node(entity_node, type="entity", name=entity)
                graph.add_edge(entity_node, chunk.chunk_id, type="mentions")
                graph.add_edge(chunk.chunk_id, entity_node, type="mentions")

        for (entity_a, entity_b), count in cooccurrence_counts.items():
            freq_a = entity_counts[entity_a]
            freq_b = entity_counts[entity_b]
            pmi = math.log((count * total_chunks) / (freq_a * freq_b) + 1e-9)
            graph.add_edge(
                f"entity::{entity_a}",
                f"entity::{entity_b}",
                type="cooccurrence",
                weight=pmi,
            )
            graph.add_edge(
                f"entity::{entity_b}",
                f"entity::{entity_a}",
                type="cooccurrence",
                weight=pmi,
            )

        title_to_first_chunk = {
            title: chunks_by_title[0].chunk_id for title, chunks_by_title in page_to_chunks.items()
        }
        for chunk in chunks:
            for link in chunk.outlinks:
                if link in title_to_first_chunk:
                    graph.add_edge(
                        chunk.chunk_id,
                        title_to_first_chunk[link],
                        type="hyperlink",
                    )

        for chunk in chunks:
            relations = self._detect_relations(chunk)
            for rel_type, _, chunk_id in relations:
                graph.nodes[chunk_id].setdefault("relations", set()).add(rel_type)

        return graph

    @staticmethod
    def save_graph(
        graph: nx.MultiDiGraph,
        graph_path: Path,
        edge_csv_path: Path | None = None,
    ) -> None:
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        with graph_path.open("wb") as f:
            pickle.dump(graph, f)
        if edge_csv_path:
            edge_csv_path.parent.mkdir(parents=True, exist_ok=True)
            with edge_csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["source", "target", "type", "weight"])
                for u, v, data in graph.edges(data=True):
                    writer.writerow([u, v, data.get("type"), data.get("weight", 1.0)])


def build_hetero_graph(
    chunks: List[Chunk],
    graph_path: Path,
    edge_csv_path: Path | None = None,
    nlp_model: str = "en_core_web_sm",
) -> nx.MultiDiGraph:
    builder = GraphBuilder(model=nlp_model)
    builder.annotate_entities(chunks)
    graph = builder.build_graph(chunks)
    builder.save_graph(graph, graph_path, edge_csv_path=edge_csv_path)
    return graph


__all__ = ["GraphBuilder", "build_hetero_graph"]
