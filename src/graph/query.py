"""
Graph query utilities including Personalized PageRank over the heterogenous graph.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx


def _aggregate_graph(graph: nx.MultiDiGraph) -> nx.DiGraph:
    """Collapse multi-edges by summing weights."""
    digraph = nx.DiGraph()
    for u, v, data in graph.edges(data=True):
        weight = data.get("weight", 1.0)
        if digraph.has_edge(u, v):
            digraph[u][v]["weight"] += weight
        else:
            digraph.add_edge(u, v, weight=weight)
    return digraph


def personalized_pagerank(
    graph: nx.MultiDiGraph,
    seed_entities: Optional[Iterable[str]] = None,
    seed_nodes: Optional[Iterable[str]] = None,
    damping: float = 0.85,
) -> Dict[str, float]:
    """
    Run PPR on the collapsed graph seeded by entity nodes.

    Parameters
    ----------
    graph:
        The heterogeneous graph.
    seed_entities:
        Entity names (not prefixed). They will be mapped to ``entity::`` nodes.
    """
    digraph = _aggregate_graph(graph)
    personalization: Dict[str, float] = {}
    seeds: List[str] = []
    if seed_entities:
        for entity in seed_entities:
            node = f"entity::{entity}"
            if node in digraph:
                seeds.append(node)
    if seed_nodes:
        for node in seed_nodes:
            if node in digraph:
                seeds.append(node)
    if not seeds:
        return {}
    weight = 1.0 / len(seeds)
    for node in seeds:
        personalization[node] = weight
    scores = nx.pagerank(
        digraph,
        alpha=damping,
        personalization=personalization,
        weight="weight",
    )
    return scores


def chunk_relation_types(graph: nx.MultiDiGraph, chunk_id: str) -> List[str]:
    node_data = graph.nodes.get(chunk_id, {})
    relations = node_data.get("relations")
    if isinstance(relations, set):
        return sorted(relations)
    if isinstance(relations, list):
        return sorted(set(relations))
    return []


__all__ = ["personalized_pagerank", "chunk_relation_types"]

