"""Utility helpers for matching natural-language queries to graph vocabulary."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set

from src.processing.chunk import Chunk


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def _generate_aliases(text: str) -> Set[str]:
    aliases = {_normalize(text)}
    cleaned = text.replace("-", " ")
    aliases.add(_normalize(cleaned))
    aliases.add(_normalize(cleaned.replace(" ", "")))
    return {alias for alias in aliases if alias}


MANUAL_ENTITY_ALIASES: Dict[str, str] = {
    # Internet of Things
    "iot": "Internet of things",
    "internet of things": "Internet of things",
    "internetofthings": "Internet of things",
    "smart devices": "Internet of things",
    "connected devices": "Internet of things",
    # Edge Computing / Fog Computing
    "edge computing": "Edge computing",
    "edge ai": "Edge computing",
    "edge analytics": "Edge computing",
    "fog computing": "Fog computing",
    "fog nodes": "Fog computing",
    "edge nodes": "Edge computing",
    # Cloud
    "cloud computing": "Cloud computing",
    "cloud": "Cloud computing",
    "cloud services": "Cloud computing",
    "cloud infrastructure": "Cloud computing",
    # Federated Learning
    "federated learning": "Federated learning",
    "federated training": "Federated learning",
    "fl": "Federated learning",
    "distributed learning": "Federated learning",
    # Knowledge Graphs / Graph Neural Networks
    "knowledge graph": "Knowledge graph",
    "knowledge graphs": "Knowledge graph",
    "kg": "Knowledge graph",
    "gnn": "Graph neural network",
    "gnns": "Graph neural network",
    "graph neural networks": "Graph neural network",
    "graph neural network": "Graph neural network",
    "graph networks": "Graph neural network",
    "gcn": "Graph neural network",
    "graph convolutional": "Graph neural network",
    # Large Language Models
    "llm": "Large language model",
    "llms": "Large language model",
    "large language models": "Large language model",
    "large language model": "Large language model",
    "language models": "Language model",
    "language model": "Language model",
    "gpt": "Large language model",
    "chatgpt": "Large language model",
    "bert": "Large language model",
    "generative ai": "Large language model",
    "generative models": "Large language model",
    # Transformers
    "transformer": "Transformer (deep learning architecture)",
    "transformers": "Transformer (deep learning architecture)",
    "transformer models": "Transformer (deep learning architecture)",
    "transformer architecture": "Transformer (deep learning architecture)",
    "attention mechanism": "Transformer (deep learning architecture)",
    "self attention": "Transformer (deep learning architecture)",
    # NLP
    "nlp": "Natural language processing",
    "natural language processing": "Natural language processing",
    "natural language": "Natural language processing",
    "text processing": "Natural language processing",
    # Augmented/Virtual Reality
    "ar": "Augmented reality",
    "augmented reality": "Augmented reality",
    "mixed reality": "Mixed reality",
    "xr": "Extended reality",
    "vr": "Virtual reality",
    "virtual reality": "Virtual reality",
    # Recommender Systems
    "recommender systems": "Recommender system",
    "recommender system": "Recommender system",
    "recommendation engines": "Recommender system",
    "recommendation system": "Recommender system",
    "recommendations": "Recommender system",
    # Data Mining
    "data mining": "Data mining",
    "data analysis": "Data mining",
    "pattern recognition": "Pattern recognition",
    # Human-Computer Interaction
    "human computer interaction": "Human–computer interaction",
    "hci": "Human–computer interaction",
    "user interface": "Human–computer interaction",
    "ui": "Human–computer interaction",
    "ux": "Human–computer interaction",
    "user experience": "Human–computer interaction",
    # Security
    "cybersecurity": "Computer security",
    "cyber security": "Computer security",
    "computer security": "Computer security",
    "information security": "Computer security",
    "network security": "Computer security",
    "privacy": "Computer security",
    # Cyber-Physical Systems
    "cyber physical systems": "Cyber-physical system",
    "cps": "Cyber-physical system",
    "cyber physical system": "Cyber-physical system",
    # High-Performance Computing
    "hpc": "High-performance computing",
    "high performance computing": "High-performance computing",
    "supercomputing": "Supercomputer",
    "supercomputers": "Supercomputer",
    "supercomputer": "Supercomputer",
    # Parallel/Distributed Computing
    "parallel computing": "Parallel computing",
    "parallel processing": "Parallel computing",
    "distributed systems": "Distributed computing",
    "distributed computing": "Distributed computing",
    # Robotics
    "autonomous robots": "Autonomous robot",
    "autonomous robot": "Autonomous robot",
    "robotics": "Robotics",
    "robots": "Robotics",
    "robot": "Robotics",
    # Hardware
    "microprocessors": "Microprocessor",
    "microprocessor": "Microprocessor",
    "cpu": "Central processing unit",
    "gpu": "Graphics processing unit",
    "gpus": "Graphics processing unit",
    "graphics processing unit": "Graphics processing unit",
    # Operating Systems
    "operating systems": "Operating system",
    "operating system": "Operating system",
    "os": "Operating system",
    # Quantum Computing
    "quantum computing": "Quantum computing",
    "quantum computers": "Quantum computing",
    "quantum computer": "Quantum computing",
    "qubits": "Quantum computing",
    # Biology/Chemistry
    "computational biology": "Computational biology",
    "bioinformatics": "Bioinformatics",
    "computational chemistry": "Computational chemistry",
    # AI / Machine Learning
    "ai": "Artificial intelligence",
    "artificial intelligence": "Artificial intelligence",
    "machine learning": "Machine learning",
    "ml": "Machine learning",
    "deep learning": "Deep learning",
    "neural networks": "Artificial neural network",
    "neural network": "Artificial neural network",
    "ann": "Artificial neural network",
    "cnn": "Convolutional neural network",
    "convolutional neural network": "Convolutional neural network",
    "rnn": "Recurrent neural network",
    "recurrent neural network": "Recurrent neural network",
    "lstm": "Long short-term memory",
    "reinforcement learning": "Reinforcement learning",
    "rl": "Reinforcement learning",
    "supervised learning": "Supervised learning",
    "unsupervised learning": "Unsupervised learning",
    # Embeddings
    "embeddings": "Word embedding",
    "word embeddings": "Word embedding",
    "embedding": "Word embedding",
    "vector representation": "Word embedding",
    # Explainable AI
    "xai": "Explainable artificial intelligence",
    "explainable ai": "Explainable artificial intelligence",
    "explainable artificial intelligence": "Explainable artificial intelligence",
    "interpretable ai": "Explainable artificial intelligence",
    # Training approaches
    "self supervised": "Self-supervised learning",
    "self supervised learning": "Self-supervised learning",
    "pre training": "Pre-training",
    "pretraining": "Pre-training",
    "fine tuning": "Fine-tuning (machine learning)",
    "finetuning": "Fine-tuning (machine learning)",
    # Scaling
    "scaling": "Scalability",
    "scalability": "Scalability",
    "scale": "Scalability",
}

# All canonical entity names that are valid for matching
VALID_ENTITY_NAMES: Set[str] = set(MANUAL_ENTITY_ALIASES.values())

MANUAL_DOC_ALIASES: Dict[str, str] = {
    # Core topics mapping to Wikipedia page titles
    "internet of things": "Internet of things",
    "iot": "Internet of things",
    "edge computing": "Edge computing",
    "fog computing": "Fog computing",
    "cloud computing": "Cloud computing",
    "augmented reality": "Augmented reality",
    "mixed reality": "Mixed reality",
    "virtual reality": "Virtual reality",
    "human computer interaction": "Human–computer interaction",
    "hci": "Human–computer interaction",
    "high performance computing": "High-performance computing",
    "supercomputer": "Supercomputer",
    "operating system": "Operating system",
    "operating systems": "Operating system",
    "artificial intelligence": "Artificial intelligence",
    "ai": "Artificial intelligence",
    "machine learning": "Machine learning",
    "deep learning": "Deep learning",
    "neural network": "Artificial neural network",
    "neural networks": "Artificial neural network",
    "large language model": "Large language model",
    "large language models": "Large language model",
    "llm": "Large language model",
    "llms": "Large language model",
    "transformer": "Transformer (deep learning architecture)",
    "transformers": "Transformer (deep learning architecture)",
    "knowledge graph": "Knowledge graph",
    "graph neural network": "Graph neural network",
    "gnn": "Graph neural network",
    "federated learning": "Federated learning",
    "recommender system": "Recommender system",
    "recommender systems": "Recommender system",
    "data mining": "Data mining",
    "computer security": "Computer security",
    "cybersecurity": "Computer security",
    "cyber physical system": "Cyber-physical system",
    "autonomous robot": "Autonomous robot",
    "robotics": "Robotics",
    "quantum computing": "Quantum computing",
    "natural language processing": "Natural language processing",
    "nlp": "Natural language processing",
    "explainable artificial intelligence": "Explainable artificial intelligence",
    "explainable ai": "Explainable artificial intelligence",
    "xai": "Explainable artificial intelligence",
    "word embedding": "Word embedding",
    "embeddings": "Word embedding",
}

VALID_DOC_TITLES: Set[str] = set(MANUAL_DOC_ALIASES.values())


@dataclass
class EntityMatcher:
    entity_aliases: Dict[str, str]
    doc_aliases: Dict[str, str]

    @classmethod
    def build(cls, graph, chunks: Iterable[Chunk]) -> "EntityMatcher":
        entity_aliases: Dict[str, str] = {}
        for node, data in graph.nodes(data=True):
            if data.get("type") != "entity":
                continue
            name = data.get("name") or node.split("::", 1)[-1]
            for alias in _generate_aliases(name):
                entity_aliases.setdefault(alias, name)
        for alias, canonical in MANUAL_ENTITY_ALIASES.items():
            entity_aliases[_normalize(alias)] = canonical

        doc_aliases: Dict[str, str] = {}
        seen_titles: Set[str] = set()
        for chunk in chunks:
            if chunk.page_title in seen_titles:
                continue
            seen_titles.add(chunk.page_title)
            for alias in _generate_aliases(chunk.page_title):
                doc_aliases.setdefault(alias, chunk.page_title)
        for alias, canonical in MANUAL_DOC_ALIASES.items():
            doc_aliases[_normalize(alias)] = canonical

        return cls(entity_aliases=entity_aliases, doc_aliases=doc_aliases)

    def match_entities(self, candidates: Iterable[str]) -> List[str]:
        """Match candidate strings to canonical entity names."""
        hits: Set[str] = set()
        for cand in candidates:
            norm = _normalize(cand)
            if not norm or len(norm) < 2:
                continue
            canonical = self.entity_aliases.get(norm)
            if canonical:
                hits.add(canonical)
        return sorted(hits)

    def match_docs(self, candidates: Iterable[str]) -> List[str]:
        """Match candidate strings to document titles."""
        hits: Set[str] = set()
        for cand in candidates:
            norm = _normalize(cand)
            if not norm or len(norm) < 2:
                continue
            canonical = self.doc_aliases.get(norm)
            if canonical:
                hits.add(canonical)
        return sorted(hits)


__all__ = ["EntityMatcher"]
