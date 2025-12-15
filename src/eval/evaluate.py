from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rapidfuzz.distance import LCSseq

from src.embeddings import load_embedder
from src.index import load_faiss_index
from src.processing import load_chunks
from src.processing.chunk import Chunk
from src.retrievers import BaselineRetriever, GraphRAGRetriever


@dataclass
class EvalExample:
    question: str
    answer: str
    page_title: str = ""
    source_seed: str = ""
    supporting_pages: Optional[List[str]] = None


def load_eval_file(path: Path) -> List[EvalExample]:
    examples: List[EvalExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            payload = json.loads(line)
            examples.append(EvalExample(**payload))
    return examples


def recall_at_k(relevant: Sequence[int], retrieved: Sequence[int], k: int) -> float:
    rel_set = set(relevant)
    hits = sum(1 for idx in retrieved[:k] if idx in rel_set)
    return hits / max(len(rel_set), 1)


def reciprocal_rank(relevant: Sequence[int], retrieved: Sequence[int]) -> float:
    rel_set = set(relevant)
    for rank, idx in enumerate(retrieved, start=1):
        if idx in rel_set:
            return 1.0 / rank
    return 0.0


def rouge_l(prediction: str, reference: str) -> float:
    lcs = LCSseq.distance(prediction, reference)
    max_len = max(len(prediction), len(reference))
    if max_len == 0:
        return 0.0
    lcs_len = max_len - lcs
    return lcs_len / max_len


def prepare_retrievers(
    chunks_path: Path,
    index_path: Path,
    graph_path: Path,
    model_name: str = "BAAI/bge-small-en-v1.5",
) -> Tuple[BaselineRetriever, Optional[GraphRAGRetriever], List[Chunk]]:
    embedder = load_embedder(model_name)
    chunks = load_chunks(chunks_path)
    index = load_faiss_index(index_path)
    baseline = BaselineRetriever(embedder, index, chunks)
    graph_retriever: Optional[GraphRAGRetriever] = None
    if graph_path.exists():
        with graph_path.open("rb") as f:
            graph = pickle.load(f)
        graph_retriever = GraphRAGRetriever(embedder, index, graph, chunks)
    return baseline, graph_retriever, chunks


def evaluate_retrievers(
    examples: List[EvalExample],
    baseline: BaselineRetriever,
    graph_retriever: Optional[GraphRAGRetriever],
    chunks: List[Chunk],
    k_values: Iterable[int] = (5, 10),
    alpha: float = 0.3,
    beta: float = 0.2,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {"baseline": {}, "graphrag": {}}
    chunk_index = {chunk.chunk_id: idx for idx, chunk in enumerate(chunks)}
    title_to_indices: Dict[str, List[int]] = {}
    for idx, chunk in enumerate(chunks):
        title_to_indices.setdefault(chunk.page_title, []).append(idx)

    def _target_indices(example: EvalExample) -> List[int]:
        titles: List[str] = []
        if example.page_title:
            titles.append(example.page_title)
        if example.supporting_pages:
            titles.extend(example.supporting_pages)
        indices: List[int] = []
        for title in titles:
            indices.extend(title_to_indices.get(title, []))
        return sorted(set(indices))

    for k in k_values:
        baseline_recalls: List[float] = []
        baseline_mrrs: List[float] = []
        graphrag_recalls: List[float] = []
        graphrag_mrrs: List[float] = []

        for example in examples:
            hits = baseline.search(example.question, top_k=k)
            retrieved_indices = [chunk_index[hit.chunk.chunk_id] for hit in hits]
            target_indices = _target_indices(example)
            baseline_recalls.append(recall_at_k(target_indices, retrieved_indices, k))
            baseline_mrrs.append(reciprocal_rank(target_indices, retrieved_indices))

            if graph_retriever:
                g_hits = graph_retriever.search(example.question, top_k=k, alpha=alpha, beta=beta)
                g_indices = [chunk_index[hit.chunk.chunk_id] for hit in g_hits]
                graphrag_recalls.append(recall_at_k(target_indices, g_indices, k))
                graphrag_mrrs.append(reciprocal_rank(target_indices, g_indices))

        results["baseline"][f"Recall@{k}"] = float(np.mean(baseline_recalls))
        results["baseline"][f"MRR@{k}"] = float(np.mean(baseline_mrrs))
        if graph_retriever and graphrag_recalls:
            results["graphrag"][f"Recall@{k}"] = float(np.mean(graphrag_recalls))
            results["graphrag"][f"MRR@{k}"] = float(np.mean(graphrag_mrrs))

    return results


def collect_examples(
    examples: List[EvalExample],
    baseline: BaselineRetriever,
    graph_retriever: Optional[GraphRAGRetriever],
    num_examples: int = 5,
    top_k: int = 5,
    alpha: float = 0.3,
    beta: float = 0.2,
) -> List[Dict[str, object]]:
    collected: List[Dict[str, object]] = []
    subset = examples[:num_examples]
    for example in subset:
        baseline_hits = baseline.search(example.question, top_k=top_k)
        baseline_rouge = (
            rouge_l(baseline_hits[0].chunk.text, example.answer) if baseline_hits else 0.0
        )
        graphrag_hits = (
            graph_retriever.search(example.question, top_k=top_k, alpha=alpha, beta=beta)
            if graph_retriever
            else []
        )
        graphrag_rouge = (
            rouge_l(graphrag_hits[0].chunk.text, example.answer) if graphrag_hits else 0.0
        )
        collected.append(
            {
                "question": example.question,
                "answer": example.answer,
                "baseline_rougeL": baseline_rouge,
                "graphrag_rougeL": graphrag_rouge,
                "baseline": [
                    {
                        "chunk_id": hit.chunk.chunk_id,
                        "score": hit.score,
                        "page_title": hit.chunk.page_title,
                        "text": hit.chunk.text,
                    }
                    for hit in baseline_hits[:3]
                ],
                "graphrag": [
                    {
                        "chunk_id": hit.chunk.chunk_id,
                        "score": hit.score,
                        "page_title": hit.chunk.page_title,
                        "text": hit.chunk.text,
                        "metadata": hit.metadata,
                    }
                    for hit in graphrag_hits[:3]
                ],
            }
        )
    return collected


def grid_search_graph_params(
    examples: List[EvalExample],
    graph_retriever: GraphRAGRetriever,
    chunks: List[Chunk],
    alpha_values: Sequence[float],
    beta_values: Sequence[float],
    k: int = 5,
) -> Dict[str, float]:
    chunk_index = {chunk.chunk_id: idx for idx, chunk in enumerate(chunks)}
    title_to_indices: Dict[str, List[int]] = {}
    for idx, chunk in enumerate(chunks):
        title_to_indices.setdefault(chunk.page_title, []).append(idx)

    def _target_indices(example: EvalExample) -> List[int]:
        titles: List[str] = []
        if example.page_title:
            titles.append(example.page_title)
        if example.supporting_pages:
            titles.extend(example.supporting_pages)
        indices: List[int] = []
        for title in titles:
            indices.extend(title_to_indices.get(title, []))
        return sorted(set(indices))

    best_score = -1.0
    best_params = {"alpha": 0.3, "beta": 0.2, "recall": 0.0}
    for alpha in alpha_values:
        for beta in beta_values:
            recalls: List[float] = []
            for example in examples:
                hits = graph_retriever.search(example.question, top_k=k, alpha=alpha, beta=beta)
                g_indices = [chunk_index[hit.chunk.chunk_id] for hit in hits]
                target_indices = _target_indices(example)
                recalls.append(recall_at_k(target_indices, g_indices, k))
            mean_recall = float(np.mean(recalls)) if recalls else 0.0
            if mean_recall > best_score:
                best_score = mean_recall
                best_params = {"alpha": alpha, "beta": beta, "recall": mean_recall}
    return best_params


__all__ = [
    "EvalExample",
    "load_eval_file",
    "evaluate_retrievers",
    "prepare_retrievers",
    "collect_examples",
    "grid_search_graph_params",
    "rouge_l",
]


def main() -> None:
    data_dir = Path("data")
    chunks_path = data_dir / "processed" / "chunks.jsonl"
    index_path = data_dir / "processed" / "faiss.index"
    graph_path = data_dir / "graph" / "wiki_graph.gpickle"
    baseline, graph_retriever, chunks = prepare_retrievers(
        chunks_path=chunks_path,
        index_path=index_path,
        graph_path=graph_path,
    )

    eval_dir = data_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        {
            "name": "single_hop",
            "path": eval_dir / "qa.jsonl",
            "metrics_path": eval_dir / "metrics_single.json",
            "examples_path": eval_dir / "examples_single.json",
            "grid_search": False,
        },
        {
            "name": "multi_hop",
            "path": eval_dir / "qa_multihop.jsonl",
            "metrics_path": eval_dir / "metrics_multihop.json",
            "examples_path": eval_dir / "examples_multihop.json",
            "grid_search": True,
        },
    ]

    alpha_default = 0.3
    beta_default = 0.2

    for dataset in datasets:
        path = dataset["path"]
        if not path.exists():
            continue
        examples = load_eval_file(path)
        alpha = alpha_default
        beta = beta_default
        tuning = None
        if dataset["grid_search"] and graph_retriever:
            tuning = grid_search_graph_params(
                examples,
                graph_retriever,
                chunks,
                alpha_values=[0.2, 0.3, 0.4],
                beta_values=[0.1, 0.2, 0.3],
            )
            alpha = tuning["alpha"]
            beta = tuning["beta"]

        metrics = evaluate_retrievers(
            examples,
            baseline,
            graph_retriever,
            chunks,
            k_values=(5, 10),
            alpha=alpha,
            beta=beta,
        )
        metrics["settings"] = {"alpha": alpha, "beta": beta}
        if tuning:
            metrics["grid_search"] = tuning
        with dataset["metrics_path"].open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        examples_summary = collect_examples(
            examples,
            baseline,
            graph_retriever,
            num_examples=min(5, len(examples)),
            top_k=5,
            alpha=alpha,
            beta=beta,
        )
        with dataset["examples_path"].open("w", encoding="utf-8") as f:
            json.dump(examples_summary, f, indent=2)


if __name__ == "__main__":
    main()

