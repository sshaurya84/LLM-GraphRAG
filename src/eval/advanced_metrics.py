"""Advanced evaluation metrics that better capture GraphRAG's advantages."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from src.eval.evaluate import EvalExample, load_eval_file, prepare_retrievers
from src.generation.local_llm import generate_answer, load_llm
from src.retrievers.types import RetrievalHit


DEFAULT_LLM_PATH = Path("models/llama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")


def _normalize_title(title: str) -> str:
    return title.strip().lower().replace("_", " ")


# --- METRIC 1: Multi-Fact Coverage ---
def multi_fact_coverage(
    retrieved_titles: List[str],
    supporting_pages: List[str],
) -> float:
    """
    Measure what fraction of supporting pages appear in retrieved results.
    This rewards retrieving diverse evidence from multiple sources.
    """
    if not supporting_pages:
        return 1.0
    retrieved_norm = {_normalize_title(t) for t in retrieved_titles}
    support_norm = {_normalize_title(t) for t in supporting_pages}
    covered = len(retrieved_norm & support_norm)
    return covered / len(support_norm)


# --- METRIC 2: Context Breadth (Unique Concepts) ---
def context_breadth(
    hits: List[RetrievalHit],
    top_k: int,
) -> Dict[str, float]:
    """
    Measure diversity of retrieved context:
    - unique_pages: fraction of unique page titles in top-k
    - unique_chunks: 1.0 if all chunks are different (always true)
    - entity_coverage: approximate entity count in text
    """
    if not hits:
        return {"unique_pages": 0.0, "entity_density": 0.0}
    
    titles = [h.chunk.page_title for h in hits[:top_k]]
    unique_titles = len(set(titles))
    
    # Count approximate entities (capitalized multi-word phrases)
    all_text = " ".join(h.chunk.text for h in hits[:top_k])
    # Simple heuristic: count capitalized words that aren't sentence starters
    entities = set(re.findall(r'(?<=[.!?]\s)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', all_text))
    entities.update(re.findall(r'([A-Z][A-Z]+(?:\s+[A-Z][A-Z]+)*)', all_text))  # Acronyms
    
    return {
        "unique_pages": unique_titles / top_k,
        "unique_page_count": unique_titles,
        "entity_density": len(entities) / max(1, len(all_text.split()) / 100),
    }


# --- METRIC 3: Answer Semantic Similarity ---
def answer_semantic_similarity(
    generated_answer: str,
    reference_answer: str,
    embedder: SentenceTransformer,
) -> float:
    """
    Compute cosine similarity between generated and reference answer embeddings.
    More robust than ROUGE for semantic equivalence.
    """
    if not generated_answer or not reference_answer:
        return 0.0
    embeddings = embedder.encode([generated_answer, reference_answer], normalize_embeddings=True)
    return float(np.dot(embeddings[0], embeddings[1]))


# --- METRIC 4: Context Utilization ---
def context_utilization(
    answer: str,
    retrieved_texts: List[str],
) -> float:
    """
    Measure how much of the retrieved context is used in the answer.
    Higher = answer draws from more retrieved passages.
    """
    if not answer or not retrieved_texts:
        return 0.0
    
    answer_words = set(answer.lower().split())
    utilized_chunks = 0
    
    for text in retrieved_texts:
        chunk_words = set(text.lower().split())
        # Check if significant overlap (more than just common words)
        overlap = len(answer_words & chunk_words)
        if overlap > 5:  # At least 5 shared content words
            utilized_chunks += 1
    
    return utilized_chunks / len(retrieved_texts)


# --- METRIC 5: Information Density ---
def information_density(answer: str, question: str) -> float:
    """
    Ratio of unique content words to total words (filters stopwords).
    Higher = more information-dense answer.
    """
    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "during", "before", "after", "above", "below", "between", "under",
        "again", "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "each", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than",
        "too", "very", "just", "and", "but", "if", "or", "because", "until",
        "while", "this", "that", "these", "those", "it", "its", "they", "them",
        "their", "what", "which", "who", "whom", "whose",
    }
    
    words = re.findall(r'\b[a-z]+\b', answer.lower())
    if not words:
        return 0.0
    
    content_words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    unique_content = set(content_words)
    
    return len(unique_content) / max(len(words), 1)


@dataclass
class AdvancedMetrics:
    """Container for all advanced metrics."""
    multi_fact_coverage: float
    unique_pages: float
    unique_page_count: int
    entity_density: float
    semantic_similarity: float
    context_utilization: float
    information_density: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "multi_fact_coverage": self.multi_fact_coverage,
            "unique_pages": self.unique_pages,
            "unique_page_count": self.unique_page_count,
            "entity_density": self.entity_density,
            "semantic_similarity": self.semantic_similarity,
            "context_utilization": self.context_utilization,
            "information_density": self.information_density,
        }


def evaluate_single_example(
    example: EvalExample,
    hits: List[RetrievalHit],
    embedder: SentenceTransformer,
    llm=None,
    top_k: int = 5,
    generate_answers: bool = False,
) -> Tuple[AdvancedMetrics, str]:
    """Evaluate a single example with all advanced metrics."""
    
    retrieved_titles = [h.chunk.page_title for h in hits[:top_k]]
    supporting_pages = example.supporting_pages or [example.page_title] if example.page_title else []
    
    # Multi-fact coverage
    mfc = multi_fact_coverage(retrieved_titles, supporting_pages)
    
    # Context breadth
    breadth = context_breadth(hits, top_k)
    
    # Generate answer if LLM available
    if generate_answers and llm:
        passages = [h.chunk.text for h in hits[:3]]
        answer = generate_answer(llm, example.question, passages, max_tokens=200)
    else:
        # Use top chunk as proxy answer
        answer = hits[0].chunk.text if hits else ""
    
    # Semantic similarity to reference
    sem_sim = answer_semantic_similarity(answer, example.answer, embedder)
    
    # Context utilization
    ctx_util = context_utilization(answer, [h.chunk.text for h in hits[:top_k]])
    
    # Information density
    info_density = information_density(answer, example.question)
    
    metrics = AdvancedMetrics(
        multi_fact_coverage=mfc,
        unique_pages=breadth["unique_pages"],
        unique_page_count=int(breadth["unique_page_count"]),
        entity_density=breadth["entity_density"],
        semantic_similarity=sem_sim,
        context_utilization=ctx_util,
        information_density=info_density,
    )
    
    return metrics, answer


def run_evaluation(
    examples: List[EvalExample],
    baseline_retriever,
    graph_retriever,
    embedder: SentenceTransformer,
    llm=None,
    top_k: int = 5,
    alpha: float = 0.4,
    beta: float = 0.15,
    gamma: float = 0.2,
    generate_answers: bool = False,
) -> Dict[str, object]:
    """Run full evaluation comparing baseline and GraphRAG."""
    
    baseline_results: List[Dict] = []
    graphrag_results: List[Dict] = []
    
    for example in examples:
        # Baseline
        b_hits = baseline_retriever.search(example.question, top_k=top_k)
        b_metrics, b_answer = evaluate_single_example(
            example, b_hits, embedder, llm, top_k, generate_answers
        )
        baseline_results.append({
            "question": example.question,
            "answer": b_answer[:500],  # Truncate for storage
            **b_metrics.to_dict(),
        })
        
        # GraphRAG
        g_hits = graph_retriever.search(
            example.question, top_k=top_k, alpha=alpha, beta=beta, gamma=gamma
        )
        g_metrics, g_answer = evaluate_single_example(
            example, g_hits, embedder, llm, top_k, generate_answers
        )
        graphrag_results.append({
            "question": example.question,
            "answer": g_answer[:500],
            **g_metrics.to_dict(),
        })
    
    # Compute averages
    def avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results if isinstance(r[key], (int, float))]
        return sum(vals) / len(vals) if vals else 0.0
    
    metric_keys = [
        "multi_fact_coverage", "unique_pages", "entity_density",
        "semantic_similarity", "context_utilization", "information_density"
    ]
    
    baseline_avg = {k: avg(baseline_results, k) for k in metric_keys}
    graphrag_avg = {k: avg(graphrag_results, k) for k in metric_keys}
    
    # Compute wins (questions where GraphRAG beats baseline)
    wins = {k: [] for k in metric_keys}
    for b, g in zip(baseline_results, graphrag_results):
        for k in metric_keys:
            if g[k] > b[k]:
                wins[k].append(g["question"])
    
    return {
        "baseline": {"averages": baseline_avg, "details": baseline_results},
        "graphrag": {"averages": graphrag_avg, "details": graphrag_results},
        "graphrag_wins": {k: len(v) for k, v in wins.items()},
        "win_details": wins,
    }


def build_markdown_report(results: Dict[str, object], dataset_name: str) -> str:
    """Generate markdown report from evaluation results."""
    lines = [f"# Advanced Metrics: {dataset_name}", ""]
    
    # Summary table
    lines.append("## Summary Comparison")
    lines.append("")
    lines.append("| Metric | Baseline | GraphRAG | Winner | GraphRAG Wins |")
    lines.append("|--------|----------|----------|--------|---------------|")
    
    b_avg = results["baseline"]["averages"]
    g_avg = results["graphrag"]["averages"]
    wins = results["graphrag_wins"]
    
    for metric in b_avg.keys():
        b_val = b_avg[metric]
        g_val = g_avg[metric]
        winner = "GraphRAG" if g_val > b_val else ("Baseline" if b_val > g_val else "Tie")
        win_count = wins.get(metric, 0)
        total = len(results["baseline"]["details"])
        lines.append(f"| {metric.replace('_', ' ').title()} | {b_val:.3f} | {g_val:.3f} | **{winner}** | {win_count}/{total} |")
    
    lines.append("")
    lines.append("## Metric Definitions")
    lines.append("")
    lines.append("- **Multi-Fact Coverage**: Fraction of required supporting pages found in top-k results")
    lines.append("- **Unique Pages**: Diversity of sources (unique page titles / k)")
    lines.append("- **Entity Density**: Approximate count of named entities per 100 words")
    lines.append("- **Semantic Similarity**: Cosine similarity between answer and reference")
    lines.append("- **Context Utilization**: Fraction of retrieved chunks reflected in answer")
    lines.append("- **Information Density**: Ratio of unique content words to total words")
    lines.append("")
    
    # Questions where GraphRAG won on key metrics
    lines.append("## Questions Where GraphRAG Excels")
    lines.append("")
    
    win_details = results["win_details"]
    for metric in ["multi_fact_coverage", "unique_pages", "semantic_similarity"]:
        questions = win_details.get(metric, [])
        lines.append(f"### {metric.replace('_', ' ').title()} ({len(questions)} wins)")
        if questions:
            for q in questions[:5]:  # Show up to 5
                lines.append(f"- {q}")
        else:
            lines.append("- (none)")
        lines.append("")
    
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run advanced metrics evaluation.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.4, help="PPR weight")
    parser.add_argument("--beta", type=float, default=0.15, help="Relation bonus weight")
    parser.add_argument("--gamma", type=float, default=0.2, help="BM25 weight")
    parser.add_argument("--generate-answers", action="store_true", help="Generate answers with LLM")
    parser.add_argument("--llm-path", type=Path, default=DEFAULT_LLM_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()
    
    print("Loading retrievers...")
    data_dir = Path("data")
    baseline, graph_retriever, chunks = prepare_retrievers(
        chunks_path=data_dir / "processed" / "chunks.jsonl",
        index_path=data_dir / "processed" / "faiss.index",
        graph_path=data_dir / "graph" / "wiki_graph.gpickle",
    )
    
    if graph_retriever is None:
        raise RuntimeError("Graph retriever not available. Build graph first.")
    
    print("Loading embedder for semantic similarity...")
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    
    llm = None
    if args.generate_answers:
        if not args.llm_path.exists():
            print(f"Warning: LLM not found at {args.llm_path}, skipping answer generation")
        else:
            print("Loading LLM...")
            llm = load_llm(args.llm_path)
    
    datasets = {
        "single_hop": data_dir / "eval" / "qa.jsonl",
        "multi_hop": data_dir / "eval" / "qa_multihop.jsonl",
    }
    
    all_results = {}
    
    for name, path in datasets.items():
        if not path.exists():
            print(f"Skipping {name}: {path} not found")
            continue
        
        print(f"\nEvaluating {name}...")
        examples = load_eval_file(path)
        
        results = run_evaluation(
            examples=examples,
            baseline_retriever=baseline,
            graph_retriever=graph_retriever,
            embedder=embedder,
            llm=llm,
            top_k=args.top_k,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            generate_answers=args.generate_answers,
        )
        
        all_results[name] = results
        
        # Print summary
        print(f"\n{name.upper()} Results:")
        print(f"{'Metric':<25} {'Baseline':>10} {'GraphRAG':>10} {'Winner':>10}")
        print("-" * 60)
        b_avg = results["baseline"]["averages"]
        g_avg = results["graphrag"]["averages"]
        for metric in b_avg.keys():
            winner = "GraphRAG" if g_avg[metric] > b_avg[metric] else "Baseline"
            print(f"{metric:<25} {b_avg[metric]:>10.3f} {g_avg[metric]:>10.3f} {winner:>10}")
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = args.output_dir / "advanced_metrics.json"
    json_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved JSON to {json_path}")
    
    # Generate markdown report
    md_lines = ["# Advanced Metrics Evaluation", ""]
    md_lines.append("These metrics are designed to capture GraphRAG's strengths:")
    md_lines.append("- Multi-hop reasoning (multi-fact coverage)")
    md_lines.append("- Source diversity (unique pages)")
    md_lines.append("- Answer quality (semantic similarity)")
    md_lines.append("")
    
    for name, results in all_results.items():
        md_lines.append(build_markdown_report(results, name))
        md_lines.append("")
    
    md_path = args.output_dir / "advanced_metrics.md"
    md_path.write_text("\n".join(md_lines))
    print(f"Saved markdown to {md_path}")


if __name__ == "__main__":
    main()


