from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from src.eval.evaluate import EvalExample, load_eval_file, prepare_retrievers
from src.generation.local_llm import generate_answer, load_llm


DEFAULT_LLM_PATH = Path("models/llama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
ACTION_WORDS = {
    "can",
    "enable",
    "enables",
    "enablement",
    "allow",
    "allows",
    "help",
    "helps",
    "support",
    "supports",
    "provide",
    "provides",
    "use",
    "using",
    "build",
    "create",
    "deploy",
    "implement",
    "step",
    "steps",
    "guide",
    "improve",
    "improves",
    "optimize",
    "optimizes",
}


def _normalize_title(title: str) -> str:
    return title.strip()


def _relevant_titles(example: EvalExample) -> List[str]:
    titles: List[str] = []
    if example.page_title:
        titles.append(_normalize_title(example.page_title))
    if example.supporting_pages:
        titles.extend(_normalize_title(t) for t in example.supporting_pages)
    return list(dict.fromkeys(titles))


def _directness(selected_titles: List[str], relevant_titles: Iterable[str], top_k: int) -> float:
    relevant_set = {_normalize_title(t) for t in relevant_titles}
    for idx, title in enumerate(selected_titles, start=1):
        if _normalize_title(title) in relevant_set:
            return 1.0 - (idx - 1) / top_k
    return 0.0


def _comprehensiveness(selected_titles: List[str], relevant_titles: Iterable[str]) -> float:
    relevant_set = {_normalize_title(t) for t in relevant_titles if t}
    if not relevant_set:
        return 0.0
    selected = {_normalize_title(t) for t in selected_titles if t}
    return len(relevant_set & selected) / len(relevant_set)


def _diversity(selected_titles: List[str], top_k: int) -> float:
    if top_k == 0:
        return 0.0
    return len({_normalize_title(t) for t in selected_titles if t}) / top_k


def _empowerment(answer: str) -> float:
    words = re.findall(r"\b[\w']+\b", answer.lower())
    if not words:
        return 0.0
    matches = sum(1 for w in words if w in ACTION_WORDS)
    return matches / len(words)


def _answer_length(answer: str) -> int:
    return len(re.findall(r"\b[\w']+\b", answer))


def _retrieve_and_score(
    retriever,
    question: str,
    relevant_titles: List[str],
    top_k: int,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
) -> Tuple[List, Dict[str, float]]:
    if alpha is not None and beta is not None:
        hits = retriever.search(question, top_k=top_k, alpha=alpha, beta=beta)
    else:
        hits = retriever.search(question, top_k=top_k)
    selected_titles = [hit.chunk.page_title for hit in hits]
    return hits, {
        "comprehensiveness": _comprehensiveness(selected_titles, relevant_titles),
        "diversity": _diversity(selected_titles, top_k),
        "directness": _directness(selected_titles, relevant_titles, top_k),
    }


def _augment_metrics(
    hits,
    base_scores: Dict[str, float],
    question: str,
    generate_answers: bool,
    llm,
) -> Dict[str, object]:
    metrics = dict(base_scores)
    if generate_answers:
        if llm is None:
            raise ValueError("LLM must be provided when generate_answers=True.")
        passages = [hit.chunk.text for hit in hits[:3]]
        answer = generate_answer(llm, question, passages, max_tokens=160)
        source = "llm"
    else:
        passages = [hit.chunk.text for hit in hits[:1]]
        answer = "\n\n".join(passages)
        source = "retrieved_chunk"

    metrics.update(
        {
            "question": question,
            "selected_titles": [hit.chunk.page_title for hit in hits],
            "answer_source": source,
            "answer": answer,
            "empowerment": _empowerment(answer),
            "answer_length": _answer_length(answer),
        }
    )
    return metrics


def _aggregate(metrics: List[Dict[str, object]], key: str) -> float:
    if not metrics:
        return 0.0
    return sum(m[key] for m in metrics) / len(metrics)


def compute_dataset_metrics(
    examples: List[EvalExample],
    baseline_retriever,
    graph_retriever,
    top_k: int,
    alpha_beta: Tuple[float, float],
    generate_answers: bool,
    llm=None,
) -> Dict[str, object]:
    alpha, beta = alpha_beta
    baseline_metrics: List[Dict[str, object]] = []
    graphrag_metrics: List[Dict[str, object]] = []

    for example in examples:
        relevant_titles = _relevant_titles(example)
        if not relevant_titles:
            continue

        baseline_hits, baseline_scores = _retrieve_and_score(
            baseline_retriever,
            example.question,
            relevant_titles,
            top_k=top_k,
        )
        graphrag_hits, graphrag_scores = _retrieve_and_score(
            graph_retriever,
            example.question,
            relevant_titles,
            top_k=top_k,
            alpha=alpha,
            beta=beta,
        )

        baseline_metrics.append(
            _augment_metrics(baseline_hits, baseline_scores, example.question, generate_answers, llm)
        )
        graphrag_metrics.append(
            _augment_metrics(graphrag_hits, graphrag_scores, example.question, generate_answers, llm)
        )

    summary = {
        "baseline": {
            "averages": {
                "comprehensiveness": _aggregate(baseline_metrics, "comprehensiveness"),
                "diversity": _aggregate(baseline_metrics, "diversity"),
                "directness": _aggregate(baseline_metrics, "directness"),
                "empowerment": _aggregate(baseline_metrics, "empowerment"),
                "answer_length": _aggregate(baseline_metrics, "answer_length"),
            },
            "details": baseline_metrics,
        },
        "graphrag": {
            "averages": {
                "comprehensiveness": _aggregate(graphrag_metrics, "comprehensiveness"),
                "diversity": _aggregate(graphrag_metrics, "diversity"),
                "directness": _aggregate(graphrag_metrics, "directness"),
                "empowerment": _aggregate(graphrag_metrics, "empowerment"),
                "answer_length": _aggregate(graphrag_metrics, "answer_length"),
            },
            "details": graphrag_metrics,
        },
        "improvements": {
            "comprehensiveness": [
                g["question"]
                for b, g in zip(baseline_metrics, graphrag_metrics)
                if g["comprehensiveness"] > b["comprehensiveness"]
            ],
            "diversity": [
                g["question"]
                for b, g in zip(baseline_metrics, graphrag_metrics)
                if g["diversity"] > b["diversity"]
            ],
            "directness": [
                g["question"]
                for b, g in zip(baseline_metrics, graphrag_metrics)
                if g["directness"] > b["directness"]
            ],
            "empowerment": [
                g["question"]
                for b, g in zip(baseline_metrics, graphrag_metrics)
                if g["empowerment"] > b["empowerment"]
            ],
        },
    }
    return summary


def build_markdown_report(results: Dict[str, object]) -> str:
    lines: List[str] = ["# Qualitative Metrics Comparison", ""]
    for dataset, data in results["datasets"].items():
        lines.append(f"## Dataset: {dataset}")
        lines.append("")
        lines.append("| Model | Comprehensiveness | Diversity | Directness | Empowerment | Answer Length |")
        lines.append("|-------|-------------------|-----------|------------|-------------|----------------|")
        for model in ("baseline", "graphrag"):
            avg = data[model]["averages"]
            lines.append(
                f"| {model.title()} | {avg['comprehensiveness']:.3f} | {avg['diversity']:.3f} "
                f"| {avg['directness']:.3f} | {avg['empowerment']:.3f} | {avg['answer_length']:.1f} |"
            )
        lines.append("")
        lines.append("### Questions where GraphRAG improved")
        lines.append("")
        for metric, questions in data["improvements"].items():
            list_items = "\n".join(f"- {question}" for question in questions) or "- (none)"
            lines.append(f"- **{metric.title()}**: {len(questions)} question(s)\n{list_items}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute qualitative metrics for baseline vs GraphRAG.")
    parser.add_argument(
        "--llm-path",
        type=Path,
        default=DEFAULT_LLM_PATH,
        help="Path to local GGUF model for answer generation.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved passages to consider.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("docs/qualitative_metrics.json"),
        help="Path to write metrics JSON.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("docs/qualitative_metrics.md"),
        help="Path to write metrics markdown summary.",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip local LLM generation; empowerment falls back to retrieved text.",
    )

    args = parser.parse_args()
    generate_answers = not args.skip_generation
    if generate_answers and not args.llm_path.exists():
        raise FileNotFoundError(
            f"Local model not found at {args.llm_path}. Download a GGUF model (e.g., scripts/download_phi3.sh)."
        )

    data_dir = Path("data")
    baseline, graph_retriever, _ = prepare_retrievers(
        chunks_path=data_dir / "processed" / "chunks.jsonl",
        index_path=data_dir / "processed" / "faiss.index",
        graph_path=data_dir / "graph" / "wiki_graph.gpickle",
    )
    llm = load_llm(args.llm_path) if generate_answers else None

    datasets = {
        "single_hop": {
            "path": data_dir / "eval" / "qa.jsonl",
            "alpha_beta": (0.3, 0.2),
        },
        "multi_hop": {
            "path": data_dir / "eval" / "qa_multihop.jsonl",
            "alpha_beta": (0.2, 0.2),
        },
    }

    results: Dict[str, object] = {"datasets": {}}
    for name, info in datasets.items():
        examples = load_eval_file(info["path"])
        summary = compute_dataset_metrics(
            examples=examples,
            baseline_retriever=baseline,
            graph_retriever=graph_retriever,
            top_k=args.top_k,
            alpha_beta=info["alpha_beta"],
            generate_answers=generate_answers,
            llm=llm,
        )
        results["datasets"][name] = summary

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2))
    args.output_md.write_text(build_markdown_report(results))


if __name__ == "__main__":
    main()

