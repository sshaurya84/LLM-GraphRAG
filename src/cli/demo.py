from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

import networkx as nx
import pickle
from llama_cpp import Llama

from src.embeddings import encode_chunks, load_embedder
from src.graph import build_hetero_graph
from src.generation.local_llm import generate_answer, load_llm
from src.index import build_faiss_index, load_faiss_index
from src.ingestion import download_corpus, load_corpus
from src.processing import chunk_records, load_chunks
from src.retrievers import BaselineRetriever, GraphRAGRetriever
from src.retrievers.types import RetrievalHit

app = typer.Typer(help="GraphRAG demo CLI")

console = Console()

DATA_DIR = Path("data")
RAW_PATH = DATA_DIR / "raw" / "science_tech_pages.jsonl"
CHUNKS_PATH = DATA_DIR / "processed" / "chunks.jsonl"
EMBEDDINGS_PATH = DATA_DIR / "processed" / "embeddings.npy"
FAISS_PATH = DATA_DIR / "processed" / "faiss.index"
GRAPH_PATH = DATA_DIR / "graph" / "wiki_graph.gpickle"
GRAPH_EDGES_CSV = DATA_DIR / "graph" / "edges.csv"

_LLM_CACHE: Dict[Path, Llama] = {}

@app.command()
def ingest(
    first_hop: int = typer.Option(20, help="First-hop links per seed."),
    second_hop: int = typer.Option(8, help="Second-hop links per first-hop page."),
    output: Path = typer.Option(RAW_PATH, help="Destination JSONL path."),
) -> None:
    records = download_corpus(
        output_path=output,
        first_hop_limit=first_hop,
        second_hop_limit=second_hop,
    )
    console.print(f"[green]Saved {len(records)} pages to {output}[/green]")


@app.command()
def chunk(
    input_path: Path = typer.Option(RAW_PATH, help="Path to pages JSONL."),
    output_path: Path = typer.Option(CHUNKS_PATH, help="Chunk output JSONL."),
    model_name: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2"),
    window: int = typer.Option(512),
    overlap: int = typer.Option(128),
) -> None:
    records = load_corpus(input_path)
    chunks = chunk_records(records, output_path, model_name=model_name, window_size=window, overlap=overlap)
    console.print(f"[green]Generated {len(chunks)} chunks at {output_path}[/green]")


@app.command()
def embed(
    chunks_path: Path = typer.Option(CHUNKS_PATH),
    embeddings_path: Path = typer.Option(EMBEDDINGS_PATH),
    index_path: Path = typer.Option(FAISS_PATH),
    model_name: str = typer.Option("BAAI/bge-small-en-v1.5"),
    metric: str = typer.Option("ip"),
) -> None:
    chunks = load_chunks(chunks_path)
    embedder = load_embedder(model_name)
    embeddings = encode_chunks(embedder, chunks)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings.astype(np.float32))
    build_faiss_index(embeddings, index_path, metric=metric)
    console.print(f"[green]Embeddings saved to {embeddings_path}, FAISS index at {index_path}[/green]")


@app.command("build-graph")
def build_graph_cmd(
    chunks_path: Path = typer.Option(CHUNKS_PATH),
    graph_path: Path = typer.Option(GRAPH_PATH),
    edges_csv: Path = typer.Option(GRAPH_EDGES_CSV),
) -> None:
    chunks = load_chunks(chunks_path)
    graph = build_hetero_graph(
        chunks,
        graph_path=graph_path,
        edge_csv_path=edges_csv,
    )
    console.print(f"[green]Graph saved to {graph_path} with {graph.number_of_nodes()} nodes[/green]")


def _render_hits(hits: List[RetrievalHit], title: str = "Results") -> None:
    table = Table(title=title)
    table.add_column("Rank", justify="center")
    table.add_column("Chunk ID")
    table.add_column("Title")
    table.add_column("Score", justify="right")
    table.add_column("Sim", justify="right")
    table.add_column("PPR", justify="right")
    for idx, hit in enumerate(hits, start=1):
        sim = hit.metadata.get("similarity", 0.0)
        ppr = hit.metadata.get("ppr", 0.0)
        table.add_row(
            str(idx),
            hit.chunk.chunk_id,
            hit.chunk.page_title,
            f"{hit.score:.3f}",
            f"{sim:.3f}",
            f"{ppr:.3f}",
        )
    console.print(table)


def _format_neighbors(graph: nx.MultiDiGraph, chunk_id: str, max_entities: int = 5, max_docs: int = 3) -> str:
    entity_neighbors: List[str] = []
    doc_neighbors: List[str] = []
    for neighbor in graph.neighbors(chunk_id):
        data = graph.nodes.get(neighbor, {})
        if data.get("type") == "entity":
            entity_neighbors.append(data.get("name", neighbor.split("::", 1)[-1]))
        elif data.get("type") == "doc":
            doc_neighbors.append(data.get("title", neighbor))
    for neighbor in graph.predecessors(chunk_id):
        data = graph.nodes.get(neighbor, {})
        if data.get("type") == "doc":
            doc_neighbors.append(data.get("title", neighbor))
    entity_neighbors = sorted(set(entity_neighbors))[:max_entities]
    doc_neighbors = sorted(set(doc_neighbors))[:max_docs]
    parts: List[str] = []
    if entity_neighbors:
        parts.append("Entities: " + ", ".join(entity_neighbors))
    if doc_neighbors:
        parts.append("Linked docs: " + ", ".join(doc_neighbors))
    return "; ".join(parts) if parts else "No graph neighbors captured."


def _show_graph_context(graph: Optional[nx.MultiDiGraph], hits: List[RetrievalHit], limit: int = 3) -> None:
    if not graph:
        return
    console.print("\n[bold]Graph context[/bold]")
    for hit in hits[:limit]:
        summary = _format_neighbors(graph, hit.chunk.chunk_id)
        console.print(f"[cyan]{hit.chunk.chunk_id}[/cyan]: {summary}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="User question"),
    mode: str = typer.Option("graphrag", help="Retriever mode: baseline or graphrag"),
    top_k: int = typer.Option(5),
    alpha: float = typer.Option(0.3),
    beta: float = typer.Option(0.2),
    model_name: str = typer.Option("BAAI/bge-small-en-v1.5"),
    llm_path: Path = typer.Option(Path("models/llama/Phi-3-mini-4k-instruct-q4.gguf")),
    show_only: bool = typer.Option(False, help="Skip generation and only display retrieval hits."),
    n_gpu_layers: Optional[int] = typer.Option(None, help="Override llama.cpp n_gpu_layers."),
    n_threads: Optional[int] = typer.Option(None, help="Override llama.cpp n_threads."),
    max_tokens: int = typer.Option(256, help="Maximum tokens to generate."),
) -> None:
    chunks = load_chunks(CHUNKS_PATH)
    index = load_faiss_index(FAISS_PATH)
    embedder = load_embedder(model_name)

    graph_for_display: Optional[nx.MultiDiGraph] = None

    if mode == "baseline":
        retriever = BaselineRetriever(embedder, index, chunks)
        hits = retriever.search(question, top_k=top_k)
    else:
        if GRAPH_PATH.exists():
            with GRAPH_PATH.open("rb") as f:
                graph = pickle.load(f)
        else:
            graph = build_hetero_graph(chunks, graph_path=GRAPH_PATH, edge_csv_path=GRAPH_EDGES_CSV)
        retriever = GraphRAGRetriever(embedder, index, graph, chunks)
        hits = retriever.search(question, top_k=top_k, alpha=alpha, beta=beta)
        graph_for_display = graph

    if not hits:
        console.print("[red]No retrieval hits found.[/red]")
        raise typer.Exit(code=1)

    _render_hits(hits, title=f"{mode.title()} top-{top_k}")
    if mode != "baseline":
        _show_graph_context(graph_for_display, hits)

    if show_only:
        return

    passages = [hit.chunk.text for hit in hits]
    llm = _LLM_CACHE.get(llm_path)
    if llm is None:
        try:
            llm = load_llm(llm_path, n_gpu_layers=n_gpu_layers, n_threads=n_threads)
        except FileNotFoundError as exc:
            console.print(f"[yellow]{exc} Skipping generation.[/yellow]")
            return
        _LLM_CACHE[llm_path] = llm

    answer = generate_answer(llm, question, passages, max_tokens=max_tokens)
    console.print("\n[bold green]Answer:[/bold green]\n")
    console.print(answer)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
