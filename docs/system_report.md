# GraphRAG Science & Technology Corpus – System Report

## 1. Overview

This project builds a local Retrieval-Augmented Generation (RAG) stack on top of a focused science & technology slice of Wikipedia. The goal is to compare a dense baseline retriever with a graph-augmented variant (GraphRAG) while keeping everything runnable on an Apple Silicon laptop using Metal acceleration. The pipeline is implemented in Python and structured so that each step can be reproduced with the Typer CLI (`src/cli/demo.py`).

```
Wikipedia ingest → Chunking → Embeddings + FAISS → Graph construction →
Baseline/GraphRAG retrievers → Local LLM generation
```

Key directories:

- `data/raw/` — raw Wikipedia JSONL snapshots (`science_tech_pages_v2.jsonl`).
- `data/processed/` — chunk metadata, embeddings (`embeddings.npy`), FAISS index.
- `data/graph/` — heterogeneous networkx graph (`wiki_graph.gpickle`, edge CSV).
- `data/eval/` — evaluation datasets, metrics, qualitative examples.
- `src/` — modular pipeline implementation (ingestion, processing, graph, retrievers, eval, cli).

## 2. Data Ingestion & Processing

**Seeds & crawling** — `src/ingestion/wikipedia.py`

- Expanded seed list (30+ topics) across AI, HPC, robotics, HCI, etc. (`DEFAULT_SEEDS`).
- First-hop and second-hop link expansion with science/tech keyword filtering and allowlist to keep the crawl targeted.
- Ingestion CLI command: `python -m src.cli.demo ingest --first-hop 20 --second-hop 8`.
- Output: `data/raw/science_tech_pages_v2.jsonl` (approx. 1,059 pages) with cleaned text, metadata, categories, edge depths.

**Chunking** — `src/processing/chunk.py`

- Uses HuggingFace tokenizer (`sentence-transformers/all-MiniLM-L6-v2` by default).
- Sliding window: 512 tokens with 128 overlap; stores token offsets, page metadata, placeholders for entities.
- CLI command: `python -m src.cli.demo chunk --input-path ...`.
- Output: `data/processed/chunks.jsonl` (~10.9k chunks).

## 3. Embeddings & FAISS Index

- Embedding model: `BAAI/bge-small-en-v1.5` (SentenceTransformers) with automatic Metal (MPS) or CUDA selection (`src/embeddings/embedder.py`, `src/utils/device.py`).
- Embedding command: `python -m src.cli.demo embed`.
- Saves `embeddings.npy` and FAISS `IndexFlatIP` at `data/processed/faiss.index`.
- Baseline retriever (`src/retrievers/baseline.py`) uses this FAISS index directly.

## 4. Heterogeneous Graph Construction

- Entities via spaCy NER (`en_core_web_sm`) per chunk; stored on `Chunk.entities`.
- Graph builder (`src/graph/build_graph.py`):
  - Nodes: `doc::<chunk_id>` and `entity::<name>`.
  - Edges: entity–doc mentions, entity–entity co-occurrence (PMI-weighted), doc–doc hyperlinks; simple relation tags (`is_a`, `part_of`, `founded_by`).
- CLI command: `python -m src.cli.demo build-graph`.
- Output: `data/graph/wiki_graph.gpickle` (~61k nodes) + `edges.csv` for inspection.

## 5. Retrieval Components

**Baseline Retriever** — cosine similarity over FAISS results.

**GraphRAG Retriever** — `src/retrievers/graphrag.py`

- Candidate generation: dense search (top-20 by default).
- Query parsing: spaCy entity + noun chunk extraction, bi-gram generation, augmented by manual synonym table (`src/graph/entity_matching.py`).
- Seeding: matched entity names plus document titles map to graph nodes; fallback uses top dense hits when no seeds found.
- Personalized PageRank (PPR) scoring (`src/graph/query.py`), combined with cosine similarity via `combined = sim + α·PPR + β·relation_bonus`.
- CLI `ask` command shows retrieved chunks and graph neighborhood summaries.

## 6. Local LLM Inference

- `src/generation/local_llm.py` wraps `llama_cpp.Llama` with sensible defaults:
  - Metal acceleration on Apple Silicon (`n_gpu_layers` auto tuned).
  - Thread count inferred from logical cores.
- Model download helper: `scripts/download_phi3.sh` (Phi-3 Mini 4K Instruct Q4_0).
- CLI caches loaded models across invocations; exposes overrides (`--n-gpu-layers`, `--n-threads`, `--max-tokens`).

## 7. Evaluation Pipeline

- Datasets:
  - `data/eval/qa.jsonl` — single-hop questions (auto generated summaries).
  - `data/eval/qa_multihop.jsonl` — curated multi-hop question set (16 items) using supporting page metadata.
- `src/eval/evaluate.py` now provides utilities:
  - `prepare_retrievers`, `evaluate_retrievers`, `collect_examples`, `grid_search_graph_params`.
  - Command-line runner writes metrics (`metrics_single.json`, `metrics_multihop.json`) and qualitative comparisons (`examples_*.json`).
- Current tuning: simple α/β grid search (0.2–0.4) for multi-hop set (best α=0.2, β=0.2).

## 8. CLI Usage Summary

```
# Reproduce raw → graph pipeline
python -m src.cli.demo ingest --first-hop 20 --second-hop 8 --output data/raw/science_tech_pages_v2.jsonl
python -m src.cli.demo chunk --input-path data/raw/science_tech_pages_v2.jsonl
python -m src.cli.demo embed
python -m src.cli.demo build-graph

# Ask questions (baseline vs GraphRAG)
python -m src.cli.demo ask "How do IoT and edge computing work together?" --mode baseline
python -m src.cli.demo ask "How do IoT and edge computing work together?" --mode graphrag --show-only

# Generate answers with local LLM (ensure models/llama/Phi-3-mini-4k-instruct-q4_0.gguf exists)
python -m src.cli.demo ask "What is edge computing?" --mode graphrag --max-tokens 200

# Run evaluation suite
python -m src.eval.evaluate
```

## 9. Implementation Notes & Limitations

- **Performance** — Dense+graph evaluation on the full 10.9k chunks is CPU heavy; expect several minutes for the evaluation script on an M4 Air.
- **Graph quality** — Current entity extraction uses spaCy small model; noise and missing entities limit PPR effectiveness. The grid search shows GraphRAG still underperforms the dense baseline on both QA sets.
- **Future work** — richer entity linking (EntityRuler, synonym lists), relation typing, hybrid BM25 reranking, manual multi-hop QA curation, front-end demo, deeper tuning.

## 10. Change Tracking

Major commits (chronological):

1. `Expand Wikipedia corpus and regenerate retrieval artifacts` — ingestion enhancements, data regeneration.
2. `Add multi-hop evaluation QA dataset` — curated evaluation questions.
3. `Improve GraphRAG entity matching with noun-phrase and synonym support` — better seeding logic.
4. `Add Metal-friendly local LLM configuration and download helper` — generation pipeline updates.
5. `Add evaluation pipeline outputs and grid-search tuning` — metrics, grid search, examples.

This report reflects repository state at commit `32d51bb` (evaluation outputs) plus subsequent documentation commits.

