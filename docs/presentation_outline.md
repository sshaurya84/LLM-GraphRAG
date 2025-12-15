## GraphRAG over Science & Technology Wikipedia – Presentation Outline

### 1. Problem motivation (1 min)
- Retrieval-augmented generation (RAG) on long-form science & technology pages struggles with multi-hop reasoning.
- Goal: augment RAG with a lightweight heterogeneous graph while staying fully local on Apple Silicon (Metal acceleration).

### 2. Dataset & preprocessing (1 min)
- 47 science/tech Wikipedia pages (10 seeds + filtered first-hop links).
- Saved raw + cleaned text, 523 overlapping chunks (512 tokens, 128 overlap).
- Chunk metadata persisted in `data/processed/chunks.jsonl`.

### 3. Baseline pipeline (2 min)
- Embeddings: `BAAI/bge-small-en-v1.5` (SentenceTransformers on MPS).
- Dense retrieval: FAISS `IndexFlatIP`.
- Local generation ready via `llama-cpp-python` (GGUF path configurable).
- CLI command: `python -m src.cli.demo ask "..." --mode baseline`.

### 4. GraphRAG method (3 min)
- Entity & doc nodes from spaCy NER + hyperlink structure (`build_hetero_graph`).
- Edges: entity-doc mentions, entity co-occurrence (PMI weighting), doc-doc hyperlinks.
- Query flow: extract entity/noun candidates, Personalized PageRank seeded on entity or fallback doc titles, hybrid score  
  `combined = similarity + α·PPR + β·relation_bonus`.
- CLI command: `python -m src.cli.demo ask "..." --mode graphrag` (prints graph neighbors).

### 5. Experiments (2 min)
- Eval set: 12 QA pairs auto-derived from summaries (`data/eval/qa.jsonl`).
- Metrics script: `src/eval/evaluate.py` → `data/eval/metrics.json`.
- Current retrieval results:
  - Baseline Recall@5 = **0.55**, MRR@5 = **0.96**.
  - GraphRAG Recall@5 = **0.29**, MRR@5 = **0.74** (needs tuning; see next steps).
- Coverage proxy (ROUGE-L between top chunk & gold answer) stored in `data/eval/rouge_proxy.json`.

### 6. Analysis & lessons (1 min)
- GraphRAG currently underperforms on single-hop factual questions (entity recall sparse).
- Graph seeds sparse for generic queries; requires enriched entity extraction or manual evaluation focused on multi-hop.
- Metal acceleration works end-to-end (embedding + graph building).

### 7. Next steps (1 min)
- Expand corpus (≥100 pages) and manual multi-hop QA to highlight graph benefits.
- Improve entity extraction (noun-phrase matcher, synonym list) & seed doc nodes directly.
- Add visualization (networkx/pyvis) and build simple FastAPI UI.
- Fine-tune α, β; consider BM25 hybrid to diversify candidates.

### 8. Quiz slide
- **Question:** Which pipeline component most differentiates GraphRAG from the baseline?  
  **Answer:** Personalized PageRank over the entity–doc graph.

### Demo checklist
- `python -m src.cli.demo ingest --first-hop 5`
- `python -m src.cli.demo chunk`
- `python -m src.cli.demo embed`
- `python -m src.cli.demo build-graph`
- `python -m src.cli.demo ask "Which organization devised the ACM Computing Classification System?" --mode graphrag --top-k 3`


