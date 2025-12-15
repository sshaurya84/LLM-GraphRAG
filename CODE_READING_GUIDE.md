# GraphRAG Code Reading Guide

## 🎯 Project Overview

This is a **GraphRAG (Graph-based Retrieval-Augmented Generation)** system that:
- Ingests Wikipedia pages (science & technology topics)
- Builds a knowledge graph from the content
- Compares two retrieval methods:
  - **Baseline**: Dense vector search (FAISS)
  - **GraphRAG**: Combines dense search with graph-based Personalized PageRank
- Generates answers using local LLMs (Phi-3, TinyLlama)

**Key Insight**: The project explores whether adding graph structure improves retrieval over pure semantic similarity.

---

## 📚 Recommended Reading Order

### Phase 1: High-Level Understanding (Start Here)

1. **`docs/system_report.md`** ⭐ **START HERE**
   - Complete system overview
   - Pipeline architecture
   - CLI usage examples
   - Implementation notes

2. **`docs/results_report.md`**
   - Evaluation results
   - Performance comparison (baseline vs GraphRAG)
   - Error analysis and limitations

3. **`src/cli/demo.py`** ⭐ **Entry Point**
   - Main CLI interface
   - Shows how all components connect
   - Commands: `ingest`, `chunk`, `embed`, `build-graph`, `ask`
   - Read this to understand the **data flow**

### Phase 2: Data Structures & Types

4. **`src/processing/chunk.py`**
   - `Chunk` dataclass - core data structure
   - How text is split into chunks
   - Tokenization and metadata storage

5. **`src/retrievers/types.py`**
   - `RetrievalHit` - what gets returned from retrieval
   - Simple but important for understanding retrieval results

### Phase 3: Pipeline Components (Follow Data Flow)

#### 3a. Ingestion & Processing

6. **`src/ingestion/wikipedia.py`**
   - How Wikipedia pages are downloaded
   - Seed-based crawling (first-hop, second-hop)
   - Output: raw JSONL files

7. **`src/processing/chunk.py`** (already read, but focus on implementation)
   - Sliding window chunking
   - Token offset tracking
   - Entity placeholders

#### 3b. Embeddings & Indexing

8. **`src/embeddings/embedder.py`**
   - SentenceTransformer wrapper
   - Device selection (MPS/CUDA/CPU)
   - Encoding functions

9. **`src/index/faiss_index.py`**
   - FAISS index building
   - Similarity search implementation
   - Used by both retrievers

#### 3c. Graph Construction

10. **`src/graph/build_graph.py`** ⭐ **Important**
    - Heterogeneous graph construction
    - Node types: `doc::<chunk_id>` and `entity::<name>`
    - Edge types: entity-doc mentions, entity-entity co-occurrence, doc-doc hyperlinks
    - Uses NetworkX

11. **`src/graph/entity_matching.py`**
    - Entity extraction and matching
    - Synonym handling
    - Used by GraphRAG retriever for query parsing

12. **`src/graph/query.py`**
    - Personalized PageRank (PPR) implementation
    - Relation type extraction
    - Core graph-based scoring logic

#### 3d. Retrieval (The Core Comparison)

13. **`src/retrievers/baseline.py`** ⭐ **Simple baseline**
    - Pure dense retrieval
    - FAISS search → return top-k
    - Good reference point

14. **`src/retrievers/graphrag.py`** ⭐ **Main innovation**
    - Combines dense search + graph signals
    - Query parsing (spaCy NER + noun chunks)
    - Entity seeding for PPR
    - Score combination: `sim + α·PPR + β·relation_bonus`
    - **This is where the magic happens**

#### 3e. Generation & Evaluation

15. **`src/generation/local_llm.py`**
    - llama.cpp wrapper
    - Metal acceleration setup
    - Answer generation from retrieved passages

16. **`src/eval/evaluate.py`**
    - Evaluation pipeline
    - Metrics: Recall@k, MRR@k
    - Grid search for hyperparameters (α, β)
    - Qualitative examples

### Phase 4: Utilities & Details

17. **`src/utils/device.py`**
    - Device detection (MPS/CUDA/CPU)
    - Used throughout for GPU acceleration

18. **`src/eval/make_eval_set.py`**
    - How evaluation datasets are created
    - Single-hop vs multi-hop question generation

---

## 🔄 Data Flow Summary

```
1. Wikipedia Pages (raw JSONL)
   ↓ [ingestion/wikipedia.py]
   
2. Chunks (processed JSONL)
   ↓ [processing/chunk.py]
   
3. Embeddings (numpy array) + FAISS Index
   ↓ [embeddings/embedder.py, index/faiss_index.py]
   
4. Knowledge Graph (NetworkX)
   ↓ [graph/build_graph.py]
   
5. Query → Retrieval
   ├─ Baseline: [retrievers/baseline.py] → FAISS search
   └─ GraphRAG: [retrievers/graphrag.py] → FAISS + PPR + relations
   
6. Retrieved Chunks → LLM → Answer
   ↓ [generation/local_llm.py]
```

---

## 🎓 Key Concepts to Understand

### 1. **Heterogeneous Graph**
- Two node types: documents (chunks) and entities
- Multiple edge types with different semantics
- Used for graph-based retrieval signals

### 2. **Personalized PageRank (PPR)**
- Graph algorithm that scores nodes based on proximity to seed nodes
- Seeds come from query entities/titles
- Higher PPR = more relevant to query context

### 3. **Score Combination**
```python
combined_score = similarity + α·PPR + β·relation_bonus
```
- `similarity`: Cosine similarity from dense embeddings
- `PPR`: Graph-based relevance score
- `relation_bonus`: Extra boost for specific relation types

### 4. **Entity Matching**
- Query → spaCy NER → entity names
- Match entities to graph nodes
- Used to seed PPR algorithm

### 5. **Evaluation Metrics**
- **Recall@k**: How many relevant chunks in top-k?
- **MRR@k**: Mean Reciprocal Rank (position of first relevant chunk)
- **ROUGE-L**: Proxy for answer quality (not implemented, just mentioned)

---

## 🔍 Questions to Answer While Reading

1. **Why does GraphRAG underperform the baseline?**
   - Check `docs/results_report.md` error analysis
   - Look at PPR magnitudes in `graphrag.py`
   - Entity extraction quality in `entity_matching.py`

2. **How does the graph structure help?**
   - Multi-hop reasoning potential
   - Entity co-occurrence signals
   - Document relationships

3. **What are the limitations?**
   - Entity extraction (spaCy small model)
   - Graph noise from PMI co-occurrence
   - Score scaling issues (PPR vs similarity magnitudes)

4. **How could this be improved?**
   - Better entity linking
   - BM25 hybrid retrieval
   - Richer relation types
   - Sentence-level graph nodes

---

## 💡 Tips for Reading

1. **Start with the CLI** (`demo.py`) - it shows the big picture
2. **Follow a query through the system** - trace `ask()` command execution
3. **Compare the two retrievers** - baseline vs graphrag side-by-side
4. **Look at actual data** - check `data/eval/examples_*.json` for concrete examples
5. **Run the code** - execute commands to see outputs

---

## 🚀 Quick Start Commands

```bash
# See the pipeline in action
python -m src.cli.demo ask "What is edge computing?" --mode baseline
python -m src.cli.demo ask "What is edge computing?" --mode graphrag

# Run evaluation
python -m src.eval.evaluate
```

---

## 📝 Next Steps After Reading

1. Experiment with different α/β values
2. Try improving entity extraction
3. Add BM25 to the retrieval pipeline
4. Create better multi-hop evaluation questions
5. Visualize the graph structure

---

**Happy reading! 🎉**

