# LLM-GraphRAG

LLM-GraphRAG is a graph-augmented retrieval system built to improve how large language models answer questions from long-form documents.

Instead of relying only on vector similarity search, this project combines semantic search with knowledge graph relationships to retrieve more relevant and connected context before generating responses.

The goal of the project was to explore how retrieval pipelines can be improved for complex question answering tasks where relationships between entities matter.

## What the project does

The system:

- Extracts entities and relationships from documents
- Builds a lightweight knowledge graph from the extracted data
- Stores document embeddings for semantic retrieval
- Combines graph traversal with vector search during retrieval
- Sends the enriched context to an LLM for final response generation

This helps the model retrieve context that is not only semantically similar, but also logically connected.

## Tech Stack

Backend:
- Python
- LangChain
- Hugging Face Transformers
- FAISS
- NetworkX

LLM / Retrieval:
- RAG pipeline
- Graph-based retrieval
- Sentence embeddings
- Knowledge graph traversal

## Project Structure

```text
data/
docs/
scripts/
src/
```

## How to run

Clone the repository:

```bash
git clone https://github.com/sshaurya84/LLM-GraphRAG.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

## What I learned

This project helped me better understand:

- Retrieval-Augmented Generation (RAG)
- Vector databases and semantic search
- Knowledge graph construction
- Prompt orchestration using LangChain
- Tradeoffs between graph retrieval and dense retrieval systems
- Designing end-to-end LLM pipelines

## Future Improvements

Some ideas for future work:

- Neo4j integration for scalable graph storage
- Hybrid ranking for retrieval quality improvement
- Multi-document conversational memory
- Streaming responses
- Evaluation metrics for retrieval accuracy
