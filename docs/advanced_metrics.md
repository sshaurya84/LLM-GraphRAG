# Advanced Metrics Evaluation

These metrics are designed to capture GraphRAG's strengths:
- Multi-hop reasoning (multi-fact coverage)
- Source diversity (unique pages)
- Answer quality (semantic similarity)

# Advanced Metrics: single_hop

## Summary Comparison

| Metric | Baseline | GraphRAG | Winner | GraphRAG Wins |
|--------|----------|----------|--------|---------------|
| Multi Fact Coverage | 0.917 | 1.000 | **GraphRAG** | 1/12 |
| Unique Pages | 0.583 | 0.600 | **GraphRAG** | 2/12 |
| Entity Density | 2.302 | 2.103 | **Baseline** | 6/12 |
| Semantic Similarity | 0.910 | 0.887 | **Baseline** | 0/12 |
| Context Utilization | 1.000 | 1.000 | **Tie** | 0/12 |
| Information Density | 0.460 | 0.467 | **GraphRAG** | 2/12 |

## Metric Definitions

- **Multi-Fact Coverage**: Fraction of required supporting pages found in top-k results
- **Unique Pages**: Diversity of sources (unique page titles / k)
- **Entity Density**: Approximate count of named entities per 100 words
- **Semantic Similarity**: Cosine similarity between answer and reference
- **Context Utilization**: Fraction of retrieved chunks reflected in answer
- **Information Density**: Ratio of unique content words to total words

## Questions Where GraphRAG Excels

### Multi Fact Coverage (1 wins)
- What is A.I. Artificial Intelligence?

### Unique Pages (2 wins)
- What is Active learning (machine learning)?
- What is 1.58-bit large language model?

### Semantic Similarity (0 wins)
- (none)


# Advanced Metrics: multi_hop

## Summary Comparison

| Metric | Baseline | GraphRAG | Winner | GraphRAG Wins |
|--------|----------|----------|--------|---------------|
| Multi Fact Coverage | 1.000 | 1.000 | **Tie** | 0/16 |
| Unique Pages | 0.425 | 0.550 | **GraphRAG** | 8/16 |
| Entity Density | 2.389 | 2.381 | **Baseline** | 9/16 |
| Semantic Similarity | 0.798 | 0.794 | **Baseline** | 3/16 |
| Context Utilization | 1.000 | 1.000 | **Tie** | 0/16 |
| Information Density | 0.463 | 0.435 | **Baseline** | 2/16 |

## Metric Definitions

- **Multi-Fact Coverage**: Fraction of required supporting pages found in top-k results
- **Unique Pages**: Diversity of sources (unique page titles / k)
- **Entity Density**: Approximate count of named entities per 100 words
- **Semantic Similarity**: Cosine similarity between answer and reference
- **Context Utilization**: Fraction of retrieved chunks reflected in answer
- **Information Density**: Ratio of unique content words to total words

## Questions Where GraphRAG Excels

### Multi Fact Coverage (0 wins)
- (none)

### Unique Pages (8 wins)
- How do the Internet of Things and edge computing complement each other when a deployment needs low-latency intelligence?
- Which data model defines a knowledge graph, and which neural network family is tailored to operate directly on that structure?
- Which training approach produces large language models, and what transformer innovation enables them to scale?
- How do augmented reality systems depend on computer vision to align overlays with the real world?
- Why does the growth of the Internet of Things raise computer security stakes?

### Semantic Similarity (3 wins)
- How do the Internet of Things and edge computing complement each other when a deployment needs low-latency intelligence?
- What privacy requirement motivates federated learning, and how does edge computing help organizations meet it?
- What capability distinguishes an autonomous robot, and how does robotics as a field enable building such machines?

