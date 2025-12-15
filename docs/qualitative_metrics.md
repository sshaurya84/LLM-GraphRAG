# Qualitative Metrics Comparison

## Dataset: single_hop

| Model | Comprehensiveness | Diversity | Directness | Empowerment | Answer Length |
|-------|-------------------|-----------|------------|-------------|----------------|
| Baseline | 0.917 | 0.583 | 0.917 | 0.011 | 358.0 |
| Graphrag | 1.000 | 0.600 | 0.967 | 0.010 | 358.9 |

### Questions where GraphRAG improved

- **Comprehensiveness**: 1 question(s)
- What is A.I. Artificial Intelligence?
- **Diversity**: 4 question(s)
- What is A.I. Artificial Intelligence?
- What is AI-assisted software development?
- What is Active learning (machine learning)?
- What is 1.58-bit large language model?
- **Directness**: 1 question(s)
- What is A.I. Artificial Intelligence?
- **Empowerment**: 0 question(s)
- (none)

## Dataset: multi_hop

| Model | Comprehensiveness | Diversity | Directness | Empowerment | Answer Length |
|-------|-------------------|-----------|------------|-------------|----------------|
| Baseline | 0.823 | 0.425 | 1.000 | 0.015 | 404.2 |
| Graphrag | 0.875 | 0.588 | 1.000 | 0.010 | 400.7 |

### Questions where GraphRAG improved

- **Comprehensiveness**: 3 question(s)
- Which data model defines a knowledge graph, and which neural network family is tailored to operate directly on that structure?
- How do augmented reality systems depend on computer vision to align overlays with the real world?
- How do computational biology and bioinformatics collaborate on biological data, and how does computational chemistry apply similar techniques elsewhere?
- **Diversity**: 9 question(s)
- How do the Internet of Things and edge computing complement each other when a deployment needs low-latency intelligence?
- Which data model defines a knowledge graph, and which neural network family is tailored to operate directly on that structure?
- Which training approach produces large language models, and what transformer innovation enables them to scale?
- How do augmented reality systems depend on computer vision to align overlays with the real world?
- Which paradigm offers elastic pools of shared resources on demand, and how does edge computing adapt that model when latency matters?
- Why does the growth of the Internet of Things raise computer security stakes?
- How are cyber-physical systems related to the Internet of Things, and what distinguishes CPS deployments?
- How does high-performance computing make use of supercomputers, and how is supercomputer performance typically compared?
- How do computational biology and bioinformatics collaborate on biological data, and how does computational chemistry apply similar techniques elsewhere?
- **Directness**: 0 question(s)
- (none)
- **Empowerment**: 2 question(s)
- What interaction challenge does human–computer interaction study, and how do augmented reality interfaces illustrate it?
- Why does the growth of the Internet of Things raise computer security stakes?
