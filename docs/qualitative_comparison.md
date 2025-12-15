# Qualitative Baseline vs GraphRAG Comparison

## Question
What is a large language model?

### Baseline (Retrieval + TinyLlama Answer)

```
Baseline top-3                                 
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ Rank ┃ Chunk ID               ┃ Title                ┃ Score ┃   Sim ┃   PPR ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩
│  1   │ Large_language_model:… │ Large language model │ 0.852 │ 0.852 │ 0.000 │
│  2   │ Large_language_model:… │ Large language model │ 0.852 │ 0.852 │ 0.000 │
│  3   │ Language_model::chunk… │ Language model       │ 0.828 │ 0.828 │ 0.000 │
└──────┴────────────────────────┴──────────────────────┴───────┴───────┴───────┘

Answer:

A large languaage model (LLM) is a languaage model trained with self-supervised 
machine learning on a vast amount of text, designed for natural languaage 
processing tasks, especially languaage generation. The largest and most capable 
LLMs are generative pre-trained transformers (GPTs) and provide the core 
capabilities of chatbot sophisticated paraellelizatation, longer context 
handling, longer context handling, few-shot learning, and compositional 
reasoning, factual accuracy, and safety. Hillman-style experiments in 1997, 
which provided the first papers andgiven byes to fine-tuned to fine-supervised 
learning.
```

### GraphRAG (Retrieval + TinyLlama Answer)

```
Graphrag top-3                                 
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ Rank ┃ Chunk ID              ┃ Title                 ┃ Score ┃   Sim ┃   PPR ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩
│  1   │ BLOOM_(language_mode… │ BLOOM (language       │ 1.102 │ 0.782 │ 0.001 │
│      │                       │ model)                │       │       │       │
│  2   │ Algebraic_modeling_l… │ Algebraic modeling    │ 1.069 │ 0.729 │ 0.000 │
│      │                       │ language              │       │       │       │
│  3   │ Large_language_model… │ Large language model  │ 1.063 │ 0.852 │ 0.039 │
└──────┴───────────────────────┴───────────────────────┴───────┴───────┴───────┘

Graph context
BLOOM_(language_model)::chunk-0: Entities: 38%, AI, BLOOM (language model), 
BigScience, Bigscience; Linked docs: AI-assisted software development, 
Actor-critic algorithm, Adversarial machine learning
Algebraic_modeling_language::chunk-0: Entities: AIMMS, AML, AMPL, Algebraic 
modeling language, DATAFORM; Linked docs: Actor model, Computer, Decision 
support system
Large_language_model::chunk-0: Entities: 2017, BERT, Benchmark, Claude, GPT; 
Linked docs: 1.58-bit large language model, AAAI Conference on Artificial 
Intelligence, AI-assisted software development

Answer:

A large language model (LLM) is a language model trained with self-supervised 
machine learning on a vast amount of text, designed for natural language 
processing tasks, especially language generation.
```

## Question
How do the Internet of Things and edge computing complement each other when a deployment needs low-latency intelligence?

### Baseline (Retrieval + TinyLlama Answer)

```
Baseline top-3                                 
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ Rank ┃ Chunk ID                 ┃ Title              ┃ Score ┃   Sim ┃   PPR ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩
│  1   │ Internet_of_things::chu… │ Internet of things │ 0.787 │ 0.787 │ 0.000 │
│  2   │ Edge_computing::chunk-0  │ Edge computing     │ 0.785 │ 0.785 │ 0.000 │
│  3   │ Edge_computing::chunk-2  │ Edge computing     │ 0.785 │ 0.785 │ 0.000 │
└──────┴──────────────────────────┴────────────────────┴───────┴───────┴───────┘

Answer:

The Internet of Things (IoT) and edge computing complement each other when a 
deployment needs low-latecy intelligence. IoT devices, such as IoT sensors, 
actuators, and IoT gateways, are networked to the cloud, and the cloud is used 
to store and process data. Edge computing is a distributed computing model that 
brings computation and data storage closer to the sources of data. More broadly,
it refers to any design that pushes computation physically closer to a user, so 
as to reduce the latency compared to when an application runs on a centralized 
data center. Edge computing is often equated with fog computing, particularly in
smaller setup, particularly for applications needing immediate data processing. 
Edge computing is often linked with edge computing services, which are used
```

### GraphRAG (Retrieval + TinyLlama Answer)

```
Graphrag top-3                                 
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ Rank ┃ Chunk ID              ┃ Title                 ┃ Score ┃   Sim ┃   PPR ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩
│  1   │ Internet_of_things::… │ Internet of things    │ 1.064 │ 0.724 │ 0.000 │
│  2   │ Fog_computing::chunk… │ Fog computing         │ 1.051 │ 0.731 │ 0.000 │
│  3   │ EDGE_(telecommunicat… │ EDGE                  │ 1.034 │ 0.694 │ 0.000 │
│      │                       │ (telecommunication)   │       │       │       │
└──────┴───────────────────────┴───────────────────────┴───────┴───────┴───────┘

Graph context
Internet_of_things::chunk-13: Entities: 16.6 billion, 2020, 2023, 2025, 
Edge/Fog; Linked docs: 3D optical data storage, Advanced driver-assistance 
system, Amazon (company)
Fog_computing::chunk-1: Entities: 2012, Assisted Living, Automated 
Identification Data Carriers, Cisco, Cisco Sr; Linked docs: Assembly language, 
Augmented reality, Cloud computing
EDGE_(telecommunication)::chunk-0: Entities: 2003, 384 kbit/s. Later, 3GPP, 
AT&T, Cingular; Linked docs: AMR radiotelephone network (Czechoslovakia), 
Advanced Mobile Phone System, Advanced Mobile Telephone System

Answer:

The Internet of Things (IoT) and edge computing complement each other when a 
deployment needs low-latency intelligence. IoT devices are often 
resource-constrained and have limited computational abilities to perform 
cryptography computations. A fog node can provide security for IoT devices by 
performing these cryptographic computation
s instead. Edge computing is a subset of cloud computing that provides storage, 
applications, and data to end-users. Edge computing is closer to end-users and 
has wider geographical distribution. Edge nodes can be deployed in the cloud, 
enabling the deployment of IoT devices in remote areas. Edge computing is a 
subset of cloud computing that provides storage, applications, and data to 
end-users. Edge computing is closer to end-users and has wider geographical 
distribution
```

## Question
Which training approach produces large language models, and what transformer innovation enables them to scale?

### Baseline (Retrieval + TinyLlama Answer)

```
Baseline top-3                                 
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ Rank ┃ Chunk ID               ┃ Title                ┃ Score ┃   Sim ┃   PPR ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩
│  1   │ Large_language_model:… │ Large language model │ 0.825 │ 0.825 │ 0.000 │
│  2   │ Large_language_model:… │ Large language model │ 0.825 │ 0.825 │ 0.000 │
│  3   │ Large_language_model:… │ Large language model │ 0.799 │ 0.799 │ 0.000 │
└──────┴────────────────────────┴──────────────────────┴───────┴───────┴───────┘

Answer:

The transformer architecture, introduced in 2017, replaced recurreccre with 
self-attention, allowing efficient paraellelizaton, longer context handling, and
scalable training on unprecendedted data volumes. This innovation enabled models
like GPT, BERT, and their successores, which demonstrated emergent behaviors at 
scale such as few-shot learning and compositional reasoning.
```

### GraphRAG (Retrieval + TinyLlama Answer)

```
Graphrag top-3                                 
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ Rank ┃ Chunk ID              ┃ Title                 ┃ Score ┃   Sim ┃   PPR ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩
│  1   │ Transformer_(deep_le… │ Transformer (deep     │ 1.124 │ 0.784 │ 0.000 │
│      │                       │ learning              │       │       │       │
│      │                       │ architecture)         │       │       │       │
│  2   │ Large_language_model… │ Large language model  │ 1.119 │ 0.799 │ 0.000 │
│  3   │ Large_language_model… │ Large language model  │ 1.119 │ 0.799 │ 0.000 │
└──────┴───────────────────────┴───────────────────────┴───────┴───────┴───────┘

Graph context
Transformer_(deep_learning_architecture)::chunk-4: Entities: 2%, 2018, 2019 
October, 2020, 2021; Linked docs: AAAI Conference on Artificial Intelligence, 
Active learning (machine learning), Adversarial machine learning
Large_language_model::chunk-1: Entities: 2000, 2001, 2012, 2013, 2014; Linked 
docs: 1.58-bit large language model, AAAI Conference on Artificial Intelligence,
Active learning (machine learning)
Large_language_model::chunk-1: Entities: 2000, 2001, 2012, 2013, 2014; Linked 
docs: 1.58-bit large language model, AAAI Conference on Artificial Intelligence,
Active learning (machine learning)

Answer:

The plain transformer architecture has difficulty in converging, in the original
paper, the authors recommended using learning rate warmup. That is, the learning
rate should linearly scale up from 0 to maximal value for the first part of the 
training (usually recommended to be 2% of the total number of training steps), 
before decaying again. In the original paper, the authors recommended using 
layer normalization before (instead of after) multihead attention and 
feedforward layers stabilizes training, not requiring learning rate warmup.
```

## Question
What is explainable artificial intelligence and why is it important?

### Baseline (Retrieval + TinyLlama Answer)

```
Baseline top-3                                 
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ Rank ┃ Chunk ID              ┃ Title                 ┃ Score ┃   Sim ┃   PPR ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩
│  1   │ Explainable_artifici… │ Explainable           │ 0.850 │ 0.850 │ 0.000 │
│      │                       │ artificial            │       │       │       │
│      │                       │ intelligence          │       │       │       │
│  2   │ Explainable_artifici… │ Explainable           │ 0.828 │ 0.828 │ 0.000 │
│      │                       │ artificial            │       │       │       │
│      │                       │ intelligence          │       │       │       │
│  3   │ Explainable_artifici… │ Explainable           │ 0.813 │ 0.813 │ 0.000 │
│      │                       │ artificial            │       │       │       │
│      │                       │ intelligence          │       │       │       │
└──────┴───────────────────────┴───────────────────────┴───────┴───────┴───────┘

Answer:

Explainable artificial intelligence (XAI) is a field of research that explores 
methods that provide humans with the ability of intellectual over-sight over AI 
algorithms. The main focus is on the reasoning behind the decision made by the 
AI algorithm, to make them more understandable and transparent. XAI aims to help
users of AI-powered systems perform more effectively by improving their 
understaning of how the AI system reasoned. XAI hopes to help users to assess 
the safety and scrutinize the AI's design. XAI generally overlaps with 
interpretable artificial intelligence or explainable machine learning (XEML), 
which is a field of research that explores methods that provide humans with the 
ability of intellectual over-sight over A
```

### GraphRAG (Retrieval + TinyLlama Answer)

```
Graphrag top-3                                 
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ Rank ┃ Chunk ID              ┃ Title                 ┃ Score ┃   Sim ┃   PPR ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩
│  1   │ Explainable_artifici… │ Explainable           │ 1.061 │ 0.850 │ 0.038 │
│      │                       │ artificial            │       │       │       │
│      │                       │ intelligence          │       │       │       │
│  2   │ Explainable_artifici… │ Explainable           │ 1.028 │ 0.828 │ 0.000 │
│      │                       │ artificial            │       │       │       │
│      │                       │ intelligence          │       │       │       │
│  3   │ Explainable_artifici… │ Explainable           │ 1.013 │ 0.813 │ 0.000 │
│      │                       │ artificial            │       │       │       │
│      │                       │ intelligence          │       │       │       │
└──────┴───────────────────────┴───────────────────────┴───────┴───────┴───────┘

Graph context
Explainable_artificial_intelligence::chunk-0: Entities: AI, Algorithmic 
Transparency, Background, Explainable artificial intelligence, ML; Linked docs: 
AI-assisted software development, Algorithm, Algorithm aversion
Explainable_artificial_intelligence::chunk-10: Entities: AI, Explainable 
artificial intelligence, ML; Linked docs: AI-assisted software development, 
Algorithm, Algorithmic transparency
Explainable_artificial_intelligence::chunk-1: Entities: AI, Algorithmic 
Transparency, Bottleneck Models, Daniel Day-Lewis', Explainable artificial 
intelligence; Linked docs: AI-assisted software development, Algorithm, 
Algorithmic transparency

Answer:

Explainable artificial intelligence (XAI) is a field of research that explores 
methods that provide humans with the ability of intellectual over-sight over AI 
algorithms. The main focus is on the reasoning behind the decision made by the 
AI algorithm, to make them more understandable and transparent. XAI aims to help
users of AI-powered systems perform more effectively by improving their 
understaning of how the AI system reasoned. XAI hopes to help users to assess 
the safety and scrutinize the AI's design. XAI generally overlaps with 
interpretable artificial intelligence or explainable machine learning (XEML), 
which is a field of research that explores methods that provide humans with the 
ability of intellectual over-sight over A
```

## Question
How do autonomous robots differ from robotics as a field?

### Baseline (Retrieval + TinyLlama Answer)

```
Baseline top-3                                 
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ Rank ┃ Chunk ID                ┃ Title               ┃ Score ┃   Sim ┃   PPR ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩
│  1   │ Autonomous_robot::chun… │ Autonomous robot    │ 0.772 │ 0.772 │ 0.000 │
│  2   │ Outline_of_robotics::c… │ Outline of robotics │ 0.772 │ 0.772 │ 0.000 │
│  3   │ Autonomous_robot::chun… │ Autonomous robot    │ 0.770 │ 0.770 │ 0.000 │
└──────┴─────────────────────────┴─────────────────────┴───────┴───────┴───────┘

Answer:

Autonomous robotics is a branch of robotics that deals with the design, 
construction, operation, and application of robotic systems that can take the 
place of humans in dangerous environments or manufacturing processes, or 
resemble humans in appearance, behavior, and or cognition. Robotic systems that 
are autonomous are not necessarily robots, but rather, they are systems that are
designed to be autonomous. Autonomous robotics differs from robotics in that it 
is concerned with developing and studying robotic systems that are entirely or 
in some way human-like. Autonomous robotics is concerned with developing and 
studying robotic systems that are either entirely or in some way human-like.
```

### GraphRAG (Retrieval + TinyLlama Answer)

```
Graphrag top-3                                 
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ Rank ┃ Chunk ID                  ┃ Title            ┃ Score ┃   Sim ┃   PPR ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩
│  1   │ Autonomous_robot::chunk-2 │ Autonomous robot │ 1.096 │ 0.756 │ 0.000 │
│  2   │ Autonomous_robot::chunk-0 │ Autonomous robot │ 0.982 │ 0.767 │ 0.050 │
│  3   │ Autonomous_robot::chunk-4 │ Autonomous robot │ 0.972 │ 0.772 │ 0.000 │
└──────┴───────────────────────────┴──────────────────┴───────┴───────┴───────┘

Graph context
Autonomous_robot::chunk-2: Entities: AI, Autonomous robot, Biological 
Inspiration to Implementation and Control, Falcon, George A. Bekey's; Linked 
docs: Aerobot, Agricultural robot, Amazon (company)
Autonomous_robot::chunk-0: Entities: 2002, Aibo, Autonomous robot, Electrolux, 
Hall; Linked docs: Aerobot, Agricultural robot, Amazon (company)
Autonomous_robot::chunk-4: Entities: 2021, AI, Autonomous robot, Biohybrid, 
Deep; Linked docs: Aerobot, Agricultural robot, Amazon (company)

Answer:

Autonomous robotics is a field that focuses on the development of autonomous 
robots, which are robots that can operate autonomously without human 
intervention. Robots that are autonomous are different from robots that are not 
autonomous. Autonomous robots are designed to operate in a specific environment,
and they are programmed to perform specific tasks without human intervention. 
Autonomous robots are different from robotics as a field because they focus on 
the development of autonomous robots, which are robots that can operate 
autonomously without human intervention. Robots that are autonomous are designed
to operate in a specific environment, and they are programmed to perform 
specific tasks without human intervention. Autonomous robots are different from 
robotics as a field
```

## Question
What privacy requirement motivates federated learning, and how does edge computing help organizations meet it?

### Baseline (Retrieval + TinyLlama Answer)

```
Baseline top-3                                 
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ Rank ┃ Chunk ID                 ┃ Title              ┃ Score ┃   Sim ┃   PPR ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩
│  1   │ Edge_computing::chunk-2  │ Edge computing     │ 0.756 │ 0.756 │ 0.000 │
│  2   │ Federated_learning::chu… │ Federated learning │ 0.755 │ 0.755 │ 0.000 │
│  3   │ Federated_learning::chu… │ Federated learning │ 0.755 │ 0.755 │ 0.000 │
└──────┴──────────────────────────┴────────────────────┴───────┴───────┴───────┘

Answer:

Privacy requirement motivates federated learning because it is a distributed 
system that allows the data to be kept close to the end-users. Edge computing 
helps organizations meet the privacy requirement by providing a secure and 
reliable infrastructure for the data to be processed. Edge computing systems are
designed to be highly scalable, and the data is kept close to the end-users. 
This ensures that the data is not exposed to the cloud, which is a privacy 
concern. The edge computing system can also provide additional security 
features, such as hardware-based encryption, that are not possible in the cloud.
Additionally, edge computing systems can provide faster response times, which is
a critical requirement for real-time applications such as autonomous driving, 
where the response time is crucial.
```

### GraphRAG (Retrieval + TinyLlama Answer)

```
Graphrag top-3                                 
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ Rank ┃ Chunk ID                 ┃ Title              ┃ Score ┃   Sim ┃   PPR ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩
│  1   │ Federated_learning::chu… │ Federated learning │ 0.966 │ 0.755 │ 0.038 │
│  2   │ Edge_computing::chunk-2  │ Edge computing     │ 0.957 │ 0.756 │ 0.001 │
│  3   │ Federated_learning::chu… │ Federated learning │ 0.955 │ 0.755 │ 0.000 │
└──────┴──────────────────────────┴────────────────────┴───────┴───────┴───────┘

Graph context
Federated_learning::chunk-0: Entities: Federated, Federated learning, IoT, 
\mathbf; Linked docs: Adversarial machine learning, Backdoor (computing), 
Bandwidth (computing)
Edge_computing::chunk-2: Entities: 370-620, Edge computing, IoT, Reliability
Management, wearer; Linked docs: Artificial intelligence, Assembly language, 
Augmented reality
Federated_learning::chunk-13: Entities: A Systematic Review of Federated 
Learning, AI, COVID-19, FL, Fed; Linked docs: Backdoor (computing), Bandwidth 
(computing), Boosting (machine learning)

Answer:

Privacy requirement motivates federated learning because it ensures that the 
data is not shared with the cloud, which is a critical concern for organizations
that are subject to data privacy laws. Edge computing helps organizations meet 
the privacy requirement by providing a local computing platform that is highly 
secure, and the data is not sent to the cloud. This ensures that the data 
remains local and is not subject to the same privacy concerns as data sent to 
the cloud.
```
