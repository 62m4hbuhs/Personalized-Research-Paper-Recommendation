# 🔄 Data Pipeline & Workflow

## End-to-End Pipeline Workflow

The project's end-to-end data lifecycle encompasses six sequential stages, transitioning from raw web-scale paper metadata to real-time recommendation inference:

```
[Raw Corpus Ingestion]
        │
        ▼
[Data Sanitization & Deduplication]
        │
        ▼
[Dual Feature Generation: SciBERT + Fallback Projection]
        │
        ▼
[FAISS Index Creation & Verification]
        │
        ▼
[Citation Graph Structuring & Edge Validation]
        │
        ▼
[GraphSAGE Inductive Training & Graph Embeddings]
        │
        ▼
[Hybrid Serving via FastAPI & React]
```

## Detailed Stage Breakdown

### 1. Ingestion (`backend/dataset_collector.py` & `expanddataset.py`)
- Source: Semantic Scholar Academic Graph API.
- Seeds: Leading papers in AI, Machine Learning, Graph Neural Networks, Computer Vision, and NLP.
- Snowball Citation Expansion: Follows outbound references and inbound citations up to 2 hops.
- Raw outputs saved in `data/raw/` (`citations.csv`, `connected_papers.jsonl`).

### 2. Sanitization & Normalization (`data/processed/papers_clean.csv`)
- Deduplication by `paperId`.
- Missing abstract filtering or synthesis.
- Citation count cleaning and publication year normalization.
- Total clean papers: **94,471 unique papers**.

### 3. Text Representation (`backend/embedding.py`)
- Model: `allenai/scibert_scivocab_uncased`.
- Sequence length: 256 tokens max with attention mask pooling (mean pooling over non-padded tokens).
- Output: 56,433 papers directly embedded with 768 dimensions (`scibert_embeddings.npy`).
- Vectors are L2-normalized so inner product equals cosine similarity.

### 4. Vector Indexing (`backend/build_faiss.py`)
- FAISS Index type: `IndexFlatIP` (Exact inner product search).
- Size: 56,433 vectors, 768 dimensions (`scibert.index`).
- Query latency: $< 5\text{ ms}$ on CPU/GPU.

### 5. Graph Topology & Feature Alignment (`backend/build_graph.py` & `preparegraph.py`)
- Graph size: 92,081 active nodes, 206,964 bidirectional message-passing edges (`edges_index.csv`).
- Feature mapping: Papers with precomputed SciBERT embeddings receive their exact 768-d vector.
- Papers in the citation graph lacking raw abstracts receive an aggregated neighborhood fallback embedding (`data/processed/graph/fallback_embeddings.npy`).
- Unified matrix: `node_features.npy` of shape `[92081, 768]`.

### 6. Inductive Graph Learning & Inference (`training/train_graphsage.py`)
- PyTorch Geometric `NeighborLoader` with neighbor fanouts $[15, 10]$.
- Leakage-safe edge split: 85% train, 5% val, 10% test.
- Binary Cross-Entropy loss with negative edge sampling.
- Forward pass generates 128-d graph embeddings saved to disk for real-time inference.

---
Back to [[System Architecture]] | Next: [[Dataset Specifications]]

