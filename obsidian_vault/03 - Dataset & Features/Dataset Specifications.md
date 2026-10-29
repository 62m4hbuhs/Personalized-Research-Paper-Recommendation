# 📊 Dataset Specifications

## Data Artifacts & Locations

All processed datasets reside within `data/processed/`:

| Artifact | File Path | Format | Record Count / Dimensions | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| Cleaned Papers | `data/processed/papers_clean.csv` | CSV | 94,471 rows | Master metadata catalog (title, abstract, year, citations) |
| SciBERT Embeddings | `data/processed/embeddings/scibert_embeddings.npy` | NumPy Binary | 56,433 × 768 (`float32`) | Normalized dense semantic vectors |
| Paper ID Array | `data/processed/embeddings/paper_ids.json` | JSON List | 56,433 string IDs | Mapping from embedding row index to `paperId` |
| FAISS Index | `data/processed/faiss/scibert.index` | Binary FAISS | 56,433 vectors, $d=768$ | High-speed inner product search index |
| Graph Node Mapping | `data/processed/graph/node_mapping.csv` | CSV | 92,081 rows | Maps `paperId` $\leftrightarrow$ `node_idx` $[0 \dots 92080]$ |
| Graph Edge Index | `data/processed/graph/edges_index.csv` | CSV | 206,964 rows | Directed edge pairs $(u, v)$ for message passing |
| Unified Node Features | `data/processed/graph/node_features.npy` | NumPy Binary | 92,081 × 768 (`float32`) | Complete feature matrix for PyTorch Geometric |
| Fallback Embeddings | `data/processed/graph/fallback_embeddings.npy` | NumPy Binary | 35,648 × 768 (`float32`) | Synthesized embeddings for graph nodes without abstracts |
| Feature Mapping Stats | `data/processed/graph/final_feature_statistics.json` | JSON | Config & counts | Validation checkpoints of feature completeness |

## Quantitative Characteristics
- **Total Unique Papers in Metadata**: 94,471
- **Total Usable Citation Relationships**: 103,482 directed links
- **Graph Nodes**: 92,081
- **Graph Edges**: 206,964 (Bidirectional message passing: for each edge $u \to v$, reverse edge $v \to u$ is added to allow symmetric information flow in GNN neighborhood aggregation).
- **Average Node Degree**: $\approx 2.25$ citations/paper in sampled subgraph.
- **Node Feature Completeness**: 100% of the 92,081 nodes have valid 768-dimensional features (no `NaN`, no `Inf`).

---
Back to [[Index]] | Next: [[SciBERT & FAISS Vector Search]]

