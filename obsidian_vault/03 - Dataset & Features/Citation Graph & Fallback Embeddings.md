# 🕸️ Citation Graph & Fallback Embeddings

## 1. Graph Topology Formulation
The citation network is represented as an attributed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{X})$:
- $\mathcal{V}$: Set of 92,081 paper nodes.
- $\mathcal{E}$: Set of 206,964 edges. Each directed citation $u \to v$ ($u$ cites $v$) is augmented with the symmetric back-edge $v \to u$ ($v$ is cited by $u$) to facilitate two-way message passing during GNN neighborhood aggregation.
- $\mathbf{X} \in \mathbb{R}^{92,081 \times 768}$: Node feature matrix.

## 2. The Missing Feature Problem (Out-of-Vocabulary / Missing Abstracts)
In web-scale scholarly graphs, citation links often point to papers where only metadata (title, year) or raw IDs are present, but full abstracts are not available.
- Directly dropping non-embedded nodes fragments the citation graph and destroys valuable structural paths.
- Initializing missing features with zeros or random noise corrupts message passing and creates negative gradients during training.

## 3. Fallback Embedding Strategy (`generate_graph_fallback_embeddings.py`)
To solve this, the pipeline employs a structured feature synthesis strategy:
1. For graph nodes with existing SciBERT embeddings (56,433 papers): feature vector $\mathbf{x}_v$ is taken directly from `scibert_embeddings.npy`.
2. For orphan graph nodes (35,648 papers): feature vector is synthesized via **1-hop neighborhood aggregation**:
   $$\mathbf{x}_v = \frac{1}{|\mathcal{N}(v) \cap \mathcal{V}_{known}|} \sum_{u \in \mathcal{N}(v) \cap \mathcal{V}_{known}} \mathbf{x}_u$$
3. For isolated orphan nodes with no embedded neighbors, title-level embeddings or global domain mean vectors with slight Gaussian noise are applied and normalized.
4. Result: `node_features.npy` has shape `(92081, 768)` with $0$ NaN and $0$ Inf values, ensuring complete tensor consistency for PyTorch Geometric.

---
Back to [[SciBERT & FAISS Vector Search]] | Next: [[GraphSAGE Architecture]]

