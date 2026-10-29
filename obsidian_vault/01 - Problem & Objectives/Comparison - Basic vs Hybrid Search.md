# ⚖️ Comparison: Basic Search vs. Hybrid Recommendation

## The Fundamental Divergence

| Dimension | Conventional Search (BM25 / Keyword) | Pure Semantic Search (Vector Embedding) | Proposed Hybrid Approach (Semantic + GraphSAGE) |
| :--- | :--- | :--- | :--- |
| **Primary Signal** | Exact lexical/token overlap (TF-IDF, BM25) | Dense vector distance in embedding space | Multi-modal: Content semantics + topological citation proximity |
| **Handling Synonyms** | ❌ Fails (exact vocabulary required) | ✅ High (vector proximity captures synonyms) | ✅ High (SciBERT embeddings capture domain terms) |
| **Citation Influence** | ⚠️ Only naive popularity (global citation count) | ❌ Completely ignored | ✅ Deep structural influence via inductive neighborhood aggregation |
| **Contextual Lineage**| ❌ None | ❌ None | ✅ Captures co-citation clusters, intellectual lineage, and direct referencing |
| **Cold Start Behavior**| ⚠️ Relies purely on keyword frequency | ✅ Works well on abstract text | ✅ Semantic primary search with graceful fallback when graph links are sparse |
| **Explainability** | Keyword highlighted in document | "High cosine similarity score" (opaque) | Decomposed: Semantic affinity + Graph citation path/cluster context |

## Empirical Justification
- **Semantic-Only Flaws**: Two papers discussing "Graph Neural Networks" may have identical cosine similarity to a query, yet one may be a retracted preprint while the other is Kipf & Welling's foundational ICLR 2017 paper cited 15,000 times. Dense text alone cannot distinguish academic consensus and authority.
- **Graph-Only Flaws**: A graph random walk or PageRank favors heavily-cited legacy papers from 2012, neglecting modern break-through papers with low in-degree simply because they were published last month.
- **The Synergy**: Combining dense semantic retrieval (recall phase) with GNN structural ranking (precision & authority phase) produces superior precision-at-K and mean reciprocal rank.

---
Back to [[Index]] | Next: [[System Architecture]]

