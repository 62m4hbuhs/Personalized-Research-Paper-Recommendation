# 🧪 Leakage-Safe Evaluation Protocol

## 1. The Critical Pitfall: Information Leakage in GNNs
In standard machine learning, train-test leakage occurs if test features contaminate training. In Graph Neural Networks, a much more pernicious danger exists: **topological link leakage**.
- If a citation edge $(u, v)$ is used during neighborhood message passing in GraphSAGE, the model directly accesses $v$'s representation when computing $u$'s embedding.
- Testing link prediction on $(u, v)$ when the edge is present in the aggregation graph yields trivially inflated, false performance scores.

## 2. Rigorous Evaluation Protocol
To guarantee sound, publication-grade results:

1. **Edge-Level Tri-Partition**:
   $$\mathcal{E}_{total} = \mathcal{E}_{train} \mathbin{\dot{\cup}} \mathcal{E}_{val} \mathbin{\dot{\cup}} \mathcal{E}_{test}$$
   - $85\%$ for training graph message passing ($\approx 175,920$ edges).
   - $5\%$ for validation ($10,348$ positive links, $10,348$ sampled negative links).
   - $10\%$ for test ($20,696$ positive links, $20,696$ sampled negative links).

2. **Graph Masking During Evaluation**:
   When computing validation or test metrics:
   $$\mathcal{G}_{eval} = (\mathcal{V}, \mathcal{E}_{train}, \mathbf{X})$$
   The validation and test edges are strictly **excised** from the message-passing adjacency matrix. The model must predict whether $(u, v)$ should exist relying solely on $u$'s and $v$'s neighborhood features gathered via training edges.

3. **Comparison Baseline vs. Proposed**:
   - **Baseline 1**: Semantic-Only (SciBERT Cosine Similarity).
   - **Baseline 2**: Pure Graph Topology (Node2Vec / Random Walk / Adamic-Adar).
   - **Proposed System**: Inductive GraphSAGE + SciBERT Hybrid Fusion.

---
Back to [[React Frontend Architecture]] | Next: [[Evaluation Metrics]]

