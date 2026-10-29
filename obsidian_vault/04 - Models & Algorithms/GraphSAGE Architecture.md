# 🔬 GraphSAGE Architecture

## 1. Why GraphSAGE?
Unlike transductive models (e.g., standard GCN, DeepWalk, Node2Vec) that require the entire graph during training and cannot generate embeddings for unseen test nodes without retraining, **GraphSAGE** (Hamilton et al., NeurIPS 2017) is an **inductive representation learning framework**. 

It learns topological aggregation functions that generalize across graph neighborhoods:
- Operates on node feature attributes (SciBERT 768-d vectors).
- Scales gracefully to large graphs via uniform neighborhood sampling.
- Capable of computing embeddings for newly added papers given their citations.

## 2. Mathematical Formulation
For node $v \in \mathcal{V}$ at layer $k \in \{1, \dots, K\}$:

1. **Neighborhood Aggregation**:
   $$\mathbf{h}_{\mathcal{N}(v)}^{(k)} \leftarrow \text{AGGREGATE}_k \left( \left\{ \mathbf{h}_u^{(k-1)}, \forall u \in \mathcal{N}(v) \right\} \right)$$
   Where the aggregation function is the **Mean Aggregator**:
   $$\mathbf{h}_{\mathcal{N}(v)}^{(k)} = \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(k-1)}$$

2. **Combination & Non-linear Projection**:
   $$\mathbf{h}_v^{(k)} \leftarrow \sigma \left( \mathbf{W}^{(k)} \cdot \left[ \mathbf{h}_v^{(k-1)} \mathbin{\Vert} \mathbf{h}_{\mathcal{N}(v)}^{(k)} \right] \right)$$
   Or in standard PyTorch Geometric `SAGEConv` formulation:
   $$\mathbf{h}_v^{(k)} = \mathbf{W}_1 \mathbf{h}_v^{(k-1)} + \mathbf{W}_2 \cdot \text{mean}_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(k-1)}$$

3. **L2 Normalization**:
   $$\mathbf{z}_v = \frac{\mathbf{h}_v^{(K)}}{\|\mathbf{h}_v^{(K)}\|_2}$$

## 3. Network Hyperparameters
- **Input Dimensions ($d_{in}$)**: 768 (SciBERT feature dimension)
- **Hidden Channels ($d_{hidden}$)**: 256
- **Output Channels ($d_{out}$)**: 128 (compact graph structural representation)
- **Number of Layers ($K$)**: 2
- **Activation Function**: ReLU + Dropout ($p = 0.2$)
- **Neighborhood Fanout**:
  - Layer 1: $S_1 = 15$ neighbors
  - Layer 2: $S_2 = 10$ neighbors
- **Total Parameters**:
  - `conv1`: $(768 \times 2) \times 256 + 256 = 393,472$
  - `conv2`: $(256 \times 2) \times 128 + 128 = 65,664$
  - Total trainable params: $\approx 459,136$

---
Back to [[Citation Graph & Fallback Embeddings]] | Next: [[Link Prediction & Training Strategy]]

