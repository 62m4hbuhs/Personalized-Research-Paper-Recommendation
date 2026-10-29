# 🔗 Link Prediction & Training Strategy

## 1. Task Definition: Unsupervised / Self-Supervised Link Prediction
In citation networks, explicit node labels (e.g. topic tags) are noisy or scarce. However, **citation links** represent direct ground-truth intellectual and methodological affinity. 

We train GraphSAGE to solve **Citation Link Prediction**:
Given paper node $u$ and paper node $v$, predict whether a citation relationship exists between them:
$$\hat{y}_{uv} = \sigma(\mathbf{z}_u^\top \mathbf{z}_v)$$
where $\mathbf{z}_u, \mathbf{z}_v \in \mathbb{R}^{128}$ are the normalized GraphSAGE node representations.

## 2. Leakage-Safe Edge Splitting
To prevent information leakage during evaluation, edges are strictly split **prior** to message passing:

```
Total Edges: 206,964
├── Train Edges (85%): ~175,920 edges used in message-passing graph during training
├── Validation Edges (5%): ~10,348 held-out positive edges + 10,348 negative edges
└── Test Edges (10%): ~20,696 held-out positive edges + 20,696 negative edges
```

> [!IMPORTANT]
> Message passing during validation and testing must strictly omit test and validation edges from the computation graph. PyTorch Geometric's `RandomLinkSplit` automatically manages:
> 1. Removing supervisory edges from `edge_index` to prevent trivial shortcut learning.
> 2. Sampling equal numbers of true negative edges $(u_{neg}, v_{neg}) \notin \mathcal{E}$.

## 3. Negative Sampling Scheme
For every true citation edge $(u, v) \in \mathcal{E}_{pos}$, a negative edge $(u, v_{neg})$ is sampled such that:
1. $v_{neg} \in \mathcal{V}$ is uniformly sampled from the corpus.
2. $(u, v_{neg}) \notin \mathcal{E}$ (verifiably non-citing).
3. Optional hard negative sampling: picking papers with high textual similarity in FAISS but zero citation link.

## 4. Objective Function (BCE with Logits)
The model is trained using Binary Cross-Entropy Loss with logits:
$$\mathcal{L} = -\sum_{(u, v) \in \mathcal{E}_{pos}} \log \sigma(\mathbf{z}_u^\top \mathbf{z}_v) - \sum_{(u, v') \in \mathcal{E}_{neg}} \log \left( 1 - \sigma(\mathbf{z}_u^\top \mathbf{z}_{v'}) \right)$$

## 5. Training Loop Hyperparameters
- **Optimizer**: AdamW ($\text{lr} = 0.001$, weight decay = $1\times 10^{-4}$)
- **Batch Size**: 256 or 512 edges per step
- **Epochs**: 5 to 15 epochs with Early Stopping on Validation ROC-AUC
- **Metrics Tracked**:
  - Validation ROC-AUC (Area Under ROC Curve)
  - Average Precision (AP)
  - Mean Reciprocal Rank (MRR)

---
Back to [[GraphSAGE Architecture]] | Next: [[Hybrid Ranking & Score Fusion]]

