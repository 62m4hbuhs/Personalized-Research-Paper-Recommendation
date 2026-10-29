# 📈 Evaluation Metrics

## 1. Classification & Discrimination Metrics
For the link prediction binary classification task on held-out edges:

### ROC-AUC (Area Under Receiver Operating Characteristic Curve)
Measures the probability that a randomly chosen true citation edge is ranked higher than a randomly chosen non-citation edge:
$$\text{ROC-AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}^{-1}(t)) \, dt$$

### AP (Average Precision / PR-AUC)
Measures precision averaged across all recall thresholds, especially robust for edge-imbalance:
$$\text{AP} = \sum_n (R_n - R_{n-1}) P_n$$

## 2. Ranking & Recommendation Metrics
For assessing recommendation quality when ranking all candidate papers for a query/seed:

### Mean Reciprocal Rank (MRR)
Evaluates how high the first truly relevant citation appears in the ranked recommendation list:
$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
where $\text{rank}_i$ is the position of the first ground-truth cited paper for seed $i$.

### Hits@K ($K \in \{5, 10, 20\}$)
Proportion of queries where at least one ground-truth relevant/cited paper appears in the top-$K$ recommendations:
$$\text{Hits@}K = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \mathbb{I}(\text{rank}_i \le K)$$

### NDCG@K (Normalized Discounted Cumulative Gain)
Penalizes relevant recommendations that appear lower in the top-$K$ list:
$$\text{DCG@}K = \sum_{i=1}^K \frac{2^{rel_i} - 1}{\log_2(i + 1)}, \quad \text{NDCG@}K = \frac{\text{DCG@}K}{\text{IDCG@}K}$$

---
Back to [[Leakage-Safe Evaluation Protocol]] | Next: [[Implementation Status & Tasks]]

