import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

splits = torch.load("backend/models/edge_splits.pt")
val_pairs = splits["val_pairs"]
num_nodes = 92081

# Paired negative sampling: for each (u, v) in val_pairs, sample (u, v_neg)
u = val_pairs[:, 0]
v = val_pairs[:, 1]
v_neg = torch.randint(0, num_nodes, (len(u),))

feat = np.load("data/processed/graph/node_features.npy")
feat = torch.from_numpy(feat)
feat = F.normalize(feat, p=2, dim=-1)

pos_sim = (feat[u] * feat[v]).sum(dim=-1).numpy()
neg_sim = (feat[u] * feat[v_neg]).sum(dim=-1).numpy()

print(f"Paired SciBERT Pos sim mean: {pos_sim.mean():.4f}")
print(f"Paired SciBERT Neg sim mean: {neg_sim.mean():.4f}")

labels = np.concatenate([np.ones_like(pos_sim), np.zeros_like(neg_sim)])
scores = np.concatenate([pos_sim, neg_sim])
auc = roc_auc_score(labels, scores)
ap = average_precision_score(labels, scores)
print(f"Paired SciBERT Val ROC-AUC: {auc:.4f}, AP: {ap:.4f}")

