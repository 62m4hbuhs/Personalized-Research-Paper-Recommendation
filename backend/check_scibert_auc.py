import torch
import numpy as np
from sklearn.metrics import roc_auc_score

splits = torch.load("backend/models/edge_splits.pt")
val_pairs = splits["val_pairs"]
val_neg_pairs = splits["val_neg_pairs"]

feat = np.load("data/processed/graph/node_features.npy", mmap_mode="r")

val_pos_u = val_pairs[:, 0].numpy()
val_pos_v = val_pairs[:, 1].numpy()
val_neg_u = val_neg_pairs[:, 0].numpy()
val_neg_v = val_neg_pairs[:, 1].numpy()

pos_u_feat = feat[val_pos_u]
pos_v_feat = feat[val_pos_v]
neg_u_feat = feat[val_neg_u]
neg_v_feat = feat[val_neg_v]

pos_sim = (pos_u_feat * pos_v_feat).sum(axis=-1)
neg_sim = (neg_u_feat * neg_v_feat).sum(axis=-1)

print("Raw SciBERT Pos sim mean:", pos_sim.mean(), "std:", pos_sim.std())
print("Raw SciBERT Neg sim mean:", neg_sim.mean(), "std:", neg_sim.std())

labels = np.concatenate([np.ones_like(pos_sim), np.zeros_like(neg_sim)])
scores = np.concatenate([pos_sim, neg_sim])

auc = roc_auc_score(labels, scores)
print("Raw SciBERT AUC:", auc)

