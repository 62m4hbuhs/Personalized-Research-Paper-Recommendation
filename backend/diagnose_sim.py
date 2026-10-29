import torch
import numpy as np
from sklearn.metrics import roc_auc_score

splits = torch.load("backend/models/edge_splits.pt")
val_pairs = splits["val_pairs"]
val_neg_pairs = splits["val_neg_pairs"]

emb = np.load("data/processed/graph/graph_embeddings.npy")
emb = torch.from_numpy(emb)

val_pos_u = val_pairs[:, 0]
val_pos_v = val_pairs[:, 1]
val_neg_u = val_neg_pairs[:, 0]
val_neg_v = val_neg_pairs[:, 1]

pos_sim = (emb[val_pos_u] * emb[val_pos_v]).sum(dim=-1).numpy()
neg_sim = (emb[val_neg_u] * emb[val_neg_v]).sum(dim=-1).numpy()

print("Pos sim mean:", pos_sim.mean(), "std:", pos_sim.std())
print("Neg sim mean:", neg_sim.mean(), "std:", neg_sim.std())

labels = np.concatenate([np.ones_like(pos_sim), np.zeros_like(neg_sim)])
scores = np.concatenate([pos_sim, neg_sim])

auc = roc_auc_score(labels, scores)
print("AUC based purely on cosine similarity:", auc)

