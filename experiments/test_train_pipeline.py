import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

print("=" * 70)
print("TESTING LEAKAGE-SAFE SPLIT & GRAPHSAGE FORWARD PASS")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# 1. Load data
features_path = "data/processed/graph/node_features.npy"
edges_path = "data/processed/graph/edges_index.csv"

print("\nLoading node features...")
x_np = np.load(features_path)
x = torch.from_numpy(x_np).to(device)
num_nodes = x.size(0)
print(f"Nodes: {num_nodes:,}, Dim: {x.size(1)}, Memory: {x.element_size() * x.nelement() / 1024**2:.2f} MB")

print("\nLoading edges...")
edges_df = pd.read_csv(edges_path)
src = torch.tensor(edges_df["source_id"].values, dtype=torch.long)
dst = torch.tensor(edges_df["target_id"].values, dtype=torch.long)
print(f"Total directed edges: {len(src):,}")

# Remove self-loops if any
mask = src != dst
src, dst = src[mask], dst[mask]

# Ensure unique undirected edge pairs for splitting
# To prevent leakage, if (u, v) is in val/test, neither u->v nor v->u should be in train!
u = torch.minimum(src, dst)
v = torch.maximum(src, dst)
unique_pairs = torch.unique(torch.stack([u, v], dim=1), dim=0)
num_pairs = unique_pairs.size(0)
print(f"Unique undirected citation pairs: {num_pairs:,}")

# Shuffle pairs
perm = torch.randperm(num_pairs)
n_val = int(0.05 * num_pairs)
n_test = int(0.10 * num_pairs)
n_train = num_pairs - n_val - n_test

train_pairs = unique_pairs[perm[:n_train]]
val_pairs = unique_pairs[perm[n_train:n_train+n_val]]
test_pairs = unique_pairs[perm[n_train+n_val:]]

print(f"Train pairs: {len(train_pairs):,} (85%)")
print(f"Val pairs:   {len(val_pairs):,} (5%)")
print(f"Test pairs:  {len(test_pairs):,} (10%)")

# Train message-passing graph (bidirectional)
train_edge_index = torch.cat([
    torch.stack([train_pairs[:, 0], train_pairs[:, 1]]),
    torch.stack([train_pairs[:, 1], train_pairs[:, 0]])
], dim=1).to(device)

print(f"Train message-passing edges: {train_edge_index.size(1):,}")

# Model
from torch_geometric.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_channels=768, hidden_channels=256, out_channels=128):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        return F.normalize(h, p=2, dim=-1)

model = GraphSAGE().to(device)

t0 = time.time()
with torch.no_grad():
    z = model(x, train_edge_index)
dt = time.time() - t0
print(f"\nForward pass successful! Output shape: {tuple(z.shape)}")
print(f"Time taken: {dt:.3f} s")
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"GPU memory reserved:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
print("Sanity check passed!")

