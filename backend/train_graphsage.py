import os
import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score, average_precision_score

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

print("=" * 70, flush=True)
print("GRAPHSAGE INDUCTIVE TRAINING PIPELINE", flush=True)
print("=" * 70, flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

FEATURES_PATH = "data/processed/graph/node_features.npy"
EDGES_PATH = "data/processed/graph/edges_index.csv"
MODEL_DIR = "backend/models"
os.makedirs(MODEL_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "graphsage_model.pt")
EMBEDDINGS_OUTPUT_PATH = "data/processed/graph/graph_embeddings.npy"
METRICS_OUTPUT_PATH = os.path.join(MODEL_DIR, "graphsage_metrics.json")
SPLITS_PATH = os.path.join(MODEL_DIR, "edge_splits.pt")

# 1. Load Data
print("\n[1/6] Loading graph features and edge index...", flush=True)
x_np = np.load(FEATURES_PATH)
x = torch.from_numpy(x_np).to(device)
# L2-normalize input features
x = F.normalize(x, p=2, dim=-1)
num_nodes = x.size(0)
print(f"Loaded {num_nodes:,} nodes with normalized feature dimension {x.size(1)}", flush=True)

edges_df = pd.read_csv(EDGES_PATH)
src_raw = torch.tensor(edges_df["source_id"].values, dtype=torch.long)
dst_raw = torch.tensor(edges_df["target_id"].values, dtype=torch.long)

mask = src_raw != dst_raw
src_raw, dst_raw = src_raw[mask], dst_raw[mask]

u = torch.minimum(src_raw, dst_raw)
v = torch.maximum(src_raw, dst_raw)
unique_pairs = torch.unique(torch.stack([u, v], dim=1), dim=0)
num_pairs = unique_pairs.size(0)
print(f"Total unique undirected citation edges: {num_pairs:,}", flush=True)

edge_set = set(zip(u.numpy(), v.numpy()))

# 2. Leakage-Safe Splits
print("\n[2/6] Preparing leakage-safe train (85%), val (5%), test (10%) splits...", flush=True)
perm = torch.randperm(num_pairs)
n_val = int(0.05 * num_pairs)
n_test = int(0.10 * num_pairs)
n_train = num_pairs - n_val - n_test

train_pairs = unique_pairs[perm[:n_train]]
val_pairs = unique_pairs[perm[n_train:n_train + n_val]]
test_pairs = unique_pairs[perm[n_train + n_val:]]

def sample_paired_negatives(pairs, num_nodes, forbidden_set):
    u_list = pairs[:, 0].numpy()
    neg_v = []
    for ui in u_list:
        while True:
            rand_node = np.random.randint(0, num_nodes)
            if rand_node == ui:
                continue
            pair = (min(ui, rand_node), max(ui, rand_node))
            if pair not in forbidden_set:
                neg_v.append(rand_node)
                break
    return torch.tensor(neg_v, dtype=torch.long)

print("Sampling fixed paired validation and test negative edges...", flush=True)
val_neg_v = sample_paired_negatives(val_pairs, num_nodes, edge_set)
test_neg_v = sample_paired_negatives(test_pairs, num_nodes, edge_set)

torch.save({
    "train_pairs": train_pairs,
    "val_pairs": val_pairs,
    "val_neg_v": val_neg_v,
    "test_pairs": test_pairs,
    "test_neg_v": test_neg_v
}, SPLITS_PATH)
print(f"Saved leakage-safe splits to {SPLITS_PATH}", flush=True)

# Train message-passing graph (bidirectional)
train_edge_index = torch.cat([
    torch.stack([train_pairs[:, 0], train_pairs[:, 1]]),
    torch.stack([train_pairs[:, 1], train_pairs[:, 0]])
], dim=1).to(device)
print(f"Train message-passing edges: {train_edge_index.size(1):,}", flush=True)

# 3. Model Definition
class GraphSAGE(nn.Module):
    def __init__(self, in_channels=768, hidden_channels=256, out_channels=128, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.ln1 = nn.LayerNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr="mean")
        self.proj = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        res = self.proj(x)
        h = self.conv1(x, edge_index)
        h = self.ln1(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        return F.normalize(res + h, p=2, dim=-1)

model = GraphSAGE().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
criterion = nn.MarginRankingLoss(margin=0.15)

print("\n[3/6] GraphSAGE Model Architecture:", flush=True)
print(model, flush=True)

# Device tensors for evaluation
val_u = val_pairs[:, 0].to(device)
val_v = val_pairs[:, 1].to(device)
val_neg_v_dev = val_neg_v.to(device)

test_u = test_pairs[:, 0].to(device)
test_v = test_pairs[:, 1].to(device)
test_neg_v_dev = test_neg_v.to(device)

def evaluate_split(z, u_dev, v_pos_dev, v_neg_dev):
    pos_scores = (z[u_dev] * z[v_pos_dev]).sum(dim=-1).cpu().numpy()
    neg_scores = (z[u_dev] * z[v_neg_dev]).sum(dim=-1).cpu().numpy()
    labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    scores = np.concatenate([pos_scores, neg_scores])
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    return auc, ap, float(pos_scores.mean()), float(neg_scores.mean())

# 4. Training Loop
EPOCHS = 12
print(f"\n[4/6] Training GraphSAGE for {EPOCHS} epochs...", flush=True)

train_u = train_pairs[:, 0].to(device)
train_v = train_pairs[:, 1].to(device)
n_train = len(train_pairs)

best_val_auc = 0.0
train_history = []

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()
    optimizer.zero_grad()
    
    z = model(x, train_edge_index)
    
    # Fast negative sampling on device
    train_neg_v = torch.randint(0, num_nodes, (n_train,), device=device)
    
    pos_sim = (z[train_u] * z[train_v]).sum(dim=-1)
    neg_sim = (z[train_u] * z[train_neg_v]).sum(dim=-1)
    
    target = torch.ones_like(pos_sim)
    loss = criterion(pos_sim, neg_sim, target)
    loss.backward()
    optimizer.step()
    
    epoch_time = time.time() - t0
    
    model.eval()
    with torch.no_grad():
        val_auc, val_ap, val_pos_mean, val_neg_mean = evaluate_split(z, val_u, val_v, val_neg_v_dev)
    
    train_history.append({
        "epoch": epoch,
        "loss": float(loss.item()),
        "val_auc": float(val_auc),
        "val_ap": float(val_ap),
        "val_pos_sim": val_pos_mean,
        "val_neg_sim": val_neg_mean,
        "time_s": float(epoch_time)
    })
    
    print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Loss: {loss.item():.4f} | "
          f"Val ROC-AUC: {val_auc:.4f} | Val AP: {val_ap:.4f} | "
          f"Pos Sim: {val_pos_mean:.4f} | Neg Sim: {val_neg_mean:.4f} | Time: {epoch_time:.2f}s", flush=True)
    
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"  -> Best model checkpoint saved (ROC-AUC: {best_val_auc:.4f})", flush=True)

# 5. Final Evaluation on Held-Out Test Set
print("\n[5/6] Final evaluation on held-out test edges (leakage-safe)...", flush=True)
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval()

with torch.no_grad():
    z_best = model(x, train_edge_index)
    test_auc, test_ap, test_pos_mean, test_neg_mean = evaluate_split(z_best, test_u, test_v, test_neg_v_dev)

print("=" * 60, flush=True)
print(f"HELD-OUT TEST SET EVALUATION:", flush=True)
print(f"  Test ROC-AUC: {test_auc:.4f}", flush=True)
print(f"  Test AP:      {test_ap:.4f}", flush=True)
print(f"  Pos Edge Cosine Sim Mean: {test_pos_mean:.4f}", flush=True)
print(f"  Neg Edge Cosine Sim Mean: {test_neg_mean:.4f}", flush=True)
print("=" * 60, flush=True)

# 6. Generate and Save Graph Embeddings
print(f"\n[6/6] Computing 128-d graph embeddings for all {num_nodes:,} nodes...", flush=True)
with torch.no_grad():
    # Use full bidirectional citation graph for final inference
    full_edge_index = torch.cat([
        torch.stack([u, v]),
        torch.stack([v, u])
    ], dim=1).to(device)
    z_all = model(x, full_edge_index)
    z_np = z_all.cpu().numpy().astype(np.float32)

print(f"Saving embeddings to {EMBEDDINGS_OUTPUT_PATH}...", flush=True)
np.save(EMBEDDINGS_OUTPUT_PATH, z_np)
print(f"Saved {z_np.shape[0]:,} embeddings ({os.path.getsize(EMBEDDINGS_OUTPUT_PATH) / 1024**2:.2f} MB)", flush=True)

summary = {
    "num_nodes": int(num_nodes),
    "num_edges": int(num_pairs),
    "train_pairs": int(len(train_pairs)),
    "val_pairs": int(len(val_pairs)),
    "test_pairs": int(len(test_pairs)),
    "best_val_auc": float(best_val_auc),
    "test_auc": float(test_auc),
    "test_ap": float(test_ap),
    "test_pos_sim": float(test_pos_mean),
    "test_neg_sim": float(test_neg_mean),
    "history": train_history
}
with open(METRICS_OUTPUT_PATH, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Metrics saved to {METRICS_OUTPUT_PATH}", flush=True)
print("\nAll training and embedding generation tasks completed successfully!", flush=True)
