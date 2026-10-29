import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score, average_precision_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

x_np = np.load("data/processed/graph/node_features.npy")
x = torch.from_numpy(x_np).to(device)
x = F.normalize(x, p=2, dim=-1)
num_nodes = x.size(0)

splits = torch.load("backend/models/edge_splits.pt")
train_pairs = splits["train_pairs"]
val_pairs = splits["val_pairs"]
val_neg_pairs = splits["val_neg_pairs"]

train_edge_index = torch.cat([
    torch.stack([train_pairs[:, 0], train_pairs[:, 1]]),
    torch.stack([train_pairs[:, 1], train_pairs[:, 0]])
], dim=1).to(device)

val_pos_u = val_pairs[:, 0].to(device)
val_pos_v = val_pairs[:, 1].to(device)
val_neg_u = val_neg_pairs[:, 0].to(device)
val_neg_v = val_neg_pairs[:, 1].to(device)

class ResidualGraphSAGE(nn.Module):
    def __init__(self, in_channels=768, hidden_channels=256, out_channels=128):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.ln1 = nn.LayerNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        # Residual projection from 768-d semantic space to 128-d
        self.proj = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index):
        # Semantic projection
        res = self.proj(x)
        # Graph neighborhood aggregation
        h = self.conv1(x, edge_index)
        h = self.ln1(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        # Combine semantic + graph
        out = F.normalize(res + h, p=2, dim=-1)
        return out

model = ResidualGraphSAGE().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
criterion = nn.MarginRankingLoss(margin=0.2)

train_u = train_pairs[:, 0].to(device)
train_v = train_pairs[:, 1].to(device)
n_train = len(train_pairs)

# Epoch 0 (untrained initialization)
model.eval()
with torch.no_grad():
    z0 = model(x, train_edge_index)
    val_pos0 = (z0[val_pos_u] * z0[val_pos_v]).sum(dim=-1).cpu().numpy()
    val_neg0 = (z0[val_neg_u] * z0[val_neg_v]).sum(dim=-1).cpu().numpy()
    labels0 = np.concatenate([np.ones_like(val_pos0), np.zeros_like(val_neg0)])
    scores0 = np.concatenate([val_pos0, val_neg0])
    auc0 = roc_auc_score(labels0, scores0)
    print(f"Epoch 00 (Initial) | Val Pos: {val_pos0.mean():.4f} | Val Neg: {val_neg0.mean():.4f} | Val ROC-AUC: {auc0:.4f}")

for epoch in range(1, 11):
    model.train()
    optimizer.zero_grad()
    
    z = model(x, train_edge_index)
    
    neg_v = torch.randint(0, num_nodes, (n_train,), device=device)
    
    pos_scores = (z[train_u] * z[train_v]).sum(dim=-1)
    neg_scores = (z[train_u] * z[neg_v]).sum(dim=-1)
    
    target = torch.ones_like(pos_scores)
    loss = criterion(pos_scores, neg_scores, target)
    
    loss.backward()
    optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_pos = (z[val_pos_u] * z[val_pos_v]).sum(dim=-1).cpu().numpy()
        val_neg = (z[val_neg_u] * z[val_neg_v]).sum(dim=-1).cpu().numpy()
        labels = np.concatenate([np.ones_like(val_pos), np.zeros_like(val_neg)])
        scores = np.concatenate([val_pos, val_neg])
        val_auc = roc_auc_score(labels, scores)
        val_ap = average_precision_score(labels, scores)
    
    print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f} | Val Pos: {val_pos.mean():.4f} | "
          f"Val Neg: {val_neg.mean():.4f} | Val ROC-AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")

