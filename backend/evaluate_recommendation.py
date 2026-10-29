import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

print("=" * 70, flush=True)
print("QUANTITATIVE COMPARATIVE EVALUATION: SEMANTIC vs. GRAPH vs. HYBRID", flush=True)
print("=" * 70, flush=True)

# 1. Load Data
SPLITS_PATH = "backend/models/edge_splits.pt"
FEATURES_PATH = "data/processed/graph/node_features.npy"
GRAPH_EMB_PATH = "data/processed/graph/graph_embeddings.npy"
NODE_MAPPING_PATH = "data/processed/graph/node_mapping.csv"
OUTPUT_METRICS_PATH = "backend/models/evaluation_comparison.json"

print("Loading test edge splits...", flush=True)
splits = torch.load(SPLITS_PATH)
test_pairs = splits["test_pairs"]
print(f"Total held-out test citation edges: {len(test_pairs):,}", flush=True)

print("Loading node features (SciBERT) and GraphSAGE embeddings...", flush=True)
x_feat = np.load(FEATURES_PATH)
x_feat = torch.from_numpy(x_feat)
x_feat = F.normalize(x_feat, p=2, dim=-1)

z_graph = np.load(GRAPH_EMB_PATH)
z_graph = torch.from_numpy(z_graph)

num_nodes = x_feat.size(0)

# 2. Benchmark Protocol
# Sample 1,000 held-out test citation edges for ranking evaluation
NUM_EVAL_SAMPLES = 1000
np.random.seed(42)
torch.manual_seed(42)

perm = torch.randperm(len(test_pairs))[:NUM_EVAL_SAMPLES]
eval_pairs = test_pairs[perm]

print(f"\nRunning 1-to-100 candidate ranking evaluation over {NUM_EVAL_SAMPLES:,} test citation queries...", flush=True)

# Ranking metrics accumulators
methods = {
    "Semantic-Only (SciBERT)": {"mrr": [], "hits5": [], "hits10": [], "hits20": []},
    "Graph-Only (GraphSAGE)": {"mrr": [], "hits5": [], "hits10": [], "hits20": []},
    "Hybrid Recommender (alpha=0.60)": {"mrr": [], "hits5": [], "hits10": [], "hits20": []}
}

K_CANDIDATES = 100
alpha = 0.60

t0 = time.time()

for idx in range(len(eval_pairs)):
    u = int(eval_pairs[idx, 0])
    v_pos = int(eval_pairs[idx, 1])

    # 1. Sample 99 negative random non-citation candidate papers for query u
    neg_pool = np.random.randint(0, num_nodes, size=K_CANDIDATES * 2)
    neg_pool = [n for n in neg_pool if n != u and n != v_pos][:K_CANDIDATES - 1]
    
    # Candidate list: true positive target is at index 0
    candidate_indices = [v_pos] + neg_pool
    cand_tensor = torch.tensor(candidate_indices, dtype=torch.long)
    
    # 2. Compute Semantic Scores (SciBERT cosine)
    u_feat = x_feat[u]
    cand_feats = x_feat[cand_tensor]
    sem_scores = (cand_feats * u_feat).sum(dim=-1).numpy()
    
    # 3. Compute Graph Scores (GraphSAGE cosine in [0, 1])
    u_graph = z_graph[u]
    cand_graphs = z_graph[cand_tensor]
    graph_cos = (cand_graphs * u_graph).sum(dim=-1).numpy()
    graph_scores = (1.0 + graph_cos) / 2.0
    
    # 4. Compute Hybrid Scores
    hybrid_scores = alpha * sem_scores + (1.0 - alpha) * graph_scores
    
    # 5. Measure Ranks for Each Method
    score_dict = {
        "Semantic-Only (SciBERT)": sem_scores,
        "Graph-Only (GraphSAGE)": graph_scores,
        "Hybrid Recommender (alpha=0.60)": hybrid_scores
    }
    
    for method_name, scores in score_dict.items():
        # Rank candidates in descending order of score
        # Target was at candidate index 0
        sorted_indices = np.argsort(-scores)
        target_rank = int(np.where(sorted_indices == 0)[0][0]) + 1  # 1-based rank
        
        methods[method_name]["mrr"].append(1.0 / target_rank)
        methods[method_name]["hits5"].append(1.0 if target_rank <= 5 else 0.0)
        methods[method_name]["hits10"].append(1.0 if target_rank <= 10 else 0.0)
        methods[method_name]["hits20"].append(1.0 if target_rank <= 20 else 0.0)

dt = time.time() - t0
print(f"Evaluation completed in {dt:.2f} seconds.\n", flush=True)

# 3. Aggregate Results Table
results_table = {}

print("=" * 80, flush=True)
print(f"{'Method / Signal':<35} | {'MRR':<8} | {'Hits@5':<8} | {'Hits@10':<8} | {'Hits@20':<8}")
print("-" * 80, flush=True)

for method_name, vals in methods.items():
    mrr = float(np.mean(vals["mrr"]))
    h5 = float(np.mean(vals["hits5"])) * 100
    h10 = float(np.mean(vals["hits10"])) * 100
    h20 = float(np.mean(vals["hits20"])) * 100
    
    results_table[method_name] = {
        "MRR": round(mrr, 4),
        "Hits@5": round(h5, 2),
        "Hits@10": round(h10, 2),
        "Hits@20": round(h20, 2)
    }
    
    print(f"{method_name:<35} | {mrr:<8.4f} | {h5:>6.2f}% | {h10:>6.2f}% | {h20:>6.2f}%", flush=True)

print("=" * 80, flush=True)

# Save JSON results
with open(OUTPUT_METRICS_PATH, "w") as f:
    json.dump({
        "num_test_queries": NUM_EVAL_SAMPLES,
        "candidates_per_query": K_CANDIDATES,
        "alpha": alpha,
        "results": results_table
    }, f, indent=2)

print(f"Successfully saved evaluation report to {OUTPUT_METRICS_PATH}", flush=True)

