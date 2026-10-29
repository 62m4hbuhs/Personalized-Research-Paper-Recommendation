import os
import json
import time
import numpy as np
import pandas as pd
from backend.hybrid_recommender import HybridRecommender

def main():
    print("=" * 80)
    print("   PERSONALIZED RESEARCH PAPER RECOMMENDATION SYSTEM - EVALUATION DEMO   ")
    print("      Fusing Deep SciBERT Semantics with Inductive GraphSAGE Topology    ")
    print("=" * 80)

    # 1. Initialize Recommender
    recommender = HybridRecommender()

    # 2. Select Canonical Seed Paper from Dataset
    seed_title = "How Powerful are Graph Neural Networks?"
    print("\n" + "-" * 80)
    print(f"STEP 1: SEED PAPER SELECTION")
    print("-" * 80)
    pid, meta = recommender.get_paper_by_title_or_id(seed_title)
    if not pid:
        # Fallback to first available title
        pid = list(recommender.metadata.keys())[0]
        meta = recommender.metadata[pid]

    print(f"Selected Paper:   '{meta['title']}'")
    print(f"Publication Year: {meta.get('year', 'N/A')}")
    print(f"Global Citations: {meta.get('citationCount', 0):,}")
    print(f"Paper ID:         {pid}")
    if meta.get("abstract"):
        print(f"Abstract Snippet: {meta['abstract'][:180]}...")

    # 3. Compare Signals: Pure Semantic vs Pure Graph vs Hybrid
    print("\n" + "-" * 80)
    print("STEP 2: MULTI-SIGNAL COMPARISON ON REAL CANDIDATES")
    print("-" * 80)

    # A. Semantic-Only (alpha = 1.0)
    res_sem = recommender.recommend(seed_paper_id=pid, top_k=3, alpha=1.00)
    print("\n[A] PURE SEMANTIC RETRIEVAL (SciBERT Content Only, alpha=1.0):")
    for r in res_sem["recommendations"]:
        print(f"  #{r['rank']} [Semantic Match: {r['semantic_score']:.1%}] {r['title']} ({r['year']}) - Citations: {r['citationCount']:,}")

    # B. Pure Citation Graph (alpha = 0.0)
    res_graph = recommender.recommend(seed_paper_id=pid, top_k=3, alpha=0.00)
    print("\n[B] PURE CITATION TOPOLOGY RETRIEVAL (GraphSAGE Only, alpha=0.0):")
    for r in res_graph["recommendations"]:
        print(f"  #{r['rank']} [Graph Affinity: {r['graph_score']:.1%}] {r['title']} ({r['year']}) - Citations: {r['citationCount']:,}")

    # C. Proposed Hybrid System (alpha = 0.60)
    res_hybrid = recommender.recommend(seed_paper_id=pid, top_k=5, alpha=0.60)
    print("\n[C] PROPOSED HYBRID SYSTEM (Balanced alpha=0.60 + Citation Prior):")
    for r in res_hybrid["recommendations"]:
        print(f"\n  #{r['rank']} {r['title']} ({r['year']})")
        print(f"     Hybrid Score:   {r['hybrid_score']:.4f} ({r['hybrid_score']*100:.1f}%)")
        print(f"     Score Breakdown: Semantic = {r['semantic_score']:.1%} | Citation Graph = {r['graph_score']:.1%} | Prior = {r['citation_prior']:.3f}")
        print(f"     Citations:      {r['citationCount']:,}")
        print(f"     Explanation:    \"{r['explanation']}\"")

    # 4. Quantitative Held-Out Benchmark Results
    print("\n" + "=" * 80)
    print("STEP 3: QUANTITATIVE BENCHMARK (Leakage-Safe Held-Out Test Edges)")
    print("=" * 80)
    eval_file = "backend/models/evaluation_comparison.json"
    if os.path.exists(eval_file):
        with open(eval_file, "r") as f:
            eval_data = json.load(f)
            res_table = eval_data.get("results", {})
            print(f"\nTested on {eval_data.get('num_test_queries', 1000):,} held-out citation test links across 100 candidates per query:")
            print(f"{'Method / Signal':<35} | {'MRR':<8} | {'Hits@5':<8} | {'Hits@10':<8} | {'Hits@20':<8}")
            print("-" * 80)
            for m, metrics in res_table.items():
                print(f"{m:<35} | {metrics['MRR']:<8.4f} | {metrics['Hits@5']:>6.2f}% | {metrics['Hits@10']:>6.2f}% | {metrics['Hits@20']:>6.2f}%")
            print("-" * 80)

    # 5. GraphSAGE Link Prediction Validation
    sage_file = "backend/models/graphsage_metrics.json"
    if os.path.exists(sage_file):
        with open(sage_file, "r") as f:
            sage_data = json.load(f)
            print(f"\nGraphSAGE Link Prediction Performance (Self-Supervised Held-Out Edges):")
            print(f"  Test ROC-AUC:                {sage_data.get('test_auc', 0):.4f} ({sage_data.get('test_auc', 0)*100:.2f}%)")
            print(f"  Test Average Precision (AP): {sage_data.get('test_ap', 0):.4f} ({sage_data.get('test_ap', 0)*100:.2f}%)")
            print(f"  Positive Edge Mean Sim:      {sage_data.get('test_pos_sim', 0):.4f}")
            print(f"  Negative Edge Mean Sim:      {sage_data.get('test_neg_sim', 0):.4f}")

    print("\n" + "=" * 80)
    print("DEMO COMPLETE! To launch the interactive web dashboard:")
    print("   .\\myenv\\Scripts\\python.exe -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload")
    print("   Open http://localhost:8000 in your browser.")
    print("=" * 80)

if __name__ == "__main__":
    main()

