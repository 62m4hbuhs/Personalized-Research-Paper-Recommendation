import os
import json
import time
import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "allenai/scibert_scivocab_uncased"
FAISS_INDEX_FILE = "data/processed/faiss/scibert.index"
PAPER_IDS_FILE = "data/processed/faiss/paper_id_mapping.json"
PAPERS_METADATA_FILE = "data/processed/papers_clean.csv"
GRAPH_EMBEDDINGS_FILE = "data/processed/graph/graph_embeddings.npy"
NODE_MAPPING_FILE = "data/processed/graph/node_mapping.csv"
EDGES_FILE = "data/processed/graph/edges_index.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HybridRecommender:
    def __init__(self):
        print("Initializing Hybrid Recommender System...")
        t0 = time.time()
        
        # 1. Load FAISS index
        print("Loading FAISS index...")
        self.faiss_index = faiss.read_index(FAISS_INDEX_FILE)
        with open(PAPER_IDS_FILE, "r", encoding="utf-8") as f:
            self.faiss_paper_ids = json.load(f)
        self.paper_id_to_faiss_idx = {pid: i for i, pid in enumerate(self.faiss_paper_ids)}
        print(f"Loaded FAISS index with {self.faiss_index.ntotal:,} vectors.")

        # 2. Load metadata
        print("Loading paper metadata catalog...")
        papers_df = pd.read_csv(
            PAPERS_METADATA_FILE,
            usecols=["paperId", "title", "abstract", "year", "citationCount"]
        ).drop_duplicates(subset=["paperId"])
        self.metadata = papers_df.set_index("paperId").to_dict("index")
        self.titles_lookup = {
            str(row["title"]).strip().lower(): pid
            for pid, row in self.metadata.items()
            if pd.notna(row["title"])
        }
        self.max_citations = max([p.get("citationCount", 0) or 0 for p in self.metadata.values()] + [1])
        print(f"Loaded metadata for {len(self.metadata):,} papers.")

        # 3. Load Graph Embeddings
        print("Loading GraphSAGE embeddings and node mapping...")
        self.graph_embeddings = np.load(GRAPH_EMBEDDINGS_FILE)
        node_map_df = pd.read_csv(NODE_MAPPING_FILE)
        self.paper_id_to_node_idx = dict(zip(node_map_df["paperId"], node_map_df["node_id"]))
        self.node_idx_to_paper_id = dict(zip(node_map_df["node_id"], node_map_df["paperId"]))
        print(f"Loaded GraphSAGE embeddings of shape {self.graph_embeddings.shape}.")

        # 4. Load SciBERT for query encoding
        print(f"Loading SciBERT encoder ({MODEL_NAME}) on {DEVICE}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.scibert_model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
        self.scibert_model.eval()
        print(f"Hybrid Recommender initialized successfully in {time.time() - t0:.2f}s!")

    def _mean_pooling(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def encode_text(self, text: str) -> np.ndarray:
        encoded = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
        with torch.inference_mode():
            outputs = self.scibert_model(**encoded)
            emb = self._mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])
        vec = emb.cpu().numpy().astype(np.float32)
        faiss.normalize_L2(vec)
        return vec

    def get_paper_by_title_or_id(self, query: str):
        if query in self.metadata:
            return query, self.metadata[query]
        q_clean = query.strip().lower()
        if q_clean in self.titles_lookup:
            pid = self.titles_lookup[q_clean]
            return pid, self.metadata[pid]
        # Partial match
        for title, pid in self.titles_lookup.items():
            if q_clean in title or title in q_clean:
                return pid, self.metadata[pid]
        return None, None

    def recommend(
        self,
        query: str = None,
        seed_paper_id: str = None,
        top_k: int = 10,
        alpha: float = 0.60,
        candidate_pool_size: int = 60
    ):
        """
        Generates hybrid recommendations.
        - alpha: weight for semantic similarity (0.0 = pure graph, 1.0 = pure semantic)
        - beta: prior weight for citation impact (0.05 default)
        """
        beta = 0.05
        is_seed_paper_mode = False
        target_meta = None
        seed_graph_emb = None

        # 1. Resolve Seed Vector & Graph Embedding
        if seed_paper_id and seed_paper_id in self.metadata:
            is_seed_paper_mode = True
            target_meta = self.metadata[seed_paper_id]
            # Get text embedding from FAISS if present, else encode abstract
            if seed_paper_id in self.paper_id_to_faiss_idx:
                faiss_idx = self.paper_id_to_faiss_idx[seed_paper_id]
                query_vec = np.zeros((1, 768), dtype=np.float32)
                self.faiss_index.reconstruct(faiss_idx, query_vec[0])
            else:
                text = str(target_meta.get("abstract") or target_meta.get("title") or "")
                query_vec = self.encode_text(text)
            
            # Seed graph embedding
            if seed_paper_id in self.paper_id_to_node_idx:
                seed_node_idx = self.paper_id_to_node_idx[seed_paper_id]
                seed_graph_emb = self.graph_embeddings[seed_node_idx]
        else:
            # Free-text search
            if not query:
                raise ValueError("Either query string or valid seed_paper_id must be provided.")
            query_vec = self.encode_text(query)

        # 2. Retrieve Candidate Papers via FAISS Semantic Search
        search_k = min(candidate_pool_size, self.faiss_index.ntotal)
        scores, indices = self.faiss_index.search(query_vec, search_k)
        
        candidates = []
        candidate_graph_embs = []
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            cand_id = self.faiss_paper_ids[idx]
            if is_seed_paper_mode and cand_id == seed_paper_id:
                continue  # Skip seed paper itself
            
            cand_meta = self.metadata.get(cand_id, {})
            in_graph = cand_id in self.paper_id_to_node_idx
            
            g_emb = None
            if in_graph:
                node_idx = self.paper_id_to_node_idx[cand_id]
                g_emb = self.graph_embeddings[node_idx]
                candidate_graph_embs.append(g_emb)
            
            candidates.append({
                "paperId": cand_id,
                "semantic_score": float(score),
                "in_graph": in_graph,
                "graph_emb": g_emb,
                "title": cand_meta.get("title", "Untitled"),
                "abstract": cand_meta.get("abstract", ""),
                "year": int(cand_meta.get("year", 0)) if pd.notna(cand_meta.get("year")) else None,
                "citationCount": int(cand_meta.get("citationCount", 0)) if pd.notna(cand_meta.get("citationCount")) else 0
            })

        if not candidates:
            return {"recommendations": [], "seed_paper": target_meta}

        # 3. Handle Graph Query Centroid for Free-Text Query
        if not is_seed_paper_mode:
            if candidate_graph_embs:
                # Compute pseudo-query graph embedding as the centroid of top semantic candidates
                seed_graph_emb = np.mean(candidate_graph_embs[:10], axis=0)
                seed_graph_emb = seed_graph_emb / (np.linalg.norm(seed_graph_emb) + 1e-9)

        # 4. Compute Graph Similarity & Combined Hybrid Scores
        ranked_results = []
        for c in candidates:
            sem_score = c["semantic_score"]
            # Graph affinity computation
            if c["in_graph"] and seed_graph_emb is not None:
                # Cosine similarity in [-1, 1], scaled to [0, 1]
                cos_sim = float(np.dot(seed_graph_emb, c["graph_emb"]))
                graph_score = max(0.0, min(1.0, (1.0 + cos_sim) / 2.0))
                fallback_used = False
            else:
                # Graceful fallback: when paper is not in graph or no graph context
                graph_score = sem_score
                fallback_used = True

            # Citation impact prior
            cites = c["citationCount"]
            citation_prior = np.log1p(cites) / np.log1p(self.max_citations)

            # Hybrid Score Fusion
            hybrid_score = (
                alpha * sem_score +
                (1.0 - alpha) * graph_score +
                beta * citation_prior
            )

            # Generate natural explanation
            explanation = self._generate_explanation(
                sem_score=sem_score,
                graph_score=graph_score,
                fallback_used=fallback_used,
                citation_count=cites,
                is_seed_paper_mode=is_seed_paper_mode
            )

            ranked_results.append({
                "paperId": c["paperId"],
                "title": c["title"],
                "abstract": c["abstract"],
                "year": c["year"],
                "citationCount": cites,
                "hybrid_score": round(float(hybrid_score), 4),
                "semantic_score": round(float(sem_score), 4),
                "graph_score": round(float(graph_score), 4),
                "citation_prior": round(float(citation_prior), 4),
                "in_graph": c["in_graph"],
                "fallback_applied": fallback_used,
                "explanation": explanation
            })

        # Sort by hybrid score descending
        ranked_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        final_recs = ranked_results[:top_k]
        for i, item in enumerate(final_recs, 1):
            item["rank"] = i

        return {
            "query_type": "seed_paper" if is_seed_paper_mode else "free_text",
            "seed_paper": target_meta,
            "alpha": alpha,
            "total_candidates_analyzed": len(candidates),
            "recommendations": final_recs
        }

    def _generate_explanation(
        self,
        sem_score: float,
        graph_score: float,
        fallback_used: bool,
        citation_count: int,
        is_seed_paper_mode: bool
    ) -> str:
        if fallback_used:
            return (
                f"Recommended based on strong semantic text alignment ({sem_score:.1%}). "
                f"Paper is newly ingested or out-of-graph, relying on content features."
            )
        
        if sem_score >= 0.82 and graph_score >= 0.75:
            return (
                f"Strong dual alignment: high semantic topic overlap ({sem_score:.1%}) "
                f"and dense citation community proximity ({graph_score:.1%})."
            )
        elif graph_score > sem_score + 0.15:
            return (
                f"Key citation network anchor ({graph_score:.1%}) with {citation_count:,} citations, "
                f"frequently co-cited with works in this intellectual lineage."
            )
        elif sem_score > graph_score + 0.15:
            return (
                f"High conceptual & thematic overlap ({sem_score:.1%}) offering fresh perspectives "
                f"from outside the immediate local citation cluster."
            )
        else:
            return (
                f"Balanced relevance: {sem_score:.1%} content similarity and "
                f"{graph_score:.1%} citation network proximity."
            )


# Self-test if executed directly
if __name__ == "__main__":
    recommender = HybridRecommender()
    
    print("\n" + "=" * 70)
    print("TEST 1: PAPER-TO-PAPER HYBRID RECOMMENDATION")
    print("=" * 70)
    target_title = "How Powerful are Graph Neural Networks?"
    pid, meta = recommender.get_paper_by_title_or_id(target_title)
    if pid:
        print(f"Target Seed: {meta['title']} ({meta['year']}) | ID: {pid}")
        res = recommender.recommend(seed_paper_id=pid, top_k=5, alpha=0.60)
        for r in res["recommendations"]:
            print(f"\n#{r['rank']} [Score: {r['hybrid_score']:.4f} | Sem: {r['semantic_score']:.3f} | Graph: {r['graph_score']:.3f}]")
            print(f"   {r['title']} ({r['year']}) - Citations: {r['citationCount']}")
            print(f"   Why: {r['explanation']}")

    print("\n" + "=" * 70)
    print("TEST 2: FREE-TEXT TOPIC HYBRID RECOMMENDATION")
    print("=" * 70)
    query = "inductive representation learning on large dynamic graphs"
    print(f"Query: '{query}'")
    res_text = recommender.recommend(query=query, top_k=5, alpha=0.60)
    for r in res_text["recommendations"]:
        print(f"\n#{r['rank']} [Score: {r['hybrid_score']:.4f} | Sem: {r['semantic_score']:.3f} | Graph: {r['graph_score']:.3f}]")
        print(f"   {r['title']} ({r['year']}) - Citations: {r['citationCount']}")
        print(f"   Why: {r['explanation']}")

