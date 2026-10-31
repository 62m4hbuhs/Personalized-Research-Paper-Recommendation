import os
import re
import json
import numpy as np
import pandas as pd
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.hybrid_recommender import HybridRecommender
from backend.agentic_workflow import LangGraphRecommenderAgent

app = FastAPI(
    title="Personalized Research Paper Recommendation System",
    description="Dual-Signal Academic Paper Recommendation Engine fusing SciBERT Semantic Similarity and GraphSAGE Citation Topology with LangGraph Agentic Orchestration.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Recommender Instances (singletons)
recommender: Optional[HybridRecommender] = None
agent_recommender: Optional[LangGraphRecommenderAgent] = None

EVAL_METRICS_PATH = "backend/models/evaluation_comparison.json"
GRAPHSAGE_METRICS_PATH = "backend/models/graphsage_metrics.json"
EDGES_FILE = "data/processed/graph/edges_index.csv"

# Pre-load edge list for fast subgraph queries
edges_df = None


@app.on_event("startup")
def startup_event():
    global recommender, agent_recommender, edges_df
    print("FastAPI server starting up...")
    recommender = HybridRecommender()
    agent_recommender = LangGraphRecommenderAgent(recommender)
    print("LangGraph Agentic Orchestrator initialized successfully!")
    if os.path.exists(EDGES_FILE):
        edges_df = pd.read_csv(EDGES_FILE)
        print(f"Loaded edge list for subgraph queries ({len(edges_df):,} edges).")


class RecommendRequest(BaseModel):
    query: Optional[str] = None
    paper_id: Optional[str] = None
    reading_list_ids: Optional[List[str]] = None
    top_k: int = 10
    alpha: float = 0.60
    candidate_pool_size: int = 60
    use_agent: bool = True


class ReadingListRequest(BaseModel):
    paper_ids: List[str]
    top_k: int = 10
    alpha: float = 0.60
    candidate_pool_size: int = 60


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "faiss_total_vectors": recommender.faiss_index.ntotal if recommender else 0,
        "graph_nodes": recommender.graph_embeddings.shape[0] if recommender else 0,
        "metadata_count": len(recommender.metadata) if recommender else 0
    }


@app.get("/api/search")
def search_papers(q: str = Query(..., min_length=2), limit: int = 8):
    """Smart fast autocomplete / paper selection endpoint with early stopping."""
    if not recommender:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    
    q_lower = q.lower().strip()
    q_tokens = [w for w in re.findall(r"\b[a-z0-9]{3,}\b", q_lower) if w not in {"the", "and", "for", "with", "that"}]
    
    scored_matches = []
    
    for title, pid in recommender.titles_lookup.items():
        # Exact contiguous substring gets top priority
        if q_lower in title:
            meta = recommender.metadata.get(pid, {})
            score = 100.0 - (len(title) - len(q_lower)) * 0.05
            scored_matches.append((score, pid, meta))
            if len(scored_matches) >= 30:
                break
        elif q_tokens and len(scored_matches) < 20:
            # Token overlap score fallback
            hits = sum(1 for tok in q_tokens if tok in title)
            if hits >= min(2, len(q_tokens)):
                meta = recommender.metadata.get(pid, {})
                score = (hits / len(q_tokens)) * 50.0
                scored_matches.append((score, pid, meta))

    # Sort descending by relevance score
    scored_matches.sort(key=lambda x: x[0], reverse=True)
    
    results = []
    seen_ids = set()
    for _, pid, meta in scored_matches:
        if pid in seen_ids:
            continue
        seen_ids.add(pid)
        results.append({
            "paperId": pid,
            "title": meta.get("title", "Untitled"),
            "year": meta.get("year"),
            "citationCount": meta.get("citationCount", 0),
            "in_graph": pid in recommender.paper_id_to_node_idx
        })
        if len(results) >= limit:
            break
                
    return {"query": q, "results": results}


@app.post("/api/recommend")
def recommend_endpoint(req: RecommendRequest):
    if not recommender:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    
    try:
        # 1. Reading list personalization
        if req.reading_list_ids and len(req.reading_list_ids) > 0:
            if req.use_agent and agent_recommender:
                return agent_recommender.invoke(
                    reading_list_ids=req.reading_list_ids,
                    alpha=req.alpha,
                    top_k=req.top_k,
                    candidate_pool_size=req.candidate_pool_size
                )
            return recommender.recommend_for_reading_list(
                paper_ids=req.reading_list_ids,
                top_k=req.top_k,
                alpha=req.alpha,
                candidate_pool_size=req.candidate_pool_size
            )

        # 2. Agentic LangGraph pipeline execution
        if req.use_agent and agent_recommender:
            return agent_recommender.invoke(
                query=req.query,
                seed_paper_id=req.paper_id,
                alpha=req.alpha,
                top_k=req.top_k,
                candidate_pool_size=req.candidate_pool_size
            )

        # 3. Direct Fast Path
        target_pid = req.paper_id
        if req.query and not target_pid:
            found_pid, _ = recommender.get_paper_by_title_or_id(req.query)
            if found_pid:
                target_pid = found_pid

        results = recommender.recommend(
            query=req.query,
            seed_paper_id=target_pid,
            top_k=req.top_k,
            alpha=req.alpha,
            candidate_pool_size=req.candidate_pool_size
        )
        return results
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/user/reading-list/recommend")
def reading_list_recommend_endpoint(req: ReadingListRequest):
    """Personalized recommendations generated from user's reading list profile."""
    if not recommender:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    
    try:
        if agent_recommender:
            return agent_recommender.invoke(
                reading_list_ids=req.paper_ids,
                alpha=req.alpha,
                top_k=req.top_k,
                candidate_pool_size=req.candidate_pool_size
            )
        return recommender.recommend_for_reading_list(
            paper_ids=req.paper_ids,
            top_k=req.top_k,
            alpha=req.alpha,
            candidate_pool_size=req.candidate_pool_size
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/paper/{paper_id}")
def get_paper_details(paper_id: str):
    if not recommender:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    
    meta = recommender.metadata.get(paper_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Paper ID not found in catalog")
    
    clean_meta = recommender._clean_paper_dict(meta)
    in_graph = paper_id in recommender.paper_id_to_node_idx
    node_idx = recommender.paper_id_to_node_idx.get(paper_id)
    
    clean_meta["in_graph"] = in_graph
    clean_meta["node_idx"] = node_idx
    return clean_meta


@app.get("/api/evaluation")
def get_evaluation_metrics():
    """Returns held-out test evaluation benchmarks"""
    eval_data = {}
    graphsage_data = {}
    
    if os.path.exists(EVAL_METRICS_PATH):
        with open(EVAL_METRICS_PATH, "r") as f:
            eval_data = json.load(f)
            
    if os.path.exists(GRAPHSAGE_METRICS_PATH):
        with open(GRAPHSAGE_METRICS_PATH, "r") as f:
            graphsage_data = json.load(f)
            
    return {
        "ranking_benchmark": eval_data,
        "graphsage_link_prediction": graphsage_data
    }


@app.get("/api/graph/subgraph/{paper_id}")
def get_citation_subgraph(paper_id: str, max_neighbors: int = 15):
    """Returns 1-hop ego network for citation visualization"""
    if not recommender or edges_df is None:
        raise HTTPException(status_code=503, detail="Graph or recommender not available")
        
    if paper_id not in recommender.paper_id_to_node_idx:
        return {"nodes": [], "links": [], "center_id": paper_id}
        
    center_node_idx = recommender.paper_id_to_node_idx[paper_id]
    
    # Direct neighbors
    sub = edges_df[edges_df["source_id"] == center_node_idx].head(max_neighbors)
    neighbor_node_ids = set(sub["target_id"].tolist())
    
    nodes = [{
        "id": paper_id,
        "label": recommender.metadata.get(paper_id, {}).get("title", "Seed Paper")[:45] + "...",
        "group": "seed",
        "citationCount": recommender.metadata.get(paper_id, {}).get("citationCount", 0)
    }]
    
    links = []
    
    for n_idx in neighbor_node_ids:
        n_pid = recommender.node_idx_to_paper_id.get(n_idx)
        if not n_pid:
            continue
        n_meta = recommender.metadata.get(n_pid, {})
        nodes.append({
            "id": n_pid,
            "label": n_meta.get("title", "Paper")[:40] + "...",
            "group": "citation",
            "citationCount": n_meta.get("citationCount", 0)
        })
        links.append({
            "source": paper_id,
            "target": n_pid
        })
        
    return {"nodes": nodes, "links": links, "center_id": paper_id}


# Mount static files directory for frontend
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_frontend():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "FastAPI Recommendation API is running. Navigate to /docs for Swagger UI."}

