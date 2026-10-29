# 🚀 FastAPI Recommendation Service

## 1. Overview
The backend microservice is built using **FastAPI** (`backend/app.py` / `backend/api.py`), providing asynchronous, high-concurrency REST endpoints for recommendation generation, metadata lookup, and evaluation inspection.

## 2. API Endpoints Specification

### 1. `POST /api/recommend`
Generates hybrid recommendations for a given query text or paper ID.
- **Request Body**:
```json
{
  "query": "graph neural networks inductive node classification",
  "paper_id": null,
  "top_k": 10,
  "alpha": 0.60,
  "include_explanation": true
}
```
- **Response**:
```json
{
  "query_mode": "free_text",
  "query_input": "graph neural networks inductive node classification",
  "total_candidates": 50,
  "alpha": 0.60,
  "recommendations": [
    {
      "rank": 1,
      "paperId": "a1b2c3d4...",
      "title": "Inductive Representation Learning on Large Graphs",
      "year": 2017,
      "citationCount": 9480,
      "hybrid_score": 0.892,
      "semantic_score": 0.884,
      "graph_score": 0.904,
      "is_in_graph": true,
      "explanation": "High textual relevance to inductive graph methods combined with top-tier citation centrality in the GNN cluster."
    }
  ]
}
```

### 2. `GET /api/paper/{paper_id}`
Retrieves complete paper metadata, authors, abstract, publication venue, and citation graph connectivity.

### 3. `GET /api/graph/subgraph/{paper_id}`
Returns ego-network nodes and edges for visualization in the interactive graph explorer.

### 4. `POST /api/evaluate`
Runs the evaluation protocol comparing Semantic-Only vs. GraphSAGE Hybrid on held-out citation test links, returning MRR, Hits@5, Hits@10, and ROC-AUC.

### 5. `GET /api/health`
Checks FAISS index status, GraphSAGE model weight readiness, and CUDA availability.

---
Back to [[Hybrid Ranking & Score Fusion]] | Next: [[LLM Explanation Engine]]

