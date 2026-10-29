# 🧠 Research Paper Recommendation System - Knowledge Base

Welcome to the central Obsidian Vault for the **Personalized Research Paper Recommendation Using Semantic and Citation Networks** project.

## 📌 Quick Overview
- **Project Goal**: Enable researchers and students to discover relevant, high-impact scientific literature by fusing textual/semantic content with topological citation-graph signals.
- **Core Paradigm**: Dual-signal hybrid ranking (`SciBERT + FAISS` for content semantic similarity, `GraphSAGE` for topological citation affinity, unified by dynamic score fusion and LLM explainability).
- **Current Development Environment**: PyTorch Geometric with GPU acceleration (NVIDIA RTX 2050 CUDA), FastAPI backend, React modern frontend.
- **Master Documentation**: Read the complete [[Project Report & Architecture]].

---

## 🗺️ Map of Content (MOC)

### 1. [[Problem Definition]] & [[Comparison - Basic vs Hybrid Search]]
- The five critical failure modes of conventional keyword search (vocabulary mismatch, unranked citation topology, lack of personalization, lack of explainability, information overload).
- Why hybrid models outperform single-signal models.

### 2. [[System Architecture]] & [[Data Pipeline & Workflow]]
- High-level 5-stage architecture: User Query -> Semantic Candidate Generation -> Graph Neighborhood Encoding -> Hybrid Fusion & Re-Ranking -> Explainability & Delivery.
- Data lineage from Semantic Scholar / ArXiv raw data to structured PyG graph datasets.

### 3. [[Dataset Specifications]], [[SciBERT & FAISS Vector Search]], & [[Citation Graph & Fallback Embeddings]]
- 94,471 unique papers with 103,482 citation relationships.
- 56,433 768-dimensional SciBERT embeddings indexed via FAISS (`IndexFlatIP`).
- 92,081-node citation graph with 206,964 bidirectional message-passing edges.
- Handling out-of-graph papers via projection and feature fallback.

### 4. [[GraphSAGE Architecture]], [[Link Prediction & Training Strategy]], & [[Hybrid Ranking & Score Fusion]]
- 2-layer inductive GraphSAGE design, neighbor sampling (`[15, 10]`), and edge mini-batching.
- Leakage-safe train/val/test edge splits and negative edge sampling.
- Mathematical formulation of hybrid score fusion: $S_{final} = \alpha \cdot S_{semantic} + (1 - \alpha) \cdot S_{graph} + \beta \cdot \text{Prior}$.

### 5. [[FastAPI Recommendation Service]] & [[LLM Explanation Engine]]
- REST API design (`/recommend/paper`, `/recommend/query`, `/evaluate`, `/graph/subgraph`).
- Human-in-the-loop explainability: synthesizing citation paths and thematic overlaps into natural language summaries.

### 6. [[React Frontend Architecture]]
- Interactive exploration interface, paper cards, semantic/graph score breakdown bars, and citation neighborhood visualization.

### 7. [[Leakage-Safe Evaluation Protocol]] & [[Evaluation Metrics]]
- Rigorous held-out edge validation avoiding transductive data snooping.
- Metrics: Mean Reciprocal Rank (MRR), Hits@K (K=5, 10, 20), ROC-AUC, and Average Precision (AP).

### 8. [[Implementation Status & Tasks]]
- Live tracker of completed components, active sprint items, and verified milestones.

---

## 🏷️ Key Tags
#overview #architecture #gnn #graphsage #scibert #faiss #recommendation #hybrid-ranking

