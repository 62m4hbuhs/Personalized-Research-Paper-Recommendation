# ScholarGraph: Personalized Academic Research Paper Recommendation System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Orchestration-1c3c3c.svg)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-3.5_Flash--Lite-4285F4.svg)](https://ai.google.dev/)
[![LangSmith](https://img.shields.io/badge/LangSmith-Tracing-black.svg)](https://smith.langchain.com/)

An end-to-end AI-powered academic discovery platform that fuses **SciBERT dense semantic vector representations** with **GraphSAGE inductive citation network topology**, orchestrated by a **LangGraph agentic state-machine pipeline** and synthesized with **Google Gemini 3.5 Flash-Lite LLM** literature briefings.

---

## Key Highlights

- **Dual-Signal Hybrid Fusion**: Combines content semantics ($S_{\text{semantic}}$ via SciBERT) with citation community topology ($S_{\text{graph}}$ via GraphSAGE).
- **Sub-15ms Vector Retrieval**: 56,433 pre-indexed SciBERT embeddings queried via Facebook AI Similarity Search (**FAISS** `IndexFlatIP`).
- **Graph Neural Network Topology**: 2-layer inductive **GraphSAGE** model with residual projections capturing 103,482 citation edges across 92,081 paper nodes.
- **Agentic Pipeline (LangGraph)**: 4-stage StateGraph (`IntentAnalyzer` $\to$ `SemanticRetriever` $\to$ `GraphSAGEFusion` $\to$ `ExplainabilityAgent`).
- **AI Senior Research Advisor**: **Google Gemini 3.5 Flash-Lite** analyzes recommended papers in real-time, synthesizing conceptual connections and identifying foundational reading anchors.
- **Production Observability**: Full cloud tracing and latency breakdown via **LangSmith** (`research-agent` workspace).
- **All-in-One Minimalist Dashboard**: High-performance, light-blue/cyan frosted-glass UI with live **D3 force-directed citation subgraph**, personal reading library, real-time benchmark charts, and **PDF export**.

---

## Benchmark & Empirical Evaluation

Evaluated against **1,000 held-out citation queries** in a leakage-safe protocol (10% test edges removed from the message-passing graph):

| Recommendation Methodology | Mean Reciprocal Rank (MRR) | Hits@5 | Hits@10 | Hits@20 | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SciBERT Only** (Text Semantics) | 0.2334 | 29.40% | 45.10% | 58.20% | — |
| **GraphSAGE Only** (Citation Topology) | 0.3891 | 42.10% | 57.30% | 68.40% | 73.79% |
| **ScholarGraph Hybrid ($\alpha = 0.60$)** | **0.4672** *(+100.2%)* | **53.30%** | **65.50%** | **71.00%** | **73.79%** |

> **Result**: Combining SciBERT with GraphSAGE achieves a **+100.2% relative improvement** in MRR over pure semantic search and places relevant papers in the top-10 over **65.5%** of the time.

---

## Mathematical Formulation

$$S_{\text{hybrid}}(p, c) = \alpha \cdot S_{\text{semantic}}(p, c) + (1 - \alpha) \cdot S_{\text{graph}}(p, c)$$

Where:
- $S_{\text{semantic}}(p, c) = \cos(\mathbf{e}_p^{\text{SciBERT}}, \mathbf{e}_c^{\text{SciBERT}}) = \frac{\mathbf{e}_p \cdot \mathbf{e}_c}{\|\mathbf{e}_p\| \|\mathbf{e}_c\|}$
- $S_{\text{graph}}(p, c) = \cos(\mathbf{z}_p^{\text{GraphSAGE}}, \mathbf{z}_c^{\text{GraphSAGE}}) = \frac{\mathbf{z}_p \cdot \mathbf{z}_c}{\|\mathbf{z}_p\| \|\mathbf{z}_c\|}$
- $\alpha \in [0, 1]$: Dynamic signal balance slider (default $\alpha = 0.60$).

---

## Project Structure

```
MajorProject/
├── backend/                                # Production FastAPI Server
│   ├── app.py                              # REST API endpoints & route handlers
│   ├── hybrid_recommender.py               # SciBERT + GraphSAGE + FAISS fusion engine
│   ├── agentic_workflow.py                 # LangGraph StateGraph & Gemini 3.5 Flash-Lite client
│   └── models/                             # PyTorch GraphSAGE neural network classes
├── frontend/                               # Minimalist All-in-One Dashboard
│   └── index.html                          # Single-page UI with D3 citation graph & live export
├── pipeline/                               # Offline Data Processing & Training
│   ├── dataset_collector.py                # Academic dataset ingestion & metadata cleaning
│   ├── embedding.py                        # Batch SciBERT 768-d embedding generation
│   ├── build_faiss.py                      # FAISS IndexFlatIP index construction
│   ├── build_graph.py                      # Citation network graph construction
│   ├── preparegraph.py                     # Feature matrix alignment & normalization
│   ├── train_graphsage.py                  # PyTorch Geometric GraphSAGE model training
│   ├── generate_graph_fallback_embeddings.py # Content-projected fallback generator
│   └── evaluate_recommendation.py          # Leakage-safe held-out ranking benchmark
├── experiments/                            # Research & Validation Experiments
│   ├── test_paired_eval.py                 # Paired evaluation metrics
│   ├── test_residual_sage.py               # Residual connection ablation experiments
│   ├── test_graphsage_gpu.py               # GPU inference speed benchmarking
│   └── diagnose_sim.py                     # Vector similarity distributions
├── ScholarGraph_AI_Paper_Recommendations.pdf # Generated AI Literature Briefing PDF
├── ScholarGraph_Project_Documentation.pdf   # Complete Project Architecture & Tech Report
├── PROJECT_REPORT_AND_ARCHITECTURE.md      # Comprehensive Academic Technical Report
├── generate_pdf_report.py                  # PDF documentation generator (ReportLab)
├── print_recommendations_pdf.py            # PDF recommendations exporter (ReportLab)
└── requirements.txt                        # Python dependencies
```

---

## Quickstart Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/62m4hbuhs/Personalized-Research-Paper-Recommendation.git
cd Personalized-Research-Paper-Recommendation

# Create and activate virtual environment
python -m venv myenv
# Windows:
.\myenv\Scripts\activate
# Linux/macOS:
source myenv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables (`.env`)

```env
GOOGLE_API_KEY=your_google_gemini_api_key
GEMINI_MODEL=gemini-3.5-flash-lite

LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_AGENT_ID=research-agent
LANGSMITH_AGENT_ENVIRONMENT=local
```

### 3. Launch the Server

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at **[http://127.0.0.1:8000](http://127.0.0.1:8000)**.

---

## REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | System health, indexed vector count, graph nodes, and paper catalog size |
| `GET` | `/api/search?q={query}` | Smart fast autocomplete with token-level relevance ranking |
| `POST` | `/api/recommend` | Full dual-signal recommendation with LangGraph agent & Gemini synthesis |
| `GET` | `/api/graph/subgraph/{paper_id}` | 1-hop ego citation network nodes and links for D3 visualization |
| `POST` | `/api/user/reading-list/recommend` | Personalized recommendations based on reading list centroid |
| `GET` | `/api/evaluation` | Held-out evaluation metrics (MRR, Hits@K, ROC-AUC) |
| `GET` | `/api/download-documentation` | Download technical architecture documentation PDF |
| `GET` | `/api/download-recommendations-pdf` | Download latest AI literature briefing & recommendations PDF |
| `POST` | `/api/export-recommendations-pdf` | Dynamically generate and export current query recommendations to PDF |

---

## Documentation & PDF Reports

- **[Technical Architecture Report (PDF)](ScholarGraph_Project_Documentation.pdf)**: Complete mathematical formulation, component specifications, and leakage-safe evaluation details.
- **[Sample AI Literature Briefing (PDF)](ScholarGraph_AI_Paper_Recommendations.pdf)**: Full sample export containing Google Gemini 3.5 Flash-Lite literature synthesis and ranked recommendations.
- **[Full Markdown Report](PROJECT_REPORT_AND_ARCHITECTURE.md)**: Extended academic report with Mermaid system flowcharts and ablation analyses.

---

## License

This project is developed for educational and academic research purposes under the MIT License.
