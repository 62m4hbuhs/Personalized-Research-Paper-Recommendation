# 💡 LLM Explanation Engine

## 1. Motivation
Standard recommender systems produce cold numeric scores (e.g. `Score: 0.842`). Researchers need to understand:
- *Why did the system rank Paper A above Paper B?*
- *Is this recommendation based on topic similarity, direct citation lineage, or shared co-citation cluster?*

## 2. Explanation Generation Strategy
The explanation engine employs a dual-tier approach:
1. **Rule-Based Signal Decomposition (Instant & Deterministic)**:
   - Evaluates the contributing vectors:
     - If `semantic_score > 0.85` and `graph_score < 0.5`: *"Recommended primarily based on deep topic overlap with your query despite low direct citation links (potential emerging paper)."*
     - If `graph_score > 0.85` and `semantic_score > 0.70`: *"Strongly recommended as a core foundational paper in the citation neighborhood with direct methodological alignment."*
     - If direct citation exists: *"Directly cited by or cites your target seed paper."*
2. **Generative LLM Explanations (LangGraph / Local or Cloud LLM)**:
   - Ingests:
     - Target paper / query summary
     - Candidate paper title, year, citations, and abstract
     - Graph relation (common citations, cluster themes)
   - Prompt format:
     ```
     You are a scholarly research assistant. Explain in 2 concise sentences why the candidate paper below is relevant to the researcher's query/seed paper, referencing both thematic alignment and citation community relevance.
     ```
   - Emits crisp, transparent rationale directly onto the paper card.

---
Back to [[FastAPI Recommendation Service]] | Next: [[React Frontend Architecture]]

