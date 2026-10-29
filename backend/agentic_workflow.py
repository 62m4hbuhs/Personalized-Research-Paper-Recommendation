"""
LangGraph Agentic Orchestration for Personalized Research Paper Recommendation.

Implements Objective 4 of Major Project Report:
"Integrate LLMs into LangGraph for orchestration of recommendations Workflow."
Orchestrates multi-step agentic graph reasoning:
IntentAnalysis -> SemanticRetrieval -> GraphSAGEFusion -> ExplainabilitySynthesis
"""

import os
import time
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
import numpy as np
from dotenv import load_dotenv

load_dotenv()


class AgentRecommendationState(TypedDict):
    # Inputs
    query: Optional[str]
    seed_paper_id: Optional[str]
    reading_list_ids: Optional[List[str]]
    alpha: float
    top_k: int
    candidate_pool_size: int
    
    # Internal agent state
    intent: str  # "seed_exploration" | "user_reading_list" | "topic_query"
    resolved_seed_id: Optional[str]
    seed_paper: Optional[Dict[str, Any]]
    reading_list_papers: Optional[List[Dict[str, Any]]]
    candidates: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    
    # Audit & Explainability trace
    agent_trace: List[Dict[str, Any]]
    agent_summary: str
    gemini_synthesis: Optional[str]
    execution_time_ms: float


class LangGraphRecommenderAgent:
    def __init__(self, recommender_instance):
        self.recommender = recommender_instance
        self.gemini_model_name = "gemini-2.5-flash"
        self.gemini_client = None
        self._init_gemini()
        self.workflow = self._build_graph()

    def _init_gemini(self):
        """Initializes Google GenAI client if GEMINI_API_KEY is present."""
        load_dotenv()
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if key and not key.startswith("your_") and len(key) > 10:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=key)
                print(f"Google Gemini LLM Agent active ({self.gemini_model_name})!")
            except Exception as e:
                print("Gemini client initialization notice:", e)

    def _call_gemini_advisor(self, prompt: str) -> Optional[str]:
        """Calls Google Gemini model with graceful fallback on error."""
        if not self.gemini_client:
            self._init_gemini()
        if not self.gemini_client:
            return None

        try:
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model_name,
                contents=prompt
            )
            return response.text.strip() if response and response.text else None
        except Exception as e:
            print(f"Gemini LLM call notice (falling back): {e}")
            return None

    def _build_graph(self) -> Any:
        builder = StateGraph(AgentRecommendationState)

        # Register Agentic Nodes
        builder.add_node("intent_analyzer", self._intent_analyzer_node)
        builder.add_node("semantic_retriever", self._semantic_retriever_node)
        builder.add_node("graphsage_fusion", self._graphsage_fusion_node)
        builder.add_node("explainability_agent", self._explainability_agent_node)

        # Set graph edges
        builder.set_entry_point("intent_analyzer")
        builder.add_edge("intent_analyzer", "semantic_retriever")
        builder.add_edge("semantic_retriever", "graphsage_fusion")
        builder.add_edge("graphsage_fusion", "explainability_agent")
        builder.add_edge("explainability_agent", END)

        return builder.compile()

    # --- Node 1: Intent Analysis Agent ---
    def _intent_analyzer_node(self, state: AgentRecommendationState) -> Dict[str, Any]:
        trace = list(state.get("agent_trace", []))
        t0 = time.time()
        
        reading_list = state.get("reading_list_ids") or []
        seed_id = state.get("seed_paper_id")
        query = (state.get("query") or "").strip()

        # Decision routing
        if reading_list:
            intent = "user_reading_list"
            resolved_seed = None
            seed_meta = None
            valid_pids = [pid for pid in reading_list if pid in self.recommender.metadata]
            reading_meta = [self.recommender._clean_paper_dict(self.recommender.metadata[pid]) for pid in valid_pids]
            detail = f"Identified User Reading Profile intent across {len(valid_pids)} bookmarked papers."
        elif seed_id and seed_id in self.recommender.metadata:
            intent = "seed_exploration"
            resolved_seed = seed_id
            seed_meta = self.recommender._clean_paper_dict(self.recommender.metadata[seed_id])
            reading_meta = None
            detail = f"Direct Seed Exploration: '{seed_meta.get('title')}' ({seed_id[:12]}...)"
        elif query:
            # Check if query targets a known paper
            matched_pid, matched_meta = self.recommender.get_paper_by_title_or_id(query)
            if matched_pid:
                intent = "seed_exploration"
                resolved_seed = matched_pid
                seed_meta = matched_meta
                detail = f"Discovered Seed Paper from title query: '{matched_meta.get('title')}'"
            else:
                intent = "topic_query"
                resolved_seed = None
                seed_meta = None
                detail = f"Free Topic Query: '{query}'"
            reading_meta = None
        else:
            intent = "topic_query"
            resolved_seed = None
            seed_meta = None
            reading_meta = None
            detail = "Default topic mode."

        trace.append({
            "stage": "1. Intent Analysis Agent",
            "agent": "IntentAnalyzerNode",
            "action": f"Classified search intent as '{intent}'",
            "detail": detail,
            "duration_ms": round((time.time() - t0) * 1000, 2)
        })

        return {
            "intent": intent,
            "resolved_seed_id": resolved_seed,
            "seed_paper": seed_meta,
            "reading_list_papers": reading_meta,
            "agent_trace": trace
        }

    # --- Node 2: Semantic Retrieval Agent ---
    def _semantic_retriever_node(self, state: AgentRecommendationState) -> Dict[str, Any]:
        trace = list(state.get("agent_trace", []))
        t0 = time.time()
        
        intent = state.get("intent")
        pool_size = state.get("candidate_pool_size", 60)
        
        if intent == "user_reading_list":
            reading_pids = state.get("reading_list_ids") or []
            valid_pids = [pid for pid in reading_pids if pid in self.recommender.metadata]
            # Average vector of reading list
            text_vectors = []
            for pid in valid_pids:
                if pid in self.recommender.paper_id_to_faiss_idx:
                    f_idx = self.recommender.paper_id_to_faiss_idx[pid]
                    vec = np.zeros((1, 768), dtype=np.float32)
                    self.recommender.faiss_index.reconstruct(f_idx, vec[0])
                    text_vectors.append(vec[0])
                else:
                    m = self.recommender.metadata[pid]
                    t = str(m.get("abstract") or m.get("title") or "")
                    vec = self.recommender.encode_text(t)
                    text_vectors.append(vec[0])
            query_vec = np.mean(text_vectors, axis=0, keepdims=True)
            self.recommender.faiss_index.reconstruct(0, np.zeros((768,), dtype=np.float32)) # dummy touch
            import faiss
            faiss.normalize_L2(query_vec)
            exclude_pids = set(valid_pids)
        elif intent == "seed_exploration" and state.get("resolved_seed_id"):
            seed_id = state.get("resolved_seed_id")
            if seed_id in self.recommender.paper_id_to_faiss_idx:
                f_idx = self.recommender.paper_id_to_faiss_idx[seed_id]
                query_vec = np.zeros((1, 768), dtype=np.float32)
                self.recommender.faiss_index.reconstruct(f_idx, query_vec[0])
            else:
                m = self.recommender.metadata.get(seed_id, {})
                t = str(m.get("abstract") or m.get("title") or "")
                query_vec = self.recommender.encode_text(t)
            exclude_pids = {seed_id}
        else:
            query_text = state.get("query") or "Graph Neural Networks"
            query_vec = self.recommender.encode_text(query_text)
            exclude_pids = set()

        search_k = min(pool_size + len(exclude_pids), self.recommender.faiss_index.ntotal)
        scores, indices = self.recommender.faiss_index.search(query_vec, search_k)

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            cand_id = self.recommender.faiss_paper_ids[idx]
            if cand_id in exclude_pids:
                continue

            cand_meta = self.recommender.metadata.get(cand_id, {})
            in_graph = cand_id in self.recommender.paper_id_to_node_idx
            candidates.append({
                "paperId": cand_id,
                "semantic_score": float(score),
                "in_graph": in_graph,
                "title": cand_meta.get("title", "Untitled"),
                "abstract": cand_meta.get("abstract", "") or "",
                "year": cand_meta.get("year"),
                "citationCount": cand_meta.get("citationCount", 0) or 0
            })

        trace.append({
            "stage": "2. Semantic Retrieval Agent",
            "agent": "SemanticRetrieverNode",
            "action": "SciBERT Deep Vector Search",
            "detail": f"Encoded input and retrieved {len(candidates)} candidates via FAISS index (56,433 vectors).",
            "duration_ms": round((time.time() - t0) * 1000, 2)
        })

        return {"candidates": candidates, "agent_trace": trace}

    # --- Node 3: GraphSAGE Fusion Agent ---
    def _graphsage_fusion_node(self, state: AgentRecommendationState) -> Dict[str, Any]:
        trace = list(state.get("agent_trace", []))
        t0 = time.time()
        
        candidates = state.get("candidates", [])
        alpha = state.get("alpha", 0.60)
        top_k = state.get("top_k", 10)
        beta = 0.05
        intent = state.get("intent")
        
        # Determine query graph embedding
        seed_graph_emb = None
        if intent == "user_reading_list":
            reading_pids = state.get("reading_list_ids") or []
            g_vecs = [
                self.recommender.graph_embeddings[self.recommender.paper_id_to_node_idx[pid]]
                for pid in reading_pids
                if pid in self.recommender.paper_id_to_node_idx
            ]
            if g_vecs:
                seed_graph_emb = np.mean(g_vecs, axis=0)
                seed_graph_emb = seed_graph_emb / (np.linalg.norm(seed_graph_emb) + 1e-9)
        elif intent == "seed_exploration" and state.get("resolved_seed_id"):
            sid = state.get("resolved_seed_id")
            if sid in self.recommender.paper_id_to_node_idx:
                seed_graph_emb = self.recommender.graph_embeddings[self.recommender.paper_id_to_node_idx[sid]]
        else:
            # Centroid of top graph-connected candidates
            top_cand_g = [
                self.recommender.graph_embeddings[self.recommender.paper_id_to_node_idx[c["paperId"]]]
                for c in candidates[:10]
                if c["in_graph"]
            ]
            if top_cand_g:
                seed_graph_emb = np.mean(top_cand_g, axis=0)
                seed_graph_emb = seed_graph_emb / (np.linalg.norm(seed_graph_emb) + 1e-9)

        ranked = []
        for c in candidates:
            sem_score = c["semantic_score"]
            if c["in_graph"] and seed_graph_emb is not None:
                n_idx = self.recommender.paper_id_to_node_idx[c["paperId"]]
                cand_g = self.recommender.graph_embeddings[n_idx]
                cos_sim = float(np.dot(seed_graph_emb, cand_g))
                graph_score = max(0.0, min(1.0, (1.0 + cos_sim) / 2.0))
                fallback = False
            else:
                graph_score = sem_score
                fallback = True

            cites = int(c["citationCount"])
            citation_prior = float(np.log1p(cites) / np.log1p(self.recommender.max_citations))
            hybrid_score = float(alpha * sem_score + (1.0 - alpha) * graph_score + beta * citation_prior)

            ranked.append({
                "paperId": c["paperId"],
                "title": c["title"],
                "abstract": c["abstract"],
                "year": c["year"],
                "citationCount": cites,
                "hybrid_score": round(hybrid_score, 4),
                "semantic_score": round(sem_score, 4),
                "graph_score": round(graph_score, 4),
                "citation_prior": round(citation_prior, 4),
                "in_graph": c["in_graph"],
                "fallback_applied": fallback
            })

        ranked.sort(key=lambda x: x["hybrid_score"], reverse=True)
        final_recs = ranked[:top_k]
        for i, item in enumerate(final_recs, 1):
            item["rank"] = i

        trace.append({
            "stage": "3. GraphSAGE Fusion Agent",
            "agent": "GraphSAGEFusionNode",
            "action": "Topological Link Affinity & Dual Signal Fusion",
            "detail": f"Fused GraphSAGE 128-d graph embeddings (weight {1.0 - alpha:.0%}) with SciBERT semantics ({alpha:.0%}) and citation priors.",
            "duration_ms": round((time.time() - t0) * 1000, 2)
        })

        return {"recommendations": final_recs, "agent_trace": trace}

    # --- Node 4: Explainability & Synthesis Agent ---
    def _explainability_agent_node(self, state: AgentRecommendationState) -> Dict[str, Any]:
        trace = list(state.get("agent_trace", []))
        t0 = time.time()
        
        recs = state.get("recommendations", [])
        intent = state.get("intent")
        
        for r in recs:
            sem = r["semantic_score"]
            g = r["graph_score"]
            fallback = r["fallback_applied"]
            cites = r["citationCount"]
            
            if fallback:
                r["explanation"] = f"Semantic match ({sem:.1%}). Paper is newly indexed without direct citation graph links."
            elif intent == "user_reading_list":
                r["explanation"] = f"Personalized to your reading collection: {sem:.1%} topic congruence and {g:.1%} citation proximity."
            elif sem >= 0.82 and g >= 0.75:
                r["explanation"] = f"Dual-signal convergence: {sem:.1%} conceptual similarity and {g:.1%} citation neighborhood proximity."
            elif g > sem + 0.15:
                r["explanation"] = f"Foundational citation network bridge ({g:.1%}) with {cites:,} citations frequently referenced in this field."
            elif sem > g + 0.15:
                r["explanation"] = f"High conceptual innovation ({sem:.1%}) bringing relevant concepts from adjacent research communities."
            else:
                r["explanation"] = f"Balanced relevance: {sem:.1%} content alignment and {g:.1%} citation network proximity."

        summary = (
            f"LangGraph Agent completed 4-stage pipeline for {len(recs)} papers. "
            f"Mode: {intent}. Signal balance: alpha={state.get('alpha', 0.60):.2f}."
        )

        # Gemini LLM Research Advisor Synthesis
        gemini_synthesis = None
        user_context = state.get("query") or (state.get("seed_paper") or {}).get("title") or "Academic Research Topic"
        if recs:
            top_papers_text = "\n".join([
                f"{i+1}. '{r['title']}' ({r.get('year', 'N/A')}, {r.get('citationCount', 0):,} citations) - Score: {r['hybrid_score']:.1%}"
                for i, r in enumerate(recs[:4])
            ])
            prompt = (
                f"You are an expert AI Research Advisor in Computer Science and Machine Learning.\n"
                f"A researcher is exploring: '{user_context}' (Intent: {intent}).\n\n"
                f"Our dual-signal AI system (SciBERT semantic embeddings + GraphSAGE citation topology) ranked these top papers:\n"
                f"{top_papers_text}\n\n"
                f"In 2-3 concise, scholarly sentences:\n"
                f"1. Explain how these recommended papers connect conceptually to the user's research topic.\n"
                f"2. Explicitly recommend which single paper they should read first as the foundational anchor and why.\n"
                f"Keep it concise, direct, and actionable."
            )
            gemini_synthesis = self._call_gemini_advisor(prompt)

        agent_label = f"ExplainabilityAgentNode (Google Gemini 2.5 Flash)" if gemini_synthesis else "ExplainabilityAgentNode (SciBERT Rationale)"
        action_label = "Generated Gemini LLM Literature Synthesis & Rationales" if gemini_synthesis else "Generated natural language rationales"

        trace.append({
            "stage": "4. Explainability & Synthesis Agent",
            "agent": agent_label,
            "action": action_label,
            "detail": "Synthesized academic literature connections and foundational reading recommendations." if gemini_synthesis else "Created multi-perspective explanations balancing topic relevance and citation prestige.",
            "duration_ms": round((time.time() - t0) * 1000, 2)
        })

        return {
            "recommendations": recs,
            "agent_summary": summary,
            "gemini_synthesis": gemini_synthesis,
            "agent_trace": trace
        }

    def invoke(
        self,
        query: Optional[str] = None,
        seed_paper_id: Optional[str] = None,
        reading_list_ids: Optional[List[str]] = None,
        alpha: float = 0.60,
        top_k: int = 10,
        candidate_pool_size: int = 60
    ) -> Dict[str, Any]:
        t_start = time.time()
        initial_state: AgentRecommendationState = {
            "query": query,
            "seed_paper_id": seed_paper_id,
            "reading_list_ids": reading_list_ids or [],
            "alpha": alpha,
            "top_k": top_k,
            "candidate_pool_size": candidate_pool_size,
            "intent": "",
            "resolved_seed_id": None,
            "seed_paper": None,
            "reading_list_papers": None,
            "candidates": [],
            "recommendations": [],
            "agent_trace": [],
            "agent_summary": "",
            "gemini_synthesis": None,
            "execution_time_ms": 0.0
        }

        final_state = self.workflow.invoke(initial_state)
        total_time_ms = round((time.time() - t_start) * 1000, 2)
        final_state["execution_time_ms"] = total_time_ms

        # Sanitize recommendations and seed_paper for strict JSON compliance
        clean_recs = [self.recommender._clean_paper_dict(r) for r in final_state.get("recommendations", [])]
        clean_seed = self.recommender._clean_paper_dict(final_state.get("seed_paper"))
        has_gemini = bool(final_state.get("gemini_synthesis"))

        return {
            "query_type": final_state.get("intent"),
            "seed_paper": clean_seed,
            "reading_list_papers": final_state.get("reading_list_papers"),
            "alpha": alpha,
            "total_candidates_analyzed": len(final_state.get("candidates", [])),
            "recommendations": clean_recs,
            "agentic_orchestration": {
                "framework": "LangGraph StateGraph",
                "llm_engine": f"Google Gemini ({self.gemini_model_name})" if has_gemini else "SciBERT (Transformer)",
                "gemini_active": has_gemini,
                "total_execution_ms": total_time_ms,
                "summary": final_state.get("agent_summary"),
                "ai_advisor_synthesis": final_state.get("gemini_synthesis"),
                "workflow_trace": final_state.get("agent_trace")
            }
        }

