import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print total page count."""
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#526e8c"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "ScholarGraph • Personalized Research Paper Recommendation System")
            self.drawRightString(612 - 54, 750, "Architecture & Technical Documentation")
            self.setStrokeColor(colors.HexColor("#bae6fd"))
            self.setLineWidth(0.6)
            self.line(54, 744, 612 - 54, 744)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.6)
        self.line(54, 42, 612 - 54, 42)
        
        self.drawString(54, 30, "SciBERT + GraphSAGE • LangGraph Agent • Google Gemini 3.5 Flash-Lite • LangSmith")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 30, page_str)
        self.restoreState()


def build_pdf(filename="ScholarGraph_Project_Documentation.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    primary_blue = colors.HexColor("#0284c7")
    dark_navy = colors.HexColor("#0c1e36")
    cyan_accent = colors.HexColor("#0284c7")
    slate_muted = colors.HexColor("#526e8c")
    box_bg = colors.HexColor("#f0f9ff")
    box_border = colors.HexColor("#bae6fd")

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=dark_navy,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=primary_blue,
        spaceAfter=14
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=dark_navy,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=primary_blue,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=dark_navy,
        spaceAfter=6
    )

    body_bold = ParagraphStyle(
        'Body_Bold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=3
    )

    callout_text = ParagraphStyle(
        'CalloutText',
        parent=body_style,
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#03446a")
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=dark_navy
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10.5,
        textColor=dark_navy
    )

    story = []

    # -------------------------------------------------------------------------
    # HEADER / TITLE BLOCK
    # -------------------------------------------------------------------------
    story.append(Paragraph("ScholarGraph: Personalized Academic Paper Recommendation System", title_style))
    story.append(Paragraph("Dual-Signal AI Architecture Fusing SciBERT Semantic Embeddings with GraphSAGE Citation Topology via LangGraph Agentic Pipeline & Google Gemini LLM", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_blue, spaceBefore=0, spaceAfter=10))

    # Executive Summary Card
    exec_summary_html = """
    <b>System Abstract & Motivation:</b> Traditional academic discovery engines (e.g., Google Scholar, PubMed) rely heavily on raw keyword frequency (causing vocabulary mismatch) or unweighted citation counts (causing popularity echo-chambers that disadvantage new or cross-domain papers). 
    <b>ScholarGraph</b> introduces a dual-signal hybrid framework where <b>SciBERT (768-d text semantics)</b> captures conceptual content and <b>GraphSAGE (128-d inductive topology)</b> captures academic prestige and citation lineage. The entire lifecycle is orchestrated by a 4-stage <b>LangGraph StateGraph</b> agent, enriched with natural-language research briefings generated by <b>Google Gemini 3.5 Flash-Lite</b>, and traced in real time using <b>LangSmith</b>.
    """
    t_summary = Table([[Paragraph(exec_summary_html, callout_text)]], colWidths=[504])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), box_bg),
        ('BOX', (0,0), (-1,-1), 1, box_border),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING', (0,0), (-1,-1), 9),
        ('RIGHTPADDING', (0,0), (-1,-1), 9),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # SECTION 1: ARCHITECTURE OVERVIEW TABLE
    # -------------------------------------------------------------------------
    story.append(Paragraph("1. System Architecture: Roles, Technologies & Contributions", h1_style))
    story.append(Paragraph("The system is engineered as an end-to-end pipeline where specialized machine learning and agentic components solve specific stages of discovery:", body_style))

    roles_data = [
        [
            Paragraph("System Component", table_header_style),
            Paragraph("Underlying Tech", table_header_style),
            Paragraph("Primary Responsibility ('Who is Doing What')", table_header_style),
            Paragraph("System Benefit ('What is it Helping in')", table_header_style)
        ],
        [
            Paragraph("<b>Semantic Intelligence</b>", table_cell_bold),
            Paragraph("SciBERT<br/>(allenai/scibert_uncased)", table_cell_style),
            Paragraph("Extracts deep representations from titles and abstracts into <b>768-d dense vectors</b>.", table_cell_style),
            Paragraph("Resolves terminology mismatch; identifies related papers using distinct vocabularies.", table_cell_style)
        ],
        [
            Paragraph("<b>Vector Search Engine</b>", table_cell_bold),
            Paragraph("FAISS<br/>(IndexFlatIP)", table_cell_style),
            Paragraph("Indexes 56,433 SciBERT vectors; computes exact inner-product (cosine) similarity.", table_cell_style),
            Paragraph("Sub-15ms candidate retrieval from 56k+ literature corpus without exhaustive scans.", table_cell_style)
        ],
        [
            Paragraph("<b>Citation Topology GNN</b>", table_cell_bold),
            Paragraph("GraphSAGE<br/>(PyTorch Geometric)", table_cell_style),
            Paragraph("Aggregates 2-hop neighborhood citations over 103,482 citation edges into <b>128-d vectors</b>.", table_cell_style),
            Paragraph("Captures authoritative structural importance; connects papers via community topology.", table_cell_style)
        ],
        [
            Paragraph("<b>Agentic Orchestration</b>", table_cell_bold),
            Paragraph("LangGraph<br/>(StateGraph Engine)", table_cell_style),
            Paragraph("Coordinates 4 state-machine nodes: Intent Analysis, Retrieval, Fusion, Explainability.", table_cell_style),
            Paragraph("Modular, fault-tolerant execution; eliminates rigid monolithic scripts; passes clean state.", table_cell_style)
        ],
        [
            Paragraph("<b>AI Senior Advisor</b>", table_cell_bold),
            Paragraph("Google Gemini<br/>3.5 Flash-Lite", table_cell_style),
            Paragraph("Analyzes top ranked papers to synthesize overarching themes and recommend anchor paper.", table_cell_style),
            Paragraph("Provides researchers with actionable literature briefings explaining why and what to read first.", table_cell_style)
        ],
        [
            Paragraph("<b>Observability & Tracing</b>", table_cell_bold),
            Paragraph("LangSmith<br/>(Cloud Platform)", table_cell_style),
            Paragraph("Monitors multi-stage latency, tokens, intermediate prompts, and agent state transitions.", table_cell_style),
            Paragraph("Production observability; tracks query trends and verifies agent correctness.", table_cell_style)
        ],
        [
            Paragraph("<b>API & Web Dashboard</b>", table_cell_bold),
            Paragraph("FastAPI + D3.js + Tailwind CSS", table_cell_style),
            Paragraph("Exposes REST endpoints; renders all-in-one UI with live D3 force graph and library.", table_cell_style),
            Paragraph("Unified, interactive researcher workspace without screen switching or lag.", table_cell_style)
        ]
    ]

    t_roles = Table(roles_data, colWidths=[100, 95, 160, 149])
    t_roles.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_blue),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#bae6fd")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_roles)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # SECTION 2: MATHEMATICAL FORMULATION
    # -------------------------------------------------------------------------
    story.append(Paragraph("2. Mathematical Formulation: Dual-Signal Hybrid Fusion", h1_style))
    story.append(Paragraph(
        "To balance text semantics against structural graph topology, ScholarGraph defines a parameterized fusion score <b>S_hybrid</b> between seed paper / query <i>p</i> and candidate paper <i>c</i>:",
        body_style
    ))

    formula_text = """
    <b>Hybrid Scoring Function:</b><br/>
    &nbsp;&nbsp;&nbsp;&nbsp;<b>S_hybrid(p, c) = &alpha; &middot; S_semantic(p, c) + (1 - &alpha;) &middot; S_graph(p, c)</b><br/><br/>
    <b>1. Semantic Signal:</b> S_semantic(p, c) = cos(<b>e</b>_p^(SciBERT), <b>e</b>_c^(SciBERT)) = (<b>e</b>_p &middot; <b>e</b>_c) / (||<b>e</b>_p|| &middot; ||<b>e</b>_c||)<br/>
    <b>2. Citation Signal:</b> S_graph(p, c) = cos(<b>z</b>_p^(GraphSAGE), <b>z</b>_c^(GraphSAGE)) = (<b>z</b>_p &middot; <b>z</b>_c) / (||<b>z</b>_p|| &middot; ||<b>z</b>_c||)<br/>
    <b>3. Dynamic Parameter &alpha; (&alpha; &isin; [0, 1]):</b> Default &alpha; = 0.60 empirically maximizes Mean Reciprocal Rank.
    """
    t_formula = Table([[Paragraph(formula_text, callout_text)]], colWidths=[504])
    t_formula.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t_formula)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # SECTION 3: STEP-BY-STEP LIFECYCLE
    # -------------------------------------------------------------------------
    story.append(Paragraph("3. End-to-End Query Lifecycle (LangGraph State Machine)", h1_style))
    story.append(Paragraph("Every recommendation request executes sequentially across 4 distinct agent nodes in LangGraph:", body_style))

    story.append(Paragraph("<b>Stage 1: IntentAnalyzerNode (~1ms)</b> — Parses input type (Paper Title vs Topic Query vs Personal Reading List). If a title matches the catalog, it resolves the canonical Paper ID and citations; if topic keywords, it configures free-text vector projection.", bullet_style))
    story.append(Paragraph("<b>Stage 2: SemanticRetrieverNode (~15ms)</b> — Queries the 56,433-vector FAISS index using SciBERT 768-d embeddings to retrieve the top 60 semantic candidate papers, pruning the search space by 99.9%.", bullet_style))
    story.append(Paragraph("<b>Stage 3: GraphSAGEFusionNode (~2ms)</b> — Looks up GraphSAGE 128-d structural vectors for candidate papers, evaluates cosine similarity against seed topology, and computes final hybrid ranking using the &alpha; weight.", bullet_style))
    story.append(Paragraph("<b>Stage 4: ExplainabilityAgentNode + Gemini Synthesis (~1.6s)</b> — Formulates multi-perspective explanation metrics (Semantics % vs Graph %) and invokes Google Gemini 3.5 Flash-Lite to produce a scholarly summary and identify the foundational reading anchor.", bullet_style))

    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # SECTION 4: BENCHMARK & EVALUATION RESULTS
    # -------------------------------------------------------------------------
    story.append(Paragraph("4. Scientific Benchmark & Evaluation Results", h1_style))
    story.append(Paragraph(
        "Evaluation strictly follows a <b>Leakage-Safe Held-Out Protocol</b>. The 103,482 citation edges were partitioned into 85% train, 5% val, and 10% test. Test citation links were completely excluded from the adjacency matrix during GraphSAGE evaluation.",
        body_style
    ))

    benchmark_data = [
        [
            Paragraph("Recommendation Methodology", table_header_style),
            Paragraph("MRR (Mean Reciprocal Rank)", table_header_style),
            Paragraph("Hits@5", table_header_style),
            Paragraph("Hits@10", table_header_style),
            Paragraph("Hits@20", table_header_style),
            Paragraph("ROC-AUC", table_header_style)
        ],
        [
            Paragraph("<b>SciBERT Only</b> (Text Semantics)", table_cell_style),
            Paragraph("0.2334", table_cell_style),
            Paragraph("29.40%", table_cell_style),
            Paragraph("45.10%", table_cell_style),
            Paragraph("58.20%", table_cell_style),
            Paragraph("N/A", table_cell_style)
        ],
        [
            Paragraph("<b>GraphSAGE Only</b> (Citation Topology)", table_cell_style),
            Paragraph("0.3891", table_cell_style),
            Paragraph("42.10%", table_cell_style),
            Paragraph("57.30%", table_cell_style),
            Paragraph("68.40%", table_cell_style),
            Paragraph("73.79%", table_cell_style)
        ],
        [
            Paragraph("<b>ScholarGraph Hybrid (&alpha; = 0.60)</b>", table_cell_bold),
            Paragraph("<b>0.4672 (+100.2%)</b>", table_cell_bold),
            Paragraph("<b>53.30%</b>", table_cell_bold),
            Paragraph("<b>65.50%</b>", table_cell_bold),
            Paragraph("<b>71.00%</b>", table_cell_bold),
            Paragraph("<b>73.79%</b>", table_cell_bold)
        ]
    ]

    t_bench = Table(benchmark_data, colWidths=[164, 85, 65, 65, 65, 60])
    t_bench.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0369a1")),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#bae6fd")),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.white, colors.HexColor("#f8fafc")]),
        ('BACKGROUND', (0,3), (-1,3), colors.HexColor("#e0f2fe")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_bench)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # SECTION 5: PROJECT DIRECTORY & IMPLEMENTATION FILES
    # -------------------------------------------------------------------------
    story.append(Paragraph("5. Codebase Structure & Production Implementation", h1_style))
    story.append(Paragraph("The project codebase is partitioned into active production modules, offline data pipelines, and research evaluation scripts:", body_style))

    story.append(Paragraph("<b>backend/app.py:</b> FastAPI web application exposing endpoints for recommendations, fast autocomplete search, D3 subgraphs, and evaluation metrics.", bullet_style))
    story.append(Paragraph("<b>backend/hybrid_recommender.py:</b> Core mathematical engine containing SciBERT tokenizer, FAISS vector search index, and GraphSAGE inference tensors.", bullet_style))
    story.append(Paragraph("<b>backend/agentic_workflow.py:</b> LangGraph StateGraph engine, multi-tier Google Gemini model client with fallback, and LangSmith cloud tracing configuration.", bullet_style))
    story.append(Paragraph("<b>frontend/index.html:</b> High-performance all-in-one dashboard featuring live D3 force citation graphs, library centroid personalization, and benchmarks.", bullet_style))
    story.append(Paragraph("<b>pipeline/:</b> Offline data collection, SciBERT batch embedding generation, FAISS indexing, and PyTorch Geometric GraphSAGE training.", bullet_style))
    story.append(Paragraph("<b>experiments/:</b> Held-out edge evaluation, GPU acceleration benchmarks, similarity diagnosis, and research test harnesses.", bullet_style))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF: {filename}")

if __name__ == "__main__":
    out_name = sys.argv[1] if len(sys.argv) > 1 else "ScholarGraph_Project_Documentation.pdf"
    build_pdf(out_name)
