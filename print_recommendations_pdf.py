import os
import sys
import json
import urllib.request
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas for dynamic total page count."""
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
            self.drawString(54, 750, "ScholarGraph • AI Literature Synthesis & Recommended Papers")
            self.drawRightString(612 - 54, 750, "Google Gemini 3.5 Flash-Lite + SciBERT + GraphSAGE")
            self.setStrokeColor(colors.HexColor("#bae6fd"))
            self.setLineWidth(0.6)
            self.line(54, 744, 612 - 54, 744)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.6)
        self.line(54, 40, 612 - 54, 40)
        
        self.drawString(54, 28, "ScholarGraph AI Literature Discovery Engine • Major Project Report")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 28, page_str)
        self.restoreState()


def fetch_recommendation_data(query="How Powerful are Graph Neural Networks?"):
    """Fetches real-time recommendation data from the local API."""
    url = "http://127.0.0.1:8000/api/recommend"
    payload = {
        "query": query,
        "alpha": 0.60,
        "top_k": 10,
        "use_agent": True
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def generate_recommendations_pdf(data, output_filename="ScholarGraph_AI_Paper_Recommendations.pdf"):
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()

    # Color Palette matching Light-Blue / Cyan Plasma theme
    primary_blue = colors.HexColor("#0284c7")
    dark_navy = colors.HexColor("#0c1e36")
    cyan_badge = colors.HexColor("#0ea5e9")
    muted_slate = colors.HexColor("#526e8c")
    advisor_bg = colors.HexColor("#f0f9ff")
    advisor_border = colors.HexColor("#38bdf8")
    card_bg = colors.HexColor("#ffffff")
    card_border = colors.HexColor("#bae6fd")
    pill_bg = colors.HexColor("#f8fafc")

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=dark_navy,
        spaceAfter=3
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=primary_blue,
        spaceAfter=10
    )

    section_header = ParagraphStyle(
        'SecHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=dark_navy,
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )

    advisor_heading = ParagraphStyle(
        'AdvisorHead',
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#03446a")
    )

    advisor_body = ParagraphStyle(
        'AdvisorBody',
        fontName='Helvetica',
        fontSize=9,
        leading=13.5,
        textColor=colors.HexColor("#0c2d48")
    )

    seed_title_style = ParagraphStyle(
        'SeedTitle',
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=dark_navy
    )

    seed_meta_style = ParagraphStyle(
        'SeedMeta',
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=muted_slate
    )

    seed_abstract_style = ParagraphStyle(
        'SeedAbstract',
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1e293b")
    )

    paper_rank_style = ParagraphStyle(
        'PaperRank',
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=primary_blue
    )

    paper_title_style = ParagraphStyle(
        'PaperTitle',
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=dark_navy
    )

    paper_meta_style = ParagraphStyle(
        'PaperMeta',
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=muted_slate
    )

    paper_abstract_style = ParagraphStyle(
        'PaperAbstract',
        fontName='Helvetica',
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor("#1e293b")
    )

    metric_badge_style = ParagraphStyle(
        'MetricBadge',
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0369a1")
    )

    table_header_style = ParagraphStyle(
        'TraceHead',
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TraceCell',
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=dark_navy
    )

    story = []

    # -------------------------------------------------------------------------
    # 1. HEADER & DOCUMENT METADATA
    # -------------------------------------------------------------------------
    story.append(Paragraph("ScholarGraph • AI Literature Synthesis & Paper Recommendations", title_style))
    story.append(Paragraph("Dual-Signal AI Retrieval (SciBERT Semantics + GraphSAGE Topology) &bull; Google Gemini LLM Synthesis", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_blue, spaceBefore=0, spaceAfter=8))

    # -------------------------------------------------------------------------
    # 2. ACTIVE SEED PAPER / RESEARCH CONTEXT
    # -------------------------------------------------------------------------
    seed = data.get("seed_paper") or {}
    if seed:
        story.append(Paragraph("Active Research Anchor (Seed Paper)", section_header))
        seed_content = [
            Paragraph(f"<b>Title:</b> {seed.get('title', 'Untitled')}", seed_title_style),
            Paragraph(f"<b>Year:</b> {seed.get('year', 'N/A')} &nbsp;&bull;&nbsp; <b>Citations:</b> {Number(seed.get('citationCount', 0)):,} &nbsp;&bull;&nbsp; <b>Paper ID:</b> {seed.get('paperId', 'N/A')[:16]}...", seed_meta_style),
            Spacer(1, 3),
            Paragraph(f"<b>Abstract:</b> {seed.get('abstract', 'No abstract available.')}", seed_abstract_style)
        ]
        t_seed = Table([[seed_content]], colWidths=[504])
        t_seed.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#bae6fd")),
            ('LINELEFT', (0,0), (-1,-1), 3.5, primary_blue),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 9),
            ('RIGHTPADDING', (0,0), (-1,-1), 9),
        ]))
        story.append(t_seed)
        story.append(Spacer(1, 9))

    # -------------------------------------------------------------------------
    # 3. GOOGLE GEMINI AI RESEARCH ADVISOR SYNTHESIS ("DATA GIVEN BY THE AI")
    # -------------------------------------------------------------------------
    orch = data.get("agentic_orchestration", {})
    gemini_synthesis = orch.get("ai_advisor_synthesis")
    
    if gemini_synthesis:
        story.append(Paragraph("AI Research Advisor Synthesis (Google Gemini 3.5 Flash-Lite)", section_header))
        advisor_box_content = [
            Paragraph("<b>Google Gemini Scholarly Synthesis & Reading Pathway:</b>", advisor_heading),
            Spacer(1, 4),
            Paragraph(gemini_synthesis.replace("\n\n", "<br/><br/>").replace("\n", " "), advisor_body)
        ]
        t_advisor = Table([[advisor_box_content]], colWidths=[504])
        t_advisor.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), advisor_bg),
            ('BOX', (0,0), (-1,-1), 1, advisor_border),
            ('LINELEFT', (0,0), (-1,-1), 3.5, colors.HexColor("#0284c7")),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ]))
        story.append(t_advisor)
        story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # 4. LANGGRAPH AGENTIC EXECUTION TRACE
    # -------------------------------------------------------------------------
    trace_steps = orch.get("workflow_trace", [])
    if trace_steps:
        story.append(Paragraph("LangGraph Agentic Execution Pipeline (4-Stage State Machine)", section_header))
        trace_table_data = [
            [
                Paragraph("Pipeline Stage", table_header_style),
                Paragraph("Agent Node", table_header_style),
                Paragraph("Action & Outcome", table_header_style),
                Paragraph("Duration", table_header_style)
            ]
        ]
        for step in trace_steps:
            trace_table_data.append([
                Paragraph(f"<b>{step.get('stage', '')}</b>", table_cell_style),
                Paragraph(step.get('agent', ''), table_cell_style),
                Paragraph(step.get('action', ''), table_cell_style),
                Paragraph(f"<b>{step.get('duration_ms', 0)} ms</b>", table_cell_style)
            ])
        
        t_trace = Table(trace_table_data, colWidths=[120, 150, 174, 60])
        t_trace.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0369a1")),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#bae6fd")),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
            ('TOPPADDING', (0,0), (-1,-1), 3.5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
            ('LEFTPADDING', (0,0), (-1,-1), 5),
            ('RIGHTPADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(t_trace)
        story.append(Spacer(1, 12))

    # -------------------------------------------------------------------------
    # 5. RECOMMENDED PAPERS ("DATA OF THE PAPERS")
    # -------------------------------------------------------------------------
    recs = data.get("recommendations", [])
    story.append(Paragraph(f"Top Recommended Papers (Ranked by SciBERT + GraphSAGE Fusion, &alpha; = {data.get('alpha', 0.60):.2f})", section_header))
    story.append(Spacer(1, 2))

    for r in recs:
        hybrid_pct = round(r.get("hybrid_score", 0) * 100)
        sem_pct = round(r.get("semantic_score", 0) * 100)
        graph_pct = round(r.get("graph_score", 0) * 100)
        citations = r.get("citationCount", 0)
        year = r.get("year", "N/A")
        title = r.get("title", "Untitled")
        abstract = r.get("abstract") or "No abstract text available in catalog."
        explanation = r.get("explanation", "High overall dual-signal relevance.")

        paper_card_elements = [
            # Top row: Rank, Title, Score
            Table([
                [
                    Paragraph(f"<b>#{r.get('rank', 1)}</b>", paper_rank_style),
                    Paragraph(f"<b>{title}</b>", paper_title_style),
                    Paragraph(f"<font color='#0284c7'><b>{hybrid_pct}%</b></font><br/><font size='6' color='#526e8c'>MATCH</font>", ParagraphStyle('Score', fontName='Helvetica-Bold', fontSize=10, leading=11, alignment=2))
                ]
            ], colWidths=[28, 410, 50], style=[
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('LEFTPADDING', (0,0), (-1,-1), 0),
                ('RIGHTPADDING', (0,0), (-1,-1), 0),
                ('TOPPADDING', (0,0), (-1,-1), 0),
                ('BOTTOMPADDING', (0,0), (-1,-1), 2),
            ]),
            # Metadata row
            Paragraph(f"<b>Year:</b> {year} &nbsp;&bull;&nbsp; <b>Citations:</b> {citations:,} &nbsp;&bull;&nbsp; <b>Semantics:</b> {sem_pct}% &nbsp;&bull;&nbsp; <b>Citation Graph:</b> {graph_pct}% &nbsp;&bull;&nbsp; <i>{explanation}</i>", paper_meta_style),
            Spacer(1, 3),
            # Abstract snippet
            Paragraph(f"<b>Abstract:</b> {abstract}", paper_abstract_style)
        ]

        t_paper = Table([[paper_card_elements]], colWidths=[504])
        t_paper.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), card_bg),
            ('BOX', (0,0), (-1,-1), 0.7, card_border),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(t_paper)
        story.append(Spacer(1, 5))

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated Recommendations PDF: {output_filename}")


def Number(n):
    try:
        return int(n)
    except Exception:
        return 0


if __name__ == "__main__":
    query_param = sys.argv[1] if len(sys.argv) > 1 else "How Powerful are Graph Neural Networks?"
    out_file = sys.argv[2] if len(sys.argv) > 2 else "ScholarGraph_AI_Paper_Recommendations.pdf"
    
    print(f"Fetching recommendation data for: '{query_param}'...")
    try:
        rec_data = fetch_recommendation_data(query_param)
        print("Data fetched successfully! Generating PDF...")
        generate_recommendations_pdf(rec_data, out_file)
    except Exception as e:
        print(f"Error fetching live API data: {e}. Falling back to internal engine...")
        # Direct fallback via recommender instance if API is not reached
        from backend.hybrid_recommender import HybridRecommender
        from backend.agentic_workflow import LangGraphRecommenderAgent
        rec = HybridRecommender()
        rec.load()
        agent = LangGraphRecommenderAgent(rec)
        rec_data = agent.invoke(query=query_param, top_k=10, alpha=0.60)
        generate_recommendations_pdf(rec_data, out_file)
