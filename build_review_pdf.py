import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and draw page numbers and running headers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4A5568"))

        # Header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(54, 750, "VerifactAI — Simple Project Review Guide")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(54, 744, 612 - 54, 744)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(54, 45, 612 - 54, 45)
        
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 32, page_text)
        self.drawString(54, 32, "VerifactAI Fact-Checking System — Project Presentation")
        self.restoreState()


def create_review_pdf(filename="review.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=64,
        bottomMargin=64,
    )

    styles = getSampleStyleSheet()

    # Friendly Color Palette
    PRIMARY = colors.HexColor("#1E3A8A")      # Deep Navy
    SECONDARY = colors.HexColor("#2563EB")    # Bright Blue
    ACCENT = colors.HexColor("#0D9488")       # Teal
    DARK_TEXT = colors.HexColor("#1F2937")    # Dark Charcoal
    LIGHT_BG = colors.HexColor("#F8FAFC")     # Soft Gray-Blue
    CARD_BG = colors.HexColor("#F1F5F9")      # Soft Slate
    BORDER_COLOR = colors.HexColor("#E2E8F0") # Border Gray

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=PRIMARY,
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=SECONDARY,
        spaceAfter=12,
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=SECONDARY,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=DARK_TEXT,
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=4,
    )

    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#1E293B"),
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.white,
        alignment=0,
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=DARK_TEXT,
    )

    story = []

    # -------------------------------------------------------------------------
    # HEADER / TITLE BLOCK
    # -------------------------------------------------------------------------
    story.append(Paragraph("VerifactAI — Misinformation Detector", title_style))
    story.append(Paragraph("Simple Project Review & Presentation Guide", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=PRIMARY, spaceBefore=0, spaceAfter=10))

    # Executive Overview Box
    exec_summary_text = (
        "<b>What is VerifactAI? (Simple Explanation):</b><br/>"
        "VerifactAI is an intelligent tool that reads any news article or social media post and checks if it's telling the truth. "
        "It ignores personal opinions, picks out real factual claims, searches the live internet for proof, "
        "compares the claims against real web evidence using AI, and gives a clear <b>Risk Score (0 to 100)</b> "
        "and <b>Risk Level (Low, Medium, High)</b>.<br/><br/>"
        "<i>This guide explains <b>HOW</b> each part works and <b>WHY</b> we built it this way in plain, simple English.</i>"
    )
    exec_table = Table(
        [[Paragraph(exec_summary_text, callout_style)]],
        colWidths=[504]
    )
    exec_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('LINELEFT', (0,0), (-1,-1), 4, ACCENT),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(exec_table)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # SECTION 1: HOW THE SYSTEM WORKS (THE BIG PICTURE)
    # -------------------------------------------------------------------------
    story.append(Paragraph("1. The Big Picture: How VerifactAI Works", h1_style))
    story.append(Paragraph(
        "VerifactAI works like a team of human fact-checkers, broken into 3 simple steps:", body_style
    ))

    # Architecture Flow Diagram Table
    arch_flow_data = [
        [Paragraph("<b>Step 0: Input Text & Quality Check</b>", table_header_style)],
        [Paragraph("Reads the whole article and checks reading difficulty and how smoothly sentences connect.", table_cell_style)],
        [Paragraph("<b>Step 1: Fact Finder & Opinion Filter</b><br/>• Splits the text into individual sentences.<br/>• Uses a smart AI model to keep ONLY factual statements (filters out personal opinions).<br/>• Checks if the text looks like it was written by ChatGPT.", table_cell_style)],
        [Paragraph("<i>⚡ Smart Shortcut: If the text is 100% opinion or chatter, it stops early to save time and money.</i>", table_cell_style)],
        [Paragraph("<b>Step 2: Live Internet Search</b><br/>• Searches the live web for each factual claim using Tavily Search.<br/>• Picks the 3 most relevant web page paragraphs as evidence.", table_cell_style)],
        [Paragraph("<b>Step 3: Truth Checker & Risk Calculator</b><br/>• Uses an AI reasoning model to check if the web evidence supports or contradicts the claim.<br/>• Calculates a final Risk Score (0 to 100) and assigns a Risk Level (Low, Medium, High).", table_cell_style)],
        [Paragraph("<b>Final Output: Simple Summary Card & Risk Report</b>", table_header_style)],
    ]
    arch_table = Table(arch_flow_data, colWidths=[504])
    arch_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('BACKGROUND', (0,-1), (-1,-1), SECONDARY),
        ('BACKGROUND', (0,1), (-1,-2), CARD_BG),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(arch_table)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # SECTION 2: HOW AND WHY FOR EACH PART (SIMPLE EXPLANATIONS)
    # -------------------------------------------------------------------------
    story.append(Paragraph("2. Step-by-Step Breakdown: HOW and WHY", h1_style))

    # Pre-Check: Article Style & Readability
    story.append(Paragraph("Pre-Check: Writing Style & Reading Difficulty", h2_style))
    story.append(Paragraph(
        "<b>HOW IT WORKS:</b> Before checking facts, VerifactAI measures how easy or hard the text is to read "
        "(reading grade level) and how smoothly ideas flow from one sentence to the next (sentence coherence).", body_style
    ))
    story.append(Paragraph(
        "<b>WHY WE DID THIS:</b> Fake news and AI-generated articles often feel 'off'. Sensational clickbait is usually written at a very basic level to grab attention, while fake AI articles often have choppy, disconnected sentences. Measuring style helps spot weird writing patterns.", bullet_style
    ))

    # Step 1: Fact Finder
    story.append(Paragraph("Step 1: Fact Finder & Opinion Filter", h2_style))
    story.append(Paragraph(
        "<b>HOW IT WORKS:</b><br/>"
        "1. Splits the article into sentences.<br/>"
        "2. Passes each sentence through a specialized AI model (DeBERTa) that scores whether the sentence is a <b>verifiable fact</b> or just a <b>personal opinion</b>.<br/>"
        "3. Passes sentences through an AI-detector model (RoBERTa) to calculate the chance it was generated by ChatGPT.<br/>"
        "4. Only sentences that are strong factual claims (score ≥ 0.65) move to the next step.", body_style
    ))
    
    stage1_table_data = [
        [Paragraph("<b>Design Decision</b>", table_header_style), Paragraph("<b>Why It Matters (In Plain English)</b>", table_header_style)],
        [Paragraph("<b>Batch Processing</b>", table_cell_style), Paragraph("We feed all sentences into the AI model at once instead of one by one. This makes the system run up to 5x faster.", table_cell_style)],
        [Paragraph("<b>Filtering Out Opinions</b>", table_cell_style), Paragraph("You can't fact-check feelings like <i>'I think rainy days are sad'</i>. Filtering out opinions ensures we only waste web search effort on real facts.", table_cell_style)],
        [Paragraph("<b>Early Exit Guard</b>", table_cell_style), Paragraph("If an article has ZERO factual claims (e.g. casual chatter), the system stops immediately. This saves 100% of web search fees and API costs.", table_cell_style)],
    ]
    st1_table = Table(stage1_table_data, colWidths=[130, 374])
    st1_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(st1_table)
    story.append(Spacer(1, 10))

    # Step 2: Web Search
    story.append(Paragraph("Step 2: Live Internet Evidence Search", h2_style))
    story.append(Paragraph(
        "<b>HOW IT WORKS:</b><br/>"
        "1. For every factual claim, VerifactAI performs a live web search using Tavily Search API.<br/>"
        "2. It downloads the text of top web pages and uses a lightweight AI model (MiniLM) to rank the snippets by how closely they match the meaning of the claim.<br/>"
        "3. It keeps the <b>top 3 best matching snippets</b> as proof.", body_style
    ))
    
    stage2_table_data = [
        [Paragraph("<b>Design Decision</b>", table_header_style), Paragraph("<b>Why It Matters (In Plain English)</b>", table_header_style)],
        [Paragraph("<b>Tavily Search API</b>", table_cell_style), Paragraph("Standard Google searches just give short titles. Tavily pulls full, clean text from web pages so our system gets real context.", table_cell_style)],
        [Paragraph("<b>Smart Relevance Ranking</b>", table_cell_style), Paragraph("Search engines rank pages by popularity. Our local AI model ranks snippets specifically by how directly they talk about the claim.", table_cell_style)],
        [Paragraph("<b>Keeping Top 3 Snippets</b>", table_cell_style), Paragraph("Focusing on the 3 best snippets removes clutter and keeps the verification fast and low-cost.", table_cell_style)],
    ]
    st2_table = Table(stage2_table_data, colWidths=[130, 374])
    st2_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(st2_table)
    story.append(Spacer(1, 10))

    # Step 3: Truth Verification & Risk Score
    story.append(Paragraph("Step 3: Truth Verification & Risk Calculation", h2_style))
    story.append(Paragraph(
        "<b>HOW IT WORKS:</b><br/>"
        "1. <b>AI Reasoning:</b> Sends the claim and top 3 web snippets to an AI model (Groq LLM). "
        "The AI is strictly instructed: <i>'Reason ONLY from the provided web proof. If evidence is missing, say insufficient evidence.'</i><br/>"
        "2. <b>Verdict:</b> The AI outputs one of three verdicts: <b>Supported</b> (True), <b>Contradicted</b> (False), or <b>Insufficient Evidence</b>.<br/>"
        "3. <b>Simple Risk Score Formula:</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>Risk Score = 100 × [ (0.05 × AI-Text Score) + (0.05 × Fact Score) + (0.90 × Contradiction Score) ]</b><br/>"
        "4. <b>Risk Levels:</b> 0–33 = Low Risk (Green) | 34–66 = Medium Risk (Yellow) | 67–100 = High Risk (Red). "
        "<i>(If evidence is missing, the risk is capped at Medium so we don't falsely accuse someone).</i>", body_style
    ))

    stage3_table_data = [
        [Paragraph("<b>Design Decision</b>", table_header_style), Paragraph("<b>Why It Matters (In Plain English)</b>", table_header_style)],
        [Paragraph("<b>Strict AI Instructions</b>", table_cell_style), Paragraph("Forcing the AI to look ONLY at web proof stops it from guessing or making up fake information (hallucinating).", table_cell_style)],
        [Paragraph("<b>'Insufficient Evidence' Option</b>", table_cell_style), Paragraph("If web search finds no clear proof, admitting 'we don't know' is much safer than making a false guess.", table_cell_style)],
        [Paragraph("<b>Fixed Math Formula</b>", table_cell_style), Paragraph("Instead of letting the AI guess a random risk number, we use a simple math formula that is 100% transparent and reliable.", table_cell_style)],
        [Paragraph("<b>Heavy Contradiction Weight (0.90)</b>", table_cell_style), Paragraph("Our tests proved that whether real web evidence contradicts a claim is by far the most important factor in catching fake news.", table_cell_style)],
    ]
    st3_table = Table(stage3_table_data, colWidths=[130, 374])
    st3_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(st3_table)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # SECTION 3: HOW WE TESTED AND IMPROVED IT
    # -------------------------------------------------------------------------
    story.append(Paragraph("3. How We Tested & Improved the System", h1_style))
    story.append(Paragraph(
        "We didn't just write code — we tested it on real claims to make sure it works accurately.", body_style
    ))

    # Calibration Summary Box
    calib_text = (
        "<b>1. Formula Tuning (<code>calibrate_weights.py</code>):</b><br/>"
        "• We tested 24 benchmark claims (True facts, False myths, and Unverifiable rumors).<br/>"
        "• We tried hundreds of weight combinations in a grid search.<br/>"
        "• Tuning the weights boosted our classification accuracy from <b>66.67%</b> up to <b>75.00%</b>!"
    )
    story.append(Table([[Paragraph(calib_text, callout_style)]], colWidths=[504], style=[
        ('BACKGROUND', (0,0), (-1,-1), LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('LINELEFT', (0,0), (-1,-1), 4, SECONDARY),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(Spacer(1, 8))

    # FEVER Academic Evaluation Summary
    fever_text = (
        "<b>2. Real Academic Benchmark (<code>evaluate.py</code>):</b><br/>"
        "• We evaluated the pipeline against 60 human-checked claims from the famous <b>FEVER</b> fact-checking dataset.<br/>"
        "• We added automatic saving so if the internet drops, the test resumes right where it left off without wasting API calls."
    )
    story.append(Table([[Paragraph(fever_text, callout_style)]], colWidths=[504], style=[
        ('BACKGROUND', (0,0), (-1,-1), LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('LINELEFT', (0,0), (-1,-1), 4, ACCENT),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # SECTION 4: USER INTERFACE & API
    # -------------------------------------------------------------------------
    story.append(Paragraph("4. Backend API & Frontend Dashboard", h1_style))
    story.append(Paragraph(
        "VerifactAI comes with a clean web server (<code>api.py</code> using FastAPI) and a simple test webpage (<code>index.html</code>).", body_style
    ))
    
    api_features_data = [
        [Paragraph("<b>Component</b>", table_header_style), Paragraph("<b>What It Does (Simple Explanation)</b>", table_header_style)],
        [Paragraph("<code>POST /analyze</code>", table_cell_style), Paragraph("The main endpoint. Send any article text, and it returns the risk score, claims, web sources, and time taken.", table_cell_style)],
        [Paragraph("<code>index.html</code>", table_cell_style), Paragraph("A visual dashboard where you can paste text, click 'Analyze', and instantly see colored risk cards and claim breakdowns.", table_cell_style)],
        [Paragraph("<b>Time Tracking</b>", table_cell_style), Paragraph("Tracks how many seconds each step took (Step 1, Step 2, Step 3) so you can see where time was spent.", table_cell_style)],
    ]
    api_table = Table(api_features_data, colWidths=[130, 374])
    api_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(api_table)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # SECTION 5: EASY PROJECT REVIEW Q&A (PRESENTATION CHEAT SHEET)
    # -------------------------------------------------------------------------
    story.append(Paragraph("5. Simple Presentation Q&A (Review Cheat Sheet)", h1_style))
    story.append(Paragraph(
        "Here are 5 easy answers to common questions reviewers might ask tomorrow:", body_style
    ))

    viva_qa = [
        ("Q1: Why not just ask ChatGPT if an article is true?",
         "A1: ChatGPT relies on old memory and can easily hallucinate (make up fake facts). VerifactAI searches the live internet for real proof and forces the AI to look ONLY at that proof."),

        ("Q2: What is the purpose of the 'Early Exit' in Step 1?",
         "A2: If an article is 100% opinions or personal banter, doing web searches and AI queries is a waste of time and money. Early exit stops the system immediately and returns 0 risk."),

        ("Q3: How did you pick the numbers in your Risk Formula?",
         "A3: We didn't guess. We tested hundreds of combinations on 24 real benchmark claims. Giving 90% weight to evidence contradiction gave the highest accuracy (75%)."),

        ("Q4: What if internet search returns no results?",
         "A4: The system marks the claim as 'Insufficient Evidence' and caps the risk tier at Medium. It never makes a wild guess when evidence is missing."),

        ("Q5: What models are you running locally vs on the cloud?",
         "A5: We run DeBERTa (fact-filter) and MiniLM (similarity ranker) locally on our computer for speed. We only call cloud APIs for web search (Tavily) and final reasoning (Groq).")
    ]

    for q, a in viva_qa:
        qa_box = [
            [Paragraph(f"<b>{q}</b>", ParagraphStyle('QStyle', parent=body_style, fontName='Helvetica-Bold', textColor=PRIMARY))],
            [Paragraph(f"{a}", body_style)]
        ]
        qa_table = Table(qa_box, colWidths=[504])
        qa_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), CARD_BG),
            ('BOX', (0,0), (-1,-1), 0.5, BORDER_COLOR),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(qa_table)
        story.append(Spacer(1, 5))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated simplified {filename}")

if __name__ == "__main__":
    create_review_pdf()
