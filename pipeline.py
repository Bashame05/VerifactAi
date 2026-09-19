"""
pipeline.py
───────────
End-to-end misinformation-detection pipeline for VerifactAI.

Chains the three modular stages into a single cohesive pipeline entry point:

  Stage 1: Claim Extraction & Check-Worthiness + AI-Text Scoring
           (via `claim_pipeline.extract_and_score_claims`)
              │
              ▼
  Stage 2: Live Web Search & Semantic Relevance Ranking
           (via `evidence_retrieval.get_evidence_for_claims`)
              │
              ▼
  Stage 3: NLI Verdict & Deterministic Risk Scoring
           (via `verdict_scoring.score_claims`)

Design notes
~~~~~~~~~~~~
- **Early-return on zero claims**:
  If an article contains only personal opinions, banter, or unsubstantiated
  chatter, Stage 1 returns an empty list. Calling Stage 2 (Tavily search API)
  and Stage 3 (Groq NLI LLM) on an empty list would be completely wasted work
  and would incur unnecessary network round-trips. An early return guarantees
  zero search API and LLM token costs on subjective or chatter texts.

- **Stage-level latency profiling**:
  Processing a long article involves local neural models (DeBERTa, RoBERTa,
  MiniLM), external web search (Tavily), and remote LLM inference (Groq).
  Measuring each stage independently allows operators to quickly identify
  bottlenecks (e.g. network latency vs. model throughput) and debug delays.

- **Unified Summary Metrics**:
  Downstream applications (like web dashboards or browser extensions) require
  both high-level summary cards (overall risk, tier, top concern claim) and
  granular claim-by-claim drill-downs.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from claim_pipeline import extract_and_score_claims
from evidence_retrieval import get_evidence_for_claims
from verdict_scoring import assign_risk_tier, score_claims
from article_metrics import compute_article_metrics


def analyze_article(
    article_text: str,
    claim_threshold: float = 0.65,
) -> Dict[str, Any]:
    """Execute the full end-to-end verification pipeline on an article.

    Parameters
    ----------
    article_text : str
        The raw article or paragraph text to be evaluated.
    claim_threshold : float, optional
        Probability threshold for check-worthiness (default 0.65).
        Only sentences with P(claim) >= threshold are forwarded to the
        search and verification stages.

    Returns
    -------
    dict
        Structured evaluation containing:
        - ``article_length``    : character count of input text
        - ``num_claims_found``  : number of check-worthy claims identified
        - ``claims``            : list of fully scored claim dictionaries
        - ``overall_risk_score``: mean risk score across all extracted claims
        - ``overall_risk_tier`` : 'Low', 'Medium', or 'High'
        - ``highest_risk_claim``: dict of the single most risky claim (or None)

    Raises
    ------
    ValueError
        If ``article_text`` is empty or consists only of whitespace.
    """
    if not article_text or not article_text.strip():
        raise ValueError("article_text is empty or whitespace-only — cannot analyze.")

    article_len = len(article_text)
    print(f"\n{'=' * 75}")
    print(f"🚀 STARTING PIPELINE ANALYSIS ({article_len} characters)")
    print(f"{'=' * 75}")

    pipeline_start = time.perf_counter()

    # ─────────────────────────────────────────────────────────────
    # ARTICLE-LEVEL METRICS: Readability & Coherence
    # ─────────────────────────────────────────────────────────────
    # Evaluates holistic stylistic and structural flow across the full text.
    print("\n[Article Metrics] Computing readability and semantic coherence...")
    article_metrics = compute_article_metrics(article_text)

    # ─────────────────────────────────────────────────────────────
    # STAGE 1: Sentence segmentation, check-worthiness & AI detection
    # ─────────────────────────────────────────────────────────────
    # Why time this stage:
    # Measures the throughput of the local transformer models (DeBERTa + RoBERTa)
    # running inference across all segmented sentences.
    print(f"\n[Stage 1/3] Extracting & scoring candidate claims (threshold={claim_threshold})...")
    s1_start = time.perf_counter()
    claims: List[Dict[str, Any]] = extract_and_score_claims(
        article_text=article_text,
        claim_threshold=claim_threshold,
    )
    s1_duration = time.perf_counter() - s1_start
    print(f"✓ Stage 1 completed in {s1_duration:.2f}s — Found {len(claims)} check-worthy claim(s).")

    # ─────────────────────────────────────────────────────────────
    # EARLY EXIT: Zero check-worthy claims
    # ─────────────────────────────────────────────────────────────
    # Why this check exists:
    # If the text is purely subjective (opinions, feelings, chatter), proceeding
    # further would perform 0 searches and 0 LLM queries anyway, but allocating
    # data structures and logging downstream stages adds noise. Returning
    # early provides immediate feedback and saves compute/resources.
    if not claims:
        print("\nℹ No check-worthy factual claims detected. Halting pipeline early.")
        return {
            "article_length": article_len,
            "num_claims_found": 0,
            "claims": [],
            "overall_risk_score": 0.0,
            "overall_risk_tier": assign_risk_tier(0.0),
            "highest_risk_claim": None,
            "article_metrics": article_metrics,
        }

    # ─────────────────────────────────────────────────────────────
    # STAGE 2: Web search (Tavily) & semantic ranking (MiniLM)
    # ─────────────────────────────────────────────────────────────
    # Why time this stage:
    # This stage depends on external HTTP search requests to Tavily and
    # embedding calculations. Search network round-trip latency often
    # dominates total runtime.
    print(f"\n[Stage 2/3] Retrieving & ranking web evidence for {len(claims)} claim(s)...")
    s2_start = time.perf_counter()
    claims = get_evidence_for_claims(claims, max_results=8, top_k=3)
    s2_duration = time.perf_counter() - s2_start
    print(f"✓ Stage 2 completed in {s2_duration:.2f}s.")

    # ─────────────────────────────────────────────────────────────
    # STAGE 3: Groq LLM NLI evaluation & composite risk calculation
    # ─────────────────────────────────────────────────────────────
    # Why time this stage:
    # Measures the time spent communicating with the Groq inference endpoint
    # and parsing structured JSON verdicts.
    print(f"\n[Stage 3/3] Evaluating NLI verdicts and computing risk scores...")
    s3_start = time.perf_counter()
    claims = score_claims(claims)
    s3_duration = time.perf_counter() - s3_start
    print(f"✓ Stage 3 completed in {s3_duration:.2f}s.")

    # ─────────────────────────────────────────────────────────────
    # METRICS AGGREGATION
    # ─────────────────────────────────────────────────────────────
    # Compute the average risk across all claims for overall assessment.
    risk_scores = [c.get("risk_score", 0.0) for c in claims]
    overall_risk_score = round(sum(risk_scores) / len(risk_scores), 2)
    overall_risk_tier = assign_risk_tier(overall_risk_score)

    # Identify the highest-risk claim to power alert banners / summary widgets.
    highest_risk_claim = max(claims, key=lambda c: c.get("risk_score", 0.0))

    total_time = time.perf_counter() - pipeline_start

    print(f"\n{'=' * 75}")
    print("🏁 PIPELINE EXECUTION SUMMARY")
    print(f"  • Stage 1 (Extract & Classify) : {s1_duration:.2f}s ({s1_duration/total_time*100:.1f}%)")
    print(f"  • Stage 2 (Evidence Retrieval) : {s2_duration:.2f}s ({s2_duration/total_time*100:.1f}%)")
    print(f"  • Stage 3 (Verdict & Scoring)  : {s3_duration:.2f}s ({s3_duration/total_time*100:.1f}%)")
    print(f"  • Total Pipeline Latency       : {total_time:.2f}s")
    print(f"  • Overall Article Risk Score   : {overall_risk_score}/100 [{overall_risk_tier}]")
    print(f"{'=' * 75}\n")

    return {
        "article_length": article_len,
        "num_claims_found": len(claims),
        "claims": claims,
        "overall_risk_score": overall_risk_score,
        "overall_risk_tier": overall_risk_tier,
        "highest_risk_claim": highest_risk_claim,
        "article_metrics": article_metrics,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Demonstration / Standalone Sanity Check
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    # A realistic multi-claim test paragraph containing:
    # 1. Verifiable fact: Global heat record in 2024 (NASA/Copernicus).
    # 2. Subjective opinion: Editorial commentary ("In my opinion...").
    # 3. Blatantly false/debunked claim: Great Wall built in 1952.
    SAMPLE_ARTICLE = (
        "In my opinion, world leaders are completely detached from everyday realities "
        "and their press conferences are frankly exhausting to watch. "
        "According to NASA, the global average surface temperature in 2024 was 1.29 °C "
        "above the pre-industrial baseline, confirming it as the hottest year on record. "
        "However, historical education has also deteriorated, with conspiracy theories "
        "alleging that the Great Wall of China was constructed in the 20th century by "
        "the Republic of China government in 1952. "
        "Regardless of what critics believe, we clearly need stronger institutional integrity."
    )

    result = analyze_article(SAMPLE_ARTICLE, claim_threshold=0.60)

    print("\n--- FINAL PIPELINE OUTPUT (JSON) ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))
