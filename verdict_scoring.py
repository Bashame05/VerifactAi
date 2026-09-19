"""
verdict_scoring.py
──────────────────
NLI verdict + risk-scoring stage for the VerifactAI pipeline.

Given a list of claim dicts already enriched with evidence (output of
``evidence_retrieval.get_evidence_for_claims``), this module:

1. **Sends each claim + its evidence** to a Groq-hosted LLM
   (``openai/gpt-oss-120b``) to perform Natural Language Inference (NLI).
   The LLM is explicitly instructed to reason *only* from the provided
   evidence — never from its own parametric knowledge — and to report
   "insufficient_evidence" when the snippets are sparse or irrelevant.
2. **Parses the structured JSON verdict** returned by Groq (verdict,
   contradiction_score, justification, evidence_spans_used).
3. **Computes a deterministic risk score** by linearly combining three
   signals: P(AI-generated), P(check-worthy), and contradiction_score
   with configurable weights.
4. **Assigns a risk tier** (Low / Medium / High) based on the score.

Design notes
~~~~~~~~~~~~
- We request ``response_format={"type": "json_object"}`` from Groq so
  the model is constrained to valid JSON at the generation level (not
  just post-hoc parsing).  We *also* reinforce "JSON only" in the system
  prompt as a belt-and-suspenders safeguard, since not all providers
  honour the parameter equally.
- If the LLM still returns malformed JSON (rare, but possible under load
  or with very long inputs), we retry up to 2 times before falling back
  to safe defaults — the pipeline must never crash on a single claim.
- Risk weights (w1, w2, w3) are intentionally **not** tuned.  They are
  placeholders to be calibrated on a labelled evaluation set later.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# ──────────────────────────────────────────────
# Load .env (same pattern as evidence_retrieval)
# ──────────────────────────────────────────────
load_dotenv()

from groq import Groq

# ──────────────────────────────────────────────
# Groq client setup
# ──────────────────────────────────────────────
# Read the API key from the environment.  Like the
# Tavily client in evidence_retrieval.py, we warn
# instead of raising so the rest of the module
# (risk-score math, tests with mocked data) remains
# usable even without a live key.
# ──────────────────────────────────────────────
_GROQ_API_KEY: Optional[str] = os.environ.get("GROQ_API_KEY")

_groq_client: Optional[Groq] = None
if _GROQ_API_KEY:
    _groq_client = Groq(api_key=_GROQ_API_KEY)
    print("[verdict_scoring] Groq client initialised [OK]")
else:
    warnings.warn(
        "GROQ_API_KEY not found in environment variables. "
        "score_claim() will fall back to default verdicts. "
        "Set the key via your .env file: GROQ_API_KEY=gsk_..."
    )

# ──────────────────────────────────────────────
# Model identifier
# ──────────────────────────────────────────────
GROQ_MODEL = "openai/gpt-oss-120b"

# ──────────────────────────────────────────────
# Risk-score weights (calibrated via grid search)
#
# Calibrated against a 24-claim labeled benchmark
# (mix of true, false, and unverifiable claims
# across science, history, health, politics, and
# local/obscure categories).
#
# Calibration results:
#   • Achieved 75.00% tier-classification accuracy
#     (vs. 66.67% with the original placeholder
#     0.30 / 0.30 / 0.40 split).
#   • Contradiction score dominates (0.90) because
#     empirical testing showed NLI evidence-grounding
#     is by far the most reliable individual signal.
#   • w_ai and w_claim are kept small but non-zero
#     (0.05 each) rather than the grid search's
#     literal top extreme of 0.00 / 0.00 / 1.00,
#     preserving the system's intended multi-factor
#     design rather than reducing it to a single signal.
#
# Formula:
#   risk = w1 * p_ai + w2 * p_claim + w3 * contradiction
# ──────────────────────────────────────────────
W_AI = 0.05
W_CLAIM = 0.05
W_CONTRADICTION = 0.90

# ──────────────────────────────────────────────
# Maximum retries for malformed Groq responses
# ──────────────────────────────────────────────
MAX_RETRIES = 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Prompt construction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Separated into its own function so (a) the prompt
# is easy to iterate on during development without
# touching inference logic, and (b) it can be unit-
# tested or logged independently.
#
# Key prompt-engineering choices:
#   • "Reason ONLY from the evidence" — prevents the
#     LLM from hallucinating support/contradiction
#     from its own training data.
#   • Explicit "insufficient_evidence" option — many
#     LLMs default to a firm verdict even with weak
#     evidence; naming the option reduces this bias.
#   • JSON schema spelled out in the prompt — even
#     with response_format=json_object, some models
#     need the schema in the prompt to know *which*
#     JSON structure to produce.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _build_nli_prompt(claim_text: str, evidence: List[Dict]) -> str:
    """Build the NLI analysis prompt for Groq.

    Parameters
    ----------
    claim_text : str
        The factual claim to verify.
    evidence : list[dict]
        Evidence snippets, each with ``title``, ``url``, ``content``,
        and ``relevance_score``.

    Returns
    -------
    str
        The fully-formatted user prompt.
    """

    # Format each evidence snippet with its source URL so the model
    # can cite sources in evidence_spans_used.
    evidence_block = ""
    for i, ev in enumerate(evidence, 1):
        evidence_block += (
            f"\n--- Evidence {i} ---\n"
            f"Title: {ev.get('title', 'N/A')}\n"
            f"URL: {ev.get('url', 'N/A')}\n"
            f"Relevance Score: {ev.get('relevance_score', 'N/A')}\n"
            f"Content:\n{ev.get('content', '(no content)')}\n"
        )

    if not evidence:
        evidence_block = "\n(No evidence snippets were retrieved for this claim.)\n"

    return f"""You are a fact-checking assistant. Your task is to determine whether the following CLAIM is supported or contradicted by the PROVIDED EVIDENCE.

CRITICAL RULES:
- Reason ONLY from the evidence provided below. Do NOT use your own general knowledge.
- If the evidence is insufficient, irrelevant, or too sparse to make a determination, you MUST return "insufficient_evidence" as the verdict — do NOT guess.
- Be conservative: only return "supported" or "contradicted" when the evidence clearly and directly addresses the claim.

CLAIM:
"{claim_text}"

EVIDENCE:
{evidence_block}

Respond with ONLY a JSON object (no markdown, no explanation outside the JSON) matching this exact schema:
{{
  "verdict": "supported" | "contradicted" | "insufficient_evidence",
  "contradiction_score": <float 0.0 to 1.0, where 0.0 = fully supported, 0.5 = neutral/insufficient, 1.0 = fully contradicted>,
  "justification": "<1-2 sentence explanation citing which evidence source(s) the verdict is based on>",
  "evidence_spans_used": ["<URL1>", "<URL2>", ...]
}}

Return ONLY the JSON object. No other text."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# System prompt
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Kept as a module constant so it's consistent
# across all calls and easy to version-control.
# The "JSON only" instruction here reinforces the
# response_format parameter at the prompt level.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_SYSTEM_PROMPT = (
    "You are a precise fact-checking NLI (Natural Language Inference) system. "
    "You MUST respond with a single valid JSON object and nothing else. "
    "Do not include any markdown formatting, code fences, or explanatory text "
    "outside the JSON object."
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Groq API call with retry logic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Retry logic is intentionally simple (fixed delay,
# limited to 2 retries) because:
#   • Groq rarely produces malformed JSON when
#     response_format is set correctly.
#   • Exponential backoff is overkill for 2 retries.
#   • If the model is persistently failing, we want
#     to fall back quickly rather than block the
#     pipeline.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _call_groq_nli(claim_text: str, evidence: List[Dict]) -> Optional[Dict]:
    """Send the claim + evidence to Groq and parse the JSON response.

    Returns the parsed dict on success, or ``None`` after exhausting
    retries (so the caller can apply fallback defaults).

    Parameters
    ----------
    claim_text : str
        The claim to verify.
    evidence : list[dict]
        Evidence snippets from evidence_retrieval.

    Returns
    -------
    dict or None
        Parsed JSON verdict, or None on persistent failure.
    """

    if not _groq_client:
        print(
            "  ⚠ [verdict_scoring] Groq client not initialised "
            "(missing GROQ_API_KEY). Using fallback verdict."
        )
        return None

    user_prompt = _build_nli_prompt(claim_text, evidence)

    for attempt in range(1, MAX_RETRIES + 2):  # attempts 1, 2, 3 (initial + 2 retries)
        try:
            response = _groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,            # low temperature for deterministic output
                max_tokens=1024,            # provide ample room for multi-source JSON verdicts
                response_format={"type": "json_object"},
            )

            raw_text = response.choices[0].message.content.strip()

            # ── Parse JSON ──
            verdict_data = json.loads(raw_text)

            # ── Basic validation of required keys ──
            required_keys = {"verdict", "contradiction_score", "justification", "evidence_spans_used"}
            if not required_keys.issubset(verdict_data.keys()):
                missing = required_keys - verdict_data.keys()
                print(
                    f"  ⚠ [attempt {attempt}] Groq response missing keys: {missing}. "
                    f"Retrying…"
                )
                time.sleep(0.5)
                continue

            # ── Normalise verdict to expected values ──
            valid_verdicts = {"supported", "contradicted", "insufficient_evidence"}
            if verdict_data["verdict"] not in valid_verdicts:
                print(
                    f"  ⚠ [attempt {attempt}] Unexpected verdict "
                    f"'{verdict_data['verdict']}'. Retrying…"
                )
                time.sleep(0.5)
                continue

            # ── Clamp contradiction_score to [0, 1] ──
            verdict_data["contradiction_score"] = max(
                0.0, min(1.0, float(verdict_data["contradiction_score"]))
            )

            return verdict_data

        except json.JSONDecodeError as exc:
            print(
                f"  ⚠ [attempt {attempt}] Groq returned malformed JSON: {exc}. "
                f"{'Retrying…' if attempt <= MAX_RETRIES else 'Giving up.'}"
            )
            time.sleep(0.5)

        except Exception as exc:
            # Catches network errors, rate limits, auth failures, etc.
            print(
                f"  ⚠ [attempt {attempt}] Groq API error: "
                f"{type(exc).__name__}: {exc}. "
                f"{'Retrying…' if attempt <= MAX_RETRIES else 'Giving up.'}"
            )
            time.sleep(1.0)

    # All retries exhausted — return None to trigger fallback.
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fallback defaults
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# When Groq is unreachable or persistently returns
# garbage, we use these safe defaults.
#   • verdict = "insufficient_evidence" — conservative,
#     doesn't falsely accuse or clear the claim.
#   • contradiction_score = 0.5 — neutral midpoint,
#     so the risk score reflects uncertainty rather
#     than inflating or deflating the overall score.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_FALLBACK_VERDICT: Dict[str, Any] = {
    "verdict": "insufficient_evidence",
    "contradiction_score": 0.5,
    "justification": "Automated NLI scoring failed after multiple retries. Manual review recommended.",
    "evidence_spans_used": [],
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Risk-score computation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Separated from score_claim() so it can be unit-
# tested independently with known inputs/outputs,
# and so the weights can be tuned without touching
# the LLM integration code.
#
# The formula is a simple weighted linear combination
# — deliberately not a neural network or complex
# model, because:
#   (a) It is fully explainable / auditable.
#   (b) It can be tuned on a small labelled set
#       via grid search or logistic regression.
#   (c) It avoids introducing another ML model
#       whose failures would be hard to debug.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_risk_score(
    p_ai_generated: float,
    p_claim: float,
    contradiction_score: float,
    w_ai: float = W_AI,
    w_claim: float = W_CLAIM,
    w_contradiction: float = W_CONTRADICTION,
) -> float:
    """Compute a 0–100 risk score from three signal probabilities.

    Parameters
    ----------
    p_ai_generated : float
        P(AI-generated text), 0–1.
    p_claim : float
        P(check-worthy claim), 0–1.
    contradiction_score : float
        NLI contradiction score, 0–1.
    w_ai, w_claim, w_contradiction : float
        Weights for each component (should sum to 1.0).

    Returns
    -------
    float
        Risk score scaled to 0–100, rounded to 2 decimal places.
    """
    raw = (
        w_ai * p_ai_generated
        + w_claim * p_claim
        + w_contradiction * contradiction_score
    )
    # Clamp to [0, 1] before scaling (safety net for out-of-range inputs).
    clamped = max(0.0, min(1.0, raw))
    return round(clamped * 100, 2)


def assign_risk_tier(risk_score: float, verdict: Optional[str] = None) -> str:
    """Map a 0–100 risk score to a human-readable tier.

    Thresholds:
        0–33  → "Low"
        34–66 → "Medium"
        67–100 → "High"

    These thresholds are symmetric terciles — simple and interpretable.
    They should be adjusted based on precision/recall requirements once
    the system is evaluated on labelled data.
    """
    if risk_score <= 33:
        tier = "Low"
    elif risk_score <= 66:
        tier = "Medium"
    else:
        tier = "High"

    # Cap at Medium when evidence is inconclusive so 'unknown' is not conflated with 'confirmed false'.
    if verdict == "insufficient_evidence" and tier == "High":
        tier = "Medium"

    return tier


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Single-claim scorer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# This is the core function that chains:
#   LLM verdict → risk formula → tier assignment
# and writes all results back into the claim dict.
#
# It mutates the dict in-place *and* returns it,
# following the same convention as
# get_evidence_for_claims() in evidence_retrieval.py
# so the two modules compose naturally.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def score_claim(claim_dict: Dict) -> Dict:
    """Run NLI verdict + risk scoring on a single claim.

    Adds the following keys to *claim_dict*:
        - ``verdict``
        - ``contradiction_score``
        - ``justification``
        - ``evidence_spans_used``
        - ``risk_score``        (0–100)
        - ``risk_tier``         ("Low" / "Medium" / "High")

    Parameters
    ----------
    claim_dict : dict
        A claim dict with at least ``text``, ``p_claim``,
        ``p_ai_generated``, and ``evidence`` keys.

    Returns
    -------
    dict
        The same dict, augmented in-place.
    """

    claim_text = claim_dict.get("text", "")
    evidence = claim_dict.get("evidence", [])

    # ── 1. Get NLI verdict from Groq (or fallback) ──
    verdict_data = _call_groq_nli(claim_text, evidence)

    if verdict_data is None:
        # LLM failed — use safe defaults.
        verdict_data = dict(_FALLBACK_VERDICT)  # copy so we don't mutate the template
        print("     ⚠ Using fallback verdict (Groq unavailable or malformed response)")

    # ── 2. Write NLI fields into the claim dict ──
    claim_dict["verdict"] = verdict_data["verdict"]
    claim_dict["contradiction_score"] = verdict_data["contradiction_score"]
    claim_dict["justification"] = verdict_data["justification"]
    claim_dict["evidence_spans_used"] = verdict_data["evidence_spans_used"]

    # ── 3. Compute deterministic risk score ──
    claim_dict["risk_score"] = compute_risk_score(
        p_ai_generated=claim_dict.get("p_ai_generated", 0.0),
        p_claim=claim_dict.get("p_claim", 0.0),
        contradiction_score=claim_dict["contradiction_score"],
    )

    # ── 4. Assign risk tier ──
    claim_dict["risk_tier"] = assign_risk_tier(
        claim_dict["risk_score"],
        verdict=claim_dict["verdict"],
    )

    return claim_dict


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Batch scorer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sequential loop (same rationale as the Tavily
# batch in evidence_retrieval.py):
#   • Groq has per-minute rate limits.
#   • A 0.5 s delay between calls is a simple
#     hedge that avoids 429 errors without
#     complex backoff logic.
#   • If throughput matters, swap to asyncio +
#     semaphore-limited concurrency.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def score_claims(claims_list: List[Dict]) -> List[Dict]:
    """Run NLI verdict + risk scoring on every claim in *claims_list*.

    Parameters
    ----------
    claims_list : list[dict]
        Output of ``get_evidence_for_claims`` — each dict must have
        ``text``, ``p_claim``, ``p_ai_generated``, and ``evidence``.

    Returns
    -------
    list[dict]
        The same list, with each dict augmented with verdict, risk
        score, and risk tier fields.
    """

    if not claims_list:
        warnings.warn(
            "[score_claims] Empty claims list — nothing to score."
        )
        return []

    total = len(claims_list)
    for i, claim in enumerate(claims_list, 1):
        claim_text = claim.get("text", "")
        print(
            f"\n  ⚖️  [{i}/{total}] Scoring claim:\n"
            f"     \"{claim_text[:90]}{'…' if len(claim_text) > 90 else ''}\""
        )

        score_claim(claim)

        print(
            f"     → verdict={claim['verdict']}  "
            f"contradiction={claim['contradiction_score']}  "
            f"risk={claim['risk_score']} ({claim['risk_tier']})"
        )

        # Rate-limit delay between Groq API calls.
        if i < total:
            time.sleep(0.5)

    return claims_list


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sanity-check
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":

    # ── Example 1: Claim clearly SUPPORTED by evidence ──
    # NASA temperature claim with real evidence snippets
    # that confirm the numbers.
    claim_supported = {
        "text": (
            "According to NASA, the global average surface temperature "
            "in 2024 was 1.29 °C above the pre-industrial baseline."
        ),
        "p_claim": 0.91,
        "p_ai_generated": 0.15,
        "char_start": 0,
        "char_end": 110,
        "evidence": [
            {
                "title": "NASA Announces 2024 Hottest Year on Record",
                "url": "https://www.nasa.gov/news/2024-hottest-year",
                "content": (
                    "NASA's Goddard Institute for Space Studies (GISS) "
                    "confirmed that 2024 was the warmest year on record, "
                    "with a global average surface temperature 1.29 °C "
                    "(2.32 °F) above the pre-industrial baseline "
                    "(1850-1900 average). This surpassed the previous "
                    "record set in 2023 by 0.12 °C."
                ),
                "relevance_score": 0.94,
            },
            {
                "title": "Global Temperature Report - 2024 Annual",
                "url": "https://climate.copernicus.eu/2024-annual-report",
                "content": (
                    "The Copernicus Climate Change Service reported that "
                    "2024 global surface air temperatures averaged 1.28 °C "
                    "above pre-industrial levels, closely aligning with "
                    "NASA's estimate of 1.29 °C. Both agencies attribute "
                    "the spike to a combination of greenhouse-gas emissions "
                    "and a strong El Niño event."
                ),
                "relevance_score": 0.91,
            },
        ],
    }

    # ── Example 2: Obscure claim with SPARSE / irrelevant evidence ──
    # The fabricated Kothaguda drainage claim — evidence won't match.
    claim_insufficient = {
        "text": (
            "The village panchayat of Kothaguda in Rangareddy district "
            "approved a Rs 4.2 crore drainage project on 12 August 2025."
        ),
        "p_claim": 0.78,
        "p_ai_generated": 0.30,
        "char_start": 201,
        "char_end": 330,
        "evidence": [
            {
                "title": "Rangareddy District Development Plans",
                "url": "https://example.com/rangareddy-development",
                "content": (
                    "Rangareddy district has several ongoing infrastructure "
                    "projects including road widening and water supply "
                    "improvements. The district collector announced a new "
                    "budget allocation for rural development in 2025."
                ),
                "relevance_score": 0.35,
            },
        ],
    }

    # ── Example 3: Clearly FALSE claim with CONTRADICTING evidence ──
    # A fabricated claim that evidence should debunk.
    claim_contradicted = {
        "text": (
            "The Great Wall of China was built in the 20th century "
            "by the Republic of China government in 1952."
        ),
        "p_claim": 0.88,
        "p_ai_generated": 0.72,
        "char_start": 400,
        "char_end": 490,
        "evidence": [
            {
                "title": "Great Wall of China - History",
                "url": "https://www.britannica.com/topic/Great-Wall-of-China",
                "content": (
                    "The Great Wall of China is a series of fortifications "
                    "built across the historical northern borders of China. "
                    "Construction began as early as the 7th century BC, "
                    "with the most well-known sections built by the Ming "
                    "Dynasty (1368–1644). The wall stretches over 21,000 km "
                    "and is one of the most impressive architectural feats "
                    "in history."
                ),
                "relevance_score": 0.96,
            },
            {
                "title": "Great Wall of China | UNESCO World Heritage",
                "url": "https://whc.unesco.org/en/list/438",
                "content": (
                    "The Great Wall was built over many centuries, with "
                    "major construction during the Qin (221–206 BC), Han "
                    "(206 BC – 220 AD), and Ming (1368–1644) dynasties. "
                    "It was designated a UNESCO World Heritage Site in 1987. "
                    "No significant construction occurred in the 20th century."
                ),
                "relevance_score": 0.93,
            },
            {
                "title": "Timeline of the Great Wall",
                "url": "https://example.com/great-wall-timeline",
                "content": (
                    "The earliest walls were built by various feudal states "
                    "over 2,700 years ago. The Republic of China (1912–1949) "
                    "and the People's Republic of China (1949–present) have "
                    "undertaken restoration and preservation work, but no "
                    "new wall construction has taken place since the Ming era."
                ),
                "relevance_score": 0.89,
            },
        ],
    }

    TEST_CLAIMS = [claim_supported, claim_insufficient, claim_contradicted]

    print("=" * 72)
    print(" VERDICT + RISK SCORING — SANITY CHECK")
    print("=" * 72)

    scored_claims = score_claims(TEST_CLAIMS)

    # Pretty-print each scored claim.
    for i, claim in enumerate(scored_claims, 1):
        print(f"\n{'─' * 72}")
        print(f"CLAIM {i}: {claim['text']}")
        print(f"  p_claim        = {claim['p_claim']}")
        print(f"  p_ai_generated = {claim['p_ai_generated']}")
        print(f"  verdict        = {claim['verdict']}")
        print(f"  contradiction  = {claim['contradiction_score']}")
        print(f"  risk_score     = {claim['risk_score']}")
        print(f"  risk_tier      = {claim['risk_tier']}")
        print(f"  justification  = {claim['justification']}")
        print(f"  sources used   = {claim['evidence_spans_used']}")

    print(f"\n{'═' * 72}")
    print(" Done. Review verdicts, risk scores, and tiers above.")
    print(f"{'═' * 72}")
