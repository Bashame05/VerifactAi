"""
claim_pipeline.py
─────────────────
Claim-detection pipeline for VerifactAI.

Wires two pretrained models into a single `extract_and_score_claims` function:

1. **Check-worthiness classifier** — fine-tuned DeBERTa-v3-small (binary:
   label 1 = check-worthy claim, label 0 = opinion / chatter).
2. **AI-text detector** — Hello-SimpleAI/chatgpt-detector-roberta (returns
   P(AI-generated) for a given span of text).

Both models are invoked in **batch mode** (one forward pass per model for the
entire article) so latency scales with model cost, not sentence count.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List

import spacy
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

# ──────────────────────────────────────────────
# Paths / model identifiers
# ──────────────────────────────────────────────
SAVE_PATH = "deberta_claim_classifier_fp32_v2"          # local fine-tuned DeBERTa
AI_DETECTOR_ID = "Hello-SimpleAI/chatgpt-detector-roberta"  # HuggingFace hub

# ──────────────────────────────────────────────
# Device selection (GPU when available)
# ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# Model / tokenizer singletons
#   Loaded once at import time so repeated calls
#   don't reload weights.  If you want lazy loading
#   instead, wrap these in a function with
#   functools.lru_cache.
# ──────────────────────────────────────────────
print(f"[claim_pipeline] Loading check-worthiness model from '{SAVE_PATH}' …")
_cw_tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH)
_cw_model = AutoModelForSequenceClassification.from_pretrained(SAVE_PATH).to(DEVICE)
_cw_model.eval()

print(f"[claim_pipeline] Loading AI-text detector from '{AI_DETECTOR_ID}' …")
# `top_k=None` returns scores for *all* labels so we can pick "ChatGPT".
_ai_pipe = pipeline(
    "text-classification",
    model=AI_DETECTOR_ID,
    top_k=None,
    device=DEVICE,
)

print("[claim_pipeline] Loading spaCy sentence segmenter …")
_nlp = spacy.load("en_core_web_sm")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper 1 – Check-worthiness (batched)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Structured as a *batch* function rather than a
# per-sentence wrapper so we tokenise the full
# list in one call and do a single forward pass.
# This avoids O(n) Python-level overhead for
# tokeniser setup and CUDA kernel launches.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_claim_probabilities(texts: List[str]) -> List[float]:
    """Return P(claim) for every text in *texts* using batched inference.

    Parameters
    ----------
    texts : list[str]
        Sentences / spans to classify.

    Returns
    -------
    list[float]
        One probability per input text, representing the model's
        confidence that the text is a **check-worthy factual claim**
        (label 1).
    """
    if not texts:
        return []

    encodings = _cw_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        logits = _cw_model(**encodings).logits          # (B, 2)
        probs = torch.softmax(logits, dim=-1)[:, 1]     # P(label=1)

    return probs.cpu().tolist()


# Convenience single-text wrapper (kept for ad-hoc use / tests).
def get_claim_probability(text: str) -> float:
    """Return P(claim) for a single text string."""
    return get_claim_probabilities([text])[0]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper 2 – AI-text detector (batched)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# The HF `pipeline` already supports batching when
# given a list, so we just parse the nested output
# structure to extract P(ChatGPT).
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_ai_probabilities(texts: List[str]) -> List[float]:
    """Return P(AI-generated) for every text in *texts*.

    The underlying pipeline returns, for each input, a list
    of ``{"label": ..., "score": ...}`` dicts.  We extract the
    ``"ChatGPT"`` label's score as a proxy for P(AI).

    Parameters
    ----------
    texts : list[str]
        Sentences / spans to classify.

    Returns
    -------
    list[float]
        One probability per input text.
    """
    if not texts:
        return []

    # pipeline(...)(list_of_str) → list[ list[{label, score}] ]
    raw_outputs = _ai_pipe(texts, batch_size=len(texts))

    scores: List[float] = []
    for label_scores in raw_outputs:
        # label_scores is e.g.
        # [{"label": "Human", "score": 0.12}, {"label": "ChatGPT", "score": 0.88}]
        ai_score = next(
            (d["score"] for d in label_scores if d["label"] == "ChatGPT"),
            0.0,
        )
        scores.append(ai_score)
    return scores


# Convenience single-text wrapper.
def get_ai_probability(text: str) -> float:
    """Return P(AI-generated) for a single text string."""
    return get_ai_probabilities([text])[0]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main pipeline function
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Designed as a pure function (no side-effects) so
# it can be called repeatedly or parallelised in a
# batch-of-articles setting.  The threshold filter
# is intentionally applied *after* both model calls
# so that the AI score is always available for any
# sentence that passed the check-worthiness gate.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def extract_and_score_claims(
    article_text: str,
    claim_threshold: float = 0.65,
) -> List[dict]:
    """Split *article_text* into sentences, score each one for
    check-worthiness **and** AI-generation probability, and return
    only the sentences whose P(claim) ≥ *claim_threshold*.

    Parameters
    ----------
    article_text : str
        Full article / paragraph text.
    claim_threshold : float, optional
        Minimum P(claim) to retain a sentence (default 0.65).

    Returns
    -------
    list[dict]
        Each dict has the keys:
        - ``text``            – the sentence string
        - ``p_claim``         – P(check-worthy claim)
        - ``p_ai_generated``  – P(AI-generated text)
        - ``char_start``      – character offset (inclusive) in original text
        - ``char_end``        – character offset (exclusive) in original text

    Raises
    ------
    ValueError
        If *article_text* is empty or consists only of whitespace.
    """

    # ── Guard: empty / blank input ────────────────
    if not article_text or not article_text.strip():
        raise ValueError(
            "article_text is empty or whitespace-only — nothing to analyse."
        )

    # ── 1. Sentence segmentation via spaCy ────────
    doc = _nlp(article_text)
    sentences = list(doc.sents)

    if not sentences:
        warnings.warn("spaCy produced zero sentences from the input text.")
        return []

    texts = [sent.text for sent in sentences]
    offsets = [(sent.start_char, sent.end_char) for sent in sentences]

    # ── 2. Batched model inference ────────────────
    p_claims = get_claim_probabilities(texts)
    p_ais    = get_ai_probabilities(texts)

    # ── 3. Filter + assemble output ───────────────
    results: List[dict] = []
    for text, (start, end), p_claim, p_ai in zip(texts, offsets, p_claims, p_ais):
        if p_claim >= claim_threshold:
            results.append(
                {
                    "text": text,
                    "p_claim": round(p_claim, 4),
                    "p_ai_generated": round(p_ai, 4),
                    "char_start": start,
                    "char_end": end,
                }
            )

    if not results:
        warnings.warn(
            f"No sentence met the claim-worthiness threshold "
            f"({claim_threshold:.2f}).  Try lowering the threshold or "
            f"providing a more factual article."
        )

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Quick sanity-check
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    import json

    EXAMPLES = [
        # ── Example 1: factual claims mixed with opinion ──
        (
            "The World Health Organization declared the pandemic over in May 2023. "
            "Honestly, I think the response was too slow and many lives could have "
            "been saved. The global death toll exceeded 6.9 million according to "
            "official reports, though some researchers believe the true figure is "
            "closer to 18 million."
        ),
        # ── Example 2: mostly opinion / chatter ──
        (
            "I just feel like nobody cares about climate change anymore. "
            "It's so frustrating! My neighbour said he saw a polar bear on the news "
            "the other day and it made him really sad."
        ),
        # ── Example 3: long compound sentence + hard facts ──
        (
            "According to NASA, the global average surface temperature in 2024 was "
            "1.29 °C above the pre-industrial baseline, which makes it the hottest "
            "year ever recorded, surpassing the previous record set in 2023 by a "
            "significant margin, and scientists attribute this largely to a "
            "combination of anthropogenic greenhouse-gas emissions and a strong "
            "El Niño event. Meanwhile, sea levels have risen by approximately 101 mm "
            "since 1993 as measured by satellite altimetry."
        ),
        # ── Example 4: real news article with verifiable claims ──
        (
            "A Booth Level Agent (BLA) of the BJP in Jharkhand's Godda district was arrested "
            "following an FIR accusing him of submitting around 200 Form 7 applications seeking "
            "deletion of voters' names, as part of the Special Intensive Revision (SIR) exercise "
            "in the state. The BLA, Vinay Kumar Mandal, is now out on bail. "
            "The complainant and the Booth Level Officer (BLO), of Booth No. 231 of Pachrukhi "
            "village in Godda block, have both submitted that all the applications by Mandal "
            "were in the name of voters belonging to the minority community. Among the voters "
            "whose names were allegedly sought to be deleted from the booth were the BLO, "
            "Sajida Bibi, and her husband. "
            "The FIR was registered at Godda Muffasil Police Station on September 3 against "
            "Mandal, identified as 'BLA 2', under Sections of the Bharatiya Nyaya Sanhita "
            "dealing with forgery and intentional insult intended to provoke breach of peace, "
            "and Section 31 of the Representation of the People Act, which covers false "
            "declaration during preparation of electoral rolls."
        ),
    ]

    for idx, article in enumerate(EXAMPLES, 1):
        print(f"\n{'═' * 72}")
        print(f" EXAMPLE {idx}")
        print(f"{'═' * 72}")
        print(f"Input ({len(article)} chars):\n  {article[:120]}…\n")

        try:
            claims = extract_and_score_claims(article, claim_threshold=0.5)
        except ValueError as exc:
            print(f"  ⚠ Error: {exc}")
            continue

        if claims:
            print(f"  ✓ {len(claims)} check-worthy claim(s) found:\n")
            print(json.dumps(claims, indent=2, ensure_ascii=False))
        else:
            print("  — No check-worthy claims detected.")
