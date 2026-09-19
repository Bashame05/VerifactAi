"""
article_metrics.py
──────────────────
Article-level metrics for the VerifactAI pipeline: readability & coherence.

Why article-level (not per-claim) metrics?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
While check-worthiness, web evidence verification, and NLI verdicts operate
at the granular sentence/claim level to judge factual accuracy, trust in
written content is multi-dimensional:
  1. Readability (Flesch-Kincaid / Flesch Reading Ease):
     Measures structural accessibility and lexical complexity of the prose.
     Low reading ease / high grade level often correlates with dense technical
     or bureaucratic reports, whereas very low grade level with sensational
     framing can indicate populist clickbait or simplified propaganda.
  2. Coherence (Semantic Sequential Embedding Similarity):
     Measures how smoothly ideas transition between adjacent sentences across
     the entire text. Disjointed, fragmented reasoning (low coherence) can
     be a hallmark of synthetic hallucinations, stitched-together scraper
     content, or manipulated disinfo snippets designed to confuse the reader.

These metrics characterize the holistic style, structure, and editorial
quality of the piece as a whole, providing complementary context to the
factual claim verification results.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import textstat

# ─────────────────────────────────────────────────────────────
# Model & Segmenter reuse
# ─────────────────────────────────────────────────────────────
# Reuse the exact sentence-transformer singleton already loaded
# in evidence_retrieval.py, avoiding loading a duplicate ~120 MB
# model into memory.
# Reuse spaCy en_core_web_sm from claim_pipeline._nlp for consistency.
# ─────────────────────────────────────────────────────────────
from evidence_retrieval import _st_model
from claim_pipeline import _nlp


def compute_readability(article_text: str, min_words: int = 30) -> Dict[str, Optional[float]]:
    """Compute Flesch-Kincaid Grade Level and Flesch Reading Ease.

    Parameters
    ----------
    article_text : str
        The full text of the article.
    min_words : int, optional
        Minimum word count required for meaningful readability scoring
        (default: 30). Very short text produces erratic/degenerate scores.

    Returns
    -------
    dict
        {"flesch_kincaid_grade": float or None, "flesch_reading_ease": float or None}
    """
    if not article_text or not article_text.strip():
        return {
            "flesch_kincaid_grade": None,
            "flesch_reading_ease": None,
        }

    word_count = len(article_text.strip().split())
    if word_count < min_words:
        warnings.warn(
            f"[compute_readability] Text contains only {word_count} words (< {min_words}). "
            "Readability formulas require longer passages; returning None."
        )
        return {
            "flesch_kincaid_grade": None,
            "flesch_reading_ease": None,
        }

    try:
        fk_grade = round(float(textstat.flesch_kincaid_grade(article_text)), 2)
        fre_score = round(float(textstat.flesch_reading_ease(article_text)), 2)
        return {
            "flesch_kincaid_grade": fk_grade,
            "flesch_reading_ease": fre_score,
        }
    except Exception as exc:
        warnings.warn(f"[compute_readability] Failed to compute readability metrics: {exc}")
        return {
            "flesch_kincaid_grade": None,
            "flesch_reading_ease": None,
        }


def compute_coherence(article_text: str) -> Dict[str, Any]:
    """Compute semantic coherence as average cosine similarity between consecutive sentences.

    Parameters
    ----------
    article_text : str
        The full text of the article.

    Returns
    -------
    dict
        {
            "avg_coherence": float or None,
            "num_sentence_pairs": int,
            "note": str (optional warning if single/empty sentence)
        }
    """
    if not article_text or not article_text.strip():
        return {
            "avg_coherence": None,
            "num_sentence_pairs": 0,
            "note": "Empty article text.",
        }

    # 1. Segment sentences using the shared spaCy pipeline
    doc = _nlp(article_text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    # Edge case: fewer than 2 sentences -> no consecutive pairs possible
    if len(sentences) < 2:
        return {
            "avg_coherence": None,
            "num_sentence_pairs": 0,
            "note": "Fewer than 2 sentences present; consecutive coherence requires at least one sentence pair.",
        }

    # 2. Batch encode all sentences using the shared sentence-transformer model
    # normalize_embeddings=True ensures vectors have unit L2 norm,
    # so cosine similarity simplifies directly to dot product.
    embeddings = _st_model.encode(
        sentences,
        batch_size=len(sentences),
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    # 3. Compute cosine similarity for each consecutive pair (i, i+1)
    # Consecutive pairs: dot product of row i with row i+1
    consecutive_sims = [
        float(np.dot(embeddings[i], embeddings[i + 1]))
        for i in range(len(embeddings) - 1)
    ]

    avg_coh = round(float(np.mean(consecutive_sims)), 4)

    return {
        "avg_coherence": avg_coh,
        "num_sentence_pairs": len(consecutive_sims),
    }


def compute_article_metrics(article_text: str) -> Dict[str, Any]:
    """Compute both readability and coherence metrics for an article.

    Parameters
    ----------
    article_text : str
        The full text of the article.

    Returns
    -------
    dict
        {
            "readability": {
                "flesch_kincaid_grade": float or None,
                "flesch_reading_ease": float or None
            },
            "coherence": {
                "avg_coherence": float or None,
                "num_sentence_pairs": int
            }
        }
    """
    return {
        "readability": compute_readability(article_text),
        "coherence": compute_coherence(article_text),
    }


if __name__ == "__main__":
    sample_text = (
        "In my opinion, world leaders are completely detached from everyday realities "
        "and their press conferences are frankly exhausting to watch. "
        "According to NASA, the global average surface temperature in 2024 was 1.29 °C "
        "above the pre-industrial baseline, confirming it as the hottest year on record. "
        "However, historical education has also deteriorated, with conspiracy theories "
        "alleging that the Great Wall of China was constructed in the 20th century by "
        "the Republic of China government in 1952. "
        "Regardless of what critics believe, we clearly need stronger institutional integrity."
    )
    coherent_sample = (
    "NASA confirmed that 2024 was the hottest year on record, with global "
    "average surface temperatures 1.29°C above the pre-industrial baseline. "
    "This surpassed the previous record set in 2023 by 0.12°C. "
    "Scientists attribute the increase to a combination of greenhouse gas "
    "emissions and a strong El Niño weather pattern."
)
    print("=" * 60)
    print("ARTICLE METRICS TEST")
    print("=" * 60)
    metrics = compute_article_metrics(sample_text)
    import json
    print(json.dumps(metrics, indent=2))
    print(json.dumps(compute_article_metrics(coherent_sample), indent=2))

