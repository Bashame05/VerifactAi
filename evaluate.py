"""
evaluate.py
───────────
Formal evaluation suite for the VerifactAI misinformation-detection pipeline.

Methodology & Viva Defense Notes:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Why FEVER (Fact Extraction and VERification)?
   FEVER (Thorne et al., 2018) is the standard academic benchmark for automated
   fact-checking and evidence-grounded claim verification. Using FEVER directly
   validates the pipeline against real human-annotated claims and closes the gap
   between synthetic sanity checks and published academic benchmarks.

2. Why track 'insufficient_evidence' separately rather than purely counting it as wrong?
   In real-world misinformation detection, returning "I don't have enough evidence
   to be sure" is an honest epistemological state. Conflating "insufficient evidence"
   with a confident mistake (e.g. confidently asserting that a true claim is
   contradicted) obscures whether the system suffers from false confidence (hallucination)
   or retrieval failure (coverage gap). We count it as an unverified/incorrect call
   for strict binary accuracy, but report its distinct frequency in the confusion
   matrix and classification breakdown.

3. Fault-Tolerant Checkpointing & Cache-Resume:
   Pipeline evaluation involves external APIs (Tavily search & Groq LLM inference).
   To prevent catastrophic data loss from rate limits, connection drops, or transient
   outages, `eval_raw_results.json` is incrementally updated after EVERY evaluated claim.
   Re-running the script resumes from the checkpoint automatically without wasting API calls.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
# ─────────────────────────────────────────────────────────────
# File Paths & Constants
# ─────────────────────────────────────────────────────────────
FEVER_DEV_URL = "https://fever.ai/download/fever/shared_task_dev.jsonl"
FEVER_EVAL_SET_PATH = "fever_eval_set.json"
EVAL_RESULTS_PATH = "eval_raw_results.json"
NUM_SAMPLES_PER_CLASS = 30  # 30 SUPPORTS + 30 REFUTES = 60 balanced claims
RANDOM_SEED = 42


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Dataset Acquisition & Sampling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_fever_sample(
    output_path: str = FEVER_EVAL_SET_PATH,
    n_per_class: int = NUM_SAMPLES_PER_CLASS,
    seed: int = RANDOM_SEED,
) -> List[Dict[str, str]]:
    """Load or sample 60 balanced claims from the FEVER dev set.

    If `output_path` already exists, loads the saved sample directly.
    Otherwise, streams `shared_task_dev.jsonl` from fever.ai, filters for
    SUPPORTS and REFUTES, samples `n_per_class` each, and saves to JSON.
    """
    if os.path.exists(output_path):
        print(f"📦 Loading existing FEVER eval set from '{output_path}'...")
        with open(output_path, "r", encoding="utf-8") as f:
            eval_set = json.load(f)
        print(f"✓ Loaded {len(eval_set)} sampled claims from cache.")
        return eval_set

    print(f"🌐 Fetching FEVER dev set from {FEVER_DEV_URL} ...")
    req = urllib.request.Request(
        FEVER_DEV_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )

    supports_pool: List[str] = []
    refutes_pool: List[str] = []

    # Stream line-by-line so we don't need to load the full 100MB file into memory at once
    with urllib.request.urlopen(req) as resp:
        for raw_line in resp:
            line_str = raw_line.decode("utf-8").strip()
            if not line_str:
                continue
            data = json.loads(line_str)
            label = data.get("label")
            claim = data.get("claim", "").strip()

            if not claim:
                continue

            if label == "SUPPORTS":
                supports_pool.append(claim)
            elif label == "REFUTES":
                refutes_pool.append(claim)

            # Once we have enough candidates to sample from, stop downloading
            if len(supports_pool) >= 500 and len(refutes_pool) >= 500:
                break

    print(f"✓ Found {len(supports_pool)} SUPPORTS and {len(refutes_pool)} REFUTES candidate claims.")

    rng = random.Random(seed)
    sampled_supports = rng.sample(supports_pool, min(n_per_class, len(supports_pool)))
    sampled_refutes = rng.sample(refutes_pool, min(n_per_class, len(refutes_pool)))

    eval_set = [
        {"claim": c, "fever_label": "SUPPORTS"} for c in sampled_supports
    ] + [
        {"claim": c, "fever_label": "REFUTES"} for c in sampled_refutes
    ]

    # Shuffle to interleave SUPPORTS and REFUTES
    rng.shuffle(eval_set)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved {len(eval_set)} balanced claims to '{output_path}'.\n")
    return eval_set


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Evaluation Runner with Incremental Checkpointing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_eval_cache(cache_path: str = EVAL_RESULTS_PATH) -> Dict[str, Dict[str, Any]]:
    """Load existing evaluation checkpoint cache."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Key by claim text for instant lookup
                return {item["claim"]: item for item in data}
        except Exception as exc:
            print(f"⚠ Warning: Could not read existing cache ({exc}), starting clean.")
    return {}


def save_eval_cache(cache_dict: Dict[str, Dict[str, Any]], cache_path: str = EVAL_RESULTS_PATH) -> None:
    """Persist current evaluation results list to disk."""
    results_list = list(cache_dict.values())
    # Atomic write to temporary file first to avoid corruption on unexpected crash
    tmp_path = cache_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, cache_path)


def run_pipeline_evaluation(
    eval_set: List[Dict[str, str]],
    cache_path: str = EVAL_RESULTS_PATH,
) -> List[Dict[str, Any]]:
    """Execute pipeline components on each claim with resume-from-cache logic."""
    cache = load_eval_cache(cache_path)
    total = len(eval_set)
    cached_count = sum(1 for item in eval_set if item["claim"] in cache)

    print(f"🔍 Pipeline Evaluation: {total} total claims ({cached_count} already cached).")

    # Import pipeline stages lazily
    from claim_pipeline import get_claim_probabilities, get_ai_probabilities
    from evidence_retrieval import get_evidence_for_claim
    from verdict_scoring import score_claim

    for idx, item in enumerate(eval_set, 1):
        claim_text = item["claim"]
        fever_label = item["fever_label"]

        if claim_text in cache:
            continue

        print(f"\n[{idx}/{total}] Evaluating: \"{claim_text[:80]}...\"")
        print(f"     Ground Truth FEVER Label: {fever_label}")

        # 1. Local Model Predictions
        p_claim = get_claim_probabilities([claim_text])[0]
        p_ai = get_ai_probabilities([claim_text])[0]

        # 2. Live Web Evidence Retrieval
        evidence = get_evidence_for_claim(claim_text, max_results=6, top_k=3)
        print(f"     Retrieved {len(evidence)} evidence snippet(s)")

        # 3. NLI + Composite Risk Scoring
        claim_dict = {
            "text": claim_text,
            "p_claim": round(p_claim, 4),
            "p_ai_generated": round(p_ai, 4),
            "evidence": evidence,
        }
        scored = score_claim(claim_dict)

        record = {
            "claim": claim_text,
            "fever_label": fever_label,
            "p_claim": scored["p_claim"],
            "p_ai_generated": scored["p_ai_generated"],
            "verdict": scored["verdict"],
            "contradiction_score": scored["contradiction_score"],
            "risk_score": scored["risk_score"],
            "risk_tier": scored["risk_tier"],
            "justification": scored.get("justification", ""),
            "evidence_count": len(evidence),
        }

        print(f"     → Verdict: {record['verdict'].upper()} (risk: {record['risk_score']}, tier: {record['risk_tier']})")

        # Cache immediately after each claim
        cache[claim_text] = record
        save_eval_cache(cache, cache_path)

        # Rate-limiting pause between web/LLM queries
        time.sleep(0.5)

    return [cache[item["claim"]] for item in eval_set if item["claim"] in cache]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Metrics Calculation & Formatted Reporting
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def calculate_and_print_metrics(results: List[Dict[str, Any]]) -> None:
    """Compute and print formal evaluation metrics."""
    if not results:
        print("No evaluation results available.")
        return

    # Mapping logic:
    # FEVER binary classes:
    #   'SUPPORTS' -> 0 (Negative / Non-misinformation)
    #   'REFUTES'  -> 1 (Positive / Misinformation to catch)
    #
    # System predictions:
    #   'supported'            -> 0
    #   'contradicted'         -> 1
    #   'insufficient_evidence'-> counted as miss/incorrect for binary evaluation
    #                            (if true is REFUTES, mapped to 0 -> False Negative;
    #                             if true is SUPPORTS, mapped to 1 -> False Positive)

    y_true: List[int] = []
    y_pred: List[int] = []

    verdict_counts = {
        "supported": 0,
        "contradicted": 0,
        "insufficient_evidence": 0,
    }

    # Breakdown matrix: [True_Label][Pred_Verdict]
    matrix_3x2 = {
        "SUPPORTS": {"supported": 0, "contradicted": 0, "insufficient_evidence": 0},
        "REFUTES": {"supported": 0, "contradicted": 0, "insufficient_evidence": 0},
    }

    for r in results:
        gt = r["fever_label"]
        pred_verdict = r["verdict"]

        verdict_counts[pred_verdict] = verdict_counts.get(pred_verdict, 0) + 1
        matrix_3x2[gt][pred_verdict] += 1

        gt_bin = 1 if gt == "REFUTES" else 0
        y_true.append(gt_bin)

        if pred_verdict == "contradicted":
            y_pred.append(1)
        elif pred_verdict == "supported":
            y_pred.append(0)
        else:
            # Insufficient evidence treated as incorrect/miss
            # If true label was REFUTES (1), predict 0 (FN)
            # If true label was SUPPORTS (0), predict 1 (FP)
            y_pred.append(0 if gt_bin == 1 else 1)

    # Compute binary metrics (Positive class = 1 / REFUTES / Misinformation)
    # TP: True=1, Pred=1 (REFUTES correctly flagged as contradicted)
    # TN: True=0, Pred=0 (SUPPORTS correctly flagged as supported)
    # FP: True=0, Pred=1 (SUPPORTS falsely flagged as contradicted/misinformation)
    # FN: True=1, Pred=0 (REFUTES missed / flagged as supported or insufficient)
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    total_samples = len(y_true)
    acc = (tp + tn) / total_samples if total_samples > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Metrics for Class 0 (SUPPORTS)
    prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = 2 * (prec_0 * rec_0) / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0.0
    support_0 = sum(1 for yt in y_true if yt == 0)
    support_1 = sum(1 for yt in y_true if yt == 1)

    macro_prec = (prec_0 + prec) / 2
    macro_rec = (rec_0 + rec) / 2
    macro_f1 = (f1_0 + f1) / 2

    weighted_prec = (prec_0 * support_0 + prec * support_1) / total_samples if total_samples > 0 else 0.0
    weighted_rec = (rec_0 * support_0 + rec * support_1) / total_samples if total_samples > 0 else 0.0
    weighted_f1 = (f1_0 * support_0 + f1 * support_1) / total_samples if total_samples > 0 else 0.0

    print("\n" + "=" * 75)
    print("📊 FORMAL PIPELINE EVALUATION REPORT (FEVER BENCHMARK)")
    print("=" * 75)
    print(f"Total Evaluated Claims: {len(results)}")
    print(f"  • SUPPORTS Ground-Truth : {support_0}")
    print(f"  • REFUTES Ground-Truth  : {support_1}")

    print("\n" + "─" * 75)
    print("📈 CORE PERFORMANCE METRICS (Positive Class = Misinformation / REFUTES)")
    print("─" * 75)
    print(f"  • Accuracy             : {acc * 100:.2f}%")
    print(f"  • Precision (REFUTES)  : {prec * 100:.2f}%  (When system flags contradiction, how often is it right)")
    print(f"  • Recall (REFUTES)     : {rec * 100:.2f}%  (Fraction of false claims successfully flagged)")
    print(f"  • F1-Score             : {f1:.4f}")
    print(f"  • False-Positive Rate  : {fpr * 100:.2f}%  (Real supported facts falsely flagged as contradicted)")

    print("\n" + "─" * 75)
    print("📋 SYSTEM VERDICT BREAKDOWN")
    print("─" * 75)
    print(f"  • 'supported'            : {verdict_counts.get('supported', 0):>3} claims ({verdict_counts.get('supported', 0)/len(results)*100:.1f}%)")
    print(f"  • 'contradicted'         : {verdict_counts.get('contradicted', 0):>3} claims ({verdict_counts.get('contradicted', 0)/len(results)*100:.1f}%)")
    print(f"  • 'insufficient_evidence': {verdict_counts.get('insufficient_evidence', 0):>3} claims ({verdict_counts.get('insufficient_evidence', 0)/len(results)*100:.1f}%)")

    print("\n" + "─" * 75)
    print("🔲 FULL 3-COLUMN CONFUSION MATRIX")
    print("─" * 75)
    print(f"{'Ground Truth':<15} | {'supported (Pred)':<18} | {'contradicted (Pred)':<20} | {'insufficient (Pred)'}")
    print("-" * 75)
    for gt in ["SUPPORTS", "REFUTES"]:
        sup_cnt = matrix_3x2[gt]["supported"]
        con_cnt = matrix_3x2[gt]["contradicted"]
        ins_cnt = matrix_3x2[gt]["insufficient_evidence"]
        print(f"{gt:<15} | {sup_cnt:<18} | {con_cnt:<20} | {ins_cnt}")
    print("─" * 75)

    print("\n" + "─" * 75)
    print("📄 CLASSIFICATION REPORT (Binary Mapping)")
    print("─" * 75)
    print(f"{'':<16}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}")
    print()
    print(f"{'SUPPORTS (0)':<16}{prec_0:>10.4f}{rec_0:>10.4f}{f1_0:>10.4f}{support_0:>10}")
    print(f"{'REFUTES (1)':<16}{prec:>10.4f}{rec:>10.4f}{f1:>10.4f}{support_1:>10}")
    print()
    print(f"{'accuracy':<16}{'':>10}{'':>10}{acc:>10.4f}{total_samples:>10}")
    print(f"{'macro avg':<16}{macro_prec:>10.4f}{macro_rec:>10.4f}{macro_f1:>10.4f}{total_samples:>10}")
    print(f"{'weighted avg':<16}{weighted_prec:>10.4f}{weighted_rec:>10.4f}{weighted_f1:>10.4f}{total_samples:>10}")
    print("=" * 75 + "\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Entry Point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    print("=" * 75)
    print("🚀 VERIFACT-AI: FULL PIPELINE FEVER EVALUATION")
    print("=" * 75)

    # 1. Acquire or load sampled 60 claims (30 SUPPORTS, 30 REFUTES)
    eval_set = get_fever_sample()

    # 2. Run evaluation with per-claim checkpointing
    results = run_pipeline_evaluation(eval_set)

    # 3. Calculate and display formal metric tables
    calculate_and_print_metrics(results)
