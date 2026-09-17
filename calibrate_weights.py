"""
calibrate_weights.py
────────────────────
Calibration and grid-search script for the VerifactAI risk scoring formula:

    risk_score = 100 * (w1 * p_ai_generated + w2 * p_claim + w3 * contradiction_score)

Design Methodology & Viva Defense
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Separation of Raw Signal Extraction vs. Optimization:
   Extracting the three raw signals involves running local neural networks
   (DeBERTa & RoBERTa), an external HTTP search query (Tavily), and remote LLM
   inference (Groq). Performing these steps takes multiple seconds per claim,
   consumes external API quota, and incurs cost.
   By decoupling the signal collection phase from the weight optimization phase
   and caching the raw signals to `calibration_raw_signals.json`, we can run
   hundreds of grid-search variations, test alternative objective functions,
   or tweak tier threshold definitions in milliseconds without re-querying APIs.

2. Benchmark Composition:
   The evaluation set consists of 24 carefully balanced claims:
   - 9 Ground-Truth TRUE: verifiable historical, scientific, and geopolitical facts.
   - 9 Ground-Truth FALSE: well-documented myths, hoaxes, and conspiracy theories.
   - 6 Ground-Truth UNVERIFIABLE: fictitious hyper-local events with no web footprint.

3. Evaluation Metric:
   Claims map to desired risk tiers:
   - "true"         -> "Low"     (target risk 0 - 33)
   - "unverifiable" -> "Medium"  (target risk 34 - 66)
   - "false"        -> "High"    (target risk 67 - 100)
   Accuracy is computed as the percentage of claims that land in their target tier.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Tuple

# ─────────────────────────────────────────────────────────────
# 1. LABELED BENCHMARK DATASET (24 CLAIMS)
# ─────────────────────────────────────────────────────────────
LABELED_CLAIMS: List[Dict[str, str]] = [
    # ── Ground Truth: TRUE (8+ claims across science, history, geography) ──
    {
        "text": "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        "ground_truth": "true",
        "category": "science",
    },
    {
        "text": "The Apollo 11 mission landed humans on the Moon in July 1969.",
        "ground_truth": "true",
        "category": "history",
    },
    {
        "text": "According to NASA, the global average surface temperature in 2024 was 1.29 °C above the pre-industrial baseline.",
        "ground_truth": "true",
        "category": "science",
    },
    {
        "text": "Alexander Fleming discovered penicillin in 1928 at St Mary's Hospital in London.",
        "ground_truth": "true",
        "category": "science",
    },
    {
        "text": "The Eiffel Tower was inaugurated in Paris in 1889 for the Exposition Universelle.",
        "ground_truth": "true",
        "category": "history",
    },
    {
        "text": "Mount Everest is the highest mountain peak above sea level on Earth.",
        "ground_truth": "true",
        "category": "geography",
    },
    {
        "text": "The United Nations was established in 1945 following the conclusion of World War II.",
        "ground_truth": "true",
        "category": "politics",
    },
    {
        "text": "Photosynthesis is the biological process by which green plants convert sunlight into chemical energy.",
        "ground_truth": "true",
        "category": "science",
    },
    {
        "text": "Brazil is the largest country by both surface area and population in South America.",
        "ground_truth": "true",
        "category": "geography",
    },

    # ── Ground Truth: FALSE (8+ claims across science, history, health) ──
    {
        "text": "The Great Wall of China was built in the 20th century by the Republic of China government in 1952.",
        "ground_truth": "false",
        "category": "history",
    },
    {
        "text": "Vaccines have been proven to cause autism according to official CDC medical research.",
        "ground_truth": "false",
        "category": "health",
    },
    {
        "text": "Humans only use 10 percent of their total brain capacity during daily functioning.",
        "ground_truth": "false",
        "category": "science",
    },
    {
        "text": "The Earth is flat and surrounded by an impenetrable wall of Antarctic ice guarded by world militaries.",
        "ground_truth": "false",
        "category": "science",
    },
    {
        "text": "Drinking highly alkaline water cures type 1 diabetes within thirty days without medical insulin.",
        "ground_truth": "false",
        "category": "health",
    },
    {
        "text": "NASA staged and faked all six Apollo moon landings on an underground movie soundstage in Nevada.",
        "ground_truth": "false",
        "category": "history",
    },
    {
        "text": "Eating carrots gives humans superhuman night vision capabilities due to military radar technology.",
        "ground_truth": "false",
        "category": "health",
    },
    {
        "text": "The Titanic ocean liner was deliberately swapped with the damaged Olympic sister ship as an insurance fraud scheme.",
        "ground_truth": "false",
        "category": "history",
    },
    {
        "text": "5G wireless cellular towers emit high-frequency radiation specifically engineered to spread biological viral infections.",
        "ground_truth": "false",
        "category": "technology",
    },

    # ── Ground Truth: UNVERIFIABLE (6 obscure / fictitious local claims) ──
    {
        "text": "The village panchayat of Kothaguda in Rangareddy district approved a Rs 4.2 crore drainage project on 12 August 2025.",
        "ground_truth": "unverifiable",
        "category": "local/obscure",
    },
    {
        "text": "A local baker in Badulla, Sri Lanka won the provincial sourdough artisan trophy on 14 March 2024.",
        "ground_truth": "unverifiable",
        "category": "local/obscure",
    },
    {
        "text": "Resident Ramesh Patel of Sector 14 Gandhinagar reported finding a 1912 silver coin under his garden driveway last Sunday.",
        "ground_truth": "unverifiable",
        "category": "local/obscure",
    },
    {
        "text": "The municipal council of San Juan Nepomuceno enacted an ordinance requiring all street lamps to be painted violet by October.",
        "ground_truth": "unverifiable",
        "category": "local/obscure",
    },
    {
        "text": "A boutique florist shop called Petals & Thorns in Ballarat closed yesterday after 37 days of continuous operation.",
        "ground_truth": "unverifiable",
        "category": "local/obscure",
    },
    {
        "text": "High school teacher Deborah Higgins of Oamaru completed a 1000-piece wooden puzzle of Lake Tekapo in under four hours.",
        "ground_truth": "unverifiable",
        "category": "local/obscure",
    },
]

CACHE_FILE = "calibration_raw_signals.json"


# ─────────────────────────────────────────────────────────────
# 2. RAW SIGNAL EXTRACTION & PERSISTENCE
# ─────────────────────────────────────────────────────────────
def collect_raw_signals(
    claims: List[Dict[str, str]],
    cache_path: str = CACHE_FILE,
    force_refresh: bool = False,
) -> List[Dict[str, Any]]:
    """Extract raw signals (p_claim, p_ai, contradiction_score) for all claims.

    If cache_path exists and force_refresh is False, loads previously saved
    signals immediately. Otherwise, executes inference across all pipeline
    stages and persists the output.
    """
    if os.path.exists(cache_path) and not force_refresh:
        print(f"📦 Loading cached raw signals from '{cache_path}'...")
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
        print(f"✓ Loaded {len(cached_data)} cached signal records.")
        return cached_data

    print(f"🚀 Cache not found or refresh requested. Extracting signals for {len(claims)} claims...")
    print("⚠️ This will run local models, Tavily web search, and Groq LLM queries.\n")

    # Import pipeline functions lazily so that offline/cached grid searches
    # do not require loading torch/transformers/DeBERTa into memory.
    from claim_pipeline import get_claim_probabilities, get_ai_probabilities
    from evidence_retrieval import get_evidence_for_claim
    from verdict_scoring import _call_groq_nli, _FALLBACK_VERDICT

    raw_results: List[Dict[str, Any]] = []
    total = len(claims)

    # 1. Batch extract check-worthiness and AI probabilities for speed
    all_texts = [c["text"] for c in claims]
    print(f"[1/3] Computing check-worthiness probabilities (batched DeBERTa)...")
    p_claims = get_claim_probabilities(all_texts)

    print(f"[2/3] Computing AI-text probabilities (batched RoBERTa)...")
    p_ais = get_ai_probabilities(all_texts)

    # 2. Sequential Evidence Retrieval + Groq NLI
    print(f"[3/3] Retrieving evidence and running Groq NLI verification...")
    for idx, (item, p_claim, p_ai) in enumerate(zip(claims, p_claims, p_ais), 1):
        text = item["text"]
        gt = item["ground_truth"]
        category = item["category"]

        print(f"\n  [{idx}/{total}] ({category}) \"{text[:75]}...\"")

        # Tavily Search + MiniLM Ranking
        snippets = get_evidence_for_claim(text, max_results=6, top_k=3)
        print(f"     → Tavily found {len(snippets)} relevant snippet(s)")

        # Groq NLI verdict
        verdict_data = _call_groq_nli(text, snippets)
        if verdict_data is None:
            verdict_data = dict(_FALLBACK_VERDICT)
            print(f"     → Groq failed or key missing, used fallback")
        else:
            print(f"     → Verdict: '{verdict_data['verdict']}' (contradiction: {verdict_data['contradiction_score']})")

        record = {
            "text": text,
            "ground_truth": gt,
            "category": category,
            "p_claim": round(p_claim, 4),
            "p_ai_generated": round(p_ai, 4),
            "verdict": verdict_data["verdict"],
            "contradiction_score": round(float(verdict_data["contradiction_score"]), 4),
            "justification": verdict_data.get("justification", ""),
            "evidence_count": len(snippets),
        }
        raw_results.append(record)

        # Rate-limiting cushion between claims
        if idx < total:
            time.sleep(0.5)

    # Save to disk
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved {len(raw_results)} signal records to '{cache_path}'.\n")

    return raw_results


# ─────────────────────────────────────────────────────────────
# 3. WEIGHT GRID SEARCH OPTIMIZER
# ─────────────────────────────────────────────────────────────
EXPECTED_TIERS = {
    "true": "Low",
    "unverifiable": "Medium",
    "false": "High",
}


def evaluate_weights(
    records: List[Dict[str, Any]],
    w1: float,
    w2: float,
    w3: float,
) -> Tuple[float, int, Dict[str, Dict[str, int]]]:
    """Compute tier accuracy for a given (w1, w2, w3) weight combination.

    Returns:
        accuracy: float percentage (0.0 to 100.0)
        correct_count: number of claims in target tier
        breakdown: per ground-truth distribution of assigned tiers
    """
    from verdict_scoring import compute_risk_score, assign_risk_tier

    correct = 0
    breakdown = {
        "true": {"Low": 0, "Medium": 0, "High": 0},
        "false": {"Low": 0, "Medium": 0, "High": 0},
        "unverifiable": {"Low": 0, "Medium": 0, "High": 0},
    }

    for r in records:
        score = compute_risk_score(
            p_ai_generated=r["p_ai_generated"],
            p_claim=r["p_claim"],
            contradiction_score=r["contradiction_score"],
            w_ai=w1,
            w_claim=w2,
            w_contradiction=w3,
        )
        tier = assign_risk_tier(score, verdict=r.get("verdict"))
        gt = r["ground_truth"]
        breakdown[gt][tier] += 1

        if tier == EXPECTED_TIERS[gt]:
            correct += 1

    accuracy = (correct / len(records)) * 100.0
    return round(accuracy, 2), correct, breakdown


def run_grid_search(
    records: List[Dict[str, Any]],
    step: float = 0.05,
) -> List[Dict[str, Any]]:
    """Iterate over all valid combinations of (w1, w2, w3) summing to 1.0."""
    results = []
    num_steps = int(round(1.0 / step))

    for i in range(num_steps + 1):
        w1 = round(i * step, 3)
        for j in range(num_steps - i + 1):
            w2 = round(j * step, 3)
            w3 = round(1.0 - w1 - w2, 3)
            if w3 < -1e-6:
                continue
            w3 = max(0.0, w3)

            acc, correct, breakdown = evaluate_weights(records, w1, w2, w3)
            results.append({
                "w1_ai": w1,
                "w2_claim": w2,
                "w3_contradiction": w3,
                "accuracy": acc,
                "correct": correct,
                "total": len(records),
                "breakdown": breakdown,
            })

    # Sort descending by accuracy, breaking ties in favor of higher contradiction weight (w3)
    results.sort(key=lambda x: (x["accuracy"], x["w3_contradiction"]), reverse=True)
    return results


# ─────────────────────────────────────────────────────────────
# 4. REPORTING & MAIN CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calibrate VerifactAI risk weights.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cache and re-run all API calls / model extractions.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Grid search step size (default: 0.05).",
    )
    args = parser.parse_args()

    print("=" * 75)
    print(" VERIFACT-AI: RISK FORMULA WEIGHT CALIBRATION")
    print("=" * 75)

    # Step 1: Collect or load raw signals
    records = collect_raw_signals(
        LABELED_CLAIMS,
        cache_path=CACHE_FILE,
        force_refresh=args.refresh,
    )

    # Step 2: Evaluate current placeholder baseline (w1=0.3, w2=0.3, w3=0.4)
    b_acc, b_corr, b_breakdown = evaluate_weights(records, 0.3, 0.3, 0.4)

    print("\n" + "─" * 75)
    print("📊 CURRENT BASELINE EVALUATION (Placeholder Weights)")
    print("   Weights: w1(AI)=0.30, w2(Claim)=0.30, w3(Contradiction)=0.40")
    print(f"   Accuracy: {b_acc:.2f}% ({b_corr}/{len(records)} claims in target tier)")
    print("   Tier Distributions:")
    for gt in ["true", "unverifiable", "false"]:
        target = EXPECTED_TIERS[gt]
        print(f"     • Ground Truth '{gt.upper()}' (Target: {target}): {dict(b_breakdown[gt])}")
    print("─" * 75)

    # Step 3: Run Grid Search
    print(f"\n🔍 Running grid search over weight triplets (step size: {args.step})...")
    grid_results = run_grid_search(records, step=args.step)
    print(f"✓ Explored {len(grid_results)} distinct weight combinations.\n")

    # Step 4: Display Top 5 Configurations
    print("=" * 75)
    print("🏆 TOP 5 CALIBRATED WEIGHT CONFIGURATIONS")
    print("=" * 75)
    print(f"{'Rank':<5} | {'w1 (AI)':<8} | {'w2 (Claim)':<10} | {'w3 (Contradiction)':<18} | {'Accuracy':<10} | {'Score'}")
    print("-" * 75)

    for rank, res in enumerate(grid_results[:5], 1):
        print(
            f"{rank:<5} | "
            f"{res['w1_ai']:<8.2f} | "
            f"{res['w2_claim']:<10.2f} | "
            f"{res['w3_contradiction']:<18.2f} | "
            f"{res['accuracy']:<6.2f}%   | "
            f"{res['correct']}/{res['total']}"
        )

    best = grid_results[0]
    print("\n" + "─" * 75)
    print(f"⭐ BEST CONFIGURATION DETAILS (Rank #1):")
    print(f"   Weights: w1={best['w1_ai']:.2f}, w2={best['w2_claim']:.2f}, w3={best['w3_contradiction']:.2f}")
    print(f"   Accuracy: {best['accuracy']:.2f}% (Improvement vs baseline: {best['accuracy'] - b_acc:+.2f}%)")
    print("   Tier Breakdown:")
    for gt in ["true", "unverifiable", "false"]:
        target = EXPECTED_TIERS[gt]
        print(f"     • {gt.upper():<12} -> Target [{target:<6}] : {dict(best['breakdown'][gt])}")
    print("─" * 75)

    print("\n💡 NOTE:")
    print("To re-run the grid search with different steps or threshold logic without hitting")
    print("APIs, simply run:")
    print("    python calibrate_weights.py --step 0.02\n")
