"""
evidence_retrieval.py
─────────────────────
Live evidence retrieval stage for the VerifactAI pipeline.

Given a list of check-worthy claim dicts (output of
``claim_pipeline.extract_and_score_claims``), this module:

1. **Searches the web** for each claim via the Tavily Search API.
2. **Ranks** retrieved snippets by semantic similarity to the claim
   using a sentence-transformer embedding model.
3. Returns each claim dict augmented with an ``"evidence"`` key
   containing the top-k most relevant snippets.

Design notes
~~~~~~~~~~~~
- The sentence-transformer model is loaded **once at module level** to
  amortise the ~1 s load time across arbitrarily many calls.  This is
  the standard pattern for inference-only models in a long-running
  pipeline or API server.
- Tavily is used instead of raw Google/Bing because it returns
  pre-extracted *page content* (not just snippets), which gives the
  downstream verifier more context to reason over.
- Relevance ranking is done **locally** with cosine similarity rather
  than relying on Tavily's own ranking, because (a) the Tavily ranking
  optimises for query-document relevance, not claim-evidence alignment,
  and (b) we need a numeric ``relevance_score`` for downstream
  thresholding and explainability.
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Dict, List, Optional

from dotenv import load_dotenv

# Load variables from .env file (if present) into os.environ
# so that os.environ.get("TAVILY_API_KEY") picks up the key
# without the user having to set it manually each session.
load_dotenv()

import numpy as np
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient

# ──────────────────────────────────────────────
# Tavily client setup
# ──────────────────────────────────────────────
# Read the API key from the environment at import
# time.  We intentionally do NOT raise here if the
# key is missing — instead, `search_claim` will
# return an empty list and print a warning.  This
# lets the rest of the module (ranking, tests with
# mocked data) work without a key.
# ──────────────────────────────────────────────
_TAVILY_API_KEY: Optional[str] = os.environ.get("TAVILY_API_KEY")

_tavily_client: Optional[TavilyClient] = None
if _TAVILY_API_KEY:
    _tavily_client = TavilyClient(api_key=_TAVILY_API_KEY)
    print("[evidence_retrieval] Tavily client initialised ✓")
else:
    warnings.warn(
        "TAVILY_API_KEY not found in environment variables. "
        "search_claim() will return empty results. "
        "Set the key via: $env:TAVILY_API_KEY = 'tvly-...'"
    )

# ──────────────────────────────────────────────
# Sentence-transformer model (singleton)
# ──────────────────────────────────────────────
# `paraphrase-multilingual-MiniLM-L12-v2` is a
# lightweight (≈120 MB) multilingual model that
# produces 384-dim embeddings.  It is optimised
# for *paraphrase detection*, which maps well to
# the "does this snippet say the same thing as
# the claim?" task.  Loaded once here so we pay
# the init cost exactly once, even if the module
# is called thousands of times in a batch run.
# ──────────────────────────────────────────────
print("[evidence_retrieval] Loading sentence-transformer model …")
_st_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("[evidence_retrieval] Sentence-transformer ready ✓")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Web search via Tavily
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Wrapped in a try/except so that transient network
# failures, rate limits, or invalid keys never crash
# the full pipeline — the caller simply gets an empty
# evidence list for that claim and a printed warning.
# Tavily's `search()` returns a dict with a "results"
# key; each result has `title`, `url`, `content`.
# We normalise into our own slim dict schema here so
# downstream code is decoupled from the Tavily SDK's
# exact response format.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def search_claim(claim_text: str, max_results: int = 8) -> List[Dict]:
    """Query Tavily for web pages related to *claim_text*.

    Parameters
    ----------
    claim_text : str
        The factual claim to search for.
    max_results : int, optional
        Maximum number of results to request (default 8).

    Returns
    -------
    list[dict]
        Each dict has keys ``content``, ``url``, ``title``.
        Returns an empty list on any failure (missing API key,
        network error, rate-limit, zero results).
    """

    if not _tavily_client:
        print(
            "  ⚠ [search_claim] Tavily client not initialised "
            "(missing TAVILY_API_KEY). Returning empty results."
        )
        return []

    if not claim_text or not claim_text.strip():
        print("  ⚠ [search_claim] Empty claim text — nothing to search.")
        return []

    try:
        response = _tavily_client.search(
            query=claim_text,
            max_results=max_results,
            search_depth="basic",           # "basic" is faster; swap to
                                            # "advanced" for deeper crawls
            include_answer=False,           # we only need raw results
        )
    except Exception as exc:
        # Catches network errors, HTTP 429 rate limits,
        # authentication failures, etc.
        print(
            f"  ⚠ [search_claim] Tavily search failed for claim:\n"
            f"    \"{claim_text[:80]}…\"\n"
            f"    Error: {type(exc).__name__}: {exc}"
        )
        return []

    raw_results = response.get("results", [])

    if not raw_results:
        print(
            f"  ⚠ [search_claim] No results returned for claim:\n"
            f"    \"{claim_text[:80]}…\""
        )
        return []

    # Normalise into a consistent schema.
    snippets = []
    for r in raw_results:
        content = r.get("content", "").strip()
        if not content:
            # Skip results that Tavily returned with no extracted text.
            continue
        snippets.append(
            {
                "title":   r.get("title", ""),
                "url":     r.get("url", ""),
                "content": content,
            }
        )

    return snippets


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Semantic relevance ranking
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rather than feeding all snippets downstream
# (which would waste verifier tokens / compute),
# we keep only the *top_k* most relevant ones.
#
# Why cosine similarity?
#   • It is the standard metric for comparing
#     normalised sentence embeddings.
#   • sentence-transformers embeddings are L2-
#     normalised by default, so cosine similarity
#     reduces to a simple dot product, which is
#     extremely fast (microseconds for k < 100).
#
# Why embed claim + snippets together in one call?
#   • `model.encode([claim, *snippets])` batches
#     everything into a single forward pass, which
#     is far more efficient than encoding them one
#     at a time.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def rank_by_relevance(
    claim_text: str,
    snippets: List[Dict],
    top_k: int = 3,
) -> List[Dict]:
    """Rank *snippets* by semantic similarity to *claim_text*
    and return the *top_k* most relevant ones.

    Parameters
    ----------
    claim_text : str
        The factual claim being verified.
    snippets : list[dict]
        Search results, each with at least a ``content`` key.
    top_k : int, optional
        Number of top results to return (default 3).

    Returns
    -------
    list[dict]
        The *top_k* snippets, sorted by descending
        ``relevance_score`` (cosine similarity, 0–1).
        Each dict is a **copy** of the original snippet
        dict with an added ``relevance_score`` key.
    """

    if not snippets:
        return []

    # Clamp top_k to the number of available snippets.
    top_k = min(top_k, len(snippets))

    # ── Encode claim + all snippet texts in one batch ──
    texts_to_encode = [claim_text] + [s["content"] for s in snippets]
    embeddings = _st_model.encode(
        texts_to_encode,
        batch_size=len(texts_to_encode),
        show_progress_bar=False,
        normalize_embeddings=True,          # L2-normalise → dot = cosine
    )

    claim_emb = embeddings[0]               # shape: (384,)
    snippet_embs = embeddings[1:]           # shape: (n_snippets, 384)

    # ── Cosine similarity (= dot product for unit vectors) ──
    similarities = np.dot(snippet_embs, claim_emb)      # shape: (n_snippets,)

    # ── Select top_k indices, descending ──
    top_indices = np.argsort(similarities)[::-1][:top_k]

    ranked: List[Dict] = []
    for idx in top_indices:
        entry = dict(snippets[idx])                       # shallow copy
        entry["relevance_score"] = round(float(similarities[idx]), 4)
        ranked.append(entry)

    return ranked


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Combined: search → rank (single claim)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# This thin wrapper exists so the rest of the
# pipeline only ever calls ONE function per claim.
# Keeping search and rank as separate functions
# underneath makes each independently testable
# and swappable (e.g. swap Tavily for Bing,
# swap MiniLM for a larger model).
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_evidence_for_claim(
    claim_text: str,
    max_results: int = 8,
    top_k: int = 3,
) -> List[Dict]:
    """Search the web for *claim_text*, then return the *top_k*
    most semantically relevant snippets.

    Parameters
    ----------
    claim_text : str
        The claim to find evidence for.
    max_results : int, optional
        Number of search results to fetch from Tavily (default 8).
    top_k : int, optional
        Number of top-ranked snippets to return (default 3).

    Returns
    -------
    list[dict]
        Up to *top_k* dicts, each with keys ``title``, ``url``,
        ``content``, and ``relevance_score``.  Empty list if the
        search or ranking produces nothing.
    """

    snippets = search_claim(claim_text, max_results=max_results)
    if not snippets:
        return []

    return rank_by_relevance(claim_text, snippets, top_k=top_k)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Batch version for the full pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Takes the *entire* output of extract_and_score_claims
# and augments each claim dict in-place with an
# "evidence" key.
#
# Why not parallelise the Tavily calls?
#   • Tavily rate limits vary by plan.  Sequential
#     calls are safer out of the box.  If you have a
#     paid plan, wrap the loop in
#     `concurrent.futures.ThreadPoolExecutor` for
#     easy parallelism — the function is thread-safe
#     because it only reads from shared state (the
#     sentence-transformer model) and the Tavily
#     client is stateless per-call.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_evidence_for_claims(
    claims_list: List[Dict],
    max_results: int = 8,
    top_k: int = 3,
) -> List[Dict]:
    """Retrieve and rank evidence for every claim in *claims_list*.

    Parameters
    ----------
    claims_list : list[dict]
        Output of ``extract_and_score_claims`` — each dict must
        have at least a ``"text"`` key.
    max_results : int, optional
        Search results to fetch per claim (default 8).
    top_k : int, optional
        Top-ranked snippets to keep per claim (default 3).

    Returns
    -------
    list[dict]
        The same claim dicts, each augmented with an ``"evidence"``
        key containing the ranked snippet list (may be empty if
        the search returned nothing useful).
    """

    if not claims_list:
        warnings.warn(
            "[get_evidence_for_claims] Empty claims list — nothing to retrieve."
        )
        return []

    total = len(claims_list)
    for i, claim in enumerate(claims_list, 1):
        claim_text = claim.get("text", "")
        print(
            f"\n  🔍 [{i}/{total}] Searching evidence for:\n"
            f"     \"{claim_text[:90]}{'…' if len(claim_text) > 90 else ''}\""
        )

        evidence = get_evidence_for_claim(
            claim_text,
            max_results=max_results,
            top_k=top_k,
        )
        claim["evidence"] = evidence

        print(f"     → {len(evidence)} relevant snippet(s) retained")

        # Brief pause between API calls to stay within Tavily's
        # rate limits when processing articles with many claims.
        if i < total:
            time.sleep(0.5)

    return claims_list


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sanity-check
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    import json

    # Simulate the output of extract_and_score_claims
    # with a hand-crafted list of claim dicts.
    TEST_CLAIMS = [
        # ── 1. Well-known, easily verifiable fact ──
        # Expect many high-quality search results and
        # high relevance scores.
        {
            "text": (
                "The Eiffel Tower was completed in 1889 and stands "
                "330 metres tall including its antenna."
            ),
            "p_claim": 0.95,
            "p_ai_generated": 0.12,
            "char_start": 0,
            "char_end": 85,
        },
        # ── 2. Specific / verifiable scientific claim ──
        # Expect decent results from news / scientific
        # sources.
        {
            "text": (
                "According to NASA, the global average surface temperature "
                "in 2024 was 1.29 °C above the pre-industrial baseline."
            ),
            "p_claim": 0.91,
            "p_ai_generated": 0.45,
            "char_start": 86,
            "char_end": 200,
        },
        # ── 3. Obscure / hyper-local claim ──
        # Expect very few or zero results and sparse
        # relevance — good for testing the empty-results
        # path.
        {
            "text": (
                "The village panchayat of Kothaguda in Rangareddy district "
                "approved a Rs 4.2 crore drainage project on 12 August 2025."
            ),
            "p_claim": 0.78,
            "p_ai_generated": 0.30,
            "char_start": 201,
            "char_end": 330,
        },
    ]

    print("=" * 72)
    print(" EVIDENCE RETRIEVAL — SANITY CHECK")
    print("=" * 72)

    enriched_claims = get_evidence_for_claims(
        TEST_CLAIMS,
        max_results=6,
        top_k=3,
    )

    # Pretty-print each claim + its evidence.
    for i, claim in enumerate(enriched_claims, 1):
        print(f"\n{'─' * 72}")
        print(f"CLAIM {i}: {claim['text']}")
        print(f"  p_claim={claim['p_claim']}  p_ai={claim['p_ai_generated']}")
        print(f"  Evidence ({len(claim['evidence'])} snippet(s)):")

        if not claim["evidence"]:
            print("    (none)")
        else:
            for j, ev in enumerate(claim["evidence"], 1):
                print(f"\n    [{j}] relevance={ev['relevance_score']}")
                print(f"        title: {ev['title']}")
                print(f"        url:   {ev['url']}")
                # Truncate long content for readability.
                content_preview = ev["content"][:200]
                if len(ev["content"]) > 200:
                    content_preview += "…"
                print(f"        content: {content_preview}")

    print(f"\n{'═' * 72}")
    print(" Done. Review the relevance scores and snippet quality above.")
    print(f"{'═' * 72}")
