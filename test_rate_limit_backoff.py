"""
test_rate_limit_backoff.py
──────────────────────────
Standalone verification script for Groq API rate-limit backoff behavior
in decontext.py and verdict_scoring.py.

Simulates HTTP 429 / RateLimitError responses via unittest.mock to confirm:
1. Rate-limit-specific detection path fires (and not generic error path).
2. Diagnostic backoff message prints ("Rate limit hit. Waiting Xs before retry...").
3. Sleep durations match expected backoff pattern (retry-after or 15s/30s/60s).
4. After retries exhaust, safe fallbacks execute properly without crashing.

Run manually:
    python test_rate_limit_backoff.py
"""

from __future__ import annotations

import contextlib
import io
import sys
from typing import Any, Dict, List
import unittest.mock as mock

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import httpx
from groq import RateLimitError

import decontext
import verdict_scoring


def test_case_1_decontext_sdk_error() -> Dict[str, Any]:
    """Case 1: decontext.py with SDK RateLimitError (no retry-after, exponential backoff)."""
    print("\n" + "─" * 72)
    print("TEST CASE 1: decontext.py — SDK RateLimitError (exponential backoff)")
    print("─" * 72)

    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp = httpx.Response(429, headers={}, request=req)
    err = RateLimitError("Rate limit reached: requests per minute exceeded", response=resp, body=None)

    mock_client = mock.MagicMock()
    mock_client.chat.completions.create.side_effect = err

    sleep_durations: List[float] = []

    def fake_sleep(duration: float) -> None:
        sleep_durations.append(duration)

    captured_out = io.StringIO()
    article = "Dr. Anita Rao led the trial. She said the drug reduced relapse by 40%."
    claims = [{
        "text": "She said the drug reduced relapse by 40%.",
        "p_claim": 0.85,
        "p_ai_generated": 0.10,
        "char_start": 29,
        "char_end": 71,
    }]

    with contextlib.redirect_stdout(captured_out):
        with mock.patch.object(decontext, "_groq_client", mock_client):
            with mock.patch("time.sleep", side_effect=fake_sleep):
                result = decontext.decontextualize_claims(article, claims)

    logs = captured_out.getvalue()
    print("Captured logs:")
    for line in logs.strip().splitlines():
        print(f"  | {line}")

    # Assertions
    assert "Rate limit hit" in logs, "Expected 'Rate limit hit' in logs"
    assert "Groq API error:" not in logs, "Generic error path should not have triggered"
    assert sleep_durations == [15.0, 30.0], f"Expected sleeps [15.0, 30.0], got {sleep_durations}"
    assert len(result) == 1
    assert result[0]["decontext_status"] == "fallback", f"Expected fallback status, got {result[0]['decontext_status']}"
    assert result[0]["search_text"] == "She said the drug reduced relapse by 40%."

    print(f"  ✓ Recorded sleep durations: {sleep_durations}")
    print(f"  ✓ Final status: {result[0]['decontext_status']}, search_text preserved.")
    return {"name": "decontext.py (SDK RateLimitError)", "passed": True, "delays": sleep_durations}


def test_case_2_decontext_fallback_detection() -> Dict[str, Any]:
    """Case 2: decontext.py with generic exception containing '429' (fallback detection path)."""
    print("\n" + "─" * 72)
    print("TEST CASE 2: decontext.py — Fallback string detection ('429')")
    print("─" * 72)

    class CustomOlderSDKError(Exception):
        pass

    fallback_err = CustomOlderSDKError("Error response from server: HTTP 429 Too Many Requests rate limit exceeded")

    mock_client = mock.MagicMock()
    mock_client.chat.completions.create.side_effect = fallback_err

    sleep_durations: List[float] = []

    def fake_sleep(duration: float) -> None:
        sleep_durations.append(duration)

    captured_out = io.StringIO()
    article = "The company announced quarterly results. It reported record profits."
    claims = [{
        "text": "It reported record profits.",
        "p_claim": 0.90,
        "p_ai_generated": 0.05,
        "char_start": 41,
        "char_end": 68,
    }]

    with contextlib.redirect_stdout(captured_out):
        with mock.patch.object(decontext, "_groq_client", mock_client):
            with mock.patch("time.sleep", side_effect=fake_sleep):
                result = decontext.decontextualize_claims(article, claims)

    logs = captured_out.getvalue()
    print("Captured logs:")
    for line in logs.strip().splitlines():
        print(f"  | {line}")

    # Assertions
    assert "Rate limit hit" in logs, "Expected 'Rate limit hit' in logs via fallback detection"
    assert "Groq API error:" not in logs, "Generic error path should not have triggered"
    assert sleep_durations == [15.0, 30.0], f"Expected sleeps [15.0, 30.0], got {sleep_durations}"
    assert result[0]["decontext_status"] == "fallback"
    assert result[0]["search_text"] == "It reported record profits."

    print(f"  ✓ Recorded sleep durations: {sleep_durations}")
    print(f"  ✓ Fallback detection fired correctly, search_text preserved.")
    return {"name": "decontext.py (Fallback '429' string)", "passed": True, "delays": sleep_durations}


def test_case_3_verdict_scoring_retry_after() -> Dict[str, Any]:
    """Case 3: verdict_scoring.py with RateLimitError carrying retry-after=5 header."""
    print("\n" + "─" * 72)
    print("TEST CASE 3: verdict_scoring.py — RateLimitError with retry-after=5 header")
    print("─" * 72)

    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp = httpx.Response(429, headers={"retry-after": "5"}, request=req)
    err = RateLimitError("Rate limit reached: retry after 5 seconds", response=resp, body=None)

    mock_client = mock.MagicMock()
    mock_client.chat.completions.create.side_effect = err

    sleep_durations: List[float] = []

    def fake_sleep(duration: float) -> None:
        sleep_durations.append(duration)

    captured_out = io.StringIO()
    claim_dict = {
        "text": "The Eiffel Tower is in Paris.",
        "p_claim": 0.95,
        "p_ai_generated": 0.02,
        "evidence": [{
            "title": "Paris Landmarks",
            "url": "https://example.com/paris",
            "content": "The Eiffel Tower stands in Paris, France.",
            "relevance_score": 0.98,
        }],
    }

    with contextlib.redirect_stdout(captured_out):
        with mock.patch.object(verdict_scoring, "_groq_client", mock_client):
            with mock.patch("time.sleep", side_effect=fake_sleep):
                result = verdict_scoring.score_claim(claim_dict)

    logs = captured_out.getvalue()
    print("Captured logs:")
    for line in logs.strip().splitlines():
        print(f"  | {line}")

    # Assertions
    # retry-after=5 + 1.0s buffer = 6.0s
    expected_delays = [6.0, 6.0]
    assert "Rate limit hit" in logs, "Expected 'Rate limit hit' in logs"
    assert "Groq API error:" not in logs, "Generic error path should not have triggered"
    assert sleep_durations == expected_delays, f"Expected {expected_delays}, got {sleep_durations}"
    assert result["verdict"] == "insufficient_evidence", f"Expected fallback verdict, got {result['verdict']}"
    assert result["contradiction_score"] == 0.5, f"Expected neutral contradiction score 0.5, got {result['contradiction_score']}"

    print(f"  ✓ Recorded sleep durations: {sleep_durations} (honored retry-after 5s + 1s buffer)")
    print(f"  ✓ Fallback verdict assigned: {result['verdict']} (contradiction={result['contradiction_score']})")
    return {"name": "verdict_scoring.py (retry-after=5)", "passed": True, "delays": sleep_durations}


def test_case_4_verdict_scoring_exponential() -> Dict[str, Any]:
    """Case 4: verdict_scoring.py with RateLimitError (no retry-after, exponential backoff)."""
    print("\n" + "─" * 72)
    print("TEST CASE 4: verdict_scoring.py — RateLimitError (exponential backoff)")
    print("─" * 72)

    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp = httpx.Response(429, headers={}, request=req)
    err = RateLimitError("Rate limit reached", response=resp, body=None)

    mock_client = mock.MagicMock()
    mock_client.chat.completions.create.side_effect = err

    sleep_durations: List[float] = []

    def fake_sleep(duration: float) -> None:
        sleep_durations.append(duration)

    captured_out = io.StringIO()
    claim_dict = {
        "text": "Water boils at 100 degrees Celsius at sea level.",
        "p_claim": 0.99,
        "p_ai_generated": 0.01,
        "evidence": [{
            "title": "Water Properties",
            "url": "https://example.com/water",
            "content": "At standard atmospheric pressure, water boils at 100 °C.",
            "relevance_score": 0.95,
        }],
    }

    with contextlib.redirect_stdout(captured_out):
        with mock.patch.object(verdict_scoring, "_groq_client", mock_client):
            with mock.patch("time.sleep", side_effect=fake_sleep):
                result = verdict_scoring.score_claim(claim_dict)

    logs = captured_out.getvalue()
    print("Captured logs:")
    for line in logs.strip().splitlines():
        print(f"  | {line}")

    # Assertions
    expected_delays = [15.0, 30.0]
    assert "Rate limit hit" in logs, "Expected 'Rate limit hit' in logs"
    assert "Groq API error:" not in logs, "Generic error path should not have triggered"
    assert sleep_durations == expected_delays, f"Expected {expected_delays}, got {sleep_durations}"
    assert result["verdict"] == "insufficient_evidence", f"Expected fallback verdict, got {result['verdict']}"

    print(f"  ✓ Recorded sleep durations: {sleep_durations} (15s, 30s exponential backoff)")
    print(f"  ✓ Fallback verdict assigned: {result['verdict']}")
    return {"name": "verdict_scoring.py (exponential backoff)", "passed": True, "delays": sleep_durations}


def main() -> int:
    print("=" * 72)
    print(" RATE-LIMIT BACKOFF VERIFICATION (STANDALONE TEST SUITE)")
    print("=" * 72)

    tests = [
        test_case_1_decontext_sdk_error,
        test_case_2_decontext_fallback_detection,
        test_case_3_verdict_scoring_retry_after,
        test_case_4_verdict_scoring_exponential,
    ]

    results: List[Dict[str, Any]] = []
    all_passed = True

    for test_fn in tests:
        try:
            res = test_fn()
            results.append(res)
        except Exception as exc:
            print(f"\n  ❌ Test failed with exception: {type(exc).__name__}: {exc}")
            results.append({"name": test_fn.__name__, "passed": False, "delays": []})
            all_passed = False

    print("\n" + "=" * 72)
    print(" FINAL TEST SUMMARY")
    print("=" * 72)
    for idx, r in enumerate(results, 1):
        status = "PASS" if r["passed"] else "FAIL"
        delays_str = f"Delays: {r['delays']}" if r["delays"] else "No delays recorded"
        print(f"  Case {idx}: {r['name']:<46} [{status}] ({delays_str})")
    print("=" * 72)

    if all_passed:
        print(" [OK] All 4 rate-limit backoff test cases passed successfully!")
        print("=" * 72 + "\n")
        return 0
    else:
        print(" [FAIL] One or more rate-limit backoff tests failed.")
        print("=" * 72 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
