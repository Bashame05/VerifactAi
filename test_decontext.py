"""
test_decontext.py
─────────────────
Standalone test script for the decontextualization module.

Tests:
1. Claims with pronoun references that need rewriting.
2. An already self-contained claim (should remain unchanged).
3. A claim whose reference cannot be resolved from the article.
4. Behaviour with USE_DECONTEXT=False (no API call, search_text == text).

Run:
    python test_decontext.py
"""

from __future__ import annotations


def main():
    # ──────────────────────────────────────────────
    # Test article with pronoun-heavy sentences
    # ──────────────────────────────────────────────
    ARTICLE = (
        "Dr. Anita Rao led the study at AIIMS Delhi. "
        "She said the drug reduced relapse by 40%. "
        "The trial was later questioned by other researchers."
    )

    # Claims that Stage 1 would extract — simulated with hand-built dicts.
    TEST_CLAIMS = [
        # (*) Pronoun "She" → should be rewritten to "Dr. Anita Rao"
        {
            "text": "She said the drug reduced relapse by 40%.",
            "p_claim": 0.85,
            "p_ai_generated": 0.10,
            "char_start": 47,
            "char_end": 89,
        },
        # (*) Vague "The trial" → should be rewritten referencing AIIMS/Rao
        {
            "text": "The trial was later questioned by other researchers.",
            "p_claim": 0.72,
            "p_ai_generated": 0.05,
            "char_start": 90,
            "char_end": 142,
        },
        # Already self-contained — should come back unchanged
        {
            "text": "The Eiffel Tower is in Paris.",
            "p_claim": 0.95,
            "p_ai_generated": 0.02,
            "char_start": 0,
            "char_end": 28,
        },
        # Deliberately tricky: article has no info to resolve "He"
        {
            "text": "He denied all the allegations during the hearing.",
            "p_claim": 0.80,
            "p_ai_generated": 0.15,
            "char_start": 200,
            "char_end": 249,
        },
    ]

    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # ──────────────────────────────────────────────
    # TEST 1: USE_DECONTEXT = True (live call)
    # ──────────────────────────────────────────────
    import decontext
    print("=" * 72)
    print(" TEST 1: USE_DECONTEXT = True (live Groq call)")
    print("=" * 72)

    # Make sure the flag is on
    decontext.USE_DECONTEXT = True

    # Deep-copy claims so we don't mutate them for the second test
    import copy
    claims_live = copy.deepcopy(TEST_CLAIMS)
    result = decontext.decontextualize_claims(ARTICLE, claims_live)

    for i, c in enumerate(result):
        print(f"\n  Claim {i}:")
        print(f"    Original   : {c['text']}")
        print(f"    search_text: {c['search_text']}")
        print(f"    status     : {c['decontext_status']}")

    # Verify original fields are untouched
    for i, c in enumerate(result):
        assert c["text"] == TEST_CLAIMS[i]["text"], f"Claim {i}: 'text' was modified!"
        assert c["char_start"] == TEST_CLAIMS[i]["char_start"], f"Claim {i}: 'char_start' was modified!"
        assert c["char_end"] == TEST_CLAIMS[i]["char_end"], f"Claim {i}: 'char_end' was modified!"
        assert "search_text" in c, f"Claim {i}: missing 'search_text'!"
        assert "decontext_status" in c, f"Claim {i}: missing 'decontext_status'!"
        assert c["decontext_status"] in {"rewritten", "unchanged", "fallback"}, (
            f"Claim {i}: invalid status '{c['decontext_status']}'"
        )
    print("\n  [OK] All assertions passed for TEST 1.")

    # ──────────────────────────────────────────────
    # TEST 2: USE_DECONTEXT = False (no API call)
    # ──────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(" TEST 2: USE_DECONTEXT = False (no API call)")
    print("=" * 72)

    decontext.USE_DECONTEXT = False
    claims_off = copy.deepcopy(TEST_CLAIMS)
    result_off = decontext.decontextualize_claims(ARTICLE, claims_off)

    for i, c in enumerate(result_off):
        assert c["search_text"] == c["text"], (
            f"Claim {i}: search_text should equal text when flag is off!"
        )
        assert c["decontext_status"] == "unchanged", (
            f"Claim {i}: status should be 'unchanged' when flag is off!"
        )
        print(f"  Claim {i}: search_text == text [OK]  status == 'unchanged' [OK]")

    print("\n  [OK] All assertions passed for TEST 2.")

    # ──────────────────────────────────────────────
    # TEST 3: Empty claims list
    # ──────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(" TEST 3: Empty claims list")
    print("=" * 72)

    decontext.USE_DECONTEXT = True
    result_empty = decontext.decontextualize_claims(ARTICLE, [])
    assert result_empty == [], "Should return empty list for empty input!"
    print("  [OK] Empty claims list handled correctly.")

    # ──────────────────────────────────────────────
    # TEST 4: Check 5 (Invented-Name & Possessive matching)
    # ──────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(" TEST 4: Check 5 validation (possessives & invented names)")
    print("=" * 72)

    article_bbc = "The BBC's chairman gave a statement yesterday."
    original_claim = "He gave a statement yesterday."

    # Positive test: "The BBC" when article has "The BBC's" -> must pass check 5
    rewrite_valid = "The BBC gave a statement yesterday."
    is_valid, reason = decontext._validate_rewrite(original_claim, rewrite_valid, article_bbc)
    assert is_valid, f"Check 5 wrongly rejected possessive rewrite: {reason}"
    print("  [OK] Positive test passed ('The BBC' matched inside 'The BBC\\'s').")

    # Negative test: "Reuters" not anywhere in article -> must be rejected with 'invented_name'
    rewrite_invalid = "The Reuters chairman gave a statement yesterday."
    is_valid_neg, reason_neg = decontext._validate_rewrite(original_claim, rewrite_invalid, article_bbc)
    assert not is_valid_neg, "Check 5 failed to reject invented name!"
    assert reason_neg == "invented_name", f"Expected 'invented_name', got '{reason_neg}'"
    print("  [OK] Negative test passed ('Reuters' correctly rejected as invented_name).")

    print(f"\n{'=' * 72}")
    print(" All tests passed!")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
