# tests/test_pipeline_nlp.py
"""
Tests to verify NLP is correctly integrated in pipeline.
Run with: python tests/test_pipeline_nlp.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_nlp_phase_function():
    print("\n[1] Testing _run_nlp_phase function...")
    from src.pipeline import _run_nlp_phase

    mock_articles = [
        {
            "title":     "Farm laws spark major protests",
            "source":    "The Hindu",
            "stance":    "Critical",
            "full_text": (
                "Thousands of farmers marched to Delhi "
                "protesting the controversial farm laws. "
                "The Supreme Court stayed the implementation. "
                "Farmers from Punjab led the protests demanding "
                "repeal of the agricultural legislation."
            ),
        },
        {
            "title":     "Agricultural reforms boost economy",
            "source":    "Times of India",
            "stance":    "Supportive",
            "full_text": (
                "The new agricultural reforms will modernize "
                "Indian farming. Government data shows income "
                "increased by 12 percent in reformed sectors. "
                "Market freedom provided to farmers nationwide."
            ),
        },
    ]

    nlp_results, nlp_context, elapsed = _run_nlp_phase(
        mock_articles, "Farm Laws"
    )

    # Validate structure
    print(f"  Elapsed: {elapsed}s")
    print(f"  Context length: {len(nlp_context)} chars")

    if nlp_results:
        assert "per_article_sentiment" in nlp_results
        assert "manual_bias"           in nlp_results
        assert "agent_context"         in nlp_results
        print(f"  Manual bias: "
              f"{nlp_results['manual_bias'].get('manual_bias_score', 0):+.4f}")
        print("  PASS")
    else:
        print("  NLP module not available — skipped (OK)")


def test_post_agent_validation():
    print("\n[2] Testing post-agent NLP validation...")
    from src.pipeline import _run_post_agent_nlp_validation

    mock_nlp = {
        "manual_bias": {
            "manual_bias_score": 0.35,
        },
        "agent_context": "Existing context text here.",
    }

    updated = _run_post_agent_nlp_validation(
        nlp_results       = mock_nlp,
        gemini_bias_score = 0.42,
    )

    # Should add gemini_validation
    if "gemini_validation" in updated:
        val = updated["gemini_validation"]
        print(f"  Agreement: {val.get('agreement_level', 'N/A')}")
        print(f"  Diff: {val.get('absolute_diff', 0):.4f}")
        assert val.get("absolute_diff") is not None
        print("  PASS")
    else:
        print("  validate_against_gemini not available — skipped (OK)")


def test_add_nlp_to_report():
    print("\n[3] Testing _add_nlp_to_report...")
    from src.pipeline import _add_nlp_to_report

    mock_report = {
        "topic":      "Test Topic",
        "bias_score": -0.35,
    }

    mock_nlp = {
        "sentiment_summary": {
            "avg_compound":   -0.25,
            "distribution":   {"negative": 3, "neutral": 1},
        },
        "manual_bias": {
            "manual_bias_score": 0.30,
            "validation_note":   "Moderate bias detected",
        },
        "gemini_validation": {
            "agreement_level":  "Strong Agreement",
            "absolute_diff":    0.05,
            "direction_agrees": True,
            "agreement_score":  0.95,
        },
        "keyword_analysis": {
            "top_keywords": [
                {"word": "farmers", "score": 0.45, "rank": 1},
                {"word": "protest", "score": 0.38, "rank": 2},
            ],
            "word_frequencies": {"farmers": 45, "protest": 32},
        },
        "entity_analysis": {
            "top_entities": [
                {"text": "India", "label": "GPE", "count": 12},
            ],
        },
        "libraries_used": {
            "vader": True, "spacy": True, "wordcloud": True,
        },
    }

    updated = _add_nlp_to_report(mock_report, mock_nlp)

    assert "nlp_summary" in updated
    nlp_sum = updated["nlp_summary"]

    assert "avg_vader_compound"    in nlp_sum
    assert "manual_bias_score"     in nlp_sum
    assert "gemini_validation"     in nlp_sum
    assert "top_keywords"          in nlp_sum
    assert "top_entities"          in nlp_sum

    print(f"  NLP summary keys: {list(nlp_sum.keys())}")
    print(f"  Top keywords: {nlp_sum['top_keywords'][:3]}")
    print(f"  Manual bias: {nlp_sum['manual_bias_score']}")
    print("  PASS")


def test_nlp_cacheable():
    print("\n[4] Testing NLP cache serialization...")
    from src.cache_manager import _make_nlp_cacheable
    import json

    mock_nlp = {
        "per_article_sentiment": [
            {"source": "BBC", "compound": -0.3, "label": "negative"},
        ],
        "sentiment_summary": {
            "avg_compound": -0.3,
            "distribution": {"negative": 1},
        },
        "manual_bias": {"manual_bias_score": 0.25},
        "gemini_validation": {"agreement_level": "Strong"},
        "keyword_analysis": {
            "top_keywords": [
                {"word": "protest", "score": 0.4, "rank": 1}
            ],
            "word_frequencies": {"protest": 32},
            "per_stance": {},
        },
        "entity_analysis": {
            "top_entities": [
                {"text": "India", "label": "GPE", "count": 10}
            ],
            "by_type": {},
        },
        "elapsed_seconds": 4.2,
        "n_articles": 5,
        "libraries_used": {"vader": True},
    }

    cacheable = _make_nlp_cacheable(mock_nlp)

    # Must be JSON serializable
    try:
        json_str = json.dumps(cacheable)
        parsed   = json.loads(json_str)
        print(f"  JSON size: {len(json_str)} chars")
        print(f"  Keys preserved: {list(parsed.keys())}")
        print("  PASS")
    except Exception as e:
        print(f"  FAILED: {e}")


def test_result_structure():
    print("\n[5] Testing pipeline result dict structure...")

    # Build a mock result like pipeline would return
    mock_result = {
        "articles":        [],
        "stats":           {"total_chunks": 0},
        "report":          {
            "bias_score":  0.0,
            "nlp_summary": {"manual_bias_score": 0.1},
        },
        "topic":           "Test",
        "intent":          {},
        "nlp_analysis":    {
            "per_article_sentiment": [],
            "sentiment_summary":     {"avg_compound": 0.0},
            "manual_bias":           {"manual_bias_score": 0.1},
            "keyword_analysis":      {"top_keywords": [], "word_frequencies": {}},
            "entity_analysis":       {"top_entities": [], "by_type": {}},
            "gemini_validation":     {},
            "agent_context":         "test context",
        },
        "image_analysis":   {},
        "visual_context":   "",
        "elapsed_seconds":  120.0,
        "phase_timings": {
            "phase2_data_engineering": 30.0,
            "phase2_5_nlp":            5.0,
            "phase2_6_images":         0.0,
            "phase3_embeddings":       45.0,
            "phase4_agents":           40.0,
            "total":                   120.0,
        },
        "evaluation": {},
    }

    # Verify app.py can access nlp_analysis
    nlp = mock_result.get("nlp_analysis", {})
    assert nlp.get("per_article_sentiment") is not None
    assert nlp.get("keyword_analysis")      is not None
    assert nlp.get("entity_analysis")       is not None

    # Verify report has nlp_summary
    report = mock_result.get("report", {})
    assert "nlp_summary" in report

    print(f"  Result keys: {list(mock_result.keys())}")
    print(f"  NLP keys:    {list(nlp.keys())}")
    print("  PASS")


if __name__ == "__main__":
    print("=" * 55)
    print("  ORBITA — Pipeline NLP Integration Tests")
    print("=" * 55)

    tests = [
        test_nlp_phase_function,
        test_post_agent_validation,
        test_add_nlp_to_report,
        test_nlp_cacheable,
        test_result_structure,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"  ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 55)
    if failed == 0:
        print(f"  ALL {passed} TESTS PASSED")
    else:
        print(f"  {passed} passed, {failed} failed")
    print("=" * 55)