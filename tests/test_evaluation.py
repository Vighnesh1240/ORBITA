# tests/test_evaluation.py
"""
Tests for the ORBITA Evaluation Framework.
Run with: python tests/test_evaluation.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_rouge_unigrams():
    print("\n[1] Testing ROUGE-1...")
    from src.evaluation.rouge_scorer import rouge_n

    hyp = "the cat sat on the mat"
    ref = "the cat sat on the mat"
    r   = rouge_n(hyp, ref, n=1)

    assert r["f1"] == 1.0, f"Perfect match should give F1=1.0, got {r['f1']}"
    print(f"  Perfect match F1: {r['f1']} — PASS")

    hyp2 = "a dog ran in the park"
    r2   = rouge_n(hyp2, ref, n=1)
    print(f"  Partial match F1: {r2['f1']}")
    assert 0.0 < r2["f1"] < 1.0, "Partial match should be between 0 and 1"
    print("  PASS")


def test_rouge_l():
    print("\n[2] Testing ROUGE-L...")
    from src.evaluation.rouge_scorer import rouge_l

    hyp = "india wins cricket series today"
    ref = "india cricket series win today"
    r   = rouge_l(hyp, ref)
    print(f"  Similar text ROUGE-L F1: {r['f1']}")
    assert r["f1"] > 0.5, f"Similar texts should score > 0.5, got {r['f1']}"
    print("  PASS")


def test_empty_inputs():
    print("\n[3] Testing edge cases — empty inputs...")
    from src.evaluation.rouge_scorer import compute_all_rouge

    r = compute_all_rouge("", "some reference text")
    assert r["rouge_1"]["f1"] == 0.0
    assert r["rouge_l"]["f1"] == 0.0
    print("  Empty hypothesis → 0.0 — PASS")


def test_ground_truth_lookup():
    print("\n[4] Testing ground truth lookup...")
    from src.evaluation.ground_truth import get_source_bias, get_topic_ground_truth

    bbc_bias = get_source_bias("BBC News")
    print(f"  BBC News bias: {bbc_bias}")
    assert -0.5 <= bbc_bias <= 0.5, "BBC should be near center"

    republic_bias = get_source_bias("Republic TV")
    print(f"  Republic TV bias: {republic_bias}")
    assert republic_bias > 0.3, "Republic TV should lean right"

    unknown_bias = get_source_bias("Unknown Source XYZ")
    print(f"  Unknown source bias: {unknown_bias}")
    assert unknown_bias == 0.0, "Unknown sources should return 0.0"

    print("  PASS")


def test_topic_ground_truth():
    print("\n[5] Testing topic ground truth...")
    from src.evaluation.ground_truth import get_topic_ground_truth

    gt = get_topic_ground_truth("upi digital payments india")
    print(f"  UPI ground truth: {gt['expected_score']} ({gt['expected_direction']})")
    assert gt["expected_direction"] == "supportive"

    gt2 = get_topic_ground_truth("completely unknown topic xyz abc")
    print(f"  Unknown topic falls back to: {gt2['expected_direction']}")
    assert gt2["match_type"] == "default"

    print("  PASS")


def test_source_expected_bias():
    print("\n[6] Testing source-level expected bias computation...")
    from src.evaluation.ground_truth import compute_expected_bias_from_sources

    articles = [
        {"source": "BBC News",     "stance": "Neutral"},
        {"source": "The Guardian", "stance": "Critical"},
        {"source": "Reuters",      "stance": "Neutral"},
    ]
    result = compute_expected_bias_from_sources(articles)
    print(f"  Expected bias: {result['expected_score']}")
    print(f"  Sources matched: {result['n_matched']}/{result['n_total']}")
    assert result["n_matched"] == 3
    print("  PASS")


def test_full_evaluator():
    print("\n[7] Testing full ORBITAEvaluator...")
    from src.evaluation.evaluator import ORBITAEvaluator

    evaluator = ORBITAEvaluator(output_dir="evaluation_results/test")

    # Build a mock pipeline result
    mock_result = {
        "topic": "cryptocurrency regulation india",
        "articles": [
            {
                "source":      "BBC News",
                "stance":      "Neutral",
                "full_text":   (
                    "India's cryptocurrency policy has been debated for years. "
                    "According to official data from March 2026, the government "
                    "introduced a 30 percent tax on crypto gains in 2022. "
                    "The Reserve Bank has raised concerns about financial stability."
                ),
                "description": "India crypto policy debate",
                "title":       "India crypto regulation overview",
            },
            {
                "source":      "The Hindu",
                "stance":      "Critical",
                "full_text":   (
                    "Critics argue the cryptocurrency tax regime is excessive "
                    "and will drive innovation abroad. Industry experts warn "
                    "that over-regulation could harm India's fintech ecosystem."
                ),
                "description": "Crypto regulation criticism",
                "title":       "Critics warn on crypto tax",
            },
            {
                "source":      "Times of India",
                "stance":      "Supportive",
                "full_text":   (
                    "The government's approach to regulating cryptocurrency "
                    "provides investor protection and market stability. "
                    "The framework aligns with global regulatory standards."
                ),
                "description": "Supportive of crypto regulation",
                "title":       "Crypto regulation benefits investors",
            },
        ],
        "stats": {"total_chunks": 15},
        "report": {
            "bias_score":   -0.10,
            "bias_vector":  {
                "composite_score":    -0.10,
                "ideological_bias":   -0.10,
                "emotional_bias":      0.12,
                "informational_bias":  0.22,
                "source_diversity":    0.65,
                "stance_entropy":      0.95,
            },
            "synthesis_report": (
                "India's cryptocurrency regulatory framework has evolved "
                "through a 30 percent capital gains tax introduced in 2022. "
                "According to government data, this aims to formalize the market. "
                "The Reserve Bank of India has raised financial stability concerns. "
                "Industry advocates argue regulation may harm fintech innovation. "
                "Both regulatory clarity and innovation concerns are valid "
                "considerations in shaping India's crypto policy framework."
            ),
            "loaded_language_removed": ["excessive", "harsh"],
            "hallucination_flags":     [],
            "key_agreements":          ["Both sides acknowledge economic impact."],
            "key_disagreements":       ["Level of regulation needed."],
            "source_citations":        ["BBC News", "The Hindu"],
            "agent_a": {
                "arguments":       ["Regulation provides investor protection."],
                "evidence":        ["30 percent tax formalized the market."],
                "confidence_score": 0.82,
            },
            "agent_b": {
                "counter_arguments": ["Over-regulation drives innovation abroad."],
                "evidence":          ["Industry warns of fintech harm."],
                "confidence_score":   0.78,
            },
            "agent_c": {
                "synthesis_report": "See above...",
                "bias_score":        -0.10,
            },
        },
        "elapsed_seconds": 145.3,
    }

    eval_report = evaluator.evaluate(
        pipeline_result = mock_result,
        topic           = "cryptocurrency regulation india",
        elapsed_seconds = 145.3,
    )

    # Validate structure
    required = [
        "bias_accuracy", "synthesis_quality",
        "coverage_diversity", "argument_quality",
        "performance", "overall_score",
    ]
    for key in required:
        assert key in eval_report, f"Missing: {key}"

    # Validate bias accuracy
    ba = eval_report["bias_accuracy"]
    assert ba.get("predicted_bias") is not None
    print(f"  Predicted: {ba['predicted_bias']:+.4f}, "
          f"MAE: {ba.get('mae', 'N/A')}")

    # Validate synthesis quality
    sq = eval_report["synthesis_quality"]
    assert sq["word_count"] > 0
    print(f"  Synthesis: {sq['word_count']} words, "
          f"covers both: {sq['covers_both_sides']}")

    # Validate overall score
    ov = eval_report["overall_score"]
    assert 0.0 <= ov["score"] <= 1.0
    print(f"  Overall score: {ov['score']:.3f} ({ov['grade']})")

    # Print full summary
    evaluator.print_summary(eval_report)

    print("  PASS")


if __name__ == "__main__":
    print("=" * 55)
    print("  ORBITA — Evaluation Framework Tests")
    print("=" * 55)

    tests = [
        test_rouge_unigrams,
        test_rouge_l,
        test_empty_inputs,
        test_ground_truth_lookup,
        test_topic_ground_truth,
        test_source_expected_bias,
        test_full_evaluator,
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