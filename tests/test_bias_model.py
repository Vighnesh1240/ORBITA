# tests/test_bias_model.py
"""
Unit tests for the multi-dimensional bias model.
Run with: python tests/test_bias_model.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_emotional_bias_low():
    print("\n[1] Testing emotional bias — neutral text...")
    from src.bias_model import compute_emotional_bias

    neutral_text = (
        "The government announced a new policy on April 5. "
        "According to official data, the program will affect "
        "approximately 2 million citizens. The budget allocation "
        "stands at Rs 5,000 crore for fiscal year 2026."
    )
    score = compute_emotional_bias(neutral_text)
    print(f"  Neutral text score: {score}")
    assert score < 0.3, f"Expected low score for neutral text, got {score}"
    print("  PASS")


def test_emotional_bias_high():
    print("\n[2] Testing emotional bias — charged text...")
    from src.bias_model import compute_emotional_bias

    charged_text = (
        "This absolutely outrageous and shocking betrayal is a "
        "catastrophic disaster destroying our future! The corrupt "
        "and disgraceful government has completely failed in this "
        "horrifying crisis. This brutal attack on citizens is "
        "totally unacceptable and deeply alarming."
    )
    score = compute_emotional_bias(charged_text)
    print(f"  Charged text score: {score}")
    assert score > 0.3, f"Expected high score for charged text, got {score}"
    print("  PASS")


def test_informational_bias_factual():
    print("\n[3] Testing informational bias — factual text...")
    from src.bias_model import compute_informational_bias

    factual_text = (
        "According to the 2025 annual report, India's GDP grew by "
        "6.8 percent. The Reserve Bank of India confirmed the data "
        "on March 15, 2026. Research published by NITI Aayog "
        "shows manufacturing increased by 4.2 billion rupees."
    )
    score = compute_informational_bias(factual_text)
    print(f"  Factual text score: {score}")
    assert score < 0.4, f"Expected low opinion score, got {score}"
    print("  PASS")


def test_informational_bias_opinion():
    print("\n[4] Testing informational bias — opinion text...")
    from src.bias_model import compute_informational_bias

    opinion_text = (
        "I think the government should clearly do better. "
        "It seems obvious that this policy is probably wrong. "
        "Everyone knows we must change this immediately. "
        "I believe this is certainly the worst decision ever made."
    )
    score = compute_informational_bias(opinion_text)
    print(f"  Opinion text score: {score}")
    assert score > 0.4, f"Expected high opinion score, got {score}"
    print("  PASS")


def test_source_diversity_similar():
    print("\n[5] Testing source diversity — similar articles...")
    from src.bias_model import compute_source_diversity

    similar_articles = [
        {
            "full_text": "Government policy is beneficial. The new law "
                         "helps farmers. Agriculture sector will improve.",
        },
        {
            "full_text": "New government policy benefits farmers. The "
                         "agricultural law will help the rural sector.",
        },
    ]
    score = compute_source_diversity(similar_articles)
    print(f"  Similar articles diversity: {score}")
    # Similar texts should have lower diversity
    print("  PASS (diversity computed)")


def test_source_diversity_different():
    print("\n[6] Testing source diversity — different articles...")
    from src.bias_model import compute_source_diversity

    different_articles = [
        {
            "full_text": "Cricket match results show India won the series. "
                         "Kohli scored a century in the final match today.",
        },
        {
            "full_text": "Stock market crashed following trade policy changes. "
                         "Technology sector lost 15 percent of valuation.",
        },
        {
            "full_text": "Medical research breakthrough in cancer treatment. "
                         "Scientists discovered new protein blocking tumor growth.",
        },
    ]
    score_diff = compute_source_diversity(different_articles)

    similar_articles = [
        {"full_text": "India wins cricket series. Team performs well."},
        {"full_text": "India cricket team wins the series today."},
        {"full_text": "Cricket India team series win today result."},
    ]
    score_sim = compute_source_diversity(similar_articles)

    print(f"  Different articles diversity: {score_diff}")
    print(f"  Similar articles diversity:  {score_sim}")
    assert score_diff > score_sim, (
        f"Different articles should have higher diversity: "
        f"{score_diff} vs {score_sim}"
    )
    print("  PASS")


def test_stance_entropy_balanced():
    print("\n[7] Testing stance entropy — balanced articles...")
    from src.bias_model import compute_stance_entropy

    balanced = [
        {"stance": "Supportive"},
        {"stance": "Supportive"},
        {"stance": "Critical"},
        {"stance": "Critical"},
        {"stance": "Neutral"},
        {"stance": "Neutral"},
    ]
    score = compute_stance_entropy(balanced)
    print(f"  Balanced stance entropy: {score}")
    assert score > 0.9, f"Expected high entropy for balanced set, got {score}"
    print("  PASS")


def test_stance_entropy_unbalanced():
    print("\n[8] Testing stance entropy — unbalanced articles...")
    from src.bias_model import compute_stance_entropy

    unbalanced = [
        {"stance": "Supportive"},
        {"stance": "Supportive"},
        {"stance": "Supportive"},
        {"stance": "Supportive"},
        {"stance": "Supportive"},
    ]
    score = compute_stance_entropy(unbalanced)
    print(f"  Unbalanced stance entropy: {score}")
    assert score == 0.0, f"Expected zero entropy for all-same stance, got {score}"
    print("  PASS")


def test_full_bias_vector():
    print("\n[9] Testing full bias vector computation...")
    from src.bias_model import compute_bias_vector

    mock_articles = [
        {
            "stance":      "Supportive",
            "source":      "NewsA",
            "full_text":   "Government policy brings great benefits to farmers. "
                           "According to official data released March 2026, "
                           "agricultural income rose by 12 percent.",
            "description": "Policy benefits farmers",
        },
        {
            "stance":      "Critical",
            "source":      "NewsB",
            "full_text":   "Critics warn the policy creates serious risks. "
                           "Research shows 3 million farmers may be affected "
                           "negatively. Opposition demands immediate review.",
            "description": "Critics warn of risks",
        },
        {
            "stance":      "Neutral",
            "source":      "NewsC",
            "full_text":   "The government announced the policy on April 1, 2026. "
                           "Parliamentary committee will review by June 30.",
            "description": "Policy announced",
        },
    ]

    mock_a = {
        "arguments":        ["Policy benefits farmers with income support."],
        "evidence":         ["Agricultural income rose by 12 percent."],
        "confidence_score": 0.80,
    }
    mock_b = {
        "counter_arguments": ["Policy creates risks for small farmers."],
        "evidence":          ["3 million may be negatively affected."],
        "confidence_score":  0.75,
    }
    mock_c = {
        "synthesis_report": (
            "The government policy presents a mixed picture. "
            "According to official data, agricultural income increased by "
            "12 percent in 2026. However, critics note that approximately "
            "3 million small farmers may face challenges under the new framework. "
            "The parliamentary committee will conduct a review by June 2026."
        ),
        "bias_score": -0.1,
    }

    vector = compute_bias_vector(
        articles        = mock_articles,
        agent_a_output  = mock_a,
        agent_b_output  = mock_b,
        agent_c_output  = mock_c,
    )

    print(f"  Ideological bias:   {vector['ideological_bias']:+.4f}")
    print(f"  Emotional bias:     {vector['emotional_bias']:.4f}")
    print(f"  Informational bias: {vector['informational_bias']:.4f}")
    print(f"  Source diversity:   {vector['source_diversity']:.4f}")
    print(f"  Stance entropy:     {vector['stance_entropy']:.4f}")
    print(f"  Composite score:    {vector['composite_score']:+.4f}")
    print(f"  Interpretation:     {vector['interpretation']}")
    print(f"  Dimension labels:   {vector['dimension_labels']}")

    # Validate structure
    required_keys = [
        "ideological_bias", "emotional_bias", "informational_bias",
        "source_diversity", "stance_entropy", "composite_score",
        "confidence", "interpretation", "dimension_labels"
    ]
    for key in required_keys:
        assert key in vector, f"Missing key: {key}"

    # Validate ranges
    assert -1.0 <= vector["ideological_bias"]   <= 1.0
    assert  0.0 <= vector["emotional_bias"]     <= 1.0
    assert  0.0 <= vector["informational_bias"] <= 1.0
    assert  0.0 <= vector["source_diversity"]   <= 1.0
    assert  0.0 <= vector["stance_entropy"]     <= 1.0
    assert -1.0 <= vector["composite_score"]    <= 1.0

    print("  All validations passed.")
    print("  PASS")


if __name__ == "__main__":
    print("=" * 55)
    print("  ORBITA — Bias Model Tests")
    print("=" * 55)

    tests = [
        test_emotional_bias_low,
        test_emotional_bias_high,
        test_informational_bias_factual,
        test_informational_bias_opinion,
        test_source_diversity_similar,
        test_source_diversity_different,
        test_stance_entropy_balanced,
        test_stance_entropy_unbalanced,
        test_full_bias_vector,
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
            print(f"  ERROR: {type(e).__name__}: {e}")
            failed += 1

    print("\n" + "=" * 55)
    if failed == 0:
        print(f"  ALL {passed} TESTS PASSED")
    else:
        print(f"  {passed} passed, {failed} failed")
    print("=" * 55)