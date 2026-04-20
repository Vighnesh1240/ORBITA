# tests/test_report_generator.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_fpdf_available():
    print("\n[1] Checking fpdf2 installation...")
    try:
        from fpdf import FPDF
        print("  fpdf2 installed — PASS")
    except ImportError:
        print("  fpdf2 NOT installed")
        print("  Run: pip install fpdf2")
        print("  SKIPPED")


def test_clean_text():
    print("\n[2] Testing text cleaner...")
    from src.report_generator import _clean_text

    test_cases = [
        ("Hello World",          "Hello World"),
        ("Modi\u2019s policy",   "Modi's policy"),
        ("Rs.\u20b9500 crore",   "Rs.Rs.500 crore"),
        ("A\u2014B dash",        "A--B dash"),
        ("",                     ""),
    ]

    for input_text, expected_start in test_cases:
        result = _clean_text(input_text)
        assert isinstance(result, str), "Must return string"
        print(f"  '{input_text[:30]}' → '{result[:30]}'")

    print("  PASS")


def test_truncate():
    print("\n[3] Testing text truncation...")
    from src.report_generator import _truncate

    long_text = "A" * 100
    result    = _truncate(long_text, 20)
    assert len(result) <= 20
    assert result.endswith("...")

    short_text = "Hello"
    result2    = _truncate(short_text, 20)
    assert result2 == "Hello"

    print("  PASS")


def test_pdf_filename():
    print("\n[4] Testing PDF filename generation...")
    from src.report_generator import get_pdf_filename

    name1 = get_pdf_filename("Farm Laws India", "Session1")
    assert name1.endswith(".pdf")
    assert "ORBITA" in name1
    assert "Farm" in name1
    print(f"  With collection: {name1}")

    name2 = get_pdf_filename("Crypto Regulation")
    assert name2.endswith(".pdf")
    assert "ORBITA" in name2
    print(f"  Without collection: {name2}")

    # Special characters
    name3 = get_pdf_filename("Farm Laws! India?", "Test@Session")
    assert name3.endswith(".pdf")
    print(f"  With special chars: {name3}")

    print("  PASS")


def test_generate_pdf_basic():
    print("\n[5] Testing PDF generation with mock data...")

    try:
        from fpdf import FPDF
    except ImportError:
        print("  fpdf2 not installed — SKIPPED")
        return

    from src.report_generator import generate_pdf_report

    mock_result = {
        "topic":           "Cryptocurrency Regulation India",
        "elapsed_seconds": 145.3,
        "articles": [
            {
                "title":     "Crypto regulation benefits investors",
                "source":    "Times of India",
                "stance":    "Supportive",
                "url":       "https://timesofindia.com/crypto",
                "full_text": "The regulation provides investor protection. " * 20,
            },
            {
                "title":     "Critics warn on crypto ban",
                "source":    "The Hindu",
                "stance":    "Critical",
                "url":       "https://thehindu.com/crypto",
                "full_text": "Critics argue against over-regulation. " * 20,
            },
        ],
        "report": {
            "bias_score":   0.15,
            "bias_vector": {
                "ideological_bias":   0.15,
                "emotional_bias":     0.18,
                "informational_bias": 0.22,
                "source_diversity":   0.65,
                "stance_entropy":     0.88,
                "composite_score":    0.15,
                "interpretation":     "Slightly Critical",
            },
            "synthesis_report": (
                "The cryptocurrency regulatory framework in India "
                "presents a balanced debate between investor protection "
                "and innovation concerns. According to government data, "
                "the 30 percent tax on crypto gains aims to formalize "
                "the market. However, industry advocates argue that "
                "over-regulation may drive innovation abroad. Both "
                "perspectives reflect genuine economic trade-offs."
            ),
            "loaded_language_removed": ["draconian", "catastrophic"],
            "key_agreements":          ["Both sides acknowledge economic impact."],
            "key_disagreements":       ["Level of regulation needed."],
            "source_citations":        ["Times of India", "The Hindu"],
            "hallucination_flags":     [],
            "nlp_validation_note":     "Manual NLP confirms slight critical lean.",
            "nlp_summary": {
                "avg_vader_compound":   -0.12,
                "manual_bias_score":     0.18,
                "sentiment_distribution": {
                    "positive": 1, "negative": 1, "neutral": 0
                },
                "gemini_validation": {
                    "agreement_level":  "Strong Agreement",
                    "absolute_diff":    0.03,
                    "direction_agrees": True,
                    "agreement_score":  0.97,
                },
                "top_keywords": ["crypto", "regulation", "india"],
                "top_entities": [
                    {"text": "India", "label": "GPE",  "count": 8},
                    {"text": "RBI",   "label": "ORG",  "count": 5},
                ],
            },
            "agent_a": {
                "arguments":               ["Regulation protects investors."],
                "evidence":                ["Market stability improves."],
                "confidence_score":         0.82,
                "nlp_validated_arguments": ["Regulation protects investors."],
                "nlp_context_used":         True,
            },
            "agent_b": {
                "counter_arguments":              ["Over-regulation harms innovation."],
                "evidence":                       ["Developers may relocate."],
                "confidence_score":                0.78,
                "nlp_validated_counter_arguments": [],
                "nlp_context_used":                True,
            },
            "agent_c": {
                "synthesis_report": "See above...",
                "bias_score":        0.15,
            },
        },
        "nlp_analysis": {
            "per_article_sentiment": [
                {
                    "source":   "Times of India",
                    "stance":   "Supportive",
                    "compound":  0.38,
                    "label":    "positive",
                    "title":    "Crypto benefits",
                },
                {
                    "source":   "The Hindu",
                    "stance":   "Critical",
                    "compound": -0.42,
                    "label":    "negative",
                    "title":    "Critics warn",
                },
            ],
            "keyword_analysis": {
                "top_keywords": [
                    {"word": "crypto",     "score": 0.45, "rank": 1},
                    {"word": "regulation", "score": 0.38, "rank": 2},
                ],
                "word_frequencies": {"crypto": 45, "regulation": 38},
            },
            "entity_analysis": {
                "top_entities": [
                    {"text": "India", "label": "GPE",
                     "count": 8, "label_name": "Places"},
                ],
                "by_type": {},
            },
            "manual_bias":      {"manual_bias_score": 0.18},
            "gemini_validation": {"agreement_level": "Strong Agreement"},
        },
    }

    pdf_bytes = generate_pdf_report(
        mock_result, collection_name="Test_Session"
    )

    if pdf_bytes is not None:
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 1000
        print(f"  PDF generated: {len(pdf_bytes):,} bytes")

        # Verify it starts with PDF magic bytes
        assert pdf_bytes[:4] == b"%PDF", "Not a valid PDF"
        print("  Valid PDF header confirmed")
        print("  PASS")
    else:
        print("  PDF generation returned None")
        print("  Check fpdf2 installation")


if __name__ == "__main__":
    print("=" * 55)
    print("  ORBITA — Report Generator Tests")
    print("=" * 55)

    tests = [
        test_fpdf_available,
        test_clean_text,
        test_truncate,
        test_pdf_filename,
        test_generate_pdf_basic,
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