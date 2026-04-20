# tests/test_nlp_analyzer.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_vader_positive():
    print("\n[1] VADER — positive text...")
    from src.nlp_analyzer import analyze_sentiment_vader

    text = (
        "The new agricultural policy has brought great benefits "
        "to farmers. Income increased significantly and rural "
        "communities are thriving with new opportunities."
    )
    result = analyze_sentiment_vader(text)
    print(f"  compound={result['compound']:+.4f}, label={result['label']}")
    assert result["label"] == "positive"
    assert result["compound"] > 0
    print("  PASS")


def test_vader_negative():
    print("\n[2] VADER — negative text...")
    from src.nlp_analyzer import analyze_sentiment_vader

    text = (
        "The controversial policy has devastated farming communities. "
        "Thousands of farmers protest against the dangerous new law. "
        "Critics warn of serious harm and economic disaster."
    )
    result = analyze_sentiment_vader(text)
    print(f"  compound={result['compound']:+.4f}, label={result['label']}")
    assert result["label"] == "negative"
    assert result["compound"] < 0
    print("  PASS")


def test_vader_neutral():
    print("\n[3] VADER — neutral text...")
    from src.nlp_analyzer import analyze_sentiment_vader

    text = (
        "The government announced the policy on March 15. "
        "According to official data, the measure affects "
        "approximately 2 million farmers in three states."
    )
    result = analyze_sentiment_vader(text)
    print(f"  compound={result['compound']:+.4f}, label={result['label']}")
    print("  PASS")


def test_spacy_entities():
    print("\n[4] spaCy NER — article text...")
    from src.nlp_analyzer import extract_entities_spacy

    text = (
        "Prime Minister Narendra Modi addressed Parliament in New Delhi "
        "regarding the new farm law. The BJP government passed the "
        "legislation despite protests from farmers in Punjab and Haryana. "
        "The Supreme Court of India stayed the implementation in January 2021."
    )
    result = extract_entities_spacy(text)
    print(f"  Entities found: {len(result['entities'])}")
    print(f"  Top persons: {result['top_persons']}")
    print(f"  Top places: {result['top_places']}")
    print(f"  Top orgs: {result['top_orgs']}")
    assert len(result["entities"]) > 0
    print("  PASS")


def test_tfidf_keywords():
    print("\n[5] TF-IDF keyword extraction...")
    from src.nlp_analyzer import extract_tfidf_keywords

    articles = [
        {
            "full_text": (
                "Farm laws passed by the central government have sparked "
                "major protests. Farmers from Punjab demand repeal of "
                "agricultural legislation. MSP guarantee is key demand."
            ),
            "stance": "Critical",
        },
        {
            "full_text": (
                "Agricultural reforms will modernize Indian farming sector. "
                "Government argues farm laws provide market freedom to farmers. "
                "New legislation allows direct sale bypassing mandis."
            ),
            "stance": "Supportive",
        },
        {
            "full_text": (
                "Parliamentary committee reviews farm law implementation. "
                "Data shows agricultural income trends after legislation. "
                "Economic analysis of farming sector reform impact."
            ),
            "stance": "Neutral",
        },
    ]

    result = extract_tfidf_keywords(articles, top_n=10)
    print(f"  Top keywords: "
          + ", ".join(kw["word"] for kw in result["top_keywords"][:5]))
    assert len(result["top_keywords"]) > 0
    print("  PASS")


def test_manual_bias_score():
    print("\n[6] Manual bias score computation...")
    from src.nlp_analyzer import compute_manual_bias_score

    articles = [
        {"stance": "Critical"},
        {"stance": "Critical"},
        {"stance": "Neutral"},
    ]
    sentiments = [
        {"compound": -0.5, "label": "negative"},
        {"compound": -0.3, "label": "negative"},
        {"compound":  0.0, "label": "neutral"},
    ]

    result = compute_manual_bias_score(articles, sentiments)
    print(f"  Manual bias: {result['manual_bias_score']:+.4f}")
    print(f"  Note: {result['validation_note']}")
    assert result["manual_bias_score"] > 0    # Should be critical
    print("  PASS")


def test_gemini_validation():
    print("\n[7] Gemini vs Manual validation...")
    from src.nlp_analyzer import validate_against_gemini

    # Test agreement
    v1 = validate_against_gemini(0.35, 0.42)
    print(f"  Close scores: {v1['agreement_level']}")
    assert v1["agreement_level"] == "Strong Agreement"

    # Test disagreement
    v2 = validate_against_gemini(-0.6, 0.5)
    print(f"  Far scores:   {v2['agreement_level']}")
    assert v2["agreement_level"] == "Low Agreement"
    assert v2["direction_agrees"] is False

    print("  PASS")


def test_full_nlp_pipeline():
    print("\n[8] Full NLP pipeline...")
    from src.nlp_analyzer import run_nlp_analysis

    mock_articles = [
        {
            "title":     "Farm laws spark protests",
            "source":    "The Hindu",
            "stance":    "Critical",
            "full_text": (
                "Thousands of farmers marched to Delhi protesting "
                "the controversial farm laws. The Supreme Court "
                "stayed the implementation in January 2021. "
                "Farmers from Punjab and Haryana led the protests "
                "demanding repeal of the agricultural legislation. "
                "The BJP government faced strong opposition from "
                "farmer unions across India."
            ),
        },
        {
            "title":     "Farm reforms boost agriculture",
            "source":    "Times of India",
            "stance":    "Supportive",
            "full_text": (
                "The new agricultural reforms will modernize Indian "
                "farming. Government data shows income increased by "
                "12 percent in reformed sectors. The legislation "
                "provides market freedom and removes barriers for "
                "farmers to sell directly to buyers nationwide."
            ),
        },
    ]

    result = run_nlp_analysis(mock_articles, gemini_bias_score=0.2)

    # Validate structure
    assert "per_article_sentiment"   in result
    assert "sentiment_summary"       in result
    assert "entity_analysis"         in result
    assert "keyword_analysis"        in result
    assert "manual_bias"             in result
    assert "agent_context"           in result

    # Check agent context is meaningful
    ctx = result["agent_context"]
    assert len(ctx) > 50
    print(f"  Agent context length: {len(ctx)} chars")
    print(f"  Manual bias: {result['manual_bias']['manual_bias_score']:+.4f}")
    print(f"  Sentiment avg: {result['sentiment_summary']['avg_compound']:+.4f}")
    print("  PASS")


if __name__ == "__main__":
    print("=" * 55)
    print("  ORBITA — NLP Analyzer Tests")
    print("=" * 55)

    tests = [
        test_vader_positive,
        test_vader_negative,
        test_vader_neutral,
        test_spacy_entities,
        test_tfidf_keywords,
        test_manual_bias_score,
        test_gemini_validation,
        test_full_nlp_pipeline,
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