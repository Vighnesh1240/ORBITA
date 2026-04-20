# tests/test_image_analyzer.py

import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_url_validation():
    print("\n[1] Testing URL validation...")
    from src.image_analyzer import _is_valid_image_url

    assert _is_valid_image_url(
        "https://bbc.com/images/article-photo.jpg"
    ) is True

    assert _is_valid_image_url(
        "https://doubleclick.net/pixel.jpg"
    ) is False

    assert _is_valid_image_url("") is False
    assert _is_valid_image_url(
        "https://site.com/favicon.ico"
    ) is False

    print("  PASS")


def test_parse_vision_response():
    print("\n[2] Testing vision response parsing...")
    from src.image_analyzer import _parse_vision_response

    valid = json.dumps({
        "emotional_tone":        "negative",
        "visual_framing":        "victim",
        "subjects_detected":     ["farmers", "police"],
        "narrative_suggested":   "Farmers facing oppression",
        "image_text_consistent": True,
        "loaded_visual_elements": ["crowd size emphasised"],
        "visual_bias_score":     -0.3,
        "visual_bias_direction": "supportive",
        "confidence":            0.75,
        "description":           "Large crowd of protesters at Delhi border",
    })

    result = _parse_vision_response(valid, "https://test.com/img.jpg")
    assert result["emotional_tone"]    == "negative"
    assert result["visual_bias_score"] == -0.3
    assert result["confidence"]        == 0.75
    print(f"  Parsed: {result['description'][:50]}")
    print("  PASS")


def test_visual_bias_summary_empty():
    print("\n[3] Testing empty visual bias summary...")
    from src.image_analyzer import compute_visual_bias_summary

    summary = compute_visual_bias_summary([])
    assert summary["visual_bias_score"]  == 0.0
    assert summary["image_count"]        == 0
    assert summary["dominant_framing"]   == "neutral"
    print("  PASS")


def test_visual_bias_summary_with_data():
    print("\n[4] Testing visual bias summary with mock data...")
    from src.image_analyzer import compute_visual_bias_summary

    mock_analyses = [
        {
            "source":         "BBC News",
            "article_url":    "https://bbc.com/test",
            "analyzed_count": 2,
            "images": [
                {
                    "visual_bias_score":      0.4,
                    "emotional_tone":         "negative",
                    "visual_framing":         "victim",
                    "confidence":             0.8,
                    "loaded_visual_elements": ["distressed faces"],
                    "description":            "Protesters in distress",
                    "error":                  None,
                },
                {
                    "visual_bias_score":      0.3,
                    "emotional_tone":         "negative",
                    "visual_framing":         "threat",
                    "confidence":             0.7,
                    "loaded_visual_elements": [],
                    "description":            "Police with batons",
                    "error":                  None,
                },
            ],
        },
    ]

    summary = compute_visual_bias_summary(mock_analyses)
    assert summary["image_count"]           == 1
    assert summary["visual_bias_score"]     > 0    # Critical framing
    assert summary["dominant_tone"]         == "negative"
    assert summary["loaded_elements_count"] == 1
    print(f"  Visual bias score: {summary['visual_bias_score']:+.4f}")
    print(f"  Dominant framing:  {summary['dominant_framing']}")
    print("  PASS")


def test_url_hash():
    print("\n[5] Testing URL hashing for cache...")
    from src.image_analyzer import _get_url_hash

    hash1 = _get_url_hash("https://test.com/image1.jpg")
    hash2 = _get_url_hash("https://test.com/image1.jpg")
    hash3 = _get_url_hash("https://test.com/image2.jpg")

    assert hash1 == hash2,  "Same URL should give same hash"
    assert hash1 != hash3,  "Different URLs should give different hashes"
    assert len(hash1) == 12, "Hash should be 12 chars"
    print(f"  Hash: {hash1}")
    print("  PASS")


if __name__ == "__main__":
    import json

    print("=" * 55)
    print("  ORBITA — Image Analyzer Tests")
    print("=" * 55)

    tests = [
        test_url_validation,
        test_parse_vision_response,
        test_visual_bias_summary_empty,
        test_visual_bias_summary_with_data,
        test_url_hash,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print("\n" + "=" * 55)
    if failed == 0:
        print(f"  ALL {passed} TESTS PASSED")
    else:
        print(f"  {passed} passed, {failed} failed")
    print("=" * 55)