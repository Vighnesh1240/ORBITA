# tests/test_cnn_analyzer.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_url_validation():
    print("\n[1] URL validation...")
    from src.cnn_image_analyzer import _is_valid_image_url

    assert _is_valid_image_url("https://bbc.com/img/photo.jpg") is True
    assert _is_valid_image_url("https://doubleclick.net/ad.jpg") is False
    assert _is_valid_image_url("https://site.com/favicon.ico")  is False
    assert _is_valid_image_url("") is False
    print("  PASS")


def test_color_analysis():
    print("\n[2] Color sentiment analysis...")
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        print("  PIL not available — SKIPPED")
        return

    from src.cnn_image_analyzer import _analyze_color_sentiment

    # Create bright image → positive
    bright_img = Image.fromarray(
        (np.ones((100, 100, 3)) * [100, 149, 237]).astype("uint8")
    )
    result = _analyze_color_sentiment(bright_img)
    print(f"  Bright blue: {result['color_sentiment_score']:+.4f}")
    assert "color_sentiment_score" in result
    print("  PASS")


def test_empty_analysis():
    print("\n[3] Empty analysis fallback...")
    from src.cnn_image_analyzer import _empty_analysis

    result = _empty_analysis("https://test.com/img.jpg", "test reason")
    assert result["visual_bias_score"] == 0.0
    assert result["error"] == "test reason"
    print("  PASS")


def test_visual_bias_summary_empty():
    print("\n[4] Empty visual bias summary...")
    from src.cnn_image_analyzer import compute_visual_bias_summary

    summary = compute_visual_bias_summary([])
    assert summary["visual_bias_score"] == 0.0
    assert summary["image_count"] == 0
    print("  PASS")


def test_visual_bias_summary_data():
    print("\n[5] Visual bias summary with mock data...")
    from src.cnn_image_analyzer import compute_visual_bias_summary

    mock = [{
        "source": "BBC",
        "images": [
            {
                "visual_bias_score": 0.4,
                "visual_framing":    "critical",
                "emotional_tone":    "negative",
                "confidence":        0.75,
                "cnn_class":         "negative",
                "description":       "Test image",
                "error":             None,
            }
        ],
    }]

    summary = compute_visual_bias_summary(mock)
    assert summary["image_count"] == 1
    print(f"  Visual bias: {summary['visual_bias_score']:+.4f}")
    print(f"  Dominant:    {summary['dominant_framing']}")
    print("  PASS")


def test_cnn_model_load():
    print("\n[6] CNN model loading...")
    try:
        import torch
    except ImportError:
        print("  PyTorch not available — SKIPPED")
        return

    from src.cnn_image_analyzer import _load_cnn_model
    model = _load_cnn_model()
    assert model is not None
    print(f"  Model loaded successfully")
    print("  PASS")


if __name__ == "__main__":
    print("=" * 55)
    print("  ORBITA — CNN Image Analyzer Tests")
    print("=" * 55)

    tests = [
        test_url_validation,
        test_color_analysis,
        test_empty_analysis,
        test_visual_bias_summary_empty,
        test_visual_bias_summary_data,
        test_cnn_model_load,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print("\n" + "=" * 55)
    if failed == 0:
        print(f"  ALL {passed} TESTS PASSED")
    else:
        print(f"  {passed} passed, {failed} failed")
    print("=" * 55)