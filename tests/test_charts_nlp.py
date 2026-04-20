# tests/test_charts_nlp.py
"""
Tests for the new NLP visualization charts.
Run with: python tests/test_charts_nlp.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_sentiment_bar_chart_basic():
    print("\n[1] Sentiment bar chart — basic...")
    from src.ui.charts import build_sentiment_bar_chart
    import plotly.graph_objects as go

    mock_data = [
        {"source": "BBC News",     "stance": "Neutral",
         "compound": -0.12, "label": "negative",
         "title": "Policy announced"},
        {"source": "The Hindu",    "stance": "Critical",
         "compound": -0.54, "label": "negative",
         "title": "Critics warn"},
        {"source": "Times of India","stance": "Supportive",
         "compound":  0.38, "label": "positive",
         "title": "Benefits highlighted"},
    ]

    fig = build_sentiment_bar_chart(mock_data, "Farm Laws India")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    print(f"  Traces: {len(fig.data)}, Type: {fig.data[0].type}")
    print("  PASS")


def test_sentiment_bar_chart_empty():
    print("\n[2] Sentiment bar chart — empty input...")
    from src.ui.charts import build_sentiment_bar_chart
    import plotly.graph_objects as go

    fig = build_sentiment_bar_chart([])
    assert isinstance(fig, go.Figure)
    print("  Empty input returns placeholder chart — PASS")


def test_word_cloud_plotly():
    print("\n[3] Word cloud — Plotly version...")
    from src.ui.charts import build_word_cloud_chart
    import plotly.graph_objects as go

    mock_frequencies = {
        "farmers":       45,
        "government":    38,
        "protest":       32,
        "legislation":   28,
        "agricultural":  25,
        "india":         40,
        "parliament":    20,
        "reform":        18,
        "market":        15,
        "subsidy":       12,
    }

    fig = build_word_cloud_chart(mock_frequencies, "Farm Laws")
    assert isinstance(fig, go.Figure)
    print(f"  Word traces: {len(fig.data)}")
    print("  PASS")


def test_word_cloud_matplotlib():
    print("\n[4] Word cloud — matplotlib version...")
    from src.ui.charts import build_word_cloud_matplotlib

    mock_frequencies = {
        "farmers": 45, "protest": 32,
        "government": 38, "reform": 18,
    }

    fig = build_word_cloud_matplotlib(mock_frequencies, "Test")
    # May return None if wordcloud not installed
    if fig is not None:
        print(f"  Matplotlib figure generated — PASS")
    else:
        print(f"  wordcloud library not installed — skipped (OK)")


def test_entity_chart_basic():
    print("\n[5] Entity frequency chart — basic...")
    from src.ui.charts import build_entity_frequency_chart
    import plotly.graph_objects as go

    mock_entity_analysis = {
        "top_entities": [
            {"text": "Narendra Modi", "label": "PERSON",
             "count": 12, "label_name": "People"},
            {"text": "Parliament",    "label": "ORG",
             "count": 8,  "label_name": "Organizations"},
            {"text": "India",         "label": "GPE",
             "count": 15, "label_name": "Places"},
        ],
        "by_type": {
            "PERSON": [
                {"text": "Narendra Modi", "count": 12,
                 "label": "PERSON", "label_name": "People"},
                {"text": "Rahul Gandhi",  "count": 7,
                 "label": "PERSON", "label_name": "People"},
            ],
            "ORG": [
                {"text": "Parliament", "count": 8,
                 "label": "ORG", "label_name": "Organizations"},
                {"text": "BJP",        "count": 6,
                 "label": "ORG", "label_name": "Organizations"},
            ],
            "GPE": [
                {"text": "India",  "count": 15,
                 "label": "GPE", "label_name": "Places"},
                {"text": "Punjab", "count": 9,
                 "label": "GPE", "label_name": "Places"},
            ],
        },
    }

    fig = build_entity_frequency_chart(
        mock_entity_analysis, "Farm Laws"
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    print(f"  Traces (entity types): {len(fig.data)}")
    print("  PASS")


def test_entity_chart_empty():
    print("\n[6] Entity chart — empty input...")
    from src.ui.charts import build_entity_frequency_chart
    import plotly.graph_objects as go

    fig = build_entity_frequency_chart({})
    assert isinstance(fig, go.Figure)
    print("  Empty input returns placeholder — PASS")


def test_sentiment_timeline():
    print("\n[7] Sentiment timeline chart...")
    from src.ui.charts import build_sentiment_timeline_chart
    import plotly.graph_objects as go

    mock_data = [
        {"source": "BBC",     "stance": "Neutral",
         "compound": -0.1,  "label": "neutral",  "title": "A"},
        {"source": "Hindu",   "stance": "Critical",
         "compound": -0.5,  "label": "negative", "title": "B"},
        {"source": "TOI",     "stance": "Supportive",
         "compound":  0.4,  "label": "positive", "title": "C"},
        {"source": "NDTV",    "stance": "Critical",
         "compound": -0.3,  "label": "negative", "title": "D"},
    ]

    fig = build_sentiment_timeline_chart(mock_data, "Test Topic")
    assert isinstance(fig, go.Figure)
    print(f"  Stance groups plotted: {len(fig.data)}")
    print("  PASS")


def test_empty_chart():
    print("\n[8] Empty chart placeholder...")
    from src.ui.charts import _empty_chart
    import plotly.graph_objects as go

    fig = _empty_chart("Test message")
    assert isinstance(fig, go.Figure)
    print("  PASS")


if __name__ == "__main__":
    print("=" * 55)
    print("  ORBITA — NLP Chart Tests")
    print("=" * 55)

    tests = [
        test_sentiment_bar_chart_basic,
        test_sentiment_bar_chart_empty,
        test_word_cloud_plotly,
        test_word_cloud_matplotlib,
        test_entity_chart_basic,
        test_entity_chart_empty,
        test_sentiment_timeline,
        test_empty_chart,
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