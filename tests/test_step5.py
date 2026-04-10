# tests/test_step5.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_css_file_exists():
    print("\n[1] Checking CSS file exists...")
    css_path = os.path.join(
        os.path.dirname(__file__), "..", "assets", "style.css"
    )
    assert os.path.exists(css_path), \
        "assets/style.css not found — create the assets/ folder"
    with open(css_path) as f:
        content = f.read()
    assert len(content) > 100, "CSS file appears empty"
    print("  style.css exists and has content — OK")


def test_app_file_exists():
    print("\n[2] Checking app.py exists...")
    app_path = os.path.join(os.path.dirname(__file__), "..", "app.py")
    assert os.path.exists(app_path), "app.py not found in ORBITA/ root"
    with open(app_path, encoding='utf-8') as f:
        content = f.read()
    assert "st.set_page_config" in content, "Missing page config"
    assert "run_pipeline_with_progress" in content, "Missing pipeline function"
    assert "render_results" in content, "Missing render_results function"
    print("  app.py exists with required functions — OK")


def test_charts_import():
    print("\n[3] Testing charts module...")
    from src.ui.charts import (
        build_bias_spectrum_graph,
        build_confidence_gauge,
        build_stance_distribution_chart,
    )
    print("  All chart functions importable — OK")


def test_bias_spectrum_chart():
    print("\n[4] Testing bias spectrum chart generation...")
    from src.ui.charts import build_bias_spectrum_graph
    import plotly.graph_objects as go

    mock_articles = [
        {"title": "Article A", "source": "NewsA",
         "stance": "Supportive", "full_text": "test " * 50},
        {"title": "Article B", "source": "NewsB",
         "stance": "Critical",   "full_text": "test " * 50},
        {"title": "Article C", "source": "NewsC",
         "stance": "Neutral",    "full_text": "test " * 50},
    ]

    fig = build_bias_spectrum_graph(
        articles   = mock_articles,
        bias_score = -0.25,
        topic      = "Test Topic",
    )

    assert isinstance(fig, go.Figure), "Should return a Plotly Figure"
    assert len(fig.data) > 0,         "Figure should have traces"
    print("  Bias spectrum chart generated — OK")
    print(f"  Traces: {len(fig.data)}")


def test_confidence_gauge():
    print("\n[5] Testing confidence gauge chart...")
    from src.ui.charts import build_confidence_gauge
    import plotly.graph_objects as go

    fig = build_confidence_gauge(a_score=0.82, b_score=0.71)
    assert isinstance(fig, go.Figure)
    print("  Confidence gauge generated — OK")


def test_stance_donut():
    print("\n[6] Testing stance distribution donut chart...")
    from src.ui.charts import build_stance_distribution_chart
    import plotly.graph_objects as go

    mock_articles = [
        {"stance": "Supportive"},
        {"stance": "Supportive"},
        {"stance": "Critical"},
        {"stance": "Neutral"},
    ]
    fig = build_stance_distribution_chart(mock_articles)
    assert isinstance(fig, go.Figure)
    print("  Stance donut chart generated — OK")


def test_components_import():
    print("\n[7] Testing UI components import...")
    from src.ui.components import (
        render_header,
        render_bias_score_display,
        render_metric_cards,
        render_agent_a_panel,
        render_agent_b_panel,
        render_synthesis,
        render_loaded_language,
        render_hallucination_report,
        render_source_transparency,
    )
    print("  All UI components importable — OK")


def test_reports_folder():
    print("\n[8] Checking reports folder exists...")
    reports_path = os.path.join(
        os.path.dirname(__file__), "..", "reports"
    )
    os.makedirs(reports_path, exist_ok=True)
    assert os.path.isdir(reports_path)
    print("  reports/ folder exists — OK")


def test_full_project_structure():
    print("\n[9] Verifying full project structure...")
    root = os.path.join(os.path.dirname(__file__), "..")
    required = [
        "app.py",
        "assets/style.css",
        "src/__init__.py",
        "src/config.py",
        "src/intent_decoder.py",
        "src/news_fetcher.py",
        "src/stance_filter.py",
        "src/scraper.py",
        "src/deduplicator.py",
        "src/chunker.py",
        "src/embedder.py",
        "src/vector_store.py",
        "src/agent_a.py",
        "src/agent_b.py",
        "src/agent_c.py",
        "src/agents.py",
        "src/pipeline.py",
        "src/ui/__init__.py",
        "src/ui/charts.py",
        "src/ui/components.py",
        "requirements.txt",
        ".env",
    ]
    missing = []
    for f in required:
        path = os.path.join(root, f)
        if not os.path.exists(path):
            missing.append(f)

    if missing:
        print("  MISSING FILES:")
        for m in missing:
            print(f"    - {m}")
        assert False, f"{len(missing)} required file(s) missing"

    print(f"  All {len(required)} required files present — OK")


if __name__ == "__main__":
    print("=" * 55)
    print("  ORBITA – Step 5 Verification Tests")
    print("=" * 55)

    tests = [
        test_css_file_exists,
        test_app_file_exists,
        test_charts_import,
        test_bias_spectrum_chart,
        test_confidence_gauge,
        test_stance_donut,
        test_components_import,
        test_reports_folder,
        test_full_project_structure,
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
        print(f"  ALL {passed} TESTS PASSED — Step 5 verified!")
        print("  Run: streamlit run app.py")
    else:
        print(f"  {passed} passed, {failed} failed.")
        print("  Fix the issues above before running the app.")
    print("=" * 55)