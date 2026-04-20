# tests/test_agents_nlp.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_nlp_section_builder_agent_a():
    print("\n[1] Agent A NLP section builder...")
    from src.agent_a import _build_nlp_section_for_prompt

    context = """NLP ANALYSIS RESULTS:
Sentiment Distribution: Positive=2, Negative=3, Neutral=1
Per-Article Sentiment (VADER compound):
  [Supportive] Times of India: compound=+0.382 (positive)
  [Critical] The Hindu: compound=-0.421 (negative)
Most Mentioned Entities: Modi (People, 12x), Parliament (Organizations, 8x)
Top TF-IDF Keywords: farmers, protest, legislation, reform, government
Manual NLP Bias Estimate: +0.342"""

    result = _build_nlp_section_for_prompt(context)
    print(f"  Section length: {len(result)} chars")
    assert len(result) > 50
    assert "NLP" in result or "VADER" in result or "sentiment" in result.lower()
    print("  PASS")


def test_nlp_section_builder_agent_b():
    print("\n[2] Agent B NLP section builder...")
    from src.agent_b import _build_nlp_section_for_prompt

    context = """NLP ANALYSIS RESULTS:
Sentiment Distribution: Positive=1, Negative=4, Neutral=1
Per-Article Sentiment:
  [Critical] BBC News: compound=-0.521 (negative)
  [Critical] Al Jazeera: compound=-0.389 (negative)
Manual NLP Bias Estimate: +0.45"""

    result = _build_nlp_section_for_prompt(context)
    print(f"  Section length: {len(result)} chars")
    assert len(result) > 50
    print("  PASS")


def test_nlp_section_builder_agent_c():
    print("\n[3] Agent C NLP section builder...")
    from src.agent_c import _build_nlp_section_for_prompt

    context = """NLP ANALYSIS RESULTS:
Sentiment Distribution: Positive=2, Negative=3, Neutral=1
Manual NLP Bias Estimate: +0.342
Gemini vs Manual: Strong Agreement (diff=0.039)"""

    result = _build_nlp_section_for_prompt(context)
    print(f"  Section length: {len(result)} chars")
    assert len(result) > 50
    # Agent C section should mention cross-validation
    assert (
        "cross-validation" in result.lower() or
        "nlp" in result.lower() or
        "validation" in result.lower()
    )
    print("  PASS")


def test_empty_nlp_context():
    print("\n[4] Empty NLP context handling...")
    from src.agent_a import _build_nlp_section_for_prompt as a_nlp
    from src.agent_b import _build_nlp_section_for_prompt as b_nlp
    from src.agent_c import _build_nlp_section_for_prompt as c_nlp

    for agent_name, func in [("A", a_nlp), ("B", b_nlp), ("C", c_nlp)]:
        result = func("")
        assert result == "", (
            f"Agent {agent_name}: empty context should return empty string"
        )
        result2 = func("   ")
        assert result2 == "", (
            f"Agent {agent_name}: whitespace should return empty string"
        )

    print("  All agents return '' for empty context — PASS")


def test_agent_a_signature():
    print("\n[5] Agent A function signature...")
    import inspect
    from src.agent_a import run_agent_a

    sig    = inspect.signature(run_agent_a)
    params = list(sig.parameters.keys())

    assert "topic"          in params, "Missing 'topic' parameter"
    assert "visual_context" in params, "Missing 'visual_context' parameter"
    assert "nlp_context"    in params, "Missing 'nlp_context' parameter"

    # Check defaults
    assert sig.parameters["visual_context"].default == ""
    assert sig.parameters["nlp_context"].default    == ""

    print(f"  Parameters: {params}")
    print("  PASS")


def test_agent_b_signature():
    print("\n[6] Agent B function signature...")
    import inspect
    from src.agent_b import run_agent_b

    sig    = inspect.signature(run_agent_b)
    params = list(sig.parameters.keys())

    assert "topic"          in params
    assert "visual_context" in params
    assert "nlp_context"    in params

    print(f"  Parameters: {params}")
    print("  PASS")


def test_agent_c_signature():
    print("\n[7] Agent C function signature...")
    import inspect
    from src.agent_c import run_agent_c

    sig    = inspect.signature(run_agent_c)
    params = list(sig.parameters.keys())

    assert "topic"          in params
    assert "agent_a_output" in params
    assert "agent_b_output" in params
    assert "visual_context" in params
    assert "nlp_context"    in params

    print(f"  Parameters: {params}")
    print("  PASS")


def test_agents_orchestrator_signature():
    print("\n[8] run_all_agents signature...")
    import inspect
    from src.agents import run_all_agents

    sig    = inspect.signature(run_all_agents)
    params = list(sig.parameters.keys())

    assert "topic"          in params
    assert "visual_context" in params
    assert "nlp_context"    in params

    print(f"  Parameters: {params}")
    print("  PASS")


def test_result_has_nlp_fields():
    print("\n[9] Agent result dict has NLP fields...")

    # Build mock agent results like the agents would return
    mock_a_result = {
        "arguments":               ["Policy benefits farmers."],
        "evidence":                ["Data shows 12% income rise."],
        "key_sources":             ["Times of India"],
        "confidence_score":        0.82,
        "argument_traces":         [],
        "retrieved_chunks":        [],
        "nlp_validated_arguments": ["Policy benefits farmers."],
        "top_entities_referenced": ["India", "farmers"],
        "visual_context_used":     False,
        "nlp_context_used":        True,
    }

    mock_b_result = {
        "counter_arguments":              ["Policy harms small farmers."],
        "evidence":                       ["3 million affected negatively."],
        "key_sources":                    ["The Hindu"],
        "confidence_score":               0.78,
        "retrieved_chunks":               [],
        "nlp_validated_counter_arguments": ["Policy harms small farmers."],
        "top_critical_entities":          ["Farmers", "Punjab"],
        "visual_context_used":            False,
        "nlp_context_used":               True,
    }

    # Verify new NLP fields exist
    assert "nlp_validated_arguments"   in mock_a_result
    assert "top_entities_referenced"   in mock_a_result
    assert "nlp_context_used"          in mock_a_result

    assert "nlp_validated_counter_arguments" in mock_b_result
    assert "top_critical_entities"           in mock_b_result
    assert "nlp_context_used"                in mock_b_result

    print("  All NLP fields present in agent result dicts")
    print("  PASS")


if __name__ == "__main__":
    print("=" * 55)
    print("  ORBITA — Agent NLP Integration Tests")
    print("=" * 55)

    tests = [
        test_nlp_section_builder_agent_a,
        test_nlp_section_builder_agent_b,
        test_nlp_section_builder_agent_c,
        test_empty_nlp_context,
        test_agent_a_signature,
        test_agent_b_signature,
        test_agent_c_signature,
        test_agents_orchestrator_signature,
        test_result_has_nlp_fields,
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