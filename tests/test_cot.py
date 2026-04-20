# tests/test_cot.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_basic_cot():
    print("\n[1] Basic CoT creation...")
    from src.chain_of_thought import ORBITACoT, CoTStepType

    cot = ORBITACoT(topic="Farm Laws India")
    assert len(cot.steps) == 0

    cot.add_pipeline_step(
        "Phase 2", "Data fetched", "Fetched 8 articles",
        ["8 articles", "3 stances"],
    )
    assert len(cot.steps) == 1
    assert cot.steps[0].step_type == CoTStepType.PIPELINE.value
    print(f"  Step added: {cot.steps[0].title}")
    print("  PASS")


def test_all_convenience_methods():
    print("\n[2] All convenience step methods...")
    from src.chain_of_thought import ORBITACoT

    cot = ORBITACoT(topic="Test")

    cot.add_retrieval_step(
        "Agent A", "find supporting args", 10,
        "Supportive", ["BBC", "TOI"],
    )
    cot.add_sentiment_step(
        [{"source": "BBC", "compound": -0.3, "label": "negative"}],
        avg_compound=-0.3,
    )
    cot.add_entity_step(
        [{"text": "India", "label_name": "Places", "count": 12}],
        n_total=5,
    )
    cot.add_keyword_step(
        [{"word": "farmers", "score": 0.45}],
    )
    cot.add_argument_step(
        "Agent A", 6, 0.82,
        ["Policy benefits farmers."], nlp_used=True,
    )
    cot.add_validation_step(0.35, 0.42, "Strong Agreement", 0.07)
    cot.add_synthesis_step(320, 2, "VADER confirms critical lean.")
    cot.add_decision_step(-0.38, "Moderately Supportive", "Reason", {})

    assert len(cot.steps) == 8
    print(f"  All 8 step types created")
    print("  PASS")


def test_serialization():
    print("\n[3] JSON serialization...")
    from src.chain_of_thought import ORBITACoT
    import json

    cot = ORBITACoT(topic="Test Topic")
    cot.add_pipeline_step("P2", "Test step", "Detail", ["ev1"])
    cot.add_argument_step("Agent A", 5, 0.8, ["arg1"])

    json_str = cot.to_json()
    parsed   = json.loads(json_str)

    assert "topic"   in parsed
    assert "steps"   in parsed
    assert "summary" in parsed
    assert len(parsed["steps"]) == 2
    print(f"  JSON length: {len(json_str)} chars")
    print("  PASS")


def test_summary():
    print("\n[4] Summary generation...")
    from src.chain_of_thought import ORBITACoT, CoTStepType

    cot = ORBITACoT("Test")
    cot.add_retrieval_step("Agent A", "query", 5)
    cot.add_retrieval_step("Agent B", "query2", 4)
    cot.add_argument_step("Agent A", 3, 0.7, [])
    cot.add_decision_step(0.3, "Critical", "reason", {})

    summary = cot.get_summary()
    assert summary["total_steps"] == 4
    assert "retrieval" in summary["step_breakdown"]
    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Breakdown: {summary['step_breakdown']}")
    print("  PASS")


def test_icon_and_color():
    print("\n[5] Icon and color lookup...")
    from src.chain_of_thought import CoTStep, CoTStepType

    step = CoTStep(
        step_type  = CoTStepType.DECISION.value,
        phase      = "Final",
        title      = "Test",
        detail     = "Test detail",
    )
    assert step.icon  != ""
    assert step.color.startswith("#")
    print(f"  Decision icon: {step.icon}, color: {step.color}")
    print("  PASS")


if __name__ == "__main__":
    print("=" * 55)
    print("  ORBITA — Chain of Thought Tests")
    print("=" * 55)

    tests = [
        test_basic_cot,
        test_all_convenience_methods,
        test_serialization,
        test_summary,
        test_icon_and_color,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 55)
    if failed == 0:
        print(f"  ALL {passed} TESTS PASSED")
    else:
        print(f"  {passed} passed, {failed} failed")
    print("=" * 55)