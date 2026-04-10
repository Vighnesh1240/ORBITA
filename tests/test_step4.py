# tests/test_step4.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _seed_chromadb_for_tests():
    """
    Seed ChromaDB with known test data so agent tests
    are not dependent on a previous pipeline run.
    """
    from src.embedder     import embed_chunks
    from src.vector_store import store_chunks

    test_chunks = [
        {
            "chunk_id":    "test_sup_1",
            "text":        "Cryptocurrency regulation brings investor protection "
                           "and market stability benefiting the economy.",
            "url":         "https://example.com/sup1",
            "title":       "Crypto Benefits Report",
            "source":      "FinanceTimes",
            "stance":      "Supportive",
            "chunk_index": 0,
        },
        {
            "chunk_id":    "test_sup_2",
            "text":        "Regulated crypto markets attract institutional investment "
                           "and support national financial growth.",
            "url":         "https://example.com/sup2",
            "title":       "Institutional Crypto",
            "source":      "EconomicPost",
            "stance":      "Supportive",
            "chunk_index": 0,
        },
        {
            "chunk_id":    "test_crit_1",
            "text":        "Strict cryptocurrency bans harm innovation and push "
                           "blockchain developers to other countries.",
            "url":         "https://example.com/crit1",
            "title":       "Crypto Ban Criticism",
            "source":      "TechReview",
            "stance":      "Critical",
            "chunk_index": 0,
        },
        {
            "chunk_id":    "test_crit_2",
            "text":        "Over-regulation of digital assets risks stifling fintech "
                           "startups and reduces India competitiveness globally.",
            "url":         "https://example.com/crit2",
            "title":       "Regulation Risk",
            "source":      "StartupIndia",
            "stance":      "Critical",
            "chunk_index": 0,
        },
        {
            "chunk_id":    "test_neu_1",
            "text":        "The government published a report analysing cryptocurrency "
                           "regulation frameworks from multiple countries.",
            "url":         "https://example.com/neu1",
            "title":       "Govt Crypto Report",
            "source":      "GovtGazette",
            "stance":      "Neutral",
            "chunk_index": 0,
        },
    ]

    embedded = embed_chunks(test_chunks)
    store_chunks(embedded, reset=True)
    print(f"  Seeded ChromaDB with {len(embedded)} test chunks")
    return embedded


def test_chromadb_has_data():
    print("\n[1] Checking ChromaDB has data for agent tests...")
    from src.vector_store import get_collection_stats

    stats = get_collection_stats()
    if stats["total_chunks"] == 0:
        print("  ChromaDB is empty — seeding with test data...")
        _seed_chromadb_for_tests()
        stats = get_collection_stats()

    assert stats["total_chunks"] > 0, "ChromaDB must have chunks"
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  By stance:    {stats['by_stance']}")
    print("  ChromaDB check — OK")


def test_agent_a():
    print("\n[2] Testing Agent A — Analyst...")
    from src.agent_a import run_agent_a

    result = run_agent_a("Cryptocurrency regulation India")

    assert isinstance(result, dict),  "Result must be a dict"
    assert "arguments"       in result, "Must have arguments key"
    assert "evidence"        in result, "Must have evidence key"
    assert "confidence_score" in result, "Must have confidence_score"
    assert isinstance(result["arguments"], list), "Arguments must be a list"

    score = result["confidence_score"]
    assert 0.0 <= score <= 1.0, f"Score must be 0-1, got {score}"

    print(f"  Arguments found:  {len(result['arguments'])}")
    print(f"  Evidence found:   {len(result['evidence'])}")
    print(f"  Confidence:       {score}")
    if result["arguments"]:
        print(f"  First argument:   {result['arguments'][0][:70]}...")
    print("  Agent A — OK")


def test_agent_b():
    print("\n[3] Testing Agent B — Critic...")
    from src.agent_b import run_agent_b

    result = run_agent_b("Cryptocurrency regulation India")

    assert isinstance(result, dict),   "Result must be a dict"
    assert "counter_arguments" in result, "Must have counter_arguments key"
    assert "evidence"          in result, "Must have evidence key"
    assert "confidence_score"  in result, "Must have confidence_score"
    assert isinstance(result["counter_arguments"], list)

    score = result["confidence_score"]
    assert 0.0 <= score <= 1.0, f"Score must be 0-1, got {score}"

    print(f"  Counter-arguments: {len(result['counter_arguments'])}")
    print(f"  Evidence found:    {len(result['evidence'])}")
    print(f"  Confidence:        {score}")
    if result["counter_arguments"]:
        print(f"  First counter-arg: {result['counter_arguments'][0][:70]}...")
    print("  Agent B — OK")


def test_hallucination_check():
    print("\n[4] Testing hallucination check...")
    from src.agent_c import _hallucination_check

    source_chunks = [
        {"text": "Cryptocurrency regulation protects retail investors from fraud."},
        {"text": "Blockchain technology enables decentralised financial transactions."},
    ]

    # This claim is closely related to source — should be supported
    grounded_claims = [
        "Cryptocurrency regulation helps protect retail investors."
    ]
    # This claim has nothing to do with sources — should be flagged
    hallucinated_claims = [
        "The moon is made of cheese and orbits Mars every Tuesday."
    ]

    supported, flagged = _hallucination_check(
        grounded_claims, source_chunks
    )
    print(f"  Grounded claim — supported: {len(supported)}, flagged: {len(flagged)}")

    supported2, flagged2 = _hallucination_check(
        hallucinated_claims, source_chunks
    )
    print(f"  Hallucinated claim — supported: {len(supported2)}, flagged: {len(flagged2)}")
    assert len(flagged2) >= 1, "Hallucinated claim should be flagged"

    print("  Hallucination check — OK")


def test_agent_c():
    print("\n[5] Testing Agent C — Arbitrator...")
    from src.agent_c import run_agent_c

    mock_a = {
        "arguments":        ["Crypto regulation protects investors from fraud."],
        "evidence":         ["Market stability improves with regulation."],
        "key_sources":      ["FinanceTimes"],
        "confidence_score": 0.75,
        "retrieved_chunks": [],
    }
    mock_b = {
        "counter_arguments": ["Regulation stifles blockchain innovation."],
        "evidence":          ["Developers leave over-regulated markets."],
        "key_sources":       ["TechReview"],
        "confidence_score":  0.70,
        "retrieved_chunks":  [],
    }

    result = run_agent_c("Cryptocurrency regulation India", mock_a, mock_b)

    assert isinstance(result, dict),         "Result must be a dict"
    assert "synthesis_report" in result,     "Must have synthesis_report"
    assert "bias_score"       in result,     "Must have bias_score"
    assert "hallucination_flags" in result,  "Must have hallucination_flags"
    assert "source_citations"    in result,  "Must have source_citations"

    bias = result["bias_score"]
    assert -1.0 <= bias <= 1.0, f"Bias score must be -1 to 1, got {bias}"

    synthesis = result.get("synthesis_report", "")
    assert len(synthesis) > 50, "Synthesis should be at least 50 characters"

    print(f"  Bias score:       {bias:+.2f}")
    print(f"  Synthesis length: {len(synthesis.split())} words")
    print(f"  Hallucination flags: {len(result.get('hallucination_flags', []))}")
    print(f"  Synthesis preview:")
    print(f"    {synthesis[:120]}...")
    print("  Agent C — OK")


def test_full_agent_pipeline():
    print("\n[6] Testing full 3-agent pipeline...")
    from src.agents import run_all_agents

    report = run_all_agents("Cryptocurrency regulation India")

    assert "topic"            in report, "Report must have topic"
    assert "bias_score"       in report, "Report must have bias_score"
    assert "synthesis_report" in report, "Report must have synthesis_report"
    assert "agent_a"          in report, "Report must have agent_a"
    assert "agent_b"          in report, "Report must have agent_b"
    assert "agent_c"          in report, "Report must have agent_c"

    bias = report["bias_score"]
    assert -1.0 <= bias <= 1.0

    print(f"\n  Full pipeline report:")
    print(f"  Topic:        {report['topic']}")
    print(f"  Bias score:   {bias:+.2f}")
    print(f"  Arguments(A): {len(report['agent_a'].get('arguments', []))}")
    print(f"  Counter(B):   {len(report['agent_b'].get('counter_arguments', []))}")
    print(f"  Synthesis:    {len(report['synthesis_report'].split())} words")
    print("  Full agent pipeline — OK")


if __name__ == "__main__":
    print("=" * 55)
    print("  ORBITA – Step 4 Verification Tests")
    print("=" * 55)

    tests = [
        test_chromadb_has_data,
        test_agent_a,
        test_agent_b,
        test_hallucination_check,
        test_agent_c,
        test_full_agent_pipeline,
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
        print(f"  ALL {passed} TESTS PASSED — Step 4 verified!")
        print("  Say 'Go to Step 5' when ready.")
    else:
        print(f"  {passed} passed, {failed} failed.")
        print("  Fix the issues above before Step 5.")
    print("=" * 55)