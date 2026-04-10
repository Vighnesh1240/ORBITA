# tests/debug_agent_c.py
# Run this to see exactly what Gemini returns and whether it parses correctly.

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent_c import (
    _parse_json_robust,
    _normalise_bias_score,
    _call_gemini_with_retry,
    _build_compact_prompt,
)
from src.embedder     import embed_chunks
from src.vector_store import store_chunks, retrieve_chunks


def seed_and_test(topic: str = "Cryptocurrency regulation India"):
    print("=" * 60)
    print(f"  ORBITA — Agent C Debug Script")
    print(f"  Topic: {topic}")
    print("=" * 60)

    # ── 1. Seed ChromaDB with test data ──────────────────────────
    print("\n[1] Seeding ChromaDB...")
    test_chunks = [
        {
            "chunk_id": "dbg_sup_1", "stance": "Supportive",
            "url": "https://a.com", "source": "FinanceNews",
            "title": "Crypto Benefits", "chunk_index": 0,
            "text": (
                "Cryptocurrency regulation in India will provide legal "
                "clarity and protect retail investors from fraud. "
                "The government's framework will attract institutional "
                "capital and boost financial inclusion across rural areas."
            ),
        },
        {
            "chunk_id": "dbg_crit_1", "stance": "Critical",
            "url": "https://b.com", "source": "TechTimes",
            "title": "Crypto Criticism", "chunk_index": 0,
            "text": (
                "Excessive regulation of cryptocurrencies risks stifling "
                "blockchain innovation in India. Startups may relocate to "
                "more crypto-friendly jurisdictions, causing brain drain "
                "and loss of competitive advantage in the fintech sector."
            ),
        },
        {
            "chunk_id": "dbg_neu_1", "stance": "Neutral",
            "url": "https://c.com", "source": "GovtReport",
            "title": "Policy Overview", "chunk_index": 0,
            "text": (
                "The Reserve Bank of India and SEBI have published reports "
                "examining various cryptocurrency regulatory models from "
                "other countries. The committee is reviewing frameworks "
                "from the EU, US, and Singapore before issuing guidelines."
            ),
        },
    ]
    embedded = embed_chunks(test_chunks)
    store_chunks(embedded, reset=True)
    print(f"  Seeded {len(embedded)} chunks")

    # ── 2. Mock agent outputs ─────────────────────────────────────
    mock_a = {
        "arguments": [
            "Regulation provides legal clarity for crypto exchanges.",
            "Investor protection mechanisms reduce fraud risk.",
        ],
        "evidence": [
            "Government framework attracts institutional capital.",
        ],
        "confidence_score": 0.78,
        "retrieved_chunks": [],
    }
    mock_b = {
        "counter_arguments": [
            "Over-regulation stifles blockchain innovation.",
            "Startups may relocate to friendlier jurisdictions.",
        ],
        "evidence": [
            "Brain drain from over-regulated fintech sector.",
        ],
        "confidence_score": 0.72,
        "retrieved_chunks": [],
    }

    # ── 3. Build prompt and call Gemini ───────────────────────────
    chunks = retrieve_chunks(topic, n_results=5)
    prompt = _build_compact_prompt(topic, mock_a, mock_b, chunks)

    print(f"\n[2] Prompt length: {len(prompt)} chars")
    print(f"\n[3] Calling Gemini...")
    raw = _call_gemini_with_retry(prompt)

    print(f"\n[4] RAW GEMINI RESPONSE:")
    print("-" * 50)
    print(raw)
    print("-" * 50)

    # ── 4. Try parsing ────────────────────────────────────────────
    print(f"\n[5] Attempting JSON parse...")
    parsed = _parse_json_robust(raw)

    if parsed is None:
        print("  FAILED: All parsing strategies failed.")
        print("  The raw response above is what Gemini returned.")
        print("  Please share it so we can diagnose further.")
    else:
        print("  SUCCESS: JSON parsed correctly.")
        bias = _normalise_bias_score(parsed)
        synth = parsed.get("synthesis_report", "")
        print(f"\n  Bias score:   {bias:+.2f}")
        print(f"  Synthesis:    {len(synth.split())} words")
        print(f"\n  Synthesis preview:")
        print(f"  {synth[:300]}...")
        print(f"\n  All keys found: {list(parsed.keys())}")


if __name__ == "__main__":
    topic = input("Enter topic to debug (or press Enter for default): ").strip()
    if not topic:
        topic = "Cryptocurrency regulation India"
    seed_and_test(topic)