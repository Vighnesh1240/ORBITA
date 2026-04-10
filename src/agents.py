# src/agents.py

import os
import json
from datetime import datetime

try:
    from .agent_a     import run_agent_a
    from .agent_b     import run_agent_b
    from .agent_c     import run_agent_c
    from .config      import REPORTS_DIR
    from .vector_store import get_collection_stats
except ImportError:
    from agent_a     import run_agent_a
    from agent_b     import run_agent_b
    from agent_c     import run_agent_c
    from config      import REPORTS_DIR
    from vector_store import get_collection_stats


def _validate_chromadb() -> None:
    """
    Check ChromaDB has data before running agents.
    Agents are useless without chunks to retrieve from.
    """
    stats = get_collection_stats()
    if stats["total_chunks"] == 0:
        raise RuntimeError(
            "ChromaDB is empty. Run the pipeline (Steps 2+3) first:\n"
            "  python src/pipeline.py"
        )
    print(f"  ChromaDB: {stats['total_chunks']} chunks available")
    print(f"  Stances: {stats['by_stance']}")


def save_report(report: dict, topic: str) -> str:
    """
    Save the full multi-agent report to a JSON file.
    Returns the file path.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    safe_topic = "".join(c if c.isalnum() else "_" for c in topic)[:40]
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename   = f"report_{safe_topic}_{timestamp}.json"
    filepath   = os.path.join(REPORTS_DIR, filename)

    # Remove retrieved_chunks from saved report (too large, not needed)
    clean_report = {k: v for k, v in report.items()
                    if k != "retrieved_chunks"}
    clean_a = {k: v for k, v in report.get("agent_a", {}).items()
               if k != "retrieved_chunks"}
    clean_b = {k: v for k, v in report.get("agent_b", {}).items()
               if k != "retrieved_chunks"}

    save_data = {
        **clean_report,
        "agent_a": clean_a,
        "agent_b": clean_b,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\n[agents] Report saved → {filepath}")
    return filepath


def print_final_report(report: dict) -> None:
    """Print a formatted summary of the final report to the terminal."""
    sep = "=" * 65

    print(f"\n{sep}")
    print("  ORBITA — FINAL ANALYSIS REPORT")
    print(sep)
    print(f"  Topic: {report.get('topic', 'Unknown')}")
    print(sep)

    # Bias score bar
    bias   = report.get("bias_score", 0.0)
    filled = int((bias + 1.0) / 2.0 * 30)
    bar    = "[" + "-" * filled + "|" + "-" * (30 - filled) + "]"
    print(f"\n  Bias Spectrum: {bar}")
    print(f"  Score: {bias:+.2f}  "
          f"({'Leans Supportive' if bias < -0.2 else 'Leans Critical' if bias > 0.2 else 'Balanced'})")

    # Agent A summary
    print(f"\n  AGENT A — Supporting Arguments "
          f"(confidence: {report.get('agent_a', {}).get('confidence_score', 0):.2f})")
    for arg in report.get("agent_a", {}).get("arguments", [])[:3]:
        print(f"    + {arg[:90]}")

    # Agent B summary
    print(f"\n  AGENT B — Counter-Arguments "
          f"(confidence: {report.get('agent_b', {}).get('confidence_score', 0):.2f})")
    for arg in report.get("agent_b", {}).get("counter_arguments", [])[:3]:
        print(f"    - {arg[:90]}")

    # Key agreements / disagreements
    agreements = report.get("key_agreements", [])
    if agreements:
        print(f"\n  KEY AGREEMENTS:")
        for a in agreements[:2]:
            print(f"    ~ {a[:90]}")

    disagreements = report.get("key_disagreements", [])
    if disagreements:
        print(f"\n  KEY DISAGREEMENTS:")
        for d in disagreements[:2]:
            print(f"    x {d[:90]}")

    # Synthesis
    synthesis = report.get("synthesis_report", "")
    if synthesis:
        print(f"\n  UNBIASED SYNTHESIS:")
        print(f"  {'-' * 61}")
        words = synthesis.split()
        lines = []
        current = "  "
        for word in words:
            if len(current) + len(word) + 1 > 65:
                lines.append(current)
                current = "  " + word
            else:
                current += (" " if current != "  " else "") + word
        if current.strip():
            lines.append(current)
        print("\n".join(lines))

    # Loaded language
    removed = report.get("loaded_language_removed", [])
    if removed:
        print(f"\n  LOADED LANGUAGE REMOVED: {len(removed)} phrase(s)")
        for phrase in removed[:3]:
            print(f"    ~ {phrase[:80]}")

    # Hallucination flags
    flags = report.get("hallucination_flags", [])
    if flags:
        print(f"\n  HALLUCINATION FLAGS ({len(flags)} unverified claim(s)):")
        for flag in flags[:2]:
            print(f"    ! {flag[:80]}")
    else:
        print(f"\n  HALLUCINATION CHECK: All claims verified against sources.")

    # Sources
    sources = report.get("source_citations", [])
    if sources:
        print(f"\n  SOURCES CITED: {', '.join(sources[:5])}")

    print(f"\n{sep}\n")


def run_all_agents(topic: str) -> dict:
    """
    Master function. Runs all three agents in sequence.

    Args:
        topic: the topic string from intent decoder

    Returns:
        complete report dict with all agent outputs merged
    """
    print("\n" + "=" * 65)
    print("  ORBITA — Step 4: Multi-Agent RAG Synthesis")
    print("=" * 65)

    # Validate ChromaDB has data
    print("\n[Preflight] Checking ChromaDB...")
    _validate_chromadb()

    # Run Agent A
    agent_a_output = run_agent_a(topic)

    # Run Agent B
    agent_b_output = run_agent_b(topic)

    # Run Agent C (receives A and B outputs)
    agent_c_output = run_agent_c(
        topic          = topic,
        agent_a_output = agent_a_output,
        agent_b_output = agent_b_output,
    )

    # Merge everything into one report
    report = {
        "topic":      topic,
        "bias_score": agent_c_output.get("bias_score", 0.0),
        "synthesis_report":        agent_c_output.get("synthesis_report", ""),
        "loaded_language_removed": agent_c_output.get("loaded_language_removed", []),
        "key_agreements":          agent_c_output.get("key_agreements", []),
        "key_disagreements":       agent_c_output.get("key_disagreements", []),
        "source_citations":        agent_c_output.get("source_citations", []),
        "hallucination_flags":     agent_c_output.get("hallucination_flags", []),
        "agent_a": agent_a_output,
        "agent_b": agent_b_output,
        "agent_c": agent_c_output,
    }

    # Print and save
    print_final_report(report)
    save_report(report, topic)

    return report


if __name__ == "__main__":
    print("ORBITA — Step 4: Multi-Agent RAG")
    print("NOTE: Run pipeline.py first to populate ChromaDB.")
    print("-" * 40)
    topic = input("Enter topic (same as pipeline run): ").strip()
    if not topic:
        topic = "Cryptocurrency regulation India"
        print(f"(Using default: '{topic}')")

    run_all_agents(topic)