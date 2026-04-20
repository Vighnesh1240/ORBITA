# demo_cache/create_demo_cache.py
"""
ORBITA Demo Cache Creator
=========================
Run this script ONCE before your viva to pre-generate
all demo results. Takes about 15-25 minutes.

Usage:
    cd ORBITA
    python demo_cache/create_demo_cache.py

Or run single topic:
    python demo_cache/create_demo_cache.py --topic "AI Regulation India"

After running:
    All demo JSON files will be in demo_cache/
    Demo Mode in the app will show them instantly
"""

import sys
import os
import argparse
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.demo_manager import DemoManager, DEMO_TOPICS
from src.pipeline     import run_pipeline


def create_single(
    topic_name: str,
    dm: DemoManager,
    force: bool = False,
) -> bool:
    """
    Run pipeline for one topic and save to cache.

    Args:
        topic_name: display name from DEMO_TOPICS
        dm:         DemoManager instance
        force:      overwrite if exists

    Returns:
        True if successful
    """
    if not force and dm.is_available(topic_name):
        print(f"\n[SKIP] Already cached: {topic_name}")
        print("       Use --force to regenerate")
        return True

    config = DEMO_TOPICS.get(topic_name)
    if not config:
        print(f"\n[ERROR] Unknown topic: {topic_name}")
        return False

    print(f"\n{'='*60}")
    print(f"  Generating cache: {topic_name}")
    print(f"  {config.get('description', '')}")
    print(f"{'='*60}")

    start = time.time()

    try:
        # Use the topic name as the search query
        result = run_pipeline(
            user_input      = topic_name,
            run_evaluation  = False,  # skip for speed
            run_nlp         = True,
            run_images      = True,
        )

        elapsed = round(time.time() - start, 1)
        print(f"\n  Pipeline completed in {elapsed}s")

        # Save to demo cache
        saved = dm.save(topic_name, result)

        if saved:
            bias = result.get("report", {}).get("bias_score", 0)
            arts = len(result.get("articles", []))
            print(f"  ✓ Saved | bias: {bias:+.3f} | articles: {arts}")
        else:
            print(f"  ✗ Save failed")

        return saved

    except Exception as e:
        print(f"\n  ✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_all(force: bool = False) -> None:
    """Create cache for all DEMO_TOPICS."""
    dm = DemoManager()

    print("\n" + "="*60)
    print("  ORBITA Demo Cache Generator")
    print("="*60)
    print(f"  Topics to cache: {len(DEMO_TOPICS)}")
    print(f"  Force overwrite: {force}")
    print(f"  Output:          demo_cache/")
    print("="*60)

    total_start = time.time()
    results = {"success": [], "failed": [], "skipped": []}

    for i, topic_name in enumerate(DEMO_TOPICS.keys(), 1):
        print(f"\n[{i}/{len(DEMO_TOPICS)}] {topic_name}")

        if not force and dm.is_available(topic_name):
            print("  Already cached — skipping")
            results["skipped"].append(topic_name)
            continue

        ok = create_single(topic_name, dm, force=force)

        if ok:
            results["success"].append(topic_name)
        else:
            results["failed"].append(topic_name)

        # Small delay between topics — respect API rate limits
        if i < len(DEMO_TOPICS):
            print(f"\n  Waiting 5s before next topic...")
            time.sleep(5)

    total_elapsed = round(time.time() - total_start, 0)

    print(f"\n{'='*60}")
    print("  Demo Cache Generation Complete")
    print(f"{'='*60}")
    print(f"  Total time:  {total_elapsed}s")
    print(f"  Successful:  {len(results['success'])}")
    print(f"  Failed:      {len(results['failed'])}")
    print(f"  Skipped:     {len(results['skipped'])}")

    if results["success"]:
        print(f"\n  ✓ Cached:")
        for t in results["success"]:
            print(f"    - {t}")

    if results["failed"]:
        print(f"\n  ✗ Failed:")
        for t in results["failed"]:
            print(f"    - {t}")

    # Show final stats
    stats = dm.get_stats()
    total_kb = stats.get("total_size_kb", 0)
    print(
        f"\n  Demo cache ready: "
        f"{stats['total_cached']} topics, "
        f"{total_kb:.0f} KB total"
    )
    print("="*60)


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ORBITA demo cache"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Single topic to cache (default: all topics)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing cache files",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        default=False,
        help="List cached topics and exit",
    )

    args = parser.parse_args()

    if args.list:
        dm    = DemoManager()
        stats = dm.get_stats()
        print(f"\nCached: {stats['total_cached']} / "
              f"{stats['total_possible']} topics\n")
        for t in dm.get_all_topics_with_status():
            status = "✓" if t["available"] else "✗"
            print(f"  {status} {t['icon']} {t['name']}")
        sys.exit(0)

    if args.topic:
        dm = DemoManager()
        ok = create_single(args.topic, dm, force=args.force)
        sys.exit(0 if ok else 1)
    else:
        create_all(force=args.force)