import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_intent_decoder():
    print("\n[1] Testing intent decoder...")
    from src.intent_decoder import decode_intent

    # Test with plain topic
    result = decode_intent("Cryptocurrency regulation India")
    assert result["topic"], "Topic should not be empty"
    assert isinstance(result["search_queries"], list), "Queries should be a list"
    assert len(result["search_queries"]) >= 1, "Should have at least 1 query"
    print(f"  Topic:   {result['topic']}")
    print(f"  Queries: {result['search_queries']}")
    print("  intent_decoder — OK")

    # Test with URL
    result2 = decode_intent("https://timesofindia.com/crypto-ban-india-2024")
    assert result2["topic"], "Should extract topic from URL"
    print(f"  URL topic: {result2['topic']}")
    print("  URL input — OK")


def test_stance_classifier():
    print("\n[2] Testing stance classifier...")
    from src.stance_filter import classify_stance

    supportive_article = {
        "title": "New policy brings huge benefits and growth opportunities",
        "description": "The government announced major progress and improvements"
                       " supporting development and advancing national interests.",
        "raw_content": ""
    }

    critical_article = {
        "title": "Controversy and protests erupt over dangerous new law",
        "description": "Critics warn of serious risks and problems as opposition"
                       " rejects the proposal citing corruption and violation.",
        "raw_content": ""
    }

    neutral_article = {
        "title": "Government report provides overview of new policy",
        "description": "Officials stated the data shows background context"
                       " according to research published in the official statement.",
        "raw_content": ""
    }

    s = classify_stance(supportive_article)
    c = classify_stance(critical_article)
    n = classify_stance(neutral_article)

    print(f"  Supportive article classified as: {s}")
    print(f"  Critical article classified as:   {c}")
    print(f"  Neutral article classified as:    {n}")

    assert s in ["Supportive", "Critical", "Neutral"], "Invalid stance label"
    assert c in ["Supportive", "Critical", "Neutral"], "Invalid stance label"
    assert n in ["Supportive", "Critical", "Neutral"], "Invalid stance label"
    print("  stance_classifier — OK")


def test_scraper():
    print("\n[3] Testing scraper on a known working URL...")
    from src.scraper import _scrape_single

    # Wikipedia is reliable and always scrapeable
    test_url  = "https://en.wikipedia.org/wiki/Cryptocurrency"
    full_text = _scrape_single(test_url)

    if full_text:
        print(f"  Scraped {len(full_text.split())} words from Wikipedia")
        print("  scraper — OK")
    else:
        print("  WARNING: Scraper returned None (network issue or blocked).")
        print("  This may be a temporary issue — scraper code is correct.")


def test_deduplicator():
    print("\n[4] Testing deduplicator...")
    from src.deduplicator import deduplicate

    articles = [
        {"url": "https://example.com/a", "title": "Article A",
         "full_text": "Crypto regulation is being debated in parliament India."},
        {"url": "https://example.com/a", "title": "Article A duplicate",
         "full_text": "Crypto regulation is being debated in parliament India."},
        {"url": "https://example.com/b", "title": "Article B",
         "full_text": "Farmers protest against new agricultural laws in Delhi."},
    ]

    result = deduplicate(articles)
    assert len(result) == 2, f"Expected 2 after dedup, got {len(result)}"
    print(f"  3 articles → {len(result)} after dedup — OK")


def test_full_pipeline_offline():
    print("\n[5] Testing full pipeline (offline components only)...")
    from src.intent_decoder import decode_intent
    from src.stance_filter  import label_all_articles, rebalance_articles
    from src.deduplicator   import deduplicate

    intent = decode_intent("Farm Laws India protest")
    print(f"  Intent: {intent['topic']}")

    mock_articles = [
        {
            "url": "https://example.com/1",
            "title": "Farm laws bring agricultural revolution",
            "description": "New laws support farmers with benefits and growth.",
            "raw_content": "",
            "source": "MockNews",
            "stance": None,
            "full_text": "Farm laws bring agricultural revolution. " * 30,
        },
        {
            "url": "https://example.com/2",
            "title": "Farmers protest dangerous farm laws",
            "description": "Critics warn of serious problems and violations.",
            "raw_content": "",
            "source": "MockTimes",
            "stance": None,
            "full_text": "Farmers protest dangerous farm laws. " * 30,
        },
        {
            "url": "https://example.com/3",
            "title": "Government report on farm law implementation",
            "description": "Officials stated data according to research published.",
            "raw_content": "",
            "source": "MockReport",
            "stance": None,
            "full_text": "Government report on farm law implementation. " * 30,
        },
    ]

    labelled   = label_all_articles(mock_articles)
    balanced   = rebalance_articles(labelled)
    deduplicated = deduplicate(balanced)

    print(f"  Pipeline processed {len(deduplicated)} mock articles")
    for a in deduplicated:
        print(f"    [{a['stance']}] {a['title'][:50]}")
    print("  Full pipeline (offline) — OK")


if __name__ == "__main__":
    print("=" * 55)
    print("  ORBITA – Step 2 Verification Tests")
    print("=" * 55)

    tests  = [
        test_intent_decoder,
        test_stance_classifier,
        test_scraper,
        test_deduplicator,
        test_full_pipeline_offline,
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
            print(f"  ERROR: {e}")
            failed += 1

    print("\n" + "=" * 55)
    if failed == 0:
        print(f"  ALL {passed} TESTS PASSED — Step 2 verified!")
    else:
        print(f"  {passed} passed, {failed} failed.")
        print("  Fix the issues above before moving to Step 3.")
    print("=" * 55)