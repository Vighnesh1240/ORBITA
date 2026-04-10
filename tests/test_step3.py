# tests/test_step3.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_chunker():
    print("\n[1] Testing chunker...")
    from src.chunker import chunk_article, chunk_all_articles

    mock_article = {
        "url":       "https://example.com/test",
        "title":     "Test Article About Crypto",
        "source":    "TestNews",
        "stance":    "Neutral",
        "full_text": (
            "Cryptocurrency regulation has been a major topic of debate. "
            "Governments around the world are discussing how to regulate "
            "digital currencies. Some argue that regulation will bring "
            "stability and protect investors from fraud and manipulation. "
            "Others believe that regulation will stifle innovation and "
            "defeat the purpose of decentralised finance. The debate "
            "continues as more institutions adopt digital assets. "
            "Central banks are exploring their own digital currencies. "
            "The technology behind cryptocurrency, blockchain, has many "
            "other applications beyond finance including supply chain "
            "management, healthcare records, and voting systems. "
        ) * 10,  # repeat to make it long enough for multiple chunks
    }

    chunks = chunk_article(mock_article)
    assert len(chunks) >= 1, "Should produce at least 1 chunk"
    assert "text"      in chunks[0], "Chunk must have text"
    assert "chunk_id"  in chunks[0], "Chunk must have chunk_id"
    assert "stance"    in chunks[0], "Chunk must have stance"
    assert "url"       in chunks[0], "Chunk must have url"

    print(f"  Single article → {len(chunks)} chunks — OK")
    print(f"  First chunk preview: {chunks[0]['text'][:80]}...")

    # Test chunk_all_articles
    all_chunks = chunk_all_articles([mock_article, mock_article])
    assert len(all_chunks) >= 2, "Should produce chunks from both articles"
    print(f"  Two articles → {len(all_chunks)} total chunks — OK")
    print("  chunker — OK")


def test_embedder_single():
    print("\n[2] Testing embedder (single chunk)...")
    from src.embedder import _embed_batch

    test_texts = ["ORBITA is a bias analysis tool for Indian news."]
    embeddings = _embed_batch(test_texts)

    assert len(embeddings) == 1, "Should return 1 embedding"
    assert isinstance(embeddings[0], list), "Embedding should be a list"
    assert len(embeddings[0]) > 100, "Embedding should have many dimensions"

    dim = len(embeddings[0])
    print(f"  Single text → embedding of {dim} dimensions — OK")
    print("  embedder single — OK")


def test_embedder_batch():
    print("\n[3] Testing embedder (batch)...")
    from src.embedder import embed_chunks

    mock_chunks = [
        {
            "chunk_id":    "test_chunk_1",
            "text":        "Farm laws have brought significant changes to Indian agriculture.",
            "url":         "https://example.com/1",
            "title":       "Farm Laws Article",
            "source":      "TestNews",
            "stance":      "Supportive",
            "chunk_index": 0,
        },
        {
            "chunk_id":    "test_chunk_2",
            "text":        "Farmers protest against the new agricultural regulations.",
            "url":         "https://example.com/2",
            "title":       "Protest Article",
            "source":      "TestTimes",
            "stance":      "Critical",
            "chunk_index": 0,
        },
    ]

    embedded = embed_chunks(mock_chunks)

    assert len(embedded) == 2, f"Expected 2 embedded chunks, got {len(embedded)}"
    assert "embedding" in embedded[0], "Chunk must have embedding field"
    assert "embedding" in embedded[1], "Chunk must have embedding field"

    print(f"  2 chunks embedded successfully — OK")
    print(f"  Embedding dimensions: {len(embedded[0]['embedding'])}")
    print("  embedder batch — OK")


def test_vector_store_store_and_retrieve():
    print("\n[4] Testing ChromaDB store and retrieve...")
    from src.embedder    import embed_chunks
    from src.vector_store import store_chunks, retrieve_chunks, get_collection_stats
    import chromadb

    mock_chunks = [
        {
            "chunk_id":    "vs_test_1",
            "text":        "Cryptocurrency regulation helps protect retail investors from scams.",
            "url":         "https://example.com/crypto1",
            "title":       "Crypto Benefits",
            "source":      "NewsA",
            "stance":      "Supportive",
            "chunk_index": 0,
        },
        {
            "chunk_id":    "vs_test_2",
            "text":        "Strict crypto bans will destroy innovation in the fintech sector.",
            "url":         "https://example.com/crypto2",
            "title":       "Crypto Criticism",
            "source":      "NewsB",
            "stance":      "Critical",
            "chunk_index": 0,
        },
        {
            "chunk_id":    "vs_test_3",
            "text":        "The government is reviewing the cryptocurrency policy framework.",
            "url":         "https://example.com/crypto3",
            "title":       "Crypto Policy",
            "source":      "NewsC",
            "stance":      "Neutral",
            "chunk_index": 0,
        },
    ]

    # Embed
    embedded = embed_chunks(mock_chunks)
    assert len(embedded) == 3, "All 3 chunks should embed"

    # Store
    store_chunks(embedded, reset=True)

    # Stats
    stats = get_collection_stats()
    assert stats["total_chunks"] >= 3, "Should have at least 3 chunks"
    print(f"  Stored {stats['total_chunks']} chunks — OK")

    # Retrieve all stances
    results = retrieve_chunks("cryptocurrency regulation", n_results=3)
    assert len(results) >= 1, "Should retrieve at least 1 result"
    print(f"  Retrieved {len(results)} chunks for 'cryptocurrency regulation'")
    for r in results:
        print(f"    [{r['stance']}] {r['text'][:60]}... (dist: {r['distance']})")

    # Retrieve with stance filter
    supportive = retrieve_chunks(
        "cryptocurrency benefits", n_results=2, stance_filter="Supportive"
    )
    print(f"  Supportive-only filter: {len(supportive)} chunk(s)")
    for r in supportive:
        assert r["stance"] == "Supportive", "Filter should only return Supportive"

    print("  vector_store store+retrieve — OK")


def test_retrieval_relevance():
    print("\n[5] Testing retrieval relevance ordering...")
    from src.vector_store import retrieve_chunks

    # Search for something specific — most relevant should come first
    results = retrieve_chunks("protect investors cryptocurrency", n_results=3)

    assert len(results) >= 1, "Should return results"
    # Distances should be in ascending order (lower = more similar)
    distances = [r["distance"] for r in results]
    assert distances == sorted(distances), \
        "Results should be ordered by distance (most similar first)"

    print(f"  Results ordered by relevance — OK")
    print(f"  Most relevant: {results[0]['text'][:70]}...")
    print(f"  Distance: {results[0]['distance']}")
    print("  retrieval relevance — OK")


if __name__ == "__main__":
    print("=" * 55)
    print("  ORBITA – Step 3 Verification Tests")
    print("=" * 55)

    tests = [
        test_chunker,
        test_embedder_single,
        test_embedder_batch,
        test_vector_store_store_and_retrieve,
        test_retrieval_relevance,
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
        print(f"  ALL {passed} TESTS PASSED — Step 3 verified!")
        print("  ChromaDB is ready. Say 'Go to Step 4' when ready.")
    else:
        print(f"  {passed} passed, {failed} failed.")
        print("  Fix the issues above before moving to Step 4.")
    print("=" * 55)