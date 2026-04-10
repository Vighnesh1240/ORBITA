import sys
import os

# ── 1. Python version check ───────────────────────────────────────────────────
def test_python_version():
    print("\n[1] Checking Python version...")
    version = sys.version_info
    assert version.major == 3 and version.minor >= 10, (
        f"Need Python 3.10+, got {version.major}.{version.minor}"
    )
    print(f"    Python {version.major}.{version.minor}.{version.micro} — OK")

# ── 2. Package import checks ──────────────────────────────────────────────────
def test_imports():
    print("\n[2] Checking all required packages...")

    packages = {
        "langchain":               "LangChain core",
        "langchain_google_genai":  "LangChain Google Gemini",
        "langchain_community":     "LangChain community",
        "google.genai":            "Google Generative AI",
        "chromadb":                "ChromaDB",
        "newspaper":               "newspaper4k",
        "requests":                "Requests",
        "spacy":                   "spaCy",
        "sklearn":                 "Scikit-learn",
        "numpy":                   "NumPy",
        "streamlit":               "Streamlit",
        "plotly":                  "Plotly",
        "dotenv":                  "python-dotenv",
        "tqdm":                    "tqdm",
    }

    all_ok = True
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"    {name} — OK")
        except ImportError as e:
            print(f"    {name} — MISSING ({e})")
            all_ok = False

    assert all_ok, "Some packages are missing. Run: pip install -r requirements.txt"

# ── 3. spaCy model check ──────────────────────────────────────────────────────
def test_spacy_model():
    print("\n[3] Checking spaCy English model...")
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("ORBITA is a bias analysis tool for Indian politics.")
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        print(f"    en_core_web_sm loaded — OK")
        print(f"    Sample NER output: {entities}")
    except OSError:
        raise AssertionError(
            "spaCy model missing. Run: python -m spacy download en_core_web_sm"
        )

# ── 4. .env file and API key checks ──────────────────────────────────────────
def test_env_keys():
    print("\n[4] Checking .env file and API keys...")
    from dotenv import load_dotenv

    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if not os.path.exists(env_path):
        raise AssertionError(
            ".env file not found. Create it in the ORBITA/ root folder."
        )

    load_dotenv(env_path)

    news_key = os.getenv("NEWS_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    assert news_key and news_key != "paste_your_newsapi_key_here", (
        "NEWS_API_KEY is missing or still a placeholder in your .env file."
    )
    assert gemini_key and gemini_key != "paste_your_gemini_api_key_here", (
        "GEMINI_API_KEY is missing or still a placeholder in your .env file."
    )

    print(f"    NEWS_API_KEY found — {news_key[:6]}{'*' * (len(news_key)-6)}")
    print(f"    GEMINI_API_KEY found — {gemini_key[:6]}{'*' * (len(gemini_key)-6)}")

# ── 5. Live NewsAPI connection test ──────────────────────────────────────────
def test_newsapi_live():
    print("\n[5] Testing live NewsAPI connection...")
    from dotenv import load_dotenv
    import requests

    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    api_key = os.getenv("NEWS_API_KEY")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "technology",
        "pageSize": 1,
        "apiKey": api_key,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if response.status_code == 200 and data.get("status") == "ok":
            article = data["articles"][0]
            print(f"    NewsAPI connection — OK")
            print(f"    Sample article: {article['title'][:60]}...")
        else:
            raise AssertionError(
                f"NewsAPI error: {data.get('message', 'Unknown error')}"
            )
    except requests.exceptions.ConnectionError:
        raise AssertionError("No internet connection or NewsAPI is unreachable.")
    except requests.exceptions.Timeout:
        raise AssertionError("NewsAPI request timed out. Check your connection.")

# ── 6. Live Gemini API connection test ───────────────────────────────────────
def test_gemini_live():
    print("\n[6] Testing live Gemini API connection...")
    from dotenv import load_dotenv
    import warnings

    # Suppress all warnings from pydantic to avoid ArbitraryTypeWarning
    warnings.filterwarnings("ignore", module="pydantic.*")

    import google.genai as genai
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
    from config import EMBEDDING_MODEL

    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    api_key = os.getenv("GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)

    try:
        print("    Testing Gemini embedding API...")

        # Test embedding instead of text generation since that's what we use
        test_text = "ORBITA is a bias analysis tool for Indian news."
        
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[test_text]
        )

        if result.embeddings and len(result.embeddings) > 0:
            embedding_dim = len(result.embeddings[0].values)
            print(f"    Embedding API connection — OK")
            print(f"    Test embedding dimensions: {embedding_dim}")
            print(f"    Gemini replied: Embedding generated successfully")
        else:
            raise AssertionError("No embeddings returned from Gemini API.")

    except Exception as e:
        raise AssertionError(f"Gemini API error: {e}")

# ── 7. ChromaDB basic test ───────────────────────────────────────────────────
def test_chromadb():
    print("\n[7] Testing ChromaDB...")
    import chromadb

    client = chromadb.Client()
    collection = client.create_collection("orbita_test")
    collection.add(
        documents=["ORBITA is a bias detection tool."],
        ids=["test_doc_1"]
    )
    results = collection.query(query_texts=["bias detection"], n_results=1)
    print(f"    ChromaDB in-memory — OK")
    print(f"    Query result: {results['documents'][0][0][:50]}...")

# ── RUNNER ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        test_python_version,
        test_imports,
        test_spacy_model,
        test_env_keys,
        test_newsapi_live,
        test_gemini_live,
        test_chromadb,
    ]

    passed = 0
    failed = 0

    print("=" * 55)
    print("   ORBITA – Step 1 Environment Verification")
    print("=" * 55)

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"    FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"    ERROR: {e}")
            failed += 1

    print("\n" + "=" * 55)
    if failed == 0:
        print(f"   ALL {passed} CHECKS PASSED — Step 1 complete!")
        print("   You are ready to proceed to Step 2.")
    else:
        print(f"   {passed} passed, {failed} failed.")
        print("   Fix the issues above before moving to Step 2.")
    print("=" * 55 + "\n")