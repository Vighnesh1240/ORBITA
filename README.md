# ORBITA

Objective Reasoning And Bias Interpretation Tool for Analysis.

ORBITA is a multi-agent AI system for news bias analysis. It combines article retrieval, NLP, vector search (RAG), visual analysis, and agent-based reasoning to produce explainable bias reports and side-by-side topic comparisons.

## What ORBITA Does

- Fetches and analyzes news coverage for a topic.
- Labels stance across articles (Supportive, Critical, Neutral).
- Runs local NLP validation (sentiment, entities, keywords).
- Builds semantic retrieval context with embeddings + ChromaDB.
- Uses a 3-agent debate pipeline:
  - Agent A: Supporting arguments
  - Agent B: Counter arguments
  - Agent C: Synthesis + arbitration
- Generates bias outputs with explainable components.
- Supports:
  - Single-topic deep analysis
  - Two-topic comparison mode
  - Historical tracking and heatmap views

## Core Features

- Streamlit multi-page UI (Home, Compare, Heatmap, History)
- 4D bias perspective (ideological, emotional, informational, diversity)
- PDF report generation
- Demo cache mode for instant presentation runs
- Optional CNN-based visual framing analysis

## Tech Stack

- Frontend: Streamlit, Plotly
- NLP: spaCy, VADER, scikit-learn
- LLM + Embeddings: Google Gemini APIs
- Vector DB: ChromaDB
- Data/Scraping: NewsAPI + newspaper4k
- Optional vision: PyTorch + torchvision

## Project Structure

- app.py: Main analysis app entry point
- pages/: Streamlit pages
  - 1_Home.py
  - 2_Compare.py
  - 3_Heatmap.py
  - 4_History.py
- src/: Core pipeline, agents, retrieval, analytics, reporting
- assets/: UI styles
- demo_cache/: Cached topic outputs for demo mode
- tests/: Environment and pipeline tests
- reports/: Generated reports
- chroma_db/: Local vector database
- data/: Saved run metadata

## Prerequisites

- Python 3.10+
- Internet connection
- NewsAPI key
- Google Gemini API key

## Setup Guide

## 1) Clone and open

```bash
git clone <your-repo-url>
cd ORBITA
```

## 2) Create and activate virtual environment

### Windows PowerShell

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3) Install dependencies

```bash
pip install -r requirements.txt
```

## 4) Install spaCy English model

```bash
python -m spacy download en_core_web_sm
```

## 5) Create .env file in project root

Create a file named .env in ORBITA root with:

```env
NEWS_API_KEY=your_newsapi_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

Notes:
- NEWS_API_KEY is required for article retrieval.
- GEMINI_API_KEY is required for embeddings and agent reasoning.

## Run ORBITA

From project root:

```bash
streamlit run app.py
```

Then open the local URL shown in terminal (usually http://localhost:8501).

## Compare Mode

Use the Compare page to analyze two topics side-by-side.

- Enter Topic A and Topic B
- Click Compare
- Enable Use Demo Cache for fast precomputed outputs

## Demo Cache (Recommended for presentations)

Generate all cached demo topics:

```bash
python demo_cache/create_demo_cache.py
```

Generate a single topic:

```bash
python demo_cache/create_demo_cache.py --topic "AI Regulation India"
```

Overwrite existing cache:

```bash
python demo_cache/create_demo_cache.py --force
```

List cached topics:

```bash
python demo_cache/create_demo_cache.py --list
```

## Testing and Validation

Environment readiness test:

```bash
python tests/test_setup.py
```

Run a specific test module:

```bash
python tests/test_pipeline_nlp.py
```

If you use pytest in your environment:

```bash
pytest -q
```

## Configuration

Main runtime constants are in src/config.py, including:

- Fetch limits and scraping thresholds
- Chunk size and overlap
- Embedding model and dimensionality
- Chroma collection settings
- Agent model and generation parameters

Default Gemini model configured currently:

- models/gemini-2.5-flash-lite

## Outputs and Artifacts

- reports/: Generated PDF reports
- data/: Saved article metadata snapshots
- chroma_db/: Persistent vector storage
- cache/: Topic-level cache
- orbita_history.db: Historical run database
- heatmap_data.json: Heatmap data

## Troubleshooting

## Missing spaCy model

Error mentions en_core_web_sm not found:

```bash
python -m spacy download en_core_web_sm
```

## API key errors

- Verify .env exists at project root.
- Confirm NEWS_API_KEY and GEMINI_API_KEY are valid.
- Restart app after changing .env.

## Chroma or embedding errors

- Ensure internet is available.
- Check Gemini API quota and key permissions.
- Delete stale local cache only if needed:
  - cache/
  - chroma_db/

## Streamlit page load issues

- Confirm dependencies installed in active virtual environment.
- Re-run from root with streamlit run app.py.

## Notes for Contributors

- Keep changes modular in src/ and avoid hardcoding credentials.
- Preserve multipage UI conventions used in pages/.
- Add tests in tests/ for new pipeline behavior.
- Keep README updated when setup or config changes.

## Responsible Use

ORBITA is an analysis aid, not a substitute for expert judgment. Model outputs can be imperfect and should be cross-checked with primary sources.

## Project Name

ORBITA
