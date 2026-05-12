<div align="center">
# 🔭 ORBITA
 
### **Objective Reasoning & Bias Interpretation Tool for Analysis**
 
*An autonomous multi-agent RAG framework for real-time ideological bias quantification in news media*
 
<br/>
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.44-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Google Gemini](https://img.shields.io/badge/Gemini-1.5_Pro-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5.5-FF6B35?style=for-the-badge)](https://trychroma.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

 
</div>
---
 
## 📌 Table of Contents
 
- [🌟 Overview](#-overview)
- [🚨 The Problem](#-the-problem)
- [💡 The Solution](#-the-solution)
- [✨ Key Features](#-key-features)
- [🏗️ System Architecture](#️-system-architecture)
- [🤖 Multi-Agent Debate System](#-multi-agent-debate-system)
- [📐 4D Bias Vector](#-4d-bias-vector)
- [🛠️ Tech Stack](#️-tech-stack)
- [📁 Project Structure](#-project-structure)
- [⚙️ Installation & Setup](#️-installation--setup)
- [🚀 Running ORBITA](#-running-orbita)
- [📊 Dashboard Pages](#-dashboard-pages)
- [📈 Results & Evaluation](#-results--evaluation)
- [👥 Team](#-team)
- [🔮 Future Scope](#-future-scope)
---
 
## 🌟 Overview
 
**ORBITA** is an autonomous, real-time news bias detection system that goes far beyond traditional left-right scalar labels. Given any news topic, ORBITA:
 
1. 🔍 **Fetches** 10–15 diverse news articles from across the ideological spectrum
2. 🏷️ **Classifies** each article's stance (Supportive / Critical / Neutral)
3. 🧠 **Analyzes** using NLP, deep learning, and 3 specialized AI agents
4. ⚖️ **Synthesizes** a neutral 360° report with a 4-dimensional bias score
5. 📊 **Visualizes** everything in a beautiful interactive dashboard
```
User Topic ──► spaCy NER ──► NewsAPI ──► Stance Filter ──► Scraper ──► Dedup
                                                                          │
                    ChromaDB ◄── Gemini Embeddings ◄── LangChain Chunker ◄┘
                        │
           ┌────────────┼────────────┐
           ▼            ▼            ▼
       Agent A      Agent B      Agent C
      (Analyst)    (Critic)   (Arbitrator)
           └────────────┼────────────┘
                        ▼
               4D Bias Vector + Synthesis Report
                        ▼
               Streamlit Dashboard 🖥️
```
 
---
 
## 🚨 The Problem
 
Media bias is everywhere — but detecting it is hard:
 
| Problem | Impact |
|---------|--------|
| 📰 Same event, wildly different coverage across outlets | Public confusion & polarization |
| 🐌 Existing tools (AllSides, MBFC) are **manually updated** | Weeks/months outdated |
| 📉 All existing tools give a **single scalar score** | Misses nuanced, multi-dimensional bias |
| ❌ No tool works in **real-time** on demand | Can't analyze breaking news |
| 🤖 LLMs hallucinate — no validation layer | Unreliable automated analysis |
 
> *"The inability to quickly identify media bias leads to misinformation, polarization, and uninformed decision-making."*
 
---
 
## 💡 The Solution
 
ORBITA solves all of the above simultaneously:
 
- ⚡ **Real-time** — full analysis in 2–4 minutes on any topic
- 🎯 **4-Dimensional** bias vector instead of a scalar score
- 🤖 **3 adversarial AI agents** debate the topic — no confirmation bias
- 🔬 **VADER cross-validation** independently verifies every Gemini output
- 👁️ **ResNet-50 CNN** visual analysis on article images — fully local, no API
- ✅ **Hallucination check** — every claim verified against source documents
- 📚 **35+ sources** rated by credibility and ideological lean
---
 
## ✨ Key Features
 
<table>
<tr>
<td width="50%">
### 🧠 AI & Analysis
- ✅ Multi-agent adversarial debate (3 agents)
- ✅ RAG with ChromaDB + Gemini embeddings
- ✅ 4-Dimensional bias vector
- ✅ VADER + Gemini cross-validation
- ✅ ResNet-50 visual sentiment (local CNN)
- ✅ Hallucination detection & flagging
- ✅ Source credibility weighting (35+ outlets)
- ✅ spaCy NER intent decoding
</td>
<td width="50%">
### 📊 Dashboard & Platform
- ✅ 4-page Streamlit application
- ✅ 10 interactive Plotly charts
- ✅ Live animated pipeline meter
- ✅ Topic comparison mode
- ✅ Source × Topic bias heatmap
- ✅ SQLite longitudinal tracking
- ✅ Demo Mode (offline, < 1 sec)
- ✅ PDF report export
</td>
</tr>
</table>
---
 
## 🏗️ System Architecture
 
### 18-Step Pipeline across 5 Phases
 
```
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE I — Data Engineering                                          │
│  spaCy NER → NewsAPI (10-15 articles) → TF-IDF Stance Filter        │
│  → newspaper4k Scraper → Semantic Deduplication                     │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE II — NLP & Visual Analysis                                    │
│  VADER Sentiment → spaCy NER Corpus → TF-IDF Keywords               │
│  → ResNet-50 CNN Visual Sentiment (PyTorch, local)                  │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE III — RAG Core                                                │
│  LangChain Chunker (500 tokens) → Gemini text-embedding-004         │
│  → ChromaDB Vector Store (768-dim, persistent)                      │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE IV — Multi-Agent Synthesis                                    │
│  Agent A (Analyst) + Agent B (Critic) → Agent C (Arbitrator)        │
│  → Hallucination Check → 4D Bias Vector                             │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE V — Output & Visualization                                    │
│  SQLite History → Heatmap Store → Streamlit Dashboard → PDF Export  │
└─────────────────────────────────────────────────────────────────────┘
```
 
---
 
## 🤖 Multi-Agent Debate System
 
The heart of ORBITA — three adversarial AI agents powered by **Google Gemini 1.5 Pro**:
 
```
                    ┌──────────────────┐
                    │   ChromaDB RAG   │
                    │  Vector Store    │
                    └────────┬─────────┘
                             │ retrieves context
              ┌──────────────┼──────────────┐
              ▼              │              ▼
   ┌─────────────────┐       │   ┌─────────────────┐
   │  🟢 AGENT A     │       │   │  🔴 AGENT B     │
   │  The Analyst    │       │   │  The Critic     │
   │                 │       │   │                 │
   │ • Supportive    │       │   │ • Critical      │
   │   stance bias   │       │   │   stance bias   │
   │ • Extracts PRO  │       │   │ • Extracts CON  │
   │   arguments     │       │   │   arguments     │
   │ • JSON output   │       │   │ • JSON output   │
   └────────┬────────┘       │   └────────┬────────┘
            └────────────────┼────────────┘
                             ▼
                  ┌──────────────────────┐
                  │  🟡 AGENT C          │
                  │  The Arbitrator      │
                  │                      │
                  │ • Hallucination check│
                  │ • Removes bias lang  │
                  │ • 360° synthesis     │
                  │ • 4D bias score      │
                  └──────────────────────┘
```
 
| Agent | Role | Output |
|-------|------|--------|
| 🟢 **Agent A** | The Analyst | Supporting arguments + evidence JSON |
| 🔴 **Agent B** | The Critic | Counter-arguments + evidence JSON |
| 🟡 **Agent C** | The Arbitrator | Neutral synthesis + bias score + hallucination flags |
 
> 💡 **Why 3 agents?** Single-model approaches agree with themselves. Adversarial debate forces opposing argument extraction — Agent C can only synthesize what's been argued from both sides.
 
---
 
## 📐 4D Bias Vector
 
**ORBITA's core research contribution** — replacing scalar scores with a multi-dimensional analysis:
 
```
Composite Score = 0.40 × Ideological
               + 0.25 × Emotional
               + 0.20 × Informational
               + 0.15 × Source Diversity
```
 
| 📏 Dimension | 🔬 Method | ⚖️ Weight |
|---|---|---|
| **Ideological Bias** | Credibility-weighted stance score (−1 to +1) | 40% |
| **Emotional Bias** | Mean absolute VADER compound score | 25% |
| **Informational Bias** | 1 − (factual sentences / total sentences) | 20% |
| **Source Diversity** | 1 − Mean pairwise TF-IDF cosine distance | 15% |
 
**Score Interpretation:**
```
-1.0 ────────────── 0.0 ────────────── +1.0
Fully Supportive   Balanced        Fully Critical
```
 
---
 
## 🛠️ Tech Stack
 
<table>
<tr><th>Category</th><th>Technology</th><th>Purpose</th></tr>
<tr><td>🐍 Language</td><td>Python 3.10+</td><td>Core programming language</td></tr>
<tr><td>🖥️ Frontend</td><td>Streamlit 1.44 + Plotly 6.0</td><td>Web dashboard & visualizations</td></tr>
<tr><td>🤖 LLM</td><td>Google Gemini 1.5 Pro</td><td>Agent reasoning & synthesis</td></tr>
<tr><td>🔗 Orchestration</td><td>LangChain 0.3.25</td><td>RAG pipeline & chunking</td></tr>
<tr><td>🗃️ Vector DB</td><td>ChromaDB 1.5.5</td><td>Semantic storage & retrieval</td></tr>
<tr><td>🔤 NLP</td><td>spaCy + VADER + scikit-learn</td><td>NER, sentiment, TF-IDF</td></tr>
<tr><td>👁️ Vision</td><td>PyTorch + ResNet-50</td><td>Visual bias detection</td></tr>
<tr><td>📰 Scraping</td><td>newspaper4k + NewsAPI</td><td>Article fetching & parsing</td></tr>
<tr><td>💾 Storage</td><td>SQLite + JSON</td><td>History tracking & heatmap</td></tr>
<tr><td>📄 Export</td><td>fpdf2</td><td>PDF report generation</td></tr>
</table>
---
 
## 📁 Project Structure
 
```
ORBITA/
│
├── 📄 app.py                      # Main Streamlit application
├── 📄 requirements.txt            # Pinned dependencies
├── 📄 .env                        # API keys (not committed)
│
├── 📁 src/                        # Core Python package
│   ├── ⚙️  config.py              # Central configuration
│   ├── 🧠 intent_decoder.py       # spaCy NER intent extraction
│   ├── 📰 news_fetcher.py         # NewsAPI article retrieval
│   ├── 🏷️  stance_filter.py       # Zero-shot TF-IDF classification
│   ├── 🕷️  scraper.py             # newspaper4k text extraction
│   ├── 🔄 deduplicator.py         # Semantic deduplication
│   ├── 📝 nlp_analyzer.py         # VADER + spaCy + TF-IDF
│   ├── 👁️  cnn_image_analyzer.py  # ResNet-50 visual analysis
│   ├── 📐 bias_model.py           # 4D bias vector computation
│   ├── ⭐ source_credibility.py   # 35+ source credibility ratings
│   ├── ✂️  chunker.py             # LangChain text chunking
│   ├── 🔢 embedder.py             # Gemini embedding generation
│   ├── 🗃️  vector_store.py        # ChromaDB management
│   ├── 🟢 agent_a.py              # Agent A — The Analyst
│   ├── 🔴 agent_b.py              # Agent B — The Critic
│   ├── 🟡 agent_c.py              # Agent C — The Arbitrator
│   ├── 🤖 agents.py               # Multi-agent orchestration
│   ├── 🔁 pipeline.py             # Master pipeline controller
│   ├── 📊 history_tracker.py      # SQLite tracking
│   ├── ⚖️  comparison_engine.py   # Two-topic comparison
│   ├── 🌡️  heatmap_manager.py     # Bias heatmap store
│   ├── 🎭 demo_manager.py         # Pre-cached demo results
│   ├── ⚡ live_meter.py           # Animated pipeline meter
│   │
│   ├── 📁 evaluation/             # Formal evaluation module
│   │   ├── evaluator.py           # ROUGE-L + MAE scoring
│   │   ├── ground_truth.py        # AllSides ratings
│   │   └── rouge_scorer.py        # ROUGE implementation
│   │
│   └── 📁 ui/                     # UI components
│       ├── charts.py              # 10 Plotly chart functions
│       ├── components.py          # Reusable UI components
│       ├── debate_viz.py          # Agent debate board
│       └── comparison_charts.py   # Comparison visualizations
│
├── 📁 pages/                      # Streamlit multi-page nav
│   ├── 1_Home.py                  # Landing page
│   ├── 2_Compare.py               # Topic comparison
│   ├── 3_Heatmap.py               # Bias heatmap
│   └── 4_History.py               # Longitudinal tracking
│
├── 📁 demo_cache/                 # Pre-run analysis JSONs
├── 📁 assets/
│   └── style.css                  # Dark navy/gold theme
└── 📁 reports/                    # Generated analysis reports
```
 
---
 
## ⚙️ Installation & Setup
 
### Prerequisites
- Python 3.10+
- Node.js (optional, for PPT generation)
- Git
### 1️⃣ Clone the Repository
 
```bash
git clone https://github.com/yourusername/ORBITA.git
cd ORBITA
```
 
### 2️⃣ Create Virtual Environment
 
```bash
python -m venv venv
 
# Windows
venv\Scripts\activate
 
# Linux / macOS
source venv/bin/activate
```
 
### 3️⃣ Install Dependencies
 
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
 
### 4️⃣ Configure API Keys
 
Create a `.env` file in the root directory:
 
```env
NEWS_API_KEY=your_newsapi_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```
 
> 🔑 Get your free keys:
> - **NewsAPI** → [newsapi.org](https://newsapi.org)
> - **Gemini API** → [Google AI Studio](https://aistudio.google.com)
 
### 5️⃣ Verify Setup
 
```bash
python tests/test_setup.py
```
 
All 7 checks should pass ✅
 
---
 
## 🚀 Running ORBITA
 
### Launch the Dashboard
 
```bash
streamlit run app.py
```
 
Open your browser at **http://localhost:8501** 🌐
 
### Run the Pipeline Directly (CLI)
 
```bash
python src/pipeline.py
# Enter topic when prompted: e.g., "Farm Laws India"
```
 
### Run Tests
 
```bash
# Step-by-step verification
python tests/test_setup.py      # Environment check
python tests/test_step2.py      # Data pipeline
python tests/test_step3.py      # ChromaDB & embeddings
python tests/test_step4.py      # Multi-agent system
python tests/test_step5.py      # UI components
```
 
### Pre-cache Demo Topics (for offline demos)
 
```bash
python demo_cache/create_demo_cache.py --topic "India Elections 2024"
python demo_cache/create_demo_cache.py --list    # see all cached topics
```
 
---
 
## 📊 Dashboard Pages
 
| Page | Description |
|------|-------------|
| 🏠 **Main Analysis** | Live pipeline meter → Bias spectrum → 4D radar → Agent debate board → Source credibility |
| ⚖️ **Compare** | Side-by-side two-topic analysis with bias differential and source overlap |
| 🌡️ **Heatmap** | Sources × Topics bias matrix — reveals systematic outlet bias patterns |
| 📈 **History** | SQLite-backed longitudinal tracking — how bias shifts over time |
 
---
 
 
### Key Findings
 
- 🤝 VADER and Gemini **agreed on 76%** of articles (MAE < 0.3) — disagreements auto-flagged
- 🔍 4D vector revealed cases where **ideological bias was low but emotional bias was high** — invisible to scalar tools
- 🖼️ ResNet-50 added **measurable visual signal** on image-heavy topics (elections, protests)
- 🛡️ Multi-agent debate **reduced hallucination** vs single-agent approach
### ORBITA vs Existing Tools
 
| Feature | AllSides | MBFC | Ad Fontes | **ORBITA** |
|---------|----------|------|-----------|-----------|
| Real-time analysis | ❌ | ❌ | ❌ | ✅ |
| Multi-dimensional score | ❌ | ❌ | ❌ | ✅ |
| Automated pipeline | ❌ | ❌ | Partial | ✅ |
| Visual bias detection | ❌ | ❌ | ❌ | ✅ |
| Hallucination check | ❌ | ❌ | ❌ | ✅ |
| Open source | ❌ | ❌ | ❌ | ✅ |
 
 

</div>
---
 
## 🔮 Future Scope
 
- 🌐 **Multilingual Support** — Hindi, Tamil, and regional Indian languages
- 🎯 **Fine-tuned CNN** — domain-specific labeled image dataset for ResNet-50
- 🤝 **Multi-round Debate** — structured rebuttal protocol between agents
- 📱 **Browser Extension** — real-time article analysis while browsing
- 📊 **Learned Bias Weights** — replace manual weights with supervised learning
- 🐦 **Social Media** — Twitter/X and Reddit API integration
- 📄 **Research Paper** — arXiv submission to cs.CL
---
 
 
<div align="center">
**⭐ If you found this project helpful, please give it a star! ⭐** 
</div>
