# 🪙 Gold Price Prediction Dashboard

> **Nippon India ETF Gold BeES (GOLDBEES.NS) — AI-powered live price forecasting & sentiment analysis**

A full-stack machine learning application that predicts the next-day price and sentiment direction of **Nippon India ETF Gold BeES** using real-time news headlines and live market data from NSE. The system combines NLP (TF-IDF), Ridge Regression, and Logistic Regression inside a unified `scikit-learn` pipeline, served through a **FastAPI** backend and visualised in a **Streamlit** dashboard with live WebSocket updates.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [How It Works — End-to-End Flow](#-how-it-works--end-to-end-flow)
- [Machine Learning Models](#-machine-learning-models)
  - [Feature Engineering](#feature-engineering)
  - [Regression Model (Ridge)](#1-regression-model--ridge-regression)
  - [Classification Model (Logistic Regression)](#2-classification-model--logistic-regression)
  - [Training Data](#training-data)
  - [Model Metrics](#model-metrics)
- [Data Sources](#-data-sources)
- [Project Structure](#-project-structure)
- [Architecture](#-architecture)
- [API Reference](#-api-reference)
- [Setup & Installation](#-setup--installation)
- [Running the App](#-running-the-app)
- [Dashboard Features](#-dashboard-features)
- [Configuration](#-configuration)
- [Tech Stack](#-tech-stack)

---

## 🔍 Overview

| Aspect | Detail |
|---|---|
| **Asset** | Nippon India ETF Gold BeES (`GOLDBEES.NS`) |
| **Prediction Target** | Next closing price (₹) + % change direction |
| **Sentiment Output** | Bullish / Neutral / Bearish |
| **Data Refresh** | Auto every 5 minutes (WebSocket push + Streamlit poll) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Streamlit with Plotly charts |
| **Storage** | SQLite (snapshot history) + JSON cache |
| **Market Data** | `yfinance` → `GOLDBEES.NS`, `HDFCGOLD.NS`, `GC=F` (fallback chain) |
| **News Data** | Google News RSS feeds (India-specific gold queries) |

---

## ⚙️ How It Works — End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SNAPSHOT BUILDER                            │
│  (triggered on startup, every 60 s via background task, or         │
│   on-demand via API / Streamlit button)                             │
└──────────────┬───────────────────────────────────┬─────────────────┘
               │                                   │
               ▼                                   ▼
   ┌───────────────────────┐          ┌─────────────────────────────┐
   │    NEWS PIPELINE      │          │      MARKET PIPELINE        │
   │                       │          │                             │
   │  Google News RSS      │          │  yfinance → GOLDBEES.NS     │
   │  (2 India-focused     │          │  fallback: HDFCGOLD.NS      │
   │   gold queries)       │          │  fallback: GC=F             │
   │  ↓ feedparser / XML   │          │  ↓ Close, Open, High, Low   │
   │  ↓ clean_text()       │          │  ↓ build_market_features()  │
   │  ↓ TF-IDF tokens      │          │  (MA3, MA7, STD3, etc.)     │
   └──────────┬────────────┘          └───────────────┬─────────────┘
              │                                       │
              └───────────────────┬───────────────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │  build_live_dataset()  │
                     │  (one row per article, │
                     │   all market features  │
                     │   attached to each)    │
                     └───────────┬────────────┘
                                 │
                                 ▼
                     ┌────────────────────────┐
                     │    ML MODEL BUNDLE     │
                     │                        │
                     │  Ridge Regressor  ──►  Predicted Price (₹)
                     │                        Predicted Change %
                     │  Logistic Classifier ► Sentiment (-1/0/+1)
                     └───────────┬────────────┘
                                 │
                                 ▼
                     ┌────────────────────────┐
                     │   aggregate_snapshot() │
                     │   avg price, avg chg%  │
                     │   bull/neut/bear count │
                     └───────────┬────────────┘
                                 │
                     ┌───────────┴───────────┐
                     ▼                       ▼
              SQLite DB               JSON cache
              (snapshots              (latest_snapshot.json)
               + articles)
                     │
                     ▼
             FastAPI REST API
             + WebSocket /ws/snapshots
                     │
                     ▼
           Streamlit Dashboard
           (KPI cards, charts,
            article table, history)
```

---

## 🤖 Machine Learning Models

The project uses **two parallel scikit-learn pipelines** bundled together in a `ModelBundle` dataclass, serialised with `joblib`.

### Feature Engineering

Both models share the same feature set — a **hybrid of NLP and quantitative financial features**:

| Feature | Type | Description |
|---|---|---|
| `News` | Text (TF-IDF) | Cleaned news headline + summary text |
| `Open` | Float | Today's opening price of GOLDBEES |
| `Prev_Close` | Float | Previous day's closing price |
| `Prev_Open` | Float | Previous day's opening price |
| `Prev_Change` | Float | Previous day's % change |
| `MA_3` | Float | 3-day moving average of closing price |
| `MA_7` | Float | 7-day moving average of closing price |
| `STD_3` | Float | 3-day rolling standard deviation (volatility) |
| `Year` | Int | Calendar year (temporal feature) |
| `Month` | Int | Calendar month |
| `Day` | Int | Calendar day |

**Text preprocessing** (`clean_text`):
- Lowercasing
- Punctuation removal
- Whitespace normalisation

**TF-IDF Vectoriser settings:**
```
max_features = 12,000
ngram_range  = (1, 2)   ← unigrams + bigrams
min_df       = 2
max_df       = 0.98
```

---

### 1. Regression Model — Ridge Regression

**Goal:** Predict the numeric closing **price** (₹) and **% change** simultaneously.

```
Pipeline:
  ColumnTransformer
    ├── TF-IDF (on "News" column)
    └── StandardScaler (on 10 numeric features)
  └── Ridge(alpha=2.0)         ← L2 regularisation to prevent overfitting
```

- The Ridge regressor is trained with a **multi-output** target: `[Price, Change %]`
- At inference time, the **predicted price** is derived as:
  ```
  Predicted Price = Current_Close × (1 + Predicted_Change% / 100)
  ```
- A **sanity check** is applied: if `Predicted Price / Prev_Close` ratio is `> 5×` or `< 0.2×`, a warning is logged (catches training-vs-live scale mismatches)

---

### 2. Classification Model — Logistic Regression

**Goal:** Predict market **sentiment** direction: `Bullish (+1)`, `Neutral (0)`, `Bearish (-1)`.

```
Pipeline:
  ColumnTransformer
    ├── TF-IDF (on "News" column)
    └── StandardScaler (on 10 numeric features)
  └── LogisticRegression(max_iter=2000, class_weight="balanced")
```

- `class_weight="balanced"` ensures minority sentiment classes (e.g., Bearish) are not overwhelmed by neutral articles
- Output labels are mapped: `-1 → Bearish`, `0 → Neutral`, `+1 → Bullish`

---

### Training Data

| Dataset | File | Description |
|---|---|---|
| News + Sentiment | `GOLDBEES_training_news.csv` | Dated gold headlines with `Price Sentiment` labels (positive/negative/neutral) |
| Market OHLC | `GOLDBEES_NS_training_data.csv` | Daily OHLCV data for GOLDBEES.NS |
| Extended Gold Dataset | `gold-dataset-sinha-khandait.csv` | Supplementary historical gold price data |
| XAU/INR Historical | `XAU_INR Historical Data.csv` | Gold spot price history in INR |

**Training procedure:**
1. News and market CSVs are merged on `Dates` (inner join)
2. Lag features (`Prev_Close`, `Prev_Open`, `Prev_Change`) and rolling windows (`MA_3`, `MA_7`, `STD_3`) are computed
3. Data is sorted chronologically and split **80% train / 20% test** (time-series aware — no data leakage)
4. Both pipelines are fit on the train split; metrics are computed on the held-out test split
5. The `ModelBundle` is saved to `artifacts/gold_forecast_bundle.joblib`

---

### Model Metrics

Metrics are computed on the test set (last 20% of the time-ordered data) and stored in the snapshot:

| Metric | Description |
|---|---|
| `rmse_price` | Root Mean Squared Error for price prediction |
| `r2_price` | R² score for price prediction |
| `rmse_change` | RMSE for % change prediction |
| `r2_change` | R² for % change |
| `sentiment_accuracy` | Classification accuracy for Bullish/Neutral/Bearish |

These are visible live in the **Current Data Status** card on the dashboard.

---

## 📡 Data Sources

### News — Google News RSS (India)

Two RSS feeds are queried in parallel:

```
https://news.google.com/rss/search?q=gold+rate+india+OR+MCX+gold+OR+24K+gold+OR+22K+gold&hl=en-IN&gl=IN
https://news.google.com/rss/search?q=gold+price+india+OR+gold+rate+today+OR+gold+market+india&hl=en-IN&gl=IN
```

- Parsed via `feedparser` (primary) → raw XML `urllib` (fallback)
- Up to **40 articles** fetched per refresh
- Deduplicated by `(url, title)`
- If both fail → **hardcoded fallback headlines** are used (3 neutral India-gold headlines)

### Market Data — yfinance

Ticker priority chain:

```
1. GOLDBEES.NS   ← Primary (same scale as training data, ₹ per unit)
2. HDFCGOLD.NS   ← Fallback ETF
3. GC=F          ← COMEX Gold Futures (USD)
```

- 1-year daily OHLCV fetched
- `MultiIndex` columns from yfinance are flattened automatically
- If all live tickers fail and `ALLOW_LOCAL_MARKET_FALLBACK=1`, loads from local CSV

---

## 📁 Project Structure

```
mp-code/
│
├── gold_app/                    # Core Python package
│   ├── __init__.py
│   ├── config.py                # Paths, RSS feeds, timing constants
│   ├── data_sources.py          # News fetching (RSS) + market data (yfinance)
│   ├── model.py                 # ML pipelines, training, prediction
│   ├── snapshot_builder.py      # Orchestrates full prediction cycle
│   ├── storage.py               # SQLite persistence + JSON cache
│   ├── utils.py                 # Text cleaning, date parsing helpers
│   └── backend.py               # FastAPI app (REST + WebSocket)
│
├── streamlit_app.py             # Streamlit dashboard UI
│
├── scripts/
│   ├── create_goldbees_training_data.py   # Training CSV builder
│   └── refresh_snapshot.py               # CLI snapshot refresh
│
├── artifacts/                   # Auto-created at runtime
│   ├── gold_forecast_bundle.joblib        # Trained ModelBundle
│   ├── gold_dashboard.sqlite3             # Snapshot history DB
│   └── latest_snapshot.json              # JSON cache of last snapshot
│
├── GOLDBEES_training_news.csv   # News + sentiment training data
├── GOLDBEES_NS_training_data.csv# Market OHLCV training data
├── gold-dataset-sinha-khandait.csv
├── XAU_INR Historical Data.csv
├── mp-code.ipynb                # Jupyter notebook (EDA / experiments)
├── requirements.txt
└── README.md
```

---

## 🏗️ Architecture

```
                 ┌──────────────────────────────────┐
                 │         Streamlit UI              │
                 │  (streamlit_app.py)               │
                 │  - KPI cards                      │
                 │  - Plotly charts (price trend,    │
                 │    sentiment pie, bar chart)       │
                 │  - Article predictions table      │
                 │  - Historical snapshot trend      │
                 └───────────┬──────────────────────┘
                             │ HTTP REST + WebSocket
                             ▼
                 ┌──────────────────────────────────┐
                 │       FastAPI Backend             │
                 │  (gold_app/backend.py)            │
                 │                                   │
                 │  GET  /health                     │
                 │  GET  /api/v1/snapshot/latest     │
                 │  GET  /api/v1/snapshot/live       │
                 │  POST /api/v1/snapshot/refresh    │
                 │  GET  /api/v1/snapshot/history    │
                 │  WS   /ws/snapshots               │
                 └───────────┬──────────────────────┘
                             │
              ┌──────────────┼───────────────┐
              ▼              ▼               ▼
   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
   │  ML Model    │ │  Data Layer  │ │   Storage    │
   │  (model.py)  │ │(data_sources)│ │ (storage.py) │
   │              │ │              │ │              │
   │  Ridge Reg.  │ │  Google RSS  │ │  SQLite DB   │
   │  Logistic    │ │  yfinance    │ │  JSON cache  │
   │  TF-IDF      │ │              │ │              │
   └──────────────┘ └──────────────┘ └──────────────┘
```

---

## 🔌 API Reference

All endpoints are served by the FastAPI backend (default: `http://127.0.0.1:8001`).

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |
| `GET` | `/api/v1/snapshot/latest` | Returns cached/stored snapshot (fast) |
| `GET` | `/api/v1/snapshot/live` | Builds a fresh snapshot on-demand |
| `POST` | `/api/v1/snapshot/refresh` | Refresh with optional `{"force_retrain": true}` |
| `GET` | `/api/v1/snapshot/history?limit=30` | Returns last N snapshot timestamps + predicted prices |
| `WS` | `/ws/snapshots` | WebSocket — pushes new snapshot JSON every 60 s |

### Snapshot JSON Schema (abbreviated)

```jsonc
{
  "created_at": "2024-05-14T06:30:00+00:00",
  "news_status": "fetched 38 live articles",
  "market_status": "fetched India market history from GOLDBEES.NS (latest close: ₹62.34)",
  "model_metrics": {
    "rmse_price": 3.45,
    "r2_price": 0.92,
    "rmse_change": 0.85,
    "r2_change": 0.78,
    "sentiment_accuracy": 0.71
  },
  "market_features": {
    "Open": 62.10, "Prev_Close": 61.80, "MA_3": 62.00, "MA_7": 61.50,
    "latest_close": 62.34, "latest_high": 62.90, "latest_low": 61.70
  },
  "summary": {
    "predicted_price": 62.89,
    "predicted_change_pct": 0.0087,
    "sentiment_score": 0.42,
    "bullish_count": 22,
    "neutral_count": 10,
    "bearish_count": 6
  },
  "articles": [ /* per-article predictions */ ],
  "history":  [ /* last 90 days OHLCV */ ]
}
```

---

## 🛠️ Setup & Installation

### Prerequisites

- Python 3.10+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
pandas>=2.2
numpy>=1.26
scikit-learn>=1.5
fastapi>=0.115
uvicorn>=0.30
requests>=2.32
streamlit>=1.38
streamlit-autorefresh>=1.0.1
plotly>=5.24
yfinance>=0.2.43
feedparser>=6.0.11
joblib>=1.4
```

---

## 🚀 Running the App

You need **two terminal windows** — one for the backend, one for the frontend.

### Terminal 1 — Start the FastAPI Backend

```bash
uvicorn gold_app.backend:app --host 0.0.0.0 --port 8001
```

On first startup:
- SQLite DB is initialised (`artifacts/gold_dashboard.sqlite3`)
- Model is trained if `artifacts/gold_forecast_bundle.joblib` does not exist
- A background WebSocket publisher loop starts (pushes snapshot every 60 s)

### Terminal 2 — Start the Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

Then open **http://localhost:8501** in your browser.

> **Note:** The Streamlit app expects the FastAPI backend at `http://127.0.0.1:8001` by default. Override with:
> ```bash
> set GOLD_BACKEND_URL=http://your-backend-host:8001   # Windows
> export GOLD_BACKEND_URL=http://your-backend-host:8001 # Linux/macOS
> ```

### Force Model Retrain

Via the Streamlit sidebar → **"Refresh & Retrain Model"** button, or via API:
```bash
curl -X POST http://127.0.0.1:8001/api/v1/snapshot/refresh \
     -H "Content-Type: application/json" \
     -d '{"force_retrain": true}'
```

---

## 📊 Dashboard Features

| Section | What you see |
|---|---|
| **Hero Banner** | Asset name, live subtitle |
| **KPI Cards (4)** | Latest GOLDBEES Close · Predicted Avg Price · Predicted Change % · Overall Sentiment badge |
| **Live WebSocket badge** | Green = connected, amber = reconnecting, red = disconnected |
| **Market Trend Chart** | 180-day area chart with 7-day MA overlay (Plotly) |
| **Current Data Status** | News feed status · Market feed status · RMSE · R² · Sentiment accuracy · Snapshot timestamp |
| **Latest Reports** | Per-article table: Source, Headline, Predicted Price, Change %, Sentiment, URL |
| **Sentiment Pie Chart** | Donut chart — Bullish / Neutral / Bearish split |
| **Top Prediction Signals** | Horizontal bar chart — top 10 articles by predicted change % |
| **Historical Snapshot Trend** | Time series of stored predicted prices from SQLite history |
| **Sidebar Controls** | Refresh snapshot · Refresh & Retrain · Backend URL info |

The dashboard **auto-refreshes every 5 minutes** via `streamlit-autorefresh`, and also reloads immediately when the WebSocket receives a newer snapshot from the backend.

---

## ⚙️ Configuration

Key constants in `gold_app/config.py`:

| Variable | Default | Description |
|---|---|---|
| `AUTO_REFRESH_MS` | `300,000` (5 min) | Streamlit polling interval |
| `PUSH_INTERVAL_SECONDS` | `60` | WebSocket broadcast interval |
| `ALLOW_LOCAL_MARKET_FALLBACK` | `0` | Set env `=1` to allow CSV fallback if all tickers fail |
| `MODEL_PATH` | `artifacts/gold_forecast_bundle.joblib` | Trained model location |
| `DATABASE_PATH` | `artifacts/gold_dashboard.sqlite3` | SQLite history DB |
| `SNAPSHOT_CACHE_PATH` | `artifacts/latest_snapshot.json` | Fast JSON cache |

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **ML / Data Science** | scikit-learn (Ridge, Logistic Regression, TF-IDF, Pipeline, ColumnTransformer) |
| **Data Processing** | pandas, numpy |
| **Market Data** | yfinance |
| **News Data** | feedparser, urllib (XML fallback) |
| **Model Persistence** | joblib |
| **Backend API** | FastAPI, Uvicorn |
| **Real-time Updates** | WebSockets (FastAPI native) |
| **Frontend / Dashboard** | Streamlit, streamlit-autorefresh |
| **Visualisation** | Plotly (go.Scatter, go.Pie, go.Bar) |
| **Storage** | SQLite3 (snapshot history), JSON file cache |
| **Fonts / Styling** | Space Grotesk, JetBrains Mono (Google Fonts), CSS variables (light/dark auto-theme) |

---

## 📝 Notes & Limitations

- The model is trained on **GOLDBEES.NS ETF prices** (₹ per unit, NSE traded). This is **not raw spot gold** (₹/10g). Keep this in mind when interpreting predicted prices.
- News sentiment is inferred **indirectly** via regression on historical price reactions to similar news text — not via a dedicated LLM or sentiment lexicon.
- Predictions are **article-level** (each fetched news article produces one prediction), then aggregated. More news articles → more stable aggregate prediction.
- The model is a **linear model** (Ridge + Logistic). It captures linear relationships between TF-IDF news tokens and price movement. Deep learning models (LSTM, transformers) are out of scope for this academic project.
- For true production use, additional data (options flow, FII/DII data, macro indicators) and a more sophisticated model would be needed.

---

<div align="center">
  <strong>Built as a Major Project — Gold Price Prediction using NLP + Market Data</strong><br/>
  Python · scikit-learn · FastAPI · Streamlit · yfinance
</div>
