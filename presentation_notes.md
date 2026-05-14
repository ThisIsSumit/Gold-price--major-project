# 🪙 Gold Price Prediction Dashboard — Presentation Script & Speaker Notes

> **Project:** Nippon India ETF Gold BeES (GOLDBEES.NS) — AI-Powered Price Forecasting & Sentiment Analysis  
> **Stack:** Python · scikit-learn · FastAPI · Streamlit · yfinance · WebSockets

---

## SLIDE 1 — Introduction / Title Slide

**What to say:**

> "Good [morning/afternoon]. Today I'll be presenting my Major Project — an AI-powered Gold Price Prediction Dashboard that combines Natural Language Processing, Machine Learning, and real-time market data to forecast the next-day price and market sentiment of gold ETFs in India."

**Key points to highlight:**
- This is a **full-stack, end-to-end ML application** — not just a notebook
- It produces **live, automated predictions** from real news and real market data
- It has a **professional dashboard** with charts, KPI cards, and live WebSocket updates

---

## SLIDE 2 — Problem Statement

**What to say:**

> "Gold is one of India's most important financial assets — both culturally and economically. Millions of Indians invest in gold ETFs listed on NSE. The problem is: most retail investors have no easy way to gauge where gold prices are heading tomorrow based on today's news and market signals. My project solves this by automating that intelligence."

**Key points:**
- Gold ETFs like GOLDBEES are traded daily on NSE
- Price is influenced by **global news, INR/USD rates, MCX trends, geopolitics**
- Manual monitoring is slow and error-prone
- **Our system fetches live news → runs ML → gives a prediction in seconds**

---

## SLIDE 3 — Why Nippon India ETF Gold BeES (GOLDBEES.NS)?

**What to say:**

> "The first question I had to answer was: which gold instrument to predict? Let me explain why I chose GOLDBEES.NS specifically."

### 5 Reasons We Chose GOLDBEES

| Reason | Explanation |
|---|---|
| **Most Liquid Gold ETF in India** | GOLDBEES has the highest trading volume among all gold ETFs on NSE — more data, more reliable signals |
| **Direct INR Pricing** | Priced in ₹ per unit on NSE — no currency conversion required. XAU/USD futures need conversion layers |
| **Reflects Indian Market Sentiment** | Its price tracks Indian gold demand, RBI policy, INR strength — exactly what Indian gold news headlines talk about |
| **Training Data Consistency** | We found historical OHLCV data for GOLDBEES going back years — the same ticker we use for live prediction |
| **yfinance Support** | The ticker `GOLDBEES.NS` is fully supported by yfinance, giving us automated live data with no API key |

**Key insight to say:**
> "Using raw spot gold (XAU/INR) would have required a different data pipeline, currency conversion, and the training data would have been inconsistent with what news in India actually talks about. GOLDBEES is what retail investors actually buy and sell — so it's the right prediction target."

---

## SLIDE 4 — Data Sources (What Goes In)

**What to say:**

> "The system runs on two live data pipelines that run in parallel every time a prediction is triggered."

### Pipeline 1: News Data — Google News RSS

```
URL: news.google.com/rss/search?q=gold+rate+india+OR+MCX+gold+...&hl=en-IN&gl=IN
```

- Fetches **up to 40 live news articles** about gold in India
- Uses India-specific queries: "gold rate india", "MCX gold", "24K gold", "gold price today"
- Parsed with `feedparser` library (fast, reliable)
- If RSS fails → falls back to raw XML parsing via `urllib`
- If both fail → uses 3 hardcoded neutral fallback headlines (so the app never crashes)

### Pipeline 2: Market Data — yfinance

```
Priority:  GOLDBEES.NS → HDFCGOLD.NS → GC=F (COMEX)
```

- Fetches **1 year of daily OHLCV** data
- Extracts: Open, Close, High, Low, Volume
- Computes derived features: MA_3, MA_7, STD_3, Prev_Close, Prev_Open, Prev_Change

**Say:**
> "Both pipelines are independent — if news fails, we still get market data, and vice versa. The system is designed to be resilient."

---

## SLIDE 5 — Feature Engineering (How We Represent the Data)

**What to say:**

> "Once we have the raw news and market data, we need to convert it into a format the ML model can understand. This is called feature engineering."

### Two Types of Features

**Text Features (NLP):**
- Each news article's title + summary is cleaned → lowercased, punctuation removed
- Then vectorised using **TF-IDF** (Term Frequency–Inverse Document Frequency)
- `max_features = 12,000` — captures 12,000 most informative word/phrase patterns
- `ngram_range = (1,2)` — captures both single words AND two-word phrases like "gold rate", "MCX fall", "RBI policy"

**Numerical / Market Features (10 columns):**

| Feature | What it means |
|---|---|
| Open | Today's opening price of GOLDBEES |
| Prev_Close | Yesterday's closing price |
| Prev_Open | Yesterday's opening price |
| Prev_Change % | Yesterday's % price change |
| MA_3 | 3-day average closing price (short trend) |
| MA_7 | 7-day average closing price (medium trend) |
| STD_3 | 3-day rolling volatility (price stability) |
| Year, Month, Day | Temporal/seasonal pattern |

**Say:**
> "The combination of text features from news and numerical features from the market is what makes this model powerful. A pure price model ignores news; a pure NLP model ignores market context. We use both."

---

## SLIDE 6 — The ML Model (How It Predicts)

**What to say:**

> "The core of the system is a ModelBundle — two scikit-learn pipelines trained together and saved as a single file."

### Model 1: Ridge Regression → Predicts PRICE

```
Input:  [TF-IDF text features] + [10 market features]
Output: Predicted Next-Day Price (₹) + Predicted % Change
```

- **Ridge Regression** = Linear Regression with L2 regularisation
- Why Ridge? — Prevents overfitting when we have 12,000+ TF-IDF features
- Alpha = 2.0 (controls regularisation strength — tuned during training)
- Sanity check: if predicted price is 5x or 0.2x the real price → logged as warning

**How the final price is calculated:**
```
Predicted Price = Current Close × (1 + Predicted Change% / 100)
```

### Model 2: Logistic Regression → Predicts SENTIMENT

```
Input:  [TF-IDF text features] + [10 market features]
Output: Bullish (+1) / Neutral (0) / Bearish (-1)
```

- **Logistic Regression** = Multi-class classifier
- `class_weight = "balanced"` — so bearish articles (rarer) aren't ignored by the model
- `max_iter = 2000` — enough iterations for convergence on large TF-IDF feature space

### scikit-learn Pipeline Architecture

```
ColumnTransformer:
  ├── TF-IDF Vectoriser  →  applied to "News" text column
  └── StandardScaler     →  applied to 10 numeric columns

→ Ridge Regressor (for price)
→ Logistic Regressor (for sentiment)
```

**Say:**
> "The Pipeline object ensures there is absolutely no data leakage — the TF-IDF vocabulary and StandardScaler means are fitted ONLY on training data, and then applied to test/live data."

---

## SLIDE 7 — Training the Model

**What to say:**

> "The model is trained on historical data once, saved to disk, and then reused for every prediction. It can be retrained on-demand from the dashboard."

### Training Dataset

| File | Content |
|---|---|
| `GOLDBEES_training_news.csv` | Gold news headlines with price sentiment labels (positive/negative/neutral) |
| `GOLDBEES_NS_training_data.csv` | Daily OHLCV for GOLDBEES.NS |
| `gold-dataset-sinha-khandait.csv` | Extended gold price history |
| `XAU_INR Historical Data.csv` | XAU/INR spot price history |

### Training Procedure
1. Merge news CSV + market CSV on **date** (inner join)
2. Compute lag features: Prev_Close, Prev_Open, Prev_Change
3. Compute rolling features: MA_3, MA_7, STD_3
4. Sort chronologically → **80% train / 20% test** (no data leakage — past predicts future)
5. Fit both pipelines on train set
6. Evaluate on test set → compute RMSE, R², Sentiment Accuracy
7. Save `ModelBundle` → `artifacts/gold_forecast_bundle.joblib`

### Model Performance Metrics (visible live on dashboard)
- **RMSE (price)** — Average rupee error in price prediction
- **R² (price)** — How well model explains price variance (1.0 = perfect)
- **Sentiment Accuracy** — % of articles correctly labelled Bullish/Neutral/Bearish

---

## SLIDE 8 — The Prediction Pipeline (How Automation Works)

**What to say:**

> "Now let me explain how the system runs automatically — this is the part that makes it a live application, not just a script."

### The Snapshot Builder (Core Automation Engine)

Every prediction cycle is called a **"snapshot"** — a complete picture at one point in time. Here's what happens:

```
TRIGGER (startup / every 60s / on-demand API call)
         │
         ├──► NEWS PIPELINE
         │     └── Google News RSS → feedparser → clean_text() → TF-IDF tokens
         │
         ├──► MARKET PIPELINE
         │     └── yfinance → GOLDBEES.NS → compute MA_3, MA_7, STD_3
         │
         ├──► build_live_dataset()
         │     └── One row per news article, with all market features attached
         │
         ├──► ML MODEL
         │     ├── Ridge → Predicted Price + Change % per article
         │     └── Logistic → Sentiment label per article
         │
         ├──► aggregate_snapshot()
         │     └── avg predicted price, avg change %, bull/neutral/bear counts
         │
         ├──► Save to SQLite (snapshot history)
         └──► Save to JSON cache (latest_snapshot.json)
```

### Why is this automated?

**Three triggers for automation:**

1. **Startup trigger** — When `uvicorn` starts, it immediately builds the first snapshot
2. **Background loop (every 60 seconds)** — A FastAPI `BackgroundTask` runs `build_snapshot()` in a loop. Pushes result to all WebSocket clients
3. **On-demand API** — Dashboard's "Refresh Snapshot" button calls `POST /api/v1/snapshot/refresh`

---

## SLIDE 9 — The Backend (FastAPI)

**What to say:**

> "The ML logic is wrapped in a FastAPI backend — this is what makes the system accessible from the dashboard and allows real-time WebSocket updates."

### REST API Endpoints

| Endpoint | What it does |
|---|---|
| `GET /health` | Checks if backend is running |
| `GET /api/v1/snapshot/latest` | Returns the cached latest snapshot (fast, no recomputation) |
| `GET /api/v1/snapshot/live` | Builds a fresh snapshot right now (slower) |
| `POST /api/v1/snapshot/refresh` | Triggers rebuild + optional model retrain |
| `GET /api/v1/snapshot/history` | Returns last N snapshot records from SQLite |
| `WS /ws/snapshots` | WebSocket — pushes new snapshot JSON every 60s |

### WebSocket — How Real-Time Updates Work

```
Backend (FastAPI)                    Frontend (Streamlit/Browser)
     │                                         │
     │  ──── WebSocket Connection ──────────►  │
     │                                         │
     │  Every 60s:                             │
     │  build_snapshot() runs                  │
     │  ──── push JSON snapshot ────────────►  │
     │                                         │  JS checks: is created_at different?
     │                                         │  YES → window.parent.location.reload()
     │                                         │  (Streamlit auto-refreshes the page)
```

**Say:**
> "This means the dashboard updates automatically without the user pressing anything. As soon as a new prediction is ready, the WebSocket pushes it and the browser refreshes."

---

## SLIDE 10 — The Dashboard (Streamlit Frontend)

**What to say:**

> "The user-facing part is a Streamlit dashboard with a gold-themed professional design. Let me walk you through what it shows."

### Dashboard Sections

| Section | What the user sees |
|---|---|
| **Hero Banner** | Asset name, live subtitle, dark gold gradient |
| **4 KPI Cards** | Latest GOLDBEES Close · Predicted Price · Predicted Change % · Overall Sentiment |
| **Live WebSocket Badge** | 🟢 Connected / 🟡 Reconnecting / 🔴 Disconnected |
| **Market Trend Chart** | 180-day area chart with 7-day MA overlay (Plotly) |
| **Current Data Status** | News status · Market status · RMSE · R² · Sentiment accuracy |
| **Latest Reports Table** | Per-article: Source, Headline, Predicted Price, Change %, Sentiment |
| **Sentiment Pie Chart** | Donut — Bullish / Neutral / Bearish split |
| **Top Prediction Signals** | Horizontal bar — top 10 articles by predicted change % |
| **Historical Snapshot Trend** | Time series of stored predictions from SQLite |
| **Sidebar Controls** | Refresh Snapshot · Refresh & Retrain Model |

### Auto-Refresh (Two mechanisms)

1. **`streamlit-autorefresh`** — Polls every 5 minutes automatically
2. **WebSocket JS** — Triggers immediate reload when backend pushes a new snapshot

---

## SLIDE 11 — End-to-End Flow Summary

**What to say:**

> "Let me give you the complete picture in one flow."

```
REAL WORLD
  │
  ├── Gold news published on Google News India
  │
  └── GOLDBEES.NS trades on NSE
         │
         ▼
    [Every 60 seconds — Automatic]
         │
    FastAPI Backend (gold_app/backend.py)
    calls snapshot_builder.py
         │
         ├── data_sources.py → fetches 40 news articles (RSS)
         ├── data_sources.py → fetches GOLDBEES price history (yfinance)
         ├── model.py        → builds features, runs Ridge + Logistic
         ├── storage.py      → saves to SQLite + JSON cache
         └── backend.py      → pushes to WebSocket clients
                   │
                   ▼
    Browser (WebSocket JS) → detects new snapshot → triggers page reload
                   │
                   ▼
    Streamlit Dashboard → fetches /api/v1/snapshot/latest → renders charts
                   │
                   ▼
    User sees: ₹XX.XX predicted price, Bullish/Bearish sentiment, article table
```

---

## SLIDE 12 — Tech Stack Summary

| Layer | Technology | Why We Used It |
|---|---|---|
| ML / NLP | scikit-learn (Ridge, Logistic, TF-IDF, Pipeline) | Mature, reliable, no GPU needed |
| Data Processing | pandas, numpy | Standard data science tools |
| Market Data | yfinance | Free, no API key, supports NSE tickers |
| News Data | feedparser, urllib | Parses Google News RSS — India-specific |
| Model Persistence | joblib | Saves/loads sklearn Pipeline efficiently |
| Backend API | FastAPI + Uvicorn | Async, fast, WebSocket support built-in |
| Real-time Updates | WebSockets (native FastAPI) | Push new predictions to dashboard without polling |
| Frontend | Streamlit + streamlit-autorefresh | Rapid dashboard development in pure Python |
| Visualisation | Plotly | Interactive charts (area, donut, bar) |
| Storage | SQLite3 + JSON cache | Lightweight, no external DB server needed |
| Fonts / Design | Space Grotesk, JetBrains Mono (Google Fonts) | Professional gold-themed premium look |

---

## SLIDE 13 — Limitations & Future Scope

**What to say:**

> "Like any academic project, there are limitations, and these point to opportunities for future improvement."

### Current Limitations
- **Linear model only** — Ridge and Logistic Regression capture only linear relationships. LSTM or Transformer models could capture sequential dependencies in price and news
- **ETF price, not spot gold** — GOLDBEES is ₹ per ETF unit on NSE, not raw gold price per gram
- **News sentiment is indirect** — Sentiment is inferred from price reactions to similar text, not from a dedicated sentiment model (like FinBERT)
- **No macro indicators** — FII/DII data, interest rates, USD/INR, options flow are not included

### Future Scope
- Replace Ridge with **LSTM or Transformer** for sequential price forecasting
- Add **FinBERT** for direct NLP sentiment (pre-trained on financial text)
- Include **macro features**: RBI rate decisions, FII data, crude oil price, USD/INR
- Deploy to **cloud (AWS/GCP)** with a public URL
- Add **alert system** (email/SMS when sentiment flips Bearish)

---

## SLIDE 14 — Conclusion

**What to say:**

> "To summarise — this project demonstrates a complete, production-grade pipeline for AI-driven financial forecasting. We:
> - Chose GOLDBEES.NS as our target because it's India's most liquid gold ETF with consistent data
> - Combined NLP text features from live Google News with quantitative market features from NSE
> - Trained Ridge Regression for price forecasting and Logistic Regression for sentiment classification
> - Built a FastAPI backend that runs predictions automatically every 60 seconds using WebSockets
> - Visualised everything in a premium Streamlit dashboard with live updates
> 
> The system runs fully automatically — once started, it requires zero human intervention to keep producing fresh predictions."

---

## 🎤 Likely Questions From the Committee (With Answers)

**Q1: Why not use a deep learning model like LSTM?**
> "LSTM requires much more data and compute. For an academic project with limited historical data, a well-regularised linear model like Ridge is actually more stable and interpretable. The project architecture is designed to swap in a better model — the pipeline abstraction makes that easy."

**Q2: How accurate is your model?**
> "The model's accuracy metrics are visible live on the dashboard — RMSE, R², and sentiment accuracy. It's important to note that price prediction in finance is inherently hard. Our goal is directional accuracy (is it going up or down?) which the sentiment classifier provides."

**Q3: Why use TF-IDF and not a pre-trained language model like BERT?**
> "TF-IDF was chosen for speed and simplicity — it processes 40 articles in milliseconds. BERT would require significant compute resources and a GPU for inference. TF-IDF with bigrams captures most gold-related financial keywords effectively."

**Q4: Is this real-time?**
> "Yes — the backend auto-refreshes every 60 seconds. News is fetched live from Google News India, market prices from yfinance (NSE data), and the WebSocket pushes each new prediction directly to the dashboard."

**Q5: What happens if the news feed is down?**
> "The system has three fallback layers: primary RSS parsing → raw XML urllib fallback → hardcoded neutral fallback headlines. The app never crashes due to news failures."

**Q6: Why SQLite and not a real database?**
> "SQLite is perfect for this project — it's serverless, requires no setup, and stores the snapshot history as a local file. For a production deployment, switching to PostgreSQL would be a straightforward change."

---

*Built as a Major Project — Gold Price Prediction using NLP + Market Data*  
*Python · scikit-learn · FastAPI · Streamlit · yfinance*
