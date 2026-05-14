from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_NEWS_PATH = ROOT_DIR / "GOLDBEES_training_news.csv"
TRAINING_MARKET_PATH = ROOT_DIR / "GOLDBEES_NS_training_data.csv"
FALLBACK_HEADLINES_PATH = ROOT_DIR / "test_headlines.md"

MODEL_PATH = ARTIFACT_DIR / "gold_forecast_bundle.joblib"
DATABASE_PATH = ARTIFACT_DIR / "gold_dashboard.sqlite3"
SNAPSHOT_CACHE_PATH = ARTIFACT_DIR / "latest_snapshot.json"

AUTO_REFRESH_MS = 5 * 60 * 1000
DEFAULT_REFRESH_MINUTES = 5
PUSH_INTERVAL_SECONDS = 60

# Keep local historical CSV as an explicit opt-in fallback.
ALLOW_LOCAL_MARKET_FALLBACK = os.getenv("ALLOW_LOCAL_MARKET_FALLBACK", "0") == "1"

RSS_FEEDS = [
    "https://news.google.com/rss/search?q=gold+rate+india+OR+MCX+gold+OR+24K+gold+OR+22K+gold&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=gold+price+india+OR+gold+rate+today+OR+gold+market+india&hl=en-IN&gl=IN&ceid=IN:en",
]

FALLBACK_NEWS = [
    {
        "source": "Fallback",
        "title": "India gold prices steady as traders watch domestic demand",
        "summary": "Markets are range bound while buyers wait for the next India-focused rate cue.",
        "url": "fallback://headline-1",
    },
    {
        "source": "Fallback",
        "title": "Gold gains in India as rupee moves and local demand improve",
        "summary": "Rupee moves and domestic buying pressure support bullion prices.",
        "url": "fallback://headline-2",
    },
    {
        "source": "Fallback",
        "title": "Gold slips in India as retail buyers turn cautious",
        "summary": "Retail demand cools and prices ease after a strong run.",
        "url": "fallback://headline-3",
    },
]
