from __future__ import annotations

import math
import re

import pandas as pd


def clean_text(value: object) -> str:
    text = str(value).lower().strip()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_change_percent(value: object) -> float:
    text = str(value).replace("%", "").strip()
    if text in {"", "nan", "None"}:
        return float("nan")
    return float(text.replace(",", ""))


def change_to_sentiment(change_value: float) -> int:
    if change_value > 0:
        return 1
    if change_value < 0:
        return -1
    return 0


def sentiment_label(sentiment_value: int) -> str:
    if sentiment_value > 0:
        return "Bullish"
    if sentiment_value < 0:
        return "Bearish"
    return "Neutral"


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
        if math.isnan(result):
            return default
        return result
    except Exception:
        return default


def ensure_datetime(value: object) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return pd.Timestamp.utcnow().tz_localize(None)
    if getattr(timestamp, "tzinfo", None) is not None:
        return timestamp.tz_convert(None)
    return timestamp

