from __future__ import annotations

import logging
from urllib.request import urlopen
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import pandas as pd

from .config import ALLOW_LOCAL_MARKET_FALLBACK, FALLBACK_HEADLINES_PATH, FALLBACK_NEWS, RSS_FEEDS, TRAINING_MARKET_PATH
from .utils import clean_text, ensure_datetime, safe_float

logger = logging.getLogger(__name__)


try:
    import feedparser
except ModuleNotFoundError:  # pragma: no cover - dependency installed in app env
    feedparser = None


try:
    import yfinance as yf
except ModuleNotFoundError:  # pragma: no cover - dependency installed in app env
    yf = None


def _source_name_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc or "unknown"
    return host.replace("www.", "")


def _local_tag_name(tag: object) -> str:
    text = str(tag)
    return text.rsplit("}", 1)[-1].lower()


def _find_first_child_text(element: ET.Element, names: set[str]) -> str:
    for child in element:
        if _local_tag_name(child.tag) in names:
            value = "".join(child.itertext()).strip()
            if value:
                return value
    return ""


def _find_link(element: ET.Element) -> str:
    for child in element:
        if _local_tag_name(child.tag) != "link":
            continue
        href = child.attrib.get("href", "").strip()
        if href:
            return href
        value = "".join(child.itertext()).strip()
        if value:
            return value
    return ""


def _parse_xml_rss_feed(feed_url: str, max_articles: int) -> list[dict[str, object]]:
    try:
        with urlopen(feed_url, timeout=20) as response:
            raw_xml = response.read()
    except Exception:
        return []

    try:
        root = ET.fromstring(raw_xml)
    except Exception:
        return []

    items = [node for node in root.iter() if _local_tag_name(node.tag) in {"item", "entry"}]
    articles: list[dict[str, object]] = []
    for entry in items[:max_articles]:
        title = _find_first_child_text(entry, {"title"})
        summary = _find_first_child_text(entry, {"summary", "description", "content", "content:encoded"})
        url = _find_link(entry)
        if not url:
            url = _find_first_child_text(entry, {"link"})
        published_raw = _find_first_child_text(entry, {"published", "pubdate", "updated"})

        source = _find_first_child_text(entry, {"source"})
        if not source:
            source_node = next((child for child in entry if _local_tag_name(child.tag) == "source"), None)
            if source_node is not None:
                source = _find_first_child_text(source_node, {"title"}) or "".join(source_node.itertext()).strip()
        if not source:
            source = _source_name_from_url(url)

        text = f"{title} {summary}".strip()
        articles.append(
            {
                "published_at": ensure_datetime(published_raw),
                "source": source,
                "title": title,
                "summary": summary,
                "url": url,
                "text": text,
                "clean_text": clean_text(text),
            }
        )

    return articles


def _frame_from_articles(articles: list[dict[str, object]], max_articles: int) -> pd.DataFrame:
    if not articles:
        return pd.DataFrame()
    frame = pd.DataFrame(articles)
    frame = frame.drop_duplicates(subset=["url", "title"]).sort_values("published_at", ascending=False)
    return frame.head(max_articles).reset_index(drop=True)


def _collect_articles_from_feedparser(feed_url: str, max_articles: int) -> list[dict[str, object]]:
    try:
        feed = feedparser.parse(feed_url)
    except Exception:
        return _parse_xml_rss_feed(feed_url, max_articles=max_articles)

    feed_entries = getattr(feed, "entries", [])
    if not feed_entries:
        return _parse_xml_rss_feed(feed_url, max_articles=max_articles)

    articles: list[dict[str, object]] = []
    for entry in feed_entries[:max_articles]:
        title = str(entry.get("title", "")).strip()
        summary = str(entry.get("summary", entry.get("description", ""))).strip()
        url = str(entry.get("link", "")).strip()
        published_raw = entry.get("published", entry.get("updated", None))
        published_at = ensure_datetime(published_raw)
        source = entry.get("source", {}).get("title") if isinstance(entry.get("source"), dict) else None
        if not source:
            source = _source_name_from_url(url)
        text = f"{title} {summary}".strip()
        articles.append(
            {
                "published_at": published_at,
                "source": source,
                "title": title,
                "summary": summary,
                "url": url,
                "text": text,
                "clean_text": clean_text(text),
            }
        )

    return articles


def _collect_live_news_articles(max_articles: int) -> list[dict[str, object]]:
    articles: list[dict[str, object]] = []
    for feed_url in RSS_FEEDS:
        if feedparser is None:
            articles.extend(_parse_xml_rss_feed(feed_url, max_articles=max_articles))
        else:
            articles.extend(_collect_articles_from_feedparser(feed_url, max_articles=max_articles))
    return articles


def _fallback_headlines() -> pd.DataFrame:
    items = list(FALLBACK_NEWS)
    if FALLBACK_HEADLINES_PATH.exists():
        text = FALLBACK_HEADLINES_PATH.read_text(encoding="utf-8")
        candidates: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(">"):
                candidates.append(stripped.lstrip("> "))
        for index, headline in enumerate(candidates[:8]):
            items.append(
                {
                    "source": "Test Headlines",
                    "title": headline,
                    "summary": headline,
                    "url": f"fallback://test-headlines/{index}",
                }
            )

    now = pd.Timestamp.utcnow().tz_localize(None)
    frame = pd.DataFrame(items)
    frame["published_at"] = now
    frame["text"] = (frame["title"].fillna("") + " " + frame["summary"].fillna("")).str.strip()
    frame["clean_text"] = frame["text"].map(clean_text)
    return frame[["published_at", "source", "title", "summary", "url", "text", "clean_text"]]


def fetch_live_news(max_articles: int = 40) -> tuple[pd.DataFrame, str]:
    articles = _collect_live_news_articles(max_articles=max_articles)
    if not articles:
        return _fallback_headlines(), "rss unavailable, using fallback headlines"

    frame = _frame_from_articles(articles, max_articles=max_articles)
    return frame, f"fetched {len(frame)} live articles"


def _download_ticker(ticker: str, period: str = "1y") -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    try:
        data = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
    except Exception:
        return pd.DataFrame()
    if data is None or data.empty:
        return pd.DataFrame()
    
    # Reset index to convert date index to column
    frame = data.reset_index()
    
    # Flatten MultiIndex columns if present (yfinance returns MultiIndex for single tickers)
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [col[0] if isinstance(col, tuple) else str(col) for col in frame.columns]
    
    # Ensure column names are strings
    frame.columns = [str(col).strip() for col in frame.columns]
    
    # Normalize Date column name
    if "Datetime" in frame.columns and "Date" not in frame.columns:
        frame = frame.rename(columns={"Datetime": "Date"})
    
    return frame


def _normalize_market_history_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "Date" not in frame.columns and "Datetime" in frame.columns:
        frame = frame.rename(columns={"Datetime": "Date"})
    if "Date" not in frame.columns:
        return pd.DataFrame()

    frame = frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)


def _load_local_market_history() -> pd.DataFrame:
    if not TRAINING_MARKET_PATH.exists():
        return pd.DataFrame()

    local = pd.read_csv(TRAINING_MARKET_PATH)
    if "Date" in local.columns:
        local["Date"] = pd.to_datetime(local["Date"], errors="coerce")
    local["Price"] = local["Price"].astype(str).str.replace(",", "", regex=False).astype(float)
    local["Open"] = local["Open"].astype(str).str.replace(",", "", regex=False).astype(float)
    local["High"] = local["High"].astype(str).str.replace(",", "", regex=False).astype(float)
    local["Low"] = local["Low"].astype(str).str.replace(",", "", regex=False).astype(float)
    local = local.rename(columns={"Price": "Close"})
    columns = [column for column in ["Date", "Open", "High", "Low", "Close", "Vol.", "Change %"] if column in local.columns]
    return local[columns].sort_values("Date").reset_index(drop=True)


def fetch_market_history(period: str = "1y") -> tuple[pd.DataFrame, str]:
    # Fetch live data - prefer GOLDBEES.NS (same scale as live data)
    # This ensures training and live data are on the same scale
    candidates = ["GOLDBEES.NS", "HDFCGOLD.NS", "GC=F"]
    for ticker in candidates:
        logger.info(f"Attempting to fetch market history from {ticker}...")
        frame = _download_ticker(ticker, period=period)
        frame = _normalize_market_history_frame(frame)
        if not frame.empty:
            latest_close = safe_float(frame.iloc[-1].get("Close", 0.0))
            message = f"fetched India market history from {ticker} (latest close: ₹{latest_close:.2f})"
            logger.info(message)
            return frame, message
        else:
            logger.warning(f"{ticker} returned empty or failed to normalize")

    if ALLOW_LOCAL_MARKET_FALLBACK:
        logger.info("All tickers failed, trying local fallback...")
        local = _load_local_market_history()
        if not local.empty:
            latest_close = safe_float(local.iloc[-1].get("Close", 0.0))
            logger.info(f"Using local fallback (latest close: ₹{latest_close:.2f})")
            return local, "using local historical market file"

    logger.error("All market data sources failed - returning empty")
    return pd.DataFrame(), "market history unavailable"


def build_market_features(market_frame: pd.DataFrame) -> dict[str, object]:
    if market_frame.empty:
        now = pd.Timestamp.utcnow().tz_localize(None)
        return {
            "Open": 0.0,
            "Prev_Close": 0.0,
            "Prev_Open": 0.0,
            "Prev_Change": 0.0,
            "MA_3": 0.0,
            "MA_7": 0.0,
            "STD_3": 0.0,
            "Year": now.year,
            "Month": now.month,
            "Day": now.day,
            "market_date": now.to_pydatetime(),
            "latest_close": 0.0,
            "latest_high": 0.0,
            "latest_low": 0.0,
            "change_pct": 0.0,
            "history_tail": pd.DataFrame(),
        }

    frame = market_frame.sort_values("Date").reset_index(drop=True).copy()
    latest_row = frame.iloc[-1]
    latest_close = safe_float(latest_row.get("Close", 0.0))
    latest_open = safe_float(latest_row.get("Open", latest_close))
    latest_high = safe_float(latest_row.get("High", latest_close))
    latest_low = safe_float(latest_row.get("Low", latest_close))

    prev_row = frame.iloc[-2] if len(frame) > 1 else None
    prev_close = safe_float(prev_row.get("Close", latest_close)) if prev_row is not None else latest_close
    prev_open = safe_float(prev_row.get("Open", prev_close)) if prev_row is not None else latest_close
    prev_change = ((latest_close - prev_close) / prev_close * 100.0) if prev_close != 0 else 0.0
    change_pct = ((latest_close - latest_open) / latest_open * 100.0) if latest_open != 0 else 0.0

    history = frame.tail(7).copy().reset_index(drop=True)
    ma_3 = history["Close"].tail(3).mean() if len(history) >= 3 else latest_close
    ma_7 = history["Close"].mean()
    std_3 = history["Close"].tail(3).std() if len(history) >= 3 else 0.0

    market_date = pd.Timestamp(latest_row.get("Date", pd.Timestamp.utcnow())).to_pydatetime()

    return {
        "Open": latest_open,
        "Prev_Close": prev_close,
        "Prev_Open": prev_open,
        "Prev_Change": prev_change,
        "MA_3": ma_3,
        "MA_7": ma_7,
        "STD_3": std_3,
        "Year": market_date.year,
        "Month": market_date.month,
        "Day": market_date.day,
        "market_date": market_date,
        "latest_close": latest_close,
        "latest_high": latest_high,
        "latest_low": latest_low,
        "change_pct": change_pct,
        "history_tail": history,
    }


def build_live_dataset(news_frame: pd.DataFrame, market_features: dict[str, object]) -> pd.DataFrame:
    if news_frame.empty or not market_features:
        return pd.DataFrame()

    dataset: list[dict[str, object]] = []
    for _, row in news_frame.iterrows():
        dataset.append(
            {
                "published_at": row.get("published_at"),
                "article_date": ensure_datetime(row.get("published_at")).date().isoformat(),
                "source": row.get("source"),
                "title": row.get("title"),
                "summary": row.get("summary"),
                "url": row.get("url"),
                "text": row.get("text"),
                "News": row.get("clean_text", ""),
                "Open": market_features.get("Open", 0.0),
                "Current_Close": market_features.get("latest_close", market_features.get("Prev_Close", 0.0)),
                "Prev_Close": market_features.get("Prev_Close", 0.0),
                "Prev_Open": market_features.get("Prev_Open", 0.0),
                "Prev_Change": market_features.get("Prev_Change", 0.0),
                "MA_3": market_features.get("MA_3", 0.0),
                "MA_7": market_features.get("MA_7", 0.0),
                "STD_3": market_features.get("STD_3", 0.0),
                "Year": market_features.get("Year", 0),
                "Month": market_features.get("Month", 0),
                "Day": market_features.get("Day", 0),
            }
        )

    return pd.DataFrame(dataset) if dataset else pd.DataFrame()




