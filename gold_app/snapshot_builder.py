from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from .data_sources import build_live_dataset, build_market_features, fetch_live_news, fetch_market_history
from .model import aggregate_snapshot, predict_live, train_model
from .storage import save_snapshot

logger = logging.getLogger(__name__)


def build_snapshot(force_retrain: bool = False) -> dict:
    bundle = train_model(force_retrain=force_retrain)
    news_frame, news_status = fetch_live_news()
    market_frame, market_status = fetch_market_history()
    
    logger.info(f"Market status: {market_status}")
    if not market_frame.empty:
        latest_close = market_frame.iloc[-1].get("Close", 0.0)
        logger.info(f"Latest market close: ₹{latest_close:.2f}")
    
    market_features = build_market_features(market_frame)
    live_frame = build_live_dataset(news_frame, market_features)
    predicted_frame = predict_live(bundle, live_frame)
    
    if not predicted_frame.empty:
        avg_price = predicted_frame["Predicted Price"].mean()
        avg_change = predicted_frame["Predicted Change %"].mean()
        logger.info(f"Predictions - Avg Price: ₹{avg_price:.2f}, Avg Change: {avg_change:.4f}%")
        
        # Validation: check consistency
        if market_features.get("latest_close", 0) > 0:
            latest_close = market_features["latest_close"]
            expected_price = latest_close * (1 + avg_change / 100)
            logger.info(f"Consistency check - Latest: ₹{latest_close:.2f}, Expected: ₹{expected_price:.2f}, Predicted: ₹{avg_price:.2f}")
            if abs(avg_price - expected_price) > expected_price * 0.1:  # More than 10% off
                logger.error(f"⚠️ MISMATCH: Predicted price (₹{avg_price:.2f}) inconsistent with change % ({avg_change:.4f}%)")
    
    summary = aggregate_snapshot(predicted_frame)

    snapshot = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "news_status": news_status,
        "market_status": market_status,
        "model_metrics": bundle.metrics,
        "market_features": {
            key: value
            for key, value in market_features.items()
            if key not in {"history_tail", "market_date"}
        },
        "summary": summary,
        "articles": predicted_frame.to_dict(orient="records"),
        "history": market_features.get("history_tail", pd.DataFrame()).tail(90).to_dict(orient="records"),
    }
    save_snapshot(snapshot)
    return snapshot
