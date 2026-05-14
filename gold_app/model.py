from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import MODEL_PATH, TRAINING_MARKET_PATH, TRAINING_NEWS_PATH
from .utils import change_to_sentiment, clean_text, parse_change_percent, sentiment_label


FEATURE_COLUMNS = ["Open", "Prev_Close", "Prev_Open", "Prev_Change", "MA_3", "MA_7", "STD_3", "Year", "Month", "Day"]


@dataclass
class ModelBundle:
    regressor: Pipeline
    classifier: Pipeline
    feature_columns: list[str]
    metrics: dict[str, float]
    version: str = "1.0"

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        regression_prediction = self.regressor.predict(frame)
        class_prediction = self.classifier.predict(frame)
        return regression_prediction, class_prediction

    def save(self, path: Path = MODEL_PATH) -> None:
        joblib.dump(self, path)


def _load_training_frame() -> pd.DataFrame:
    news_frame = pd.read_csv(TRAINING_NEWS_PATH)
    market_frame = pd.read_csv(TRAINING_MARKET_PATH)

    news_frame["Dates"] = pd.to_datetime(news_frame["Dates"].astype(str).str.strip(), format="%d-%m-%Y", errors="coerce")
    if "Date" in market_frame.columns:
        market_frame["Date"] = pd.to_datetime(market_frame["Date"].astype(str).str.strip(), format="%m/%d/%Y", errors="coerce")
        market_frame = market_frame.rename(columns={"Date": "Dates"})
    market_frame["Dates"] = pd.to_datetime(market_frame["Dates"], errors="coerce")

    merged = pd.merge(news_frame, market_frame, on="Dates", how="inner", validate="many_to_many")
    merged["Price"] = merged["Price"].astype(str).str.replace(",", "", regex=False).astype(float)
    merged["Open"] = merged["Open"].astype(str).str.replace(",", "", regex=False).astype(float)
    merged["High"] = merged["High"].astype(str).str.replace(",", "", regex=False).astype(float)
    merged["Low"] = merged["Low"].astype(str).str.replace(",", "", regex=False).astype(float)
    merged["Change_num"] = merged["Change %"].map(parse_change_percent)
    merged["News"] = merged["News"].map(clean_text)
    merged["Price Sentiment"] = merged["Price Sentiment"].astype(str).str.lower()
    merged["SentimentClass"] = merged["Price Sentiment"].map({"negative": -1, "neutral": 0, "positive": 1, "none": 0}).fillna(0).astype(int)

    merged = merged.sort_values("Dates").reset_index(drop=True)
    merged["Year"] = merged["Dates"].dt.year
    merged["Month"] = merged["Dates"].dt.month
    merged["Day"] = merged["Dates"].dt.day
    merged["Prev_Close"] = merged["Price"].shift(1)
    merged["Prev_Open"] = merged["Open"].shift(1)
    merged["Prev_Change"] = merged["Change_num"].shift(1)
    merged["MA_3"] = merged["Price"].rolling(3).mean()
    merged["MA_7"] = merged["Price"].rolling(7).mean()
    merged["STD_3"] = merged["Price"].rolling(3).std()

    merged = merged.dropna(subset=["News", "Price", "Change_num", *FEATURE_COLUMNS, "SentimentClass"]).reset_index(drop=True)
    return merged


def _build_regression_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline([
        ("scale", StandardScaler()),
    ])
    preprocessing = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=2, max_df=0.98), "News"),
            ("num", numeric_pipeline, FEATURE_COLUMNS),
        ]
    )
    return Pipeline([
        ("preprocess", preprocessing),
        ("regressor", Ridge(alpha=2.0)),
    ])


def _build_classifier_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline([
        ("scale", StandardScaler()),
    ])
    preprocessing = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=2, max_df=0.98), "News"),
            ("num", numeric_pipeline, FEATURE_COLUMNS),
        ]
    )
    return Pipeline([
        ("preprocess", preprocessing),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def train_model(force_retrain: bool = False) -> ModelBundle:
    if MODEL_PATH.exists() and not force_retrain:
        return load_model()

    frame = _load_training_frame()
    split_index = max(int(len(frame) * 0.8), 1)

    train_frame = frame.iloc[:split_index].copy()
    test_frame = frame.iloc[split_index:].copy()

    regressor = _build_regression_pipeline()
    classifier = _build_classifier_pipeline()

    regressor.fit(train_frame[["News", *FEATURE_COLUMNS]], train_frame[["Price", "Change_num"]])
    classifier.fit(train_frame[["News", *FEATURE_COLUMNS]], train_frame["SentimentClass"])

    reg_predictions = regressor.predict(test_frame[["News", *FEATURE_COLUMNS]])
    cls_predictions = classifier.predict(test_frame[["News", *FEATURE_COLUMNS]])

    metrics = {
        "rmse_price": float(np.sqrt(mean_squared_error(test_frame["Price"], reg_predictions[:, 0]))),
        "r2_price": float(r2_score(test_frame["Price"], reg_predictions[:, 0])),
        "rmse_change": float(np.sqrt(mean_squared_error(test_frame["Change_num"], reg_predictions[:, 1]))),
        "r2_change": float(r2_score(test_frame["Change_num"], reg_predictions[:, 1])),
        "sentiment_accuracy": float(accuracy_score(test_frame["SentimentClass"], cls_predictions)),
    }

    bundle = ModelBundle(
        regressor=regressor,
        classifier=classifier,
        feature_columns=FEATURE_COLUMNS,
        metrics=metrics,
        version="1.0",
    )
    bundle.save()
    return bundle


def load_model() -> ModelBundle:
    if not MODEL_PATH.exists():
        return train_model(force_retrain=True)
    bundle = joblib.load(MODEL_PATH)
    if not isinstance(bundle, ModelBundle):
        raise TypeError("Saved model artifact is not a ModelBundle")
    return bundle


def predict_live(bundle: ModelBundle, live_frame: pd.DataFrame) -> pd.DataFrame:
    if live_frame.empty:
        return live_frame.copy()

    frame = live_frame.copy()
    for column in FEATURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0.0

    frame["News"] = frame["News"].map(clean_text)
    predictions = bundle.regressor.predict(frame[["News", *FEATURE_COLUMNS]])
    class_predictions = bundle.classifier.predict(frame[["News", *FEATURE_COLUMNS]])

    result = frame.copy()
    result["Raw Predicted Price"] = predictions[:, 0]
    result["Predicted Change %"] = predictions[:, 1]
    if "Current_Close" in result.columns:
        current_close = pd.to_numeric(result["Current_Close"], errors="coerce").fillna(0.0)
    else:
        current_close = pd.Series(0.0, index=result.index)
    result["Predicted Price"] = current_close * (1 + result["Predicted Change %"] / 100.0)
    result["Model Sentiment"] = class_predictions
    result["Model Sentiment Label"] = result["Model Sentiment"].map(sentiment_label)
    result["Predicted Direction"] = result["Predicted Change %"].map(lambda value: sentiment_label(change_to_sentiment(float(value))))
    
    # Validation: warn if predictions seem off (e.g., predicted price is 10x the input features)
    if not result.empty and "Prev_Close" in result.columns:
        avg_prev_close = result["Prev_Close"].mean()
        avg_pred_price = result["Predicted Price"].mean()
        if avg_prev_close > 0 and avg_pred_price > 0:
            ratio = avg_pred_price / avg_prev_close
            # If ratio is extreme (>5 or <0.2), likely a scale mismatch
            if ratio > 5 or ratio < 0.2:
                import warnings
                warnings.warn(
                    f"Predicted price (₹{avg_pred_price:.2f}) vs input price (₹{avg_prev_close:.2f}) "
                    f"ratio is {ratio:.2f}x - possible scale mismatch between training and live data"
                )
    
    return result


def aggregate_snapshot(predicted_frame: pd.DataFrame) -> dict[str, float | int | str]:
    if predicted_frame.empty:
        return {
            "predicted_price": 0.0,
            "predicted_change_pct": 0.0,
            "sentiment_score": 0.0,
            "bullish_count": 0,
            "neutral_count": 0,
            "bearish_count": 0,
        }

    sentiment_map = predicted_frame["Model Sentiment"].astype(int)
    return {
        "predicted_price": float(predicted_frame["Predicted Price"].mean()),
        "predicted_change_pct": float(predicted_frame["Predicted Change %"].mean()),
        "sentiment_score": float(sentiment_map.mean()),
        "bullish_count": int((sentiment_map > 0).sum()),
        "neutral_count": int((sentiment_map == 0).sum()),
        "bearish_count": int((sentiment_map < 0).sum()),
    }
