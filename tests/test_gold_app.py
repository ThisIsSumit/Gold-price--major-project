from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gold_app import backend, data_sources, model, snapshot_builder, storage, utils


class UtilsTestCase(unittest.TestCase):
    def test_text_and_sentiment_helpers(self) -> None:
        self.assertEqual(utils.clean_text(" Gold prices jump!  "), "gold prices jump")
        self.assertTrue(np.isnan(utils.parse_change_percent("nan")))
        self.assertEqual(utils.parse_change_percent("1.25%"), 1.25)
        self.assertEqual(utils.change_to_sentiment(0.1), 1)
        self.assertEqual(utils.change_to_sentiment(-0.1), -1)
        self.assertEqual(utils.sentiment_label(0), "Neutral")
        self.assertEqual(utils.sentiment_label(1), "Bullish")
        self.assertEqual(utils.sentiment_label(-1), "Bearish")
        self.assertEqual(utils.safe_float("bad", default=7.5), 7.5)
        self.assertEqual(utils.ensure_datetime("2026-05-12").date().isoformat(), "2026-05-12")


class DataSourcesTestCase(unittest.TestCase):
    def test_fetch_live_news_uses_feedparser(self) -> None:
        fake_entry = {
            "title": "Gold jumps on inflation data",
            "summary": "Bullion rises after market release",
            "link": "https://example.com/story-1",
            "published": "Tue, 12 May 2026 10:00:00 GMT",
            "source": {"title": "Example News"},
        }
        fake_feedparser = SimpleNamespace(parse=lambda _url: SimpleNamespace(entries=[fake_entry]))

        with patch.object(data_sources, "feedparser", fake_feedparser), patch.object(
            data_sources, "RSS_FEEDS", ["https://example.com/feed"]
        ):
            frame, status = data_sources.fetch_live_news(max_articles=5)

        self.assertEqual(len(frame), 1)
        self.assertIn("fetched 1 live articles", status)
        self.assertEqual(frame.iloc[0]["source"], "Example News")
        self.assertIn("gold jumps on inflation data", frame.iloc[0]["clean_text"])

    def test_fetch_live_news_uses_fallback_without_parser(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "headlines.md"
            temp_path.write_text(
                "# Headlines\n\n> Custom fallback headline one\n> Custom fallback headline two\n",
                encoding="utf-8",
            )

            with patch.object(data_sources, "feedparser", None), patch.object(
                data_sources, "FALLBACK_HEADLINES_PATH", temp_path
            ), patch.object(data_sources, "_parse_xml_rss_feed", return_value=[]):
                frame, status = data_sources.fetch_live_news(max_articles=5)

        self.assertGreaterEqual(len(frame), 2)
        self.assertIn("fallback", status)
        self.assertIn("Custom fallback headline one", frame.iloc[3]["title"])

    def test_fetch_market_history_uses_local_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "market.csv"
            temp_path.write_text(
                "Date,Open,High,Low,Price,Vol.,Change %\n"
                "2026-05-10,100,110,90,105,1000,1.50%\n"
                "2026-05-11,105,112,101,108,1100,2.86%\n",
                encoding="utf-8",
            )

            with patch.object(data_sources, "yf", None), patch.object(
                data_sources, "TRAINING_MARKET_PATH", temp_path
            ), patch.object(
                data_sources, "ALLOW_LOCAL_MARKET_FALLBACK", True
            ):
                frame, status = data_sources.fetch_market_history()

        self.assertEqual(status, "using local historical market file")
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.iloc[-1]["Close"], 108.0)

    def test_build_live_dataset_copies_market_features(self) -> None:
        articles = pd.DataFrame(
            [
                {
                    "published_at": pd.Timestamp("2026-05-12"),
                    "text": "Gold reacts to inflation",
                }
            ]
        )
        market_features = {
            "Open": 1.0,
            "Prev_Close": 2.0,
            "Prev_Open": 3.0,
            "Prev_Change": 4.0,
            "MA_3": 5.0,
            "MA_7": 6.0,
            "STD_3": 7.0,
            "Year": 2026,
            "Month": 5,
            "Day": 12,
            "market_date": pd.Timestamp("2026-05-12"),
        }
        frame = data_sources.build_live_dataset(articles, market_features)
        self.assertEqual(frame.iloc[0]["Open"], 1.0)
        self.assertEqual(frame.iloc[0]["article_date"], "2026-05-12")


class ModelTestCase(unittest.TestCase):
    def test_predict_live_and_aggregate_snapshot(self) -> None:
        class FakeRegressor:
            def predict(self, frame: pd.DataFrame) -> np.ndarray:
                return np.array([[100.0, 1.5], [120.0, -0.5]])

        class FakeClassifier:
            def predict(self, frame: pd.DataFrame) -> np.ndarray:
                return np.array([1, -1])

        bundle = model.ModelBundle(
            regressor=FakeRegressor(),
            classifier=FakeClassifier(),
            feature_columns=model.FEATURE_COLUMNS,
            metrics={},
        )
        first_features = dict.fromkeys(model.FEATURE_COLUMNS, 1.0)
        second_features = dict.fromkeys(model.FEATURE_COLUMNS, 2.0)
        live_frame = pd.DataFrame(
            [
                {"News": "gold rises", **first_features},
                {"News": "gold falls", **second_features},
            ]
        )

        predicted = model.predict_live(bundle, live_frame)
        summary = model.aggregate_snapshot(predicted)

        self.assertEqual(list(predicted["Model Sentiment Label"]), ["Bullish", "Bearish"])
        self.assertAlmostEqual(summary["predicted_price"], 110.0)
        self.assertEqual(summary["bullish_count"], 1)
        self.assertEqual(summary["bearish_count"], 1)


class StorageTestCase(unittest.TestCase):
    def test_save_and_load_snapshot_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            db_path = temp_root / "gold.sqlite3"
            cache_path = temp_root / "latest_snapshot.json"

            with patch.object(storage, "ARTIFACT_DIR", temp_root), patch.object(storage, "DATABASE_PATH", db_path), patch.object(
                storage, "SNAPSHOT_CACHE_PATH", cache_path
            ):
                snapshot = {
                    "created_at": "2026-05-12T00:00:00+00:00",
                    "articles": [
                        {
                            "article_date": "2026-05-12",
                            "source": "Example",
                            "title": "Gold news",
                            "summary": "Test article",
                            "url": "https://example.com",
                            "predicted_price": 100.0,
                            "predicted_change": 1.0,
                            "predicted_sentiment": 1,
                            "model_sentiment": 1,
                        }
                    ],
                }

                storage.save_snapshot(snapshot)
                latest = storage.load_latest_snapshot()
                history = storage.load_snapshot_history(limit=5)

        self.assertIsNotNone(latest)
        self.assertEqual(latest["created_at"], snapshot["created_at"])
        self.assertEqual(len(history), 1)


class SnapshotBuilderTestCase(unittest.TestCase):
    def test_build_snapshot_wires_modules_together(self) -> None:
        fake_bundle = SimpleNamespace(metrics={"rmse_price": 1.0})
        news_frame = pd.DataFrame([{"text": "gold news", "published_at": pd.Timestamp("2026-05-12")}])
        market_frame = pd.DataFrame([{"Date": pd.Timestamp("2026-05-12"), "Close": 100.0, "Open": 99.0, "High": 101.0, "Low": 98.0}])
        market_features = {"history_tail": market_frame, "market_date": pd.Timestamp("2026-05-12")}
        live_frame = pd.DataFrame([{"News": "gold news"}])
        predicted_frame = pd.DataFrame([{"Predicted Price": 100.0, "Predicted Change %": 1.0, "Model Sentiment": 1}])
        summary = {"predicted_price": 100.0, "predicted_change_pct": 1.0, "sentiment_score": 1.0, "bullish_count": 1, "neutral_count": 0, "bearish_count": 0}

        with patch.object(snapshot_builder, "train_model", return_value=fake_bundle), patch.object(
            snapshot_builder, "fetch_live_news", return_value=(news_frame, "fetched 1 live articles")
        ), patch.object(snapshot_builder, "fetch_market_history", return_value=(market_frame, "fetched market history from XAUINR=X")), patch.object(
            snapshot_builder, "build_market_features", return_value=market_features
        ), patch.object(snapshot_builder, "build_live_dataset", return_value=live_frame), patch.object(
            snapshot_builder, "predict_live", return_value=predicted_frame
        ), patch.object(snapshot_builder, "aggregate_snapshot", return_value=summary), patch.object(
            snapshot_builder, "save_snapshot"
        ) as save_snapshot:
            snapshot = snapshot_builder.build_snapshot(force_retrain=False)

        self.assertEqual(snapshot["news_status"], "fetched 1 live articles")
        self.assertEqual(snapshot["summary"]["predicted_price"], 100.0)
        save_snapshot.assert_called_once()


class BackendTestCase(unittest.TestCase):
    def test_backend_snapshot_endpoints_are_wired(self) -> None:
        snapshot = {"created_at": "2026-05-12T00:00:00+00:00", "summary": {"predicted_price": 1.0}}

        with patch.object(backend, "load_latest_snapshot", return_value=snapshot), patch.object(
            backend, "build_snapshot", return_value=snapshot
        ), patch.object(backend, "load_snapshot_history", return_value=pd.DataFrame([{"created_at": "2026-05-12T00:00:00+00:00", "payload": {"summary": {"predicted_price": 1.0}}}])), patch.object(
            backend, "_create_background_task", return_value=None
        ):
            self.assertEqual(backend.health(), {"status": "ok"})
            self.assertEqual(backend.get_latest_snapshot()["created_at"], snapshot["created_at"])
            self.assertEqual(backend.get_live_snapshot()["created_at"], snapshot["created_at"])
            self.assertEqual(backend.refresh_snapshot(backend.RefreshRequest()).get("created_at"), snapshot["created_at"])
            history = backend.get_snapshot_history(limit=1)

        self.assertEqual(history["items"][0]["predicted_price"], 1.0)


if __name__ == "__main__":
    unittest.main()
