# Gold Forecast Web App

This project turns the notebook prototype into a runnable web app that:

- automatically fetches gold-related news from RSS feeds,
- pulls live market data with a fallback to local CSV history,
- trains and loads a reusable prediction bundle,
- shows live analytics and article-level predictions in a Streamlit dashboard,
- stores snapshots for historical trend tracking.

## Run locally

```bash
pip install -r requirements.txt
uvicorn gold_app.backend:app --host 0.0.0.0 --port 8000
streamlit run streamlit_app.py
```

The dashboard is frontend-only now. It calls the FastAPI service for predictions and snapshot data.
It also opens a WebSocket connection for instant push updates when a new snapshot is published.

If your backend runs on a different URL:

```bash
set GOLD_BACKEND_URL=http://127.0.0.1:8000
streamlit run streamlit_app.py
```

## Refresh snapshots automatically

Run the background refresh script manually or schedule it with Windows Task Scheduler:

```bash
python scripts/refresh_snapshot.py
```

## API endpoints

- GET `/health`
- GET `/api/v1/snapshot/latest`
- GET `/api/v1/snapshot/live?force_retrain=false`
- POST `/api/v1/snapshot/refresh`
- GET `/api/v1/snapshot/history?limit=30`
- WS `/ws/snapshots`

## Live push updates

- Backend pushes snapshot events over `/ws/snapshots`.
- Streamlit listens to this WebSocket and reloads immediately when a new snapshot ID arrives.
- Configure push interval with `GOLD_PUSH_INTERVAL_SECONDS` (default `60`, minimum `15`).

## Files used

- `gold-dataset-sinha-khandait.csv` for historical news-label training
- `XAU_INR Historical Data.csv` for historical price training
- `test_headlines.md` as an offline fallback headline source
