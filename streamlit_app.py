from __future__ import annotations

import os
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh


BACKEND_URL = os.getenv("GOLD_BACKEND_URL", "http://127.0.0.1:8001")
AUTO_REFRESH_MS = 15 * 1000
REQUEST_TIMEOUT = 120
PREDICTED_PRICE_COL = "Predicted Price"
PREDICTED_CHANGE_COL = "Predicted Change %"
MODEL_SENTIMENT_LABEL_COL = "Model Sentiment Label"
PLOT_TRANSPARENT = "rgba(0,0,0,0)"
PLOT_GRID = "rgba(180,155,80,0.10)"
PLOT_AXIS = "rgba(180,155,80,0.20)"

st.set_page_config(page_title="Gold Forecast Dashboard", page_icon="🪙", layout="wide")

INIT_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Space Grotesk', sans-serif;
    }

    /* ─── CSS Variables: auto light / dark ─── */
    :root {
        --gold-primary:   #C9941A;
        --gold-light:     #F5D97A;
        --gold-muted:     #A07010;
        --gold-faint:     #FFF8E6;

        --surface:        #FFFFFF;
        --surface-raised: #FAFAF7;
        --surface-border: rgba(180,155,80,0.22);

        --text-primary:   #1A1410;
        --text-secondary: #5C4E38;
        --text-muted:     #8C7A60;

        --badge-bull-bg:  #E6F5EC;
        --badge-bull-fg:  #1A6636;
        --badge-bear-bg:  #FDEEEE;
        --badge-bear-fg:  #971F1F;
        --badge-neut-bg:  #F0EDE6;
        --badge-neut-fg:  #5C4E38;

        --chart-gold:     #C9941A;
        --chart-dark:     #3A2B12;
        --chart-grid:     rgba(180,155,80,0.12);

        --hero-bg-start:  #1C1712;
        --hero-bg-mid:    #3A2B12;
        --hero-bg-end:    #7A5A17;
        --hero-text:      #FFF8E6;
        --hero-sub:       #F0D898;
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --gold-primary:   #E8B84B;
            --gold-light:     #FFE08A;
            --gold-muted:     #C9941A;
            --gold-faint:     #2A2010;

            --surface:        #1E1A14;
            --surface-raised: #252018;
            --surface-border: rgba(200,160,60,0.20);

            --text-primary:   #F5EDD8;
            --text-secondary: #C8AA78;
            --text-muted:     #8C7A60;

            --badge-bull-bg:  #0E3020;
            --badge-bull-fg:  #66D48A;
            --badge-bear-bg:  #2E1010;
            --badge-bear-fg:  #F09090;
            --badge-neut-bg:  #252018;
            --badge-neut-fg:  #C8AA78;

            --chart-gold:     #E8B84B;
            --chart-dark:     #F5EDD8;
            --chart-grid:     rgba(200,160,60,0.10);

            --hero-bg-start:  #0E0C08;
            --hero-bg-mid:    #1C1408;
            --hero-bg-end:    #3A2B0C;
            --hero-text:      #FFF8E6;
            --hero-sub:       #E8C878;
        }
    }

    /* ─── Hero Banner ─── */
    .hero {
        background: linear-gradient(135deg, var(--hero-bg-start) 0%, var(--hero-bg-mid) 55%, var(--hero-bg-end) 100%);
        color: var(--hero-text);
        padding: 28px 32px;
        border-radius: 20px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -40px; right: -40px;
        width: 200px; height: 200px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(200,160,60,0.18) 0%, transparent 70%);
    }
    .hero-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--gold-light);
        margin-bottom: 8px;
    }
    .hero h1 {
        margin: 0 0 6px 0;
        font-size: 2rem;
        font-weight: 700;
        color: var(--hero-text);
        letter-spacing: -0.01em;
    }
    .hero p {
        margin: 0;
        color: var(--hero-sub);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
    }

    /* ─── Metric Cards ─── */
    .metric-card {
        background: var(--surface);
        border: 1px solid var(--surface-border);
        border-radius: 16px;
        padding: 20px 22px;
        height: 100%;
        box-sizing: border-box;
        position: relative;
        overflow: hidden;
    }
    .metric-card::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--gold-primary), var(--gold-light));
        opacity: 0.7;
        border-radius: 0 0 16px 16px;
    }
    .metric-label {
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        font-size: 0.68rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: var(--hero-text);
        letter-spacing: -0.02em;
        line-height: 1.1;
        margin-bottom: 8px;
    }
    .metric-value.positive { color: #1A8040; }
    .metric-value.negative { color: #B02020; }
    .metric-value.gold     { color: var(--gold-primary); }

    @media (prefers-color-scheme: dark) {
        .metric-value.positive { color: #55CC80; }
        .metric-value.negative { color: #E87070; }
    }

    .metric-note {
        color: var(--text-muted);
        font-size: 0.78rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ─── Status Card ─── */
    .status-card {
        background: linear-gradient(180deg, rgba(20, 23, 31, 0.98) 0%, rgba(14, 17, 24, 0.98) 100%);
        border: 1px solid var(--surface-border);
        border-radius: 16px;
        padding: 20px 22px;
        height: 100%;
        box-sizing: border-box;
    }
    .status-card h4 {
        margin: 0 0 14px 0;
        font-size: 0.68rem;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
    }
    .status-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 7px 0;
        border-bottom: 1px solid var(--surface-border);
        font-size: 0.84rem;
    }
    .status-row:last-child { border-bottom: none; }
    .status-row .key {
        color: var(--hero-sub);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
    }
    .status-row .val {
        color: var(--hero-text);
        font-weight: 600;
        font-size: 0.84rem;
    }

    .stCaption, .stMarkdown, .stText, .stAlert, .stInfo, .stWarning, .stError {
        color: var(--hero-text);
    }

    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--surface-border);
    }

    /* ─── Section Headers ─── */
    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        letter-spacing: -0.01em;
        margin: 28px 0 14px 0;
        padding-left: 12px;
        border-left: 3px solid var(--gold-primary);
    }

    /* ─── Sentiment Badges ─── */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.04em;
    }
    .badge-bull { background: var(--badge-bull-bg); color: var(--badge-bull-fg); }
    .badge-bear { background: var(--badge-bear-bg); color: var(--badge-bear-fg); }
    .badge-neut { background: var(--badge-neut-bg); color: var(--badge-neut-fg); }

    /* ─── Sidebar ─── */
    .sidebar-header {
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin-bottom: 8px;
    }

    /* ─── Global Streamlit overrides ─── */
    .stDataFrame { border-radius: 12px; overflow: hidden; }
    div[data-testid="metric-container"] { display: none; }
</style>
"""

st.markdown(INIT_CSS, unsafe_allow_html=True)


def _websocket_url(base_url: str) -> str:
    if base_url.startswith("https://"):
        return f"wss://{base_url[len('https://'):]}"
    if base_url.startswith("http://"):
        return f"ws://{base_url[len('http://'):]}"
    return f"ws://{base_url}"


def _endpoint(path: str) -> str:
    return f"{BACKEND_URL}{path}"


@st.cache_data(ttl=10, show_spinner=False)
def fetch_latest_snapshot() -> dict[str, Any]:
    response = requests.get(_endpoint("/api/v1/snapshot/latest"), timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=20, show_spinner=False)
def fetch_snapshot_history(limit: int = 30) -> list[dict[str, Any]]:
    response = requests.get(
        _endpoint("/api/v1/snapshot/history"),
        params={"limit": limit},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("items", [])


def trigger_refresh(force_retrain: bool = False) -> dict[str, Any]:
    response = requests.post(
        _endpoint("/api/v1/snapshot/refresh"),
        json={"force_retrain": force_retrain},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    fetch_latest_snapshot.clear()
    fetch_snapshot_history.clear()
    return response.json()


def get_backend_status() -> tuple[bool, str]:
    try:
        response = requests.get(_endpoint("/health"), timeout=20)
        response.raise_for_status()
    except Exception as exc:
        return False, f"Backend not reachable at {BACKEND_URL}: {exc}"
    return True, "ok"


# ─── Backend health check ─────────────────────────────────────────────────────
backend_ok, backend_message = get_backend_status()
if not backend_ok:
    st.error(backend_message)
    st.info("Start API server: uvicorn gold_app.backend:app --host 0.0.0.0 --port 8001")
    st.stop()

snapshot = fetch_latest_snapshot()
summary = snapshot.get("summary", {})
market_features = snapshot.get("market_features", {})
model_metrics = snapshot.get("model_metrics", {})
news_status = snapshot.get("news_status", "unknown")
market_status = snapshot.get("market_status", "unknown")
current_snapshot_id = str(snapshot.get("created_at", ""))

# ─── WebSocket live-update badge ─────────────────────────────────────────────
ws_base = _websocket_url(BACKEND_URL)
components.html(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500&display=swap');
        .ws-wrap {{
            display: flex;
            justify-content: flex-end;
            padding: 0;
            margin: 0;
        }}
        .ws-badge {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            font-weight: 500;
            padding: 5px 12px;
            border-radius: 999px;
            border: 1px solid;
            letter-spacing: 0.04em;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .ws-badge::before {{
            content: '';
            width: 7px;
            height: 7px;
            border-radius: 50%;
            flex-shrink: 0;
        }}
        .ws-connected    {{ background: #E6F5EC; color: #1A6636; border-color: #A8D8BC; }}
        .ws-connected::before {{ background: #1A8040; }}
        .ws-reconnecting {{ background: #FFF6E0; color: #8A5A00; border-color: #E8C878; }}
        .ws-reconnecting::before {{ background: #C9941A; animation: pulse 1s infinite; }}
        .ws-disconnected {{ background: #FDEEEE; color: #971F1F; border-color: #F0A8A8; }}
        .ws-disconnected::before {{ background: #B02020; }}
        @media (prefers-color-scheme: dark) {{
            .ws-connected    {{ background: #0E3020; color: #66D48A; border-color: #255530; }}
            .ws-reconnecting {{ background: #2A1E08; color: #E8C878; border-color: #5A3A10; }}
            .ws-disconnected {{ background: #2E1010; color: #F09090; border-color: #601818; }}
        }}
        @keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.3; }} }}
    </style>
    <div class="ws-wrap">
        <div id="ws-badge" class="ws-badge ws-reconnecting">Live: Reconnecting…</div>
    </div>
    <script>
    (function() {{
        const currentId = {current_snapshot_id!r};
        const socketUrl = {f"{ws_base}/ws/snapshots"!r};
        const badge = document.getElementById("ws-badge");
        let socket = null;

        function setStatus(kind, text) {{
            badge.className = "ws-badge " + kind;
            badge.textContent = text;
        }}

        function connect() {{
            setStatus("ws-reconnecting", "Live: Reconnecting…");
            socket = new WebSocket(socketUrl);
            socket.onopen = function() {{
                setStatus("ws-connected", "Live: Connected");
                socket.send("subscribe");
            }};
            socket.onmessage = function(event) {{
                try {{
                    const payload = JSON.parse(event.data);
                    const nextId = payload && payload.created_at ? String(payload.created_at) : "";
                    if (nextId && nextId !== currentId) {{
                        window.parent.location.reload();
                    }}
                }} catch (e) {{}}
            }};
            socket.onclose = function() {{
                setStatus("ws-disconnected", "Live: Disconnected");
                setTimeout(connect, 1500);
            }};
            socket.onerror = function() {{
                try {{ socket.close(); }} catch (e) {{}}
            }};
        }}
        connect();
    }})();
    </script>
    """,
    height=36,
)

articles = pd.DataFrame(snapshot.get("articles", []))
history_data = pd.DataFrame(snapshot.get("history", []))

if articles.empty:
    st.warning("No article-level predictions are available yet.")


# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <div class="hero-label">🪙 Market Intelligence</div>
        <h1>Gold Forecast Dashboard</h1>
        <p>Predictions and analytics served by the FastAPI backend — live via WebSocket</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─── KPI Cards ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

latest_close = float(market_features.get("latest_close", 0.0))
pred_price   = float(summary.get("predicted_price", 0.0))
pred_chg     = float(summary.get("predicted_change_pct", 0.0))
sentiment_score = float(summary.get("sentiment_score", 0.0))
bullish_count   = int(summary.get("bullish_count", 0))
bearish_count   = int(summary.get("bearish_count", 0))

market_asof = None
try:
    market_asof = pd.Timestamp(
        year=int(market_features.get("Year", 0)),
        month=int(market_features.get("Month", 0)),
        day=int(market_features.get("Day", 0)),
    )
except Exception:
    market_asof = None

market_note = market_status
if market_asof is not None and not pd.isna(market_asof):
    market_note = f"{market_status} | as of {market_asof.date().isoformat()}"
    age_days = (pd.Timestamp.utcnow().tz_localize(None).date() - market_asof.date()).days
    if age_days > 7:
        market_note += " | historical snapshot"

if sentiment_score > 0:
    sentiment_label = "Bullish"
    sentiment_badge = "badge-bull"
elif sentiment_score < 0:
    sentiment_label = "Bearish"
    sentiment_badge = "badge-bear"
else:
    sentiment_label = "Neutral"
    sentiment_badge = "badge-neut"

chg_class = "positive" if pred_chg >= 0 else "negative"
chg_sign  = "+" if pred_chg >= 0 else ""

with c1:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">Latest Market Close</div>
            <div class="metric-value gold">₹{latest_close:,.2f}</div>
            <div class="metric-note">{market_note}</div>
        </div>""",
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">Predicted Avg Price</div>
            <div class="metric-value">₹{pred_price:,.2f}</div>
            <div class="metric-note">{len(articles)} articles sampled</div>
        </div>""",
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">Predicted Change</div>
            <div class="metric-value {chg_class}">{chg_sign}{pred_chg:.4f}%</div>
            <div class="metric-note">Model-driven aggregate</div>
        </div>""",
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">Sentiment</div>
            <div class="metric-value" style="font-size:1.4rem; margin-bottom:12px;">
                <span class="badge {sentiment_badge}">{sentiment_label}</span>
            </div>
            <div class="metric-note">🟢 {bullish_count} bull &nbsp;|&nbsp; 🔴 {bearish_count} bear</div>
        </div>""",
        unsafe_allow_html=True,
    )


# ─── Live Analytics ───────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Live Analytics</div>', unsafe_allow_html=True)

chart_col, status_col = st.columns([1.2, 0.8])

with chart_col:
    if not history_data.empty and "Date" in history_data.columns:
        cf = history_data.copy()
        cf["Date"]  = pd.to_datetime(cf["Date"], errors="coerce")
        cf["Close"] = pd.to_numeric(cf["Close"], errors="coerce")
        cf = cf.dropna(subset=["Date", "Close"]).tail(180)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cf["Date"], y=cf["Close"],
            name="Market Close",
            line={"color": "#C9941A", "width": 2.5},
            fill="tozeroy",
            fillcolor="rgba(201,148,26,0.06)",
        ))
        fig.add_trace(go.Scatter(
            x=cf["Date"], y=cf["Close"].rolling(7).mean(),
            name="7-Day MA",
            line={"color": "#4A90A4", "width": 2, "dash": "dot"},
        ))
        fig.update_layout(
            height=360,
            margin={"l": 8, "r": 8, "t": 28, "b": 8},
            paper_bgcolor=PLOT_TRANSPARENT,
            plot_bgcolor=PLOT_TRANSPARENT,
            font={"family": "Space Grotesk, sans-serif", "color": "#8C7A60"},
            title={"text": "Market Trend", "font": {"size": 14, "color": "#C9941A", "family": "Space Grotesk"}},
            xaxis={
                "gridcolor": PLOT_GRID,
                "linecolor": PLOT_AXIS,
                "tickcolor": PLOT_TRANSPARENT,
                "tickfont": {"size": 11},
            },
            yaxis={
                "gridcolor": PLOT_GRID,
                "linecolor": PLOT_AXIS,
                "tickprefix": "₹",
                "tickfont": {"size": 11},
            },
            legend={"bgcolor": PLOT_TRANSPARENT, "font": {"size": 12}},
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Market history is not available yet.")

with status_col:
    rmse_p  = float(model_metrics.get("rmse_price", 0.0))
    r2_p    = float(model_metrics.get("r2_price", 0.0))
    sent_ac = float(model_metrics.get("sentiment_accuracy", 0.0))
    snap_ts = snapshot.get("created_at", "—")

    st.markdown(
        f"""<div class="status-card">
            <h4>Current Data Status</h4>
            <div class="status-row">
                <span class="key">News feed</span>
                <span class="val">{news_status}</span>
            </div>
            <div class="status-row">
                <span class="key">Market feed</span>
                <span class="val">{market_status}</span>
            </div>
            <div class="status-row">
                <span class="key">RMSE (price)</span>
                <span class="val">{rmse_p:.2f}</span>
            </div>
            <div class="status-row">
                <span class="key">R² (price)</span>
                <span class="val">{r2_p:.4f}</span>
            </div>
            <div class="status-row">
                <span class="key">Sentiment acc.</span>
                <span class="val">{sent_ac:.2%}</span>
            </div>
            <div class="status-row">
                <span class="key">Snapshot at</span>
                <span class="val" style="font-size:0.74rem;">{snap_ts}</span>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


# ─── Latest Reports ───────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Latest Reports</div>', unsafe_allow_html=True)

if not articles.empty:
    rf = articles.copy()
    rf["published_at"] = pd.to_datetime(rf.get("published_at"), errors="coerce")
    rf = rf.sort_values("published_at", ascending=False)

    st.dataframe(
        rf[[
            "published_at", "source", "title",
            PREDICTED_PRICE_COL, PREDICTED_CHANGE_COL,
            MODEL_SENTIMENT_LABEL_COL, "url",
        ]].rename(columns={
            "published_at": "Published At",
            "source": "Source",
            "title": "Headline",
            PREDICTED_PRICE_COL: "Pred. Price",
            PREDICTED_CHANGE_COL: "Chg %",
            MODEL_SENTIMENT_LABEL_COL: "Sentiment",
            "url": "URL",
        }),
        use_container_width=True,
        height=400,
    )


# ─── Prediction Breakdown ─────────────────────────────────────────────────────
st.markdown('<div class="section-title">Prediction Breakdown</div>', unsafe_allow_html=True)

pie_col, bar_col = st.columns(2)

BULL_COLOR  = "#22A855"
BEAR_COLOR  = "#E24B4A"
NEUT_COLOR  = "#8C7A60"
GOLD_COLOR  = "#C9941A"
DARK_COLOR  = "#4A90A4"

_common_layout = {
    "paper_bgcolor": PLOT_TRANSPARENT,
    "plot_bgcolor": PLOT_TRANSPARENT,
    "font": {"family": "Space Grotesk, sans-serif", "color": "#8C7A60"},
    "margin": {"l": 8, "r": 8, "t": 36, "b": 8},
    "height": 360,
}

with pie_col:
    if not articles.empty:
        counts = (
            articles[MODEL_SENTIMENT_LABEL_COL]
            .value_counts()
            .reindex(["Bullish", "Neutral", "Bearish"])
            .fillna(0)
        )
        pie_fig = go.Figure(go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            hole=0.58,
            marker={"colors": [BULL_COLOR, NEUT_COLOR, BEAR_COLOR]},
            textfont={"size": 13, "family": "Space Grotesk"},
            hovertemplate="%{label}: %{value}<extra></extra>",
        ))
        pie_fig.update_layout(
            **_common_layout,
            title={"text": "Sentiment Split", "font": {"size": 14, "color": GOLD_COLOR}},
            legend={"bgcolor": PLOT_TRANSPARENT, "font": {"size": 13}},
        )
        st.plotly_chart(pie_fig, use_container_width=True)

with bar_col:
    if not articles.empty:
        cmp = (
            articles[["title", PREDICTED_CHANGE_COL, MODEL_SENTIMENT_LABEL_COL]]
            .head(10)
            .copy()
            .sort_values(PREDICTED_CHANGE_COL, ascending=True)
        )
        bar_colors = [BEAR_COLOR if v < 0 else BULL_COLOR for v in cmp[PREDICTED_CHANGE_COL]]

        # Truncate long titles
        cmp["short_title"] = cmp["title"].str[:46] + "…"

        bar_fig = go.Figure(go.Bar(
            x=cmp[PREDICTED_CHANGE_COL],
            y=cmp["short_title"],
            orientation="h",
            marker_color=bar_colors,
            hovertemplate="%{y}<br>Change: %{x:.4f}%<extra></extra>",
        ))
        bar_fig.update_layout(
            **_common_layout,
            title={"text": "Top Prediction Signals", "font": {"size": 14, "color": GOLD_COLOR}},
            xaxis={
                "gridcolor": PLOT_GRID,
                "ticksuffix": "%",
                "tickfont": {"size": 11},
                "linecolor": PLOT_AXIS,
            },
            yaxis={
                "tickfont": {"size": 10},
                "automargin": True,
            },
        )
        st.plotly_chart(bar_fig, use_container_width=True)


# ─── Historical Snapshot Store ────────────────────────────────────────────────
st.markdown('<div class="section-title">Historical Snapshot Store</div>', unsafe_allow_html=True)

backend_history = fetch_snapshot_history(limit=30)
history_frame = pd.DataFrame(backend_history)

if not history_frame.empty:
    history_frame["created_at"]      = pd.to_datetime(history_frame["created_at"], errors="coerce")
    history_frame["predicted_price"] = pd.to_numeric(history_frame["predicted_price"], errors="coerce")
    history_frame = history_frame.sort_values("created_at")

    hfig = go.Figure()
    hfig.add_trace(go.Scatter(
        x=history_frame["created_at"],
        y=history_frame["predicted_price"],
        name="Predicted Price",
        line={"color": GOLD_COLOR, "width": 2.5},
        mode="lines+markers",
        marker={"size": 6, "color": GOLD_COLOR, "line": {"width": 1.5, "color": "#FFF8E6"}},
        hovertemplate="₹%{y:,.2f}<extra></extra>",
    ))
    hfig.update_layout(
        **_common_layout,
        title={"text": "Stored Snapshot Trend", "font": {"size": 14, "color": GOLD_COLOR}},
        xaxis={
                "gridcolor": PLOT_GRID,
                "linecolor": PLOT_AXIS,
            "tickfont": {"size": 11},
        },
        yaxis={
                "gridcolor": PLOT_GRID,
            "tickprefix": "₹",
            "tickfont": {"size": 11},
        },
    )
    st.plotly_chart(hfig, use_container_width=True)
else:
    st.caption("No snapshot history stored yet. Use refresh to generate snapshots.")


# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.markdown("### 🪙 Controls")

if st.sidebar.button("⟳ Refresh Snapshot", use_container_width=True):
    with st.spinner("Refreshing snapshot from backend…"):
        trigger_refresh(force_retrain=False)
    st.rerun()

if st.sidebar.button("⟳ Refresh & Retrain Model", use_container_width=True):
    with st.spinner("Refreshing snapshot and forcing retrain…"):
        trigger_refresh(force_retrain=True)
    st.rerun()

st.sidebar.divider()
st.sidebar.markdown("**Backend**")
st.sidebar.code(BACKEND_URL, language=None)
st.sidebar.caption("Set `GOLD_BACKEND_URL` env var if FastAPI is on a different host.")