# 📊 FinServ Market Dashboard & Swing Signal Bot

A web app that analyzes the **Nasdaq-100 (NQ)** and **S&P 500 (ES)** futures and their
heavyweight component stocks using real daily market data, detects four families of
technical trading patterns, and produces a clear **LONG / SHORT / NEUTRAL** swing-trade
signal for each — with the reasoning behind every call and a historical backtest.

> ⚠️ **Not financial advice.** Signals are alerts to review, not orders. Futures are
> leveraged and can lose money quickly; a backtest is not a promise about the future.
> This app does not connect to any brokerage and does not place trades.

## 🔗 Live app

- **Railway (primary):** _add your Railway URL here after deploy_
- **Streamlit Community Cloud (mirror):** https://finserv-swing-bot.streamlit.app

## What it does

The app is organized into tabs:

- **Overview** — ranked view of all instruments (best-to-worst signal score), key metrics, and a market heatmap.
- **🤖 Signal Bot** — focused LONG/SHORT/NEUTRAL calls for NQ and ES, confirmed by leader-stock breadth, with plain-English reasons.
- **Finder** — pick any instrument to see its candlestick chart, moving averages, signal, and the pattern breakdown behind it.
- **Analytics** — average signal score by sector and summary statistics.
- **Backtesting** — run the strategy over real history and see win rate, total return, trade count, and an equity curve.
- **Reports** — the full scored table with CSV export.
- **About** — methodology and disclosures.

## How the signal engine works

Each instrument is scored on four independent pattern families:

1. **Trend** — 20-day vs 50-day EMA position and golden/death crossovers.
2. **Momentum** — RSI (overbought/oversold) and MACD crossovers.
3. **Breakout** — closing above a 20-day high (breakout) or below a 20-day low (breakdown).
4. **Candlesticks** — bullish/bearish engulfing, hammer, and shooting-star formations.

For the **index futures**, the score is further confirmed by **leader-stock breadth** —
how many of the index's biggest component stocks (e.g. AAPL, MSFT, NVDA for the Nasdaq)
are currently in an uptrend. The idea: an index move backed by most of its leaders is
more durable than one carried by a couple of names. The weighted blend yields the final
signal and a strength score (0–100%).

## Tech stack

Python · Streamlit (UI) · yfinance (real market data) · Plotly (charts) · Pandas / NumPy (analysis).

## Repository structure

```
├── streamlit_app.py    # the app UI (all tabs)
├── signal_engine.py    # data fetching, indicators, pattern detection, scoring, backtest
├── test_engine.py      # correctness tests for every pattern detector
├── requirements.txt    # dependencies
├── Procfile            # start command for Railway
├── runtime.txt         # Python version
└── .streamlit/config.toml
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Then open http://localhost:8501.

## Deploy

**Railway:** connect this GitHub repo to a new Railway project. Railway reads the
`Procfile` and starts the app on its assigned `$PORT`. No extra configuration needed.

**Streamlit Community Cloud:** point a new app at this repo with `streamlit_app.py` as
the main file.

## Testing

The pattern detectors are verified against hand-crafted price sequences with known
outcomes:

```bash
python test_engine.py
```
All 15 checks pass (trend crossovers, RSI extremes, breakouts, and each candlestick pattern).

## Data & reliability

Prices come from `yfinance` (free, end-of-day). If the host briefly can't reach the data
feed, the app falls back to clearly-labelled demo data rather than crashing, so it stays
up during market days.

---

*Built for the IST 495 internship (Stock Market Project), Penn State — College of
Information Sciences and Technology.*
