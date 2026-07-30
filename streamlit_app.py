"""FinServ Market Dashboard + Swing Signal Bot — combined single app.

Two things in one:
  1. A market dashboard: ranked view of index futures and their leader stocks,
     with a finder, analytics, backtesting and reports.
  2. A swing signal bot: focused LONG/SHORT/NEUTRAL calls for NQ and ES, confirmed
     by leader-stock breadth, with plain-English reasons.

Real daily data via yfinance (falls back to clearly-labelled demo data if the host
has no internet). NOT financial advice. Futures are leveraged and risky. Signals are
alerts you act on, not orders. No broker connection, no auto-trading.
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import signal_engine as se

st.set_page_config(page_title="FinServ Market Dashboard", layout="wide", page_icon="📊")

# ----- universe -----
SECTORS = {
    "NQ=F": "Index Future", "ES=F": "Index Future",
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOGL": "Technology", "META": "Technology", "AVGO": "Technology",
    "AMZN": "Consumer", "TSLA": "Consumer",
    "BRK-B": "Financials", "JPM": "Financials",
}
FUTURE_TICKERS = list(se.FUTURES.keys())
STOCK_TICKERS = sorted({t for m in se.FUTURES.values() for t in m["components"]})
ALL_TICKERS = FUTURE_TICKERS + STOCK_TICKERS


@st.cache_data(ttl=3600, show_spinner=False)
def load(ticker, period):
    return se.fetch_daily(ticker, period)


def demo_df(seed, n=400, drift=0.0006):
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, 0.012, n)
    c = 100 * np.cumprod(1 + r)
    idx = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="B")
    return pd.DataFrame({
        "open": c * (1 + rng.normal(0, 0.004, n)),
        "high": c * (1 + np.abs(rng.normal(0, 0.006, n))),
        "low": c * (1 - np.abs(rng.normal(0, 0.006, n))),
        "close": c, "volume": rng.integers(1_000_000, 20_000_000, n)}, index=idx)


def get_all_data(period):
    """Return (data dict, demo_mode bool). One fetch per unique ticker."""
    data, demo = {}, False
    try:
        for t in ALL_TICKERS:
            data[t] = load(t, period)
    except Exception:
        demo = True
        data = {t: demo_df(abs(hash(t)) % 9999,
                           drift=0.0009 if t in FUTURE_TICKERS else 0.0006)
                for t in ALL_TICKERS}
    return data, demo


def components_for(fut, data):
    return {t: data[t] for t in se.FUTURES[fut]["components"] if t in data}


def build_ranked(data):
    rows = []
    for t in ALL_TICKERS:
        df = data[t]
        comps = components_for(t, data) if t in FUTURE_TICKERS else None
        res = se.evaluate(df, comps)
        signal = res["label"].split(" ")[0]  # LONG / SHORT / NEUTRAL
        rows.append({
            "ticker": t,
            "kind": "Future" if t in FUTURE_TICKERS else "Stock",
            "sector": SECTORS.get(t, "Other"),
            "last_close": res["last_close"],
            "score": res["score"],
            "strength": res["strength"],
            "signal": signal,
            "volume": int(df["volume"].iloc[-1]),
        })
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


SIG_COLOR = {"LONG": "#199e70", "SHORT": "#e66767", "NEUTRAL": "#8a897e"}

# ============================ load =====================================
with st.sidebar:
    st.header("Settings")
    period = st.selectbox("History window", ["1y", "2y", "5y"], index=1)
    st.markdown("---")
    st.markdown("**Signal key**\n\n- **LONG** = bullish\n- **SHORT** = bearish\n"
                "- **NEUTRAL** = stand aside\n\n*Strength* = how lopsided the patterns are.")
    st.markdown("---")
    st.caption("⚠️ Not financial advice. Leveraged futures can lose money fast. "
               "Backtest ≠ future results. This tool does not place trades.")

st.title("📊 FinServ Market Dashboard")
st.caption("Real market data · pattern-based swing signals for index futures and their leader stocks.")

data, demo_mode = get_all_data(period)
ranked = build_ranked(data)

if demo_mode:
    st.warning("Couldn't reach the live data feed from this host — showing **DEMO (simulated) "
               "data**. Runs on real market data automatically when deployed with internet.")
else:
    st.success(f"Live data loaded for {len(ALL_TICKERS)} instruments "
               f"({len(FUTURE_TICKERS)} futures, {len(STOCK_TICKERS)} leader stocks).")

overview, bot, finder, analytics, backtesting, reports, about = st.tabs(
    ["Overview", "🤖 Signal Bot", "Finder", "Analytics", "Backtesting", "Reports", "About"])

# ---------------------------- Overview --------------------------------
with overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Instruments", len(ranked))
    c2.metric("LONG signals", int((ranked["signal"] == "LONG").sum()))
    c3.metric("SHORT signals", int((ranked["signal"] == "SHORT").sum()))
    c4.metric("Avg strength", f"{ranked['strength'].mean():.0f}%")

    st.subheader("Top Opportunities")
    st.dataframe(ranked[["ticker", "kind", "sector", "last_close", "signal", "strength", "score"]],
                 use_container_width=True, hide_index=True)

    st.subheader("Market Heatmap")
    st.caption("Area = latest volume · color = signal score (red bearish → green bullish).")
    fig = px.treemap(ranked, path=[px.Constant("Market"), "sector", "ticker"],
                     values="volume", color="score",
                     color_continuous_scale=[[0, "#e66767"], [0.5, "#6b6a63"], [1, "#199e70"]],
                     color_continuous_midpoint=0)
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------- Signal Bot ------------------------------
with bot:
    st.subheader("🤖 Swing Signal Bot — NQ & ES")
    st.caption("Focused futures calls, confirmed by how many leader stocks are trending.")
    cols = st.columns(len(FUTURE_TICKERS))
    for col, fut in zip(cols, FUTURE_TICKERS):
        res = se.evaluate(data[fut], components_for(fut, data))
        sig = res["label"].split(" ")[0]
        with col:
            st.markdown(f"### {fut}")
            st.caption(se.FUTURES[fut]["name"])
            st.markdown(f"<div style='font-size:26px;font-weight:700;color:{SIG_COLOR.get(sig)}'>"
                        f"{res['label']}</div>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Strength", f"{res['strength']:.0f}%")
            m2.metric("Last close", res["last_close"])
            if res["breadth_pct"] is not None:
                st.metric("Leader stocks bullish", f"{res['breadth_pct']:.0f}%")
            st.markdown("**Why:**")
            for r in res["reasons"]:
                st.markdown(f"- {r}")

# ---------------------------- Finder ----------------------------------
with finder:
    st.subheader("🔎 Stock & Futures Finder")
    ticker = st.selectbox("Pick an instrument", ALL_TICKERS)
    df = data[ticker]
    comps = components_for(ticker, data) if ticker in FUTURE_TICKERS else None
    res = se.evaluate(df, comps)
    sig = res["label"].split(" ")[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Signal", res["label"])
    c2.metric("Strength", f"{res['strength']:.0f}%")
    c3.metric("Last close", res["last_close"])
    ema20, ema50 = se.ema(df["close"], 20), se.ema(df["close"], 50)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"],
                                 close=df["close"], name="price",
                                 increasing_line_color="#199e70", decreasing_line_color="#e66767"))
    fig.add_trace(go.Scatter(x=df.index, y=ema20, name="EMA20", line=dict(color="#3987e5", width=1.3)))
    fig.add_trace(go.Scatter(x=df.index, y=ema50, name="EMA50", line=dict(color="#c98500", width=1.3)))
    fig.update_layout(height=430, xaxis_rangeslider_visible=False, title=f"{ticker} (real data)",
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Why this signal:**")
    for r in res["reasons"]:
        st.markdown(f"- {r}")
    st.json(res["votes"])

# ---------------------------- Analytics -------------------------------
with analytics:
    st.subheader("📈 Analytics")
    sp = ranked.groupby("sector")["score"].mean().reset_index().sort_values("score")
    bar = go.Figure(go.Bar(x=sp["score"], y=sp["sector"], orientation="h",
                           marker=dict(color=["#199e70" if v >= 0 else "#e66767" for v in sp["score"]])))
    bar.update_layout(height=300, title="Average signal score by sector", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(bar, use_container_width=True)
    st.subheader("Statistics")
    st.dataframe(ranked[["last_close", "score", "strength", "volume"]].describe(),
                 use_container_width=True)

# ---------------------------- Backtesting -----------------------------
with backtesting:
    st.subheader("🧪 Backtesting")
    st.caption("Aggregate signal, held until it flips (max 20 days). Illustrative only — not advice.")
    bt_ticker = st.selectbox("Instrument to backtest", ALL_TICKERS, key="bt")
    bt = se.backtest(data[bt_ticker])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total return", f"{bt['total_return_pct']}%")
    c2.metric("Win rate", f"{bt['win_rate']}%")
    c3.metric("Trades", bt["trades"])
    c4.metric("Avg / worst", f"{bt['avg_trade_pct']}% / {bt['worst_trade_pct']}%")
    eq = bt["equity_curve"].reset_index()
    eq.columns = ["date", "equity"]
    st.plotly_chart(px.line(eq, x="date", y="equity",
                            title=f"{bt_ticker} strategy equity curve (x starting capital)"),
                    use_container_width=True)

# ---------------------------- Reports ---------------------------------
with reports:
    st.subheader("🗂️ Reports")
    st.dataframe(ranked, use_container_width=True, hide_index=True)
    st.download_button("Export CSV", ranked.to_csv(index=False).encode("utf-8"),
                       "finserv_signals.csv", "text/csv")

# ---------------------------- About -----------------------------------
with about:
    st.subheader("ℹ️ About")
    st.markdown("""
**What this is.** A market dashboard and swing-signal tool for the Nasdaq-100 (NQ) and
S&P 500 (ES) futures and their heavyweight component stocks.

**Methodology.** Each instrument is scored on four pattern families — EMA 20/50 trend &
crossovers, RSI + MACD momentum, 20-day breakout/breakdown, and candlestick patterns.
For the index futures, the score is also confirmed by *leader-stock breadth*: how many of
the index's biggest stocks are currently trending up. The blend yields a LONG / SHORT /
NEUTRAL signal with a strength score.

**Data.** Real daily prices via yfinance (free, end-of-day). Falls back to clearly-labelled
demo data only if the host has no internet.

**Important.** Not financial advice. Signals are alerts you act on, not orders. Futures are
leveraged and can lose money quickly; a backtest is not a promise about the future. This
tool does not connect to any brokerage and does not place trades.

Built with Streamlit · Plotly · Pandas · NumPy.
""")
