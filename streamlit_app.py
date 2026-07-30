"""FinServ Swing Signal Bot — Streamlit front end.

Pulls REAL daily data for NQ/ES and their heavyweight components (via yfinance),
runs the tested pattern engine, and shows current swing signals + a backtest.

NOT financial advice. Futures are leveraged and risky. Signals mean "your rule
fired — take a look"; you decide. No broker connection, no auto-trading.
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

import signal_engine as se

st.set_page_config(page_title="FinServ Swing Signal Bot", layout="wide", page_icon="📈")

st.title("📈 FinServ Swing Signal Bot")
st.caption("Real daily-data swing signals for NQ / ES, confirmed by their leader stocks. "
           "Not financial advice — signals are alerts you act on, not orders.")

with st.sidebar:
    st.header("Settings")
    period = st.selectbox("History window", ["1y", "2y", "5y"], index=1)
    st.markdown("---")
    st.markdown(
        "**How to read a signal**\n\n"
        "- **LONG** = bullish patterns outweigh bearish\n"
        "- **SHORT** = bearish outweigh bullish\n"
        "- **NEUTRAL** = no edge, stand aside\n\n"
        "*Strength* is how lopsided the patterns are (0–100%).")
    st.markdown("---")
    st.caption("⚠️ Leveraged futures can lose money fast. Backtest ≠ future results. "
               "This tool does not place trades.")


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
        "close": c, "volume": np.full(n, 1e6)}, index=idx)


# Try real data; if the host has no internet, fall back to a clearly-labelled demo.
demo_mode = False
data = {}
try:
    for fut, meta in se.FUTURES.items():
        data[fut] = {"df": load(fut, period), "comps": {}}
        for t in meta["components"]:
            try:
                data[fut]["comps"][t] = load(t, period)
            except Exception:
                pass
except Exception as e:
    demo_mode = True
    for i, (fut, meta) in enumerate(se.FUTURES.items()):
        data[fut] = {"df": demo_df(i, drift=0.0009),
                     "comps": {t: demo_df(hash(t) % 999) for t in meta["components"]}}

if demo_mode:
    st.warning("Couldn't reach the live data feed from this host, so the app is showing "
               "**DEMO (simulated) data**. Deploy it somewhere with internet (e.g. Streamlit "
               "Community Cloud) and it will switch to real market data automatically.")
else:
    st.success(f"Live data loaded for {', '.join(data.keys())} and their component stocks.")

# ---- current signals ----
st.subheader("Current Signals")
cols = st.columns(len(data))
for col, (fut, d) in zip(cols, data.items()):
    res = se.evaluate(d["df"], d["comps"])
    color = {"LONG (bullish)": "#199e70", "SHORT (bearish)": "#e66767"}.get(res["label"], "#8a897e")
    with col:
        st.markdown(f"### {fut}")
        st.caption(se.FUTURES[fut]["name"])
        st.markdown(
            f"<div style='font-size:26px;font-weight:700;color:{color}'>{res['label']}</div>",
            unsafe_allow_html=True)
        st.metric("Strength", f"{res['strength']:.0f}%")
        st.metric("Last close", res["last_close"])
        if res["breadth_pct"] is not None:
            st.metric("Leader stocks bullish", f"{res['breadth_pct']:.0f}%")
        st.markdown("**Why:**")
        for r in res["reasons"]:
            st.markdown(f"- {r}")

st.markdown("---")

# ---- charts + backtest per instrument ----
for fut, d in data.items():
    st.subheader(f"{fut} — price & backtest")
    df = d["df"]
    ema20, ema50 = se.ema(df["close"], 20), se.ema(df["close"], 50)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"],
                                 low=df["low"], close=df["close"], name="price",
                                 increasing_line_color="#199e70", decreasing_line_color="#e66767"))
    fig.add_trace(go.Scatter(x=df.index, y=ema20, name="EMA20", line=dict(color="#3987e5", width=1.3)))
    fig.add_trace(go.Scatter(x=df.index, y=ema50, name="EMA50", line=dict(color="#c98500", width=1.3)))
    fig.update_layout(height=430, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    bt = se.backtest(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Backtest total return", f"{bt['total_return_pct']}%")
    c2.metric("Win rate", f"{bt['win_rate']}%")
    c3.metric("Trades", bt["trades"])
    c4.metric("Avg / worst trade", f"{bt['avg_trade_pct']}% / {bt['worst_trade_pct']}%")
    eq = bt["equity_curve"].reset_index()
    eq.columns = ["date", "equity"]
    st.plotly_chart(px.line(eq, x="date", y="equity",
                            title=f"{fut} strategy equity curve (x starting capital)"),
                    use_container_width=True)
    st.caption("Backtest holds a long/short until the aggregate signal flips (max 20 days). "
               "Illustrative only — not a promise of future results.")
    st.markdown("---")

st.caption("Methodology: four pattern families (EMA trend, RSI+MACD momentum, 20-day breakout, "
           "candlesticks) plus leader-stock breadth are combined into one signal. "
           "Data via yfinance. This tool does not connect to a broker or place trades.")
