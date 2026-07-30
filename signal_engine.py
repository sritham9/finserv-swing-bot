"""FinServ swing-signal engine.

Pulls REAL daily market data (via yfinance) for index futures and their heavyweight
component stocks, detects four families of technical patterns, and combines them into
a single swing-trade signal with plain-English reasons.

IMPORTANT — read this:
  * Signals are NOT financial advice. They mean "a rule you defined just triggered,
    take a look" — you decide whether to trade.
  * Futures are leveraged and can lose money quickly. Backtest before risking money,
    and remember a backtest is not a promise about the future.
  * This module does not connect to a broker and does not place trades.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# --- instruments ---------------------------------------------------------------
# Each index future is confirmed by the trend of its heavyweight component stocks
# (your "pattern on the prospective stock" idea: watch the leaders, trade the index).
FUTURES = {
    "NQ=F": {
        "name": "Nasdaq-100 future (micro: MNQ=F)",
        "components": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "TSLA"],
    },
    "ES=F": {
        "name": "S&P 500 future (micro: MES=F)",
        "components": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "BRK-B", "JPM"],
    },
}


# --- data layer ----------------------------------------------------------------
def fetch_daily(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch real daily OHLCV via yfinance. Returns columns open/high/low/close/volume.

    Raises RuntimeError if data can't be retrieved (e.g. no internet / bad ticker),
    so callers can fall back or report clearly.
    """
    import yfinance as yf
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    df.index.name = "date"
    return df


# --- indicators ----------------------------------------------------------------
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    out = out.mask(avg_loss == 0, 100.0)                       # no down-days -> overbought
    out = out.mask((avg_gain == 0) & (avg_loss == 0), 50.0)    # perfectly flat -> neutral
    return out.fillna(50)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def _crossed_up(a: pd.Series, b: pd.Series) -> bool:
    return a.iloc[-2] <= b.iloc[-2] and a.iloc[-1] > b.iloc[-1]


def _crossed_down(a: pd.Series, b: pd.Series) -> bool:
    return a.iloc[-2] >= b.iloc[-2] and a.iloc[-1] < b.iloc[-1]


# --- pattern detectors: each returns (vote:int in {-1,0,1}, reason:str|None) -----
def trend_signal(df: pd.DataFrame, fast: int = 20, slow: int = 50):
    ef, es = ema(df["close"], fast), ema(df["close"], slow)
    if _crossed_up(ef, es):
        return 1, f"Golden cross: {fast}-day EMA crossed above {slow}-day EMA"
    if _crossed_down(ef, es):
        return -1, f"Death cross: {fast}-day EMA crossed below {slow}-day EMA"
    if ef.iloc[-1] > es.iloc[-1]:
        return 1, f"Uptrend: price EMA{fast} above EMA{slow}"
    return -1, f"Downtrend: price EMA{fast} below EMA{slow}"


def momentum_signal(df: pd.DataFrame):
    r = rsi(df["close"]).iloc[-1]
    macd_line, signal_line, _ = macd(df["close"])
    votes, reasons = [], []
    if r < 30:
        votes.append(1); reasons.append(f"RSI oversold ({r:.0f})")
    elif r > 70:
        votes.append(-1); reasons.append(f"RSI overbought ({r:.0f})")
    if _crossed_up(macd_line, signal_line):
        votes.append(1); reasons.append("MACD crossed up (bullish momentum)")
    elif _crossed_down(macd_line, signal_line):
        votes.append(-1); reasons.append("MACD crossed down (bearish momentum)")
    if not votes:
        return 0, None
    v = int(np.sign(sum(votes)))
    return v, "; ".join(reasons)


def breakout_signal(df: pd.DataFrame, lookback: int = 20):
    prior = df.iloc[-(lookback + 1):-1]
    hi, lo = prior["high"].max(), prior["low"].min()
    close = df["close"].iloc[-1]
    if close > hi:
        return 1, f"Breakout: closed above the {lookback}-day high ({hi:.2f})"
    if close < lo:
        return -1, f"Breakdown: closed below the {lookback}-day low ({lo:.2f})"
    return 0, None


def candlestick_signal(df: pd.DataFrame):
    o, h, l, c = (df[x].iloc[-1] for x in ("open", "high", "low", "close"))
    po, pc = df["open"].iloc[-2], df["close"].iloc[-2]
    body = abs(c - o)
    rng = max(h - l, 1e-9)
    upper = h - max(o, c)
    lower = min(o, c) - l
    # Bullish / bearish engulfing
    if pc < po and c > o and o <= pc and c >= po:
        return 1, "Bullish engulfing candle"
    if pc > po and c < o and o >= pc and c <= po:
        return -1, "Bearish engulfing candle"
    # Hammer (long lower shadow, small body up top)
    if lower >= 2 * body and upper <= body and body / rng < 0.4:
        return 1, "Hammer (potential reversal up)"
    # Shooting star (long upper shadow, small body at bottom)
    if upper >= 2 * body and lower <= body and body / rng < 0.4:
        return -1, "Shooting star (potential reversal down)"
    return 0, None


# --- component breadth (your stock->future confirmation) -----------------------
def component_breadth(components: dict) -> tuple[float, str]:
    """components: {ticker: dataframe}. Returns (% bullish, reason)."""
    if not components:
        return 0.5, "No component data"
    bulls = 0
    for _, cdf in components.items():
        if len(cdf) >= 51 and trend_signal(cdf)[0] > 0:
            bulls += 1
    pct = bulls / len(components)
    return pct, f"{bulls}/{len(components)} leader stocks in an uptrend ({pct*100:.0f}%)"


# --- aggregate -----------------------------------------------------------------
WEIGHTS = {"trend": 2.0, "momentum": 2.0, "breakout": 2.0, "candlestick": 1.0, "breadth": 1.5}


def evaluate(df: pd.DataFrame, components: dict | None = None) -> dict:
    """Combine every pattern into one swing signal for a single instrument."""
    detectors = {
        "trend": trend_signal(df),
        "momentum": momentum_signal(df),
        "breakout": breakout_signal(df),
        "candlestick": candlestick_signal(df),
    }
    score = 0.0
    reasons = []
    for name, (vote, reason) in detectors.items():
        score += WEIGHTS[name] * vote
        if reason:
            reasons.append(reason)

    breadth_pct = None
    if components:
        breadth_pct, breadth_reason = component_breadth(components)
        score += WEIGHTS["breadth"] * (breadth_pct - 0.5) * 2  # map 0..1 -> -1..1
        reasons.append(breadth_reason)

    max_score = sum(WEIGHTS[k] for k in ("trend", "momentum", "breakout", "candlestick")) + \
        (WEIGHTS["breadth"] if components else 0)
    norm = score / max_score if max_score else 0.0
    if norm > 0.20:
        label = "LONG (bullish)"
    elif norm < -0.20:
        label = "SHORT (bearish)"
    else:
        label = "NEUTRAL — stand aside"
    return {
        "label": label,
        "score": round(score, 2),
        "strength": round(abs(norm) * 100, 0),
        "last_close": round(df["close"].iloc[-1], 2),
        "breadth_pct": None if breadth_pct is None else round(breadth_pct * 100, 0),
        "reasons": reasons,
        "votes": {k: v[0] for k, v in detectors.items()},
    }


# --- backtest ------------------------------------------------------------------
def backtest(df: pd.DataFrame, hold_max: int = 20) -> dict:
    """Swing backtest: each day compute trend+momentum+breakout aggregate, go long
    when net-positive / short when net-negative, hold until the signal flips (or
    hold_max days). Returns performance stats on REAL history when run live.
    """
    ef, es = ema(df["close"], 20), ema(df["close"], 50)
    r = rsi(df["close"])
    macd_line, signal_line, _ = macd(df["close"])
    hi = df["high"].rolling(20).max().shift(1)
    lo = df["low"].rolling(20).min().shift(1)

    sig = np.zeros(len(df))
    for i in range(51, len(df)):
        s = 0
        s += 1 if ef.iloc[i] > es.iloc[i] else -1
        if r.iloc[i] < 30: s += 1
        elif r.iloc[i] > 70: s -= 1
        if macd_line.iloc[i] > signal_line.iloc[i]: s += 1
        else: s -= 1
        if df["close"].iloc[i] > hi.iloc[i]: s += 1
        elif df["close"].iloc[i] < lo.iloc[i]: s -= 1
        sig[i] = np.sign(s)

    position = pd.Series(sig, index=df.index).shift(1).fillna(0)
    rets = df["close"].pct_change().fillna(0)
    strat = position * rets
    equity = (1 + strat).cumprod()

    # trade-level stats (position changes)
    changes = position.diff().fillna(0)
    trade_starts = position.index[(changes != 0) & (position != 0)]
    pnls = []
    for start in trade_starts:
        idx = position.index.get_loc(start)
        dir_ = position.loc[start]
        j = idx
        while j < len(position) and position.iloc[j] == dir_ and (j - idx) < hold_max:
            j += 1
        j = min(j, len(df) - 1)
        pnl = (df["close"].iloc[j] / df["close"].iloc[idx] - 1) * dir_ * 100
        pnls.append(pnl)
    pnls = pd.Series(pnls)
    wins = pnls[pnls > 0]
    win_rate = round(len(wins) / len(pnls) * 100, 1) if len(pnls) else 0.0
    return {
        "equity_curve": equity,
        "total_return_pct": round((equity.iloc[-1] - 1) * 100, 1),
        "win_rate": win_rate,
        "trades": len(pnls),
        "avg_trade_pct": round(pnls.mean(), 2) if len(pnls) else 0.0,
        "best_trade_pct": round(pnls.max(), 2) if len(pnls) else 0.0,
        "worst_trade_pct": round(pnls.min(), 2) if len(pnls) else 0.0,
    }


def scan(period: str = "2y") -> list[dict]:
    """Fetch real data for every future + its components and return current signals."""
    out = []
    for fut, meta in FUTURES.items():
        fdf = fetch_daily(fut, period)
        comps = {}
        for t in meta["components"]:
            try:
                comps[t] = fetch_daily(t, period)
            except Exception:
                pass
        res = evaluate(fdf, comps)
        res["ticker"] = fut
        res["name"] = meta["name"]
        out.append(res)
    return out


if __name__ == "__main__":
    print("Scanning real market data...\n")
    for s in scan():
        print(f"{s['ticker']}  {s['name']}")
        print(f"  Signal: {s['label']}  (strength {s['strength']}%, last {s['last_close']})")
        for r in s["reasons"]:
            print(f"    - {r}")
        print()
