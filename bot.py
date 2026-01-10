import yfinance as yf
import pandas as pd
import requests
import os
import sys
from datetime import datetime, time as dtime
import pytz
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ================= CONFIG =================

DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")

SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
    "AMD", "INTC", "AVGO", "ORCL", "IBM",
    "JPM", "BAC", "GS", "MS", "V",
    "WMT", "COST", "HD", "MCD", "NKE",
    "BA", "CAT", "GE", "LMT",
    "XOM", "CVX"
]

INTERVAL = "1h"
PERIOD = "60d"

RSI_PERIOD = 14
EMA_PERIOD = 200
ATR_PERIOD = 14

SL_MULT = 1.5
TP_MULT = 3

SENTIMENT_BLOCK = -0.2

# ==========================================

sentiment_analyzer = SentimentIntensityAnalyzer()


def send_discord(message):
    if DISCORD_WEBHOOK:
        requests.post(DISCORD_WEBHOOK, json={"content": message})


# ---------- HORAIRES MARCHÉ US (FRANCE) ----------

def us_market_is_open():
    tz = pytz.timezone("Europe/Paris")
    now = datetime.now(tz)

    # Week-end
    if now.weekday() >= 5:
        return False

    market_open = dtime(15, 30)
    market_close = dtime(22, 0)

    return market_open <= now.time() <= market_close


# ---------- MESSAGE ÉTAT MARCHÉ ----------

def check_market_and_notify():
    if not us_market_is_open():
        send_discord(
            "🔴 **Marché US fermé**\n"
            "⛔ Aucune analyse effectuée"
        )
        return False
    else:
        send_discord(
            "🟢 **Marché US ouvert**\n"
            "📊 Analyse 1H en cours…"
        )
        return True


# ---------- INDICATEURS ----------

def compute_indicators(df):
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))
    df["EMA200"] = df["Close"].ewm(span=EMA_PERIOD).mean()

    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)

    df["ATR"] = tr.rolling(ATR_PERIOD).mean()
    return df


# ---------- SENTIMENT YAHOO ----------

def get_yahoo_sentiment(symbol):
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news

        if not news:
            return 0.0

        scores = []
        for article in news[:10]:
            text = f"{article.get('title', '')}. {article.get('summary', '')}"
            score = sentiment_analyzer.polarity_scores(text)["compound"]
            scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0

    except Exception:
        return 0.0


# ---------- SIGNAL ----------

def check_signal(symbol):
    df = yf.download(symbol, interval=INTERVAL, period=PERIOD, progress=False)

    if df.empty or len(df) < EMA_PERIOD + 5:
        return

    df = compute_indicators(df)

    c = df.iloc[-1]
    p1 = df.iloc[-2]
    p2 = df.iloc[-3]

    close = c["Close"].item()
    open_ = c["Open"].item()
    ema = c["EMA200"].item()
    vwap = c["VWAP"].item()
    rsi = c["RSI"].item()
    atr = c["ATR"].item()

    sentiment = get_yahoo_sentiment(symbol)
    if sentiment < SENTIMENT_BLOCK:
        return

    # ===== BUY =====
    if (
        close > ema and close > vwap and
        p2["RSI"] < 40 and p1["RSI"] < 40 and rsi >= 40 and
        close > open_
    ):
        sl = close - SL_MULT * atr
        tp = close + TP_MULT * atr

        send_discord(
            f"🟢 **BUY 1H — {symbol}**\n"
            f"💰 Prix : {close:.2f}\n"
            f"📈 RSI : {rsi:.1f}\n"
            f"📰 Sentiment : {sentiment:.2f}\n"
            f"🛑 SL : {sl:.2f}\n"
            f"🎯 TP : {tp:.2f}"
        )

    # ===== SELL =====
    if (
        close < ema and close < vwap and
        p2["RSI"] > 60 and p1["RSI"] > 60 and rsi <= 60 and
        close < open_
    ):
        sl = close + SL_MULT * atr
        tp = close - TP_MULT * atr

        send_discord(
            f"🔴 **SELL 1H — {symbol}**\n"
            f"💰 Prix : {close:.2f}\n"
            f"📉 RSI : {rsi:.1f}\n"
            f"📰 Sentiment : {sentiment:.2f}\n"
            f"🛑 SL : {sl:.2f}\n"
            f"🎯 TP : {tp:.2f}"
        )


# ---------- EXECUTION ----------

if not check_market_and_notify():
    sys.exit()

for symbol in SYMBOLS:
    check_signal(symbol)


