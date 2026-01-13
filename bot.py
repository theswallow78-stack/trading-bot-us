import yfinance as yf
import pandas as pd
import requests
import os
import sys
from datetime import datetime, time as dtime
import pytz
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ================= CONFIG OPTIMISÉE =================

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

# Amélioration du Risk/Reward (Ratio 1:2.5)
SL_MULT = 1.2  
TP_MULT = 3.0  

# Sentiment moins restrictif pour privilégier le prix
SENTIMENT_BLOCK = -0.4 

# ===================================================

sentiment_analyzer = SentimentIntensityAnalyzer()

def send_discord(message):
    if DISCORD_WEBHOOK:
        requests.post(DISCORD_WEBHOOK, json={"content": message})

def us_market_is_open():
    tz = pytz.timezone("Europe/Paris")
    now = datetime.now(tz)
    if now.weekday() >= 5: return False
    market_open = dtime(15, 30)
    market_close = dtime(22, 0)
    return market_open <= now.time() <= market_close

def check_market_and_notify():
    if not us_market_is_open():
        return False
    else:
        # On réduit le spam Discord, on n'envoie que si le marché est ouvert
        return True

def compute_indicators(df):
    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # EMA 200
    df["EMA200"] = df["Close"].ewm(span=EMA_PERIOD).mean()

    # VWAP (Calculé sur la session actuelle)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()

    # ATR
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(ATR_PERIOD).mean()
    
    return df

def get_yahoo_sentiment(symbol):
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        if not news: return 0.0
        scores = []
        for article in news[:5]: # Top 5 articles pour plus de rapidité
            text = f"{article.get('title', '')}. {article.get('summary', '')}"
            score = sentiment_analyzer.polarity_scores(text)["compound"]
            scores.append(score)
        return sum(scores) / len(scores) if scores else 0.0
    except:
        return 0.0

def check_signal(symbol):
    df = yf.download(symbol, interval=INTERVAL, period=PERIOD, progress=False)
    if df.empty or len(df) < EMA_PERIOD: return

    df = compute_indicators(df)
    c = df.iloc[-1]
    p1 = df.iloc[-2]

    close = float(c["Close"])
    open_ = float(c["Open"])
    ema = float(c["EMA200"])
    vwap = float(c["VWAP"])
    rsi = float(c["RSI"])
    rsi_p1 = float(p1["RSI"])
    atr = float(c["ATR"])
    
    sentiment = get_yahoo_sentiment(symbol)

    # LOG DE DIAGNOSTIC (Visible dans GitHub Actions)
    print(f"[{symbol}] Prix:{close:.2f} | EMA:{ema:.2f} | RSI:{rsi:.1f} | Sent:{sentiment:.2f}")

    # ===== NOUVELLE STRATÉGIE BUY (Tendance + Momentum) =====
    # Condition : Prix > EMA200 ET Prix > VWAP ET RSI croise les 50 à la hausse
    if (
        close > ema and 
        close > vwap and 
        rsi_p1 < 50 and rsi >= 50 and 
        sentiment >= SENTIMENT_BLOCK and
        close > open_
    ):
        sl = close - SL_MULT * atr
        tp = close + TP_MULT * atr
        send_discord(
            f"🚀 **SIGNAL BUY (Optimisé) — {symbol}**\n"
            f"💰 Prix : {close:.2f}\n"
            f"📈 RSI : {rsi:.1f} (Crossover 50)\n"
            f"📰 Sentiment : {sentiment:.2f}\n"
            f"🛑 SL : {sl:.2f} | 🎯 TP : {tp:.2f}"
        )

    # ===== NOUVELLE STRATÉGIE SELL (Short) =====
    if (
        close < ema and 
        close < vwap and 
        rsi_p1 > 50 and rsi <= 50 and 
        sentiment >= SENTIMENT_BLOCK and # On ne short pas si sentiment trop négatif (risque de rebond violent)
        close < open_
    ):
        sl = close + SL_MULT * atr
        tp = close - TP_MULT * atr
        send_discord(
            f"📉 **SIGNAL SELL (Optimisé) — {symbol}**\n"
            f"💰 Prix : {close:.2f}\n"
            f"📉 RSI : {rsi:.1f} (Crossover 50)\n"
            f"📰 Sentiment : {sentiment:.2f}\n"
            f"🛑 SL : {sl:.2f} | 🎯 TP : {tp:.2f}"
        )

# ---------- EXECUTION ----------

if __name__ == "__main__":
    if not us_market_is_open():
        print("Marché fermé.")
        sys.exit()

    print(f"--- Analyse du {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    for symbol in SYMBOLS:
        check_signal(symbol)
