import yfinance as yf
import pandas as pd
import requests
import os
import sys
import warnings
from datetime import datetime, time as dtime
import pytz
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Supprimer les avertissements inutiles dans les logs
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
SL_MULT = 1.2  
TP_MULT = 3.0  
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

def get_company_info(symbol):
    """Récupère le nom complet et le sentiment"""
    try:
        ticker = yf.Ticker(symbol)
        # On tente de récupérer le nom complet
        full_name = ticker.info.get('longName', symbol)
        
        news = ticker.news
        if not news: 
            return full_name, 0.0
        
        scores = []
        for article in news[:5]:
            text = f"{article.get('title', '')}. {article.get('summary', '')}"
            score = sentiment_analyzer.polarity_scores(text)["compound"]
            scores.append(score)
        sentiment = sum(scores) / len(scores) if scores else 0.0
        return full_name, sentiment
    except:
        return symbol, 0.0

def check_signal(symbol):
    df = yf.download(symbol, interval=INTERVAL, period=PERIOD, progress=False, auto_adjust=True)
    if df.empty or len(df) < EMA_PERIOD: return

    df = compute_indicators(df)
    c = df.iloc[-1]
    p1 = df.iloc[-2]

    def to_f(val):
        if isinstance(val, pd.Series): return float(val.iloc[0])
        return float(val)

    close, open_ = to_f(c["Close"]), to_f(c["Open"])
    ema, vwap = to_f(c["EMA200"]), to_f(c["VWAP"])
    rsi, rsi_p1 = to_f(c["RSI"]), to_f(p1["RSI"])
    atr = to_f(c["ATR"])
    
    # Récupération Nom + Sentiment
    company_name, sentiment = get_company_info(symbol)

    print(f"[{symbol}] {company_name} | RSI:{rsi:.1f} | Sent:{sentiment:.2f}")

    # ===== SIGNAL BUY =====
    if (close > ema and close > vwap and rsi_p1 < 50 and rsi >= 50 and 
        sentiment >= SENTIMENT_BLOCK and close > open_):
        sl, tp = close - SL_MULT * atr, close + TP_MULT * atr
        send_discord(f"🚀 **BUY — {company_name} ({symbol})**\n💰 Prix : {close:.2f}\n📈 RSI : {rsi:.1f}\n🛑 SL : {sl:.2f} | 🎯 TP : {tp:.2f}")

    # ===== SIGNAL SELL =====
    if (close < ema and close < vwap and rsi_p1 > 50 and rsi <= 50 and 
        sentiment >= SENTIMENT_BLOCK and close < open_):
        sl, tp = close + SL_MULT * atr, close - TP_MULT * atr
        send_discord(f"📉 **SELL — {company_name} ({symbol})**\n💰 Prix : {close:.2f}\n📉 RSI : {rsi:.1f}\n🛑 SL : {sl:.2f} | 🎯 TP : {tp:.2f}")

# ---------- EXECUTION ----------

if __name__ == "__main__":
    if not us_market_is_open():
        print("Marché fermé.")
        sys.exit()

    print(f"--- Analyse du {datetime.now(pytz.timezone('Europe/Paris')).strftime('%Y-%m-%d %H:%M:%S')} ---")
    for symbol in SYMBOLS:
        try:
            check_signal(symbol)
        except Exception as e:
            print(f"Erreur sur {symbol}: {e}")
