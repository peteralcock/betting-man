

import sqlite3
import time
import json
import requests
import math
from datetime import datetime, timezone
from kalshi_python import Configuration, KalshiClient
from urllib.parse import quote_plus

# (Optional) for news / web search
from bs4 import BeautifulSoup

# ========== Configuration & setup ==========

API_KEY_ID = "your_api_key_id"
PRIVATE_KEY_PEM = """-----BEGIN PRIVATE KEY-----
...
-----END PRIVATE KEY-----"""

# Base API host (use demo or production as appropriate)
API_HOST = "https://api.elections.kalshi.com/trade-api/v2"

config = Configuration(host=API_HOST)
config.api_key_id = API_KEY_ID
config.private_key_pem = PRIVATE_KEY_PEM
client = KalshiClient(config)

# SQLite setup
DB_PATH = "kalshi_econ.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # event table
    c.execute("""
    CREATE TABLE IF NOT EXISTS events (
        event_ticker TEXT PRIMARY KEY,
        name TEXT,
        category TEXT,
        close_ts INTEGER,
        resolution TEXT
    )""")
    # market table
    c.execute("""
    CREATE TABLE IF NOT EXISTS markets (
        market_ticker TEXT PRIMARY KEY,
        event_ticker TEXT,
        yes_price REAL,
        no_price REAL,
        last_trade_ts INTEGER,
        volume REAL,
        FOREIGN KEY(event_ticker) REFERENCES events(event_ticker)
    )""")
    # trade history / snapshots
    c.execute("""
    CREATE TABLE IF NOT EXISTS market_snapshots (
        snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
        market_ticker TEXT,
        ts INTEGER,
        yes_price REAL,
        no_price REAL,
        volume REAL
    )""")
    # optional: your model signals / bets
    c.execute("""
    CREATE TABLE IF NOT EXISTS model_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        market_ticker TEXT,
        ts INTEGER,
        implied_prob REAL,
        model_prob REAL,
        signal TEXT,
        confidence REAL
    )""")
    conn.commit()
    conn.close()

# ========== Fetch & store data ==========

def fetch_all_markets(limit=1000):
    """Fetch all markets via pagination"""
    all_markets = []
    cursor = None
    while True:
        resp = client.get_markets(limit=limit, cursor=cursor)
        data = resp.data
        for m in data.markets:
            all_markets.append(m)
        cursor = data.cursor
        if not cursor:
            break
        # rate-limit sleep if needed
        time.sleep(0.2)
    return all_markets

def filter_economic_markets(markets):
    """Filter markets whose underlying event is economic in nature."""
    econ = []
    for m in markets:
        # Some heuristics: check event ticker or market name or category containing “CPI”, “Fed”, “inflation”, “GDP”, etc.
        name = m.name.lower() if hasattr(m, "name") else ""
        ticker = m.market_ticker.lower()
        if any(tok in name for tok in ["cpi","inflation","fed","gdp","unemployment","rate","ppi"]):
            econ.append(m)
    return econ

def store_markets(markets):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for m in markets:
        # store event
        ev = m.event
        c.execute("""
            INSERT OR IGNORE INTO events(event_ticker, name, category, close_ts, resolution)
            VALUES (?, ?, ?, ?, ?)
        """, (ev.event_ticker, ev.name, ev.category if hasattr(ev, "category") else None,
              ev.close_ts, ev.resolution if hasattr(ev, "resolution") else None))
        # store market
        yes_price = None; no_price = None
        # The API may return “last_price” for yes side and no = 1 - yes (depending on representation). Adapt as needed.
        # Here, assume m.last_price is yes side, and no_price = 1 - yes_price.
        yes_price = m.last_price
        no_price = 1.0 - yes_price
        c.execute("""
            INSERT OR REPLACE INTO markets(market_ticker, event_ticker, yes_price, no_price, last_trade_ts, volume)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (m.market_ticker, ev.event_ticker, yes_price, no_price, m.last_trade_ts, m.volume))
        # snapshot
        c.execute("""
            INSERT INTO market_snapshots(market_ticker, ts, yes_price, no_price, volume)
            VALUES (?, ?, ?, ?, ?)
        """, (m.market_ticker, int(time.time()), yes_price, no_price, (m.volume or 0)))
    conn.commit()
    conn.close()

# ========== External news / background fetch ==========

def search_news_for_event(event_name, num=5):
    """Do a simple web search and return a list of (title, snippet, url)."""
    query = quote_plus(event_name + " outlook analysis 2025")
    url = f"https://www.google.com/search?q={query}"
    # (Note: Google search may block automated requests; you may need to use a search API.)
    headers = {"User-Agent": "Mozilla/5.0 (compatible)"}
    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for g in soup.select(".kCrYT a"):
        href = g.get("href")
        if href and href.startswith("/url?q="):
            actual = href.split("/url?q=")[1].split("&sa=")[0]
            title = g.text
            results.append((title, "", actual))
            if len(results) >= num:
                break
    return results

# ========== Simple “model” & bet suggestion logic ==========

def compute_signal_for_market(market_ticker):
    """
    Get latest market, compute implied probability, build a naive model for true probability,
    then compute signal and confidence.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT yes_price FROM markets WHERE market_ticker = ?", (market_ticker,))
    row = c.fetchone()
    if not row:
        conn.close()
        return None
    implied = row[0]
    # *** Simple model: treat implied as base, then adjust by news sentiment ***
    # For demonstration: if latest news has strong language (“sharp rise inflation”) push model a bit.
    # A real model would parse economic forecasts, time series, etc.
    # Here we fetch news:
    c.execute("SELECT event_ticker FROM markets WHERE market_ticker = ?", (market_ticker,))
    evt = c.fetchone()[0]
    c.execute("SELECT name FROM events WHERE event_ticker = ?", (evt,))
    ev_name = c.fetchone()[0]
    news = search_news_for_event(ev_name, num=3)
    # Very crude sentiment: if news titles contain “rise”, “surge”, “jump” → upward bias
    bias = 0.0
    for title, _, _ in news:
        t = title.lower()
        if "surge" in t or "rise" in t or "increase" in t or "jump" in t:
            bias += 0.02
        if "fall" in t or "decline" in t or "drop" in t:
            bias -= 0.02
    model_prob = implied + bias
    # clamp
    model_prob = max(0.01, min(0.99, model_prob))
    signal = None
    if model_prob > implied + 0.01:
        signal = "bet_yes"
    elif model_prob < implied - 0.01:
        signal = "bet_no"
    else:
        signal = "no_bet"
    # Confidence: based on magnitude of difference and number of sentiment signals
    diff = abs(model_prob - implied)
    confidence = min(1.0, diff * 5)  # e.g. if diff=0.1 → confidence=0.5
    conn.close()
    return {
        "market_ticker": market_ticker,
        "implied_prob": implied,
        "model_prob": model_prob,
        "signal": signal,
        "confidence": confidence,
        "news": news
    }

def choose_best_bet(signals):
    """
    Among signals, pick the one with highest confidence (and non-neutral) as the “best bet”.
    """
    best = None
    for s in signals:
        if s["signal"] != "no_bet":
            if best is None or s["confidence"] > best["confidence"]:
                best = s
    return best

# ========== Main orchestration ==========

def main():
    init_db()
    print("Fetching markets …")
    markets = fetch_all_markets(limit=500)
    print(f"Fetched {len(markets)} markets")
    econ_markets = filter_economic_markets(markets)
    print(f"Filtered {len(econ_markets)} economic markets")
    store_markets(econ_markets)
    # compute signals for each econ market
    signals = []
    for m in econ_markets:
        sig = compute_signal_for_market(m.market_ticker)
        if sig:
            signals.append(sig)
    # pick best bet
    best = choose_best_bet(signals)
    if best:
        print("=== Best bet recommendation ===")
        print(f"Market: {best['market_ticker']}")
        print(f"Signal: {best['signal']}")
        print(f"Model prob: {best['model_prob']:.3f}, Implied prob: {best['implied_prob']:.3f}")
        print(f"Confidence: {best['confidence']:.3f}")
        print("News influencing decision:")
        for title, _, url in best["news"]:
            print(f" - {title} → {url}")
    else:
        print("No strong bet signal at this time.")

if __name__ == "__main__":
    main()
