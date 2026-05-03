import warnings
warnings.filterwarnings("ignore")

import os
import importlib.util
import requests
import yfinance as yf
import feedparser

import os

# --- GEN 5: Cloud detection ---
# yfinance 1.2.x uses curl_cffi internally — passing a requests.Session breaks it.
# We detect cloud and skip Ollama; yfinance handles anti-ban itself in 1.2.x.
_IS_CLOUD = (
    os.environ.get("STREAMLIT_SHARING_MODE") == "streamlit"
    or os.environ.get("HOME", "").startswith("/home/adminuser")
    or not os.path.exists("/usr/bin/ollama")
)

import pandas as pd
import numpy as np
from urllib.parse import quote
from datetime import datetime
import time
import scipy.stats as scipy_stats

# ── Lightweight temporal cache (in-process TTL cache) ───────────────────────
_FUNDAMENTAL_CACHE = {}
_HEADLINES_CACHE = {}
_FUNDAMENTAL_TTL_SECONDS = 60 * 30
_HEADLINES_TTL_SECONDS = 60 * 15

def _cache_get(cache, key, ttl_seconds, allow_stale=False):
    item = cache.get(key)
    if not item:
        return None
    age = time.time() - item["ts"]
    if age <= ttl_seconds or allow_stale:
        return item["value"]
    return None

def _cache_set(cache, key, value):
    cache[key] = {"ts": time.time(), "value": value}

def _is_rate_limited_error(err_msg):
    msg = str(err_msg).lower()
    return (
        "too many requests" in msg or
        "rate limit" in msg or
        "429" in msg
    )

# ── Manual TA (no pandas_ta — works Python 3.11+) ───────────────────────────
def _ta_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _ta_macd(close, fast=12, slow=26, signal=9):
    ema_fast    = close.ewm(span=fast,   adjust=False).mean()
    ema_slow    = close.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def _ta_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ── Sentiment: Ollama preferred, VADER fallback ───────────────────────────────
def _get_sentiment_labels(headlines):
    texts = [h["headline"] for h in headlines]

    # On cloud: VADER only, no Ollama attempt at all
    if _IS_CLOUD:
        return _vader_labels(texts)

    # Locally: try Ollama with a short timeout, fall back to VADER
    try:
        import ollama as _ollama_check
        # Quick connectivity check before spinning up classifier
        _ollama_check.list()  # raises if Ollama not running
        from skollama.models.ollama.classification.zero_shot import ZeroShotOllamaClassifier
        clf = ZeroShotOllamaClassifier(model="gemma3")
        clf.fit(None, ["positive", "negative", "neutral"])
        return clf.predict(texts)
    except Exception:
        return _vader_labels(texts)

def _vader_labels(texts):
    """
    VADER-based labeler with safe fallback.
    Returns neutral labels if nltk/vader resources are unavailable.
    """
    if not texts:
        return []
    if importlib.util.find_spec("nltk") is None:
        return ["neutral"] * len(texts)

    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        try:
            nltk.download("vader_lexicon", quiet=True)
        except Exception:
            return ["neutral"] * len(texts)

    try:
        analyzer = SentimentIntensityAnalyzer()
    except Exception:
        return ["neutral"] * len(texts)

    labels = []
    for txt in texts:
        score = analyzer.polarity_scores(str(txt)).get("compound", 0.0)
        if score >= 0.05:
            labels.append("positive")
        elif score <= -0.05:
            labels.append("negative")
        else:
            labels.append("neutral")
    return labels

# ═══════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════
IDX_UNIVERSE = [
    "ADRO.JK","AGRO.JK","AKRA.JK","AMRT.JK","ANTM.JK","ASII.JK",
    "BBCA.JK","BBNI.JK","BBRI.JK","BBTN.JK","BMRI.JK","BRIS.JK",
    "BUKA.JK","CPIN.JK","EMTK.JK","EXCL.JK","GOTO.JK","HRUM.JK",
    "ICBP.JK","INCO.JK","INDF.JK","INDY.JK","ITMG.JK","KLBF.JK",
    "MBMA.JK","MDKA.JK","MEDC.JK","MIKA.JK","MYOR.JK","PGAS.JK",
    "PTBA.JK","SIDO.JK","SMGR.JK","TINS.JK","TLKM.JK","TOWR.JK",
    "UNTR.JK","UNVR.JK","ESSA.JK","BYAN.JK","CTRA.JK","PWON.JK",
    "SMRA.JK","JPFA.JK","HEAL.JK","BJTM.JK","BJBR.JK","INKP.JK",
    "INTP.JK","ISAT.JK",
]

# LQ45 — top 45 most liquid IDX stocks
LQ45_UNIVERSE = [
    "ADRO.JK","AKRA.JK","AMRT.JK","ANTM.JK","ASII.JK",
    "BBCA.JK","BBNI.JK","BBRI.JK","BBTN.JK","BMRI.JK","BRIS.JK",
    "BUKA.JK","CPIN.JK","EXCL.JK","GOTO.JK","HRUM.JK",
    "ICBP.JK","INCO.JK","INDF.JK","ITMG.JK","KLBF.JK",
    "MDKA.JK","MEDC.JK","MIKA.JK","MYOR.JK","PGAS.JK",
    "PTBA.JK","SMGR.JK","TINS.JK","TLKM.JK","TOWR.JK",
    "UNTR.JK","UNVR.JK","BYAN.JK","CTRA.JK","PWON.JK",
    "SMRA.JK","JPFA.JK","HEAL.JK","INKP.JK","INTP.JK","ISAT.JK",
    "MBMA.JK","EMTK.JK","ESSA.JK",
]


US_UNIVERSE = [
    # Mega-cap leaders (deep liquidity, strong trend persistence)
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","AVGO","TSLA","BRK-B","LLY",
    # Financials
    "JPM","BAC","WFC","GS","MS","BLK","SCHW","C",
    # Semis / AI infra
    "AMD","QCOM","MU","TXN","INTC","AMAT","LRCX","KLAC","ANET","SMCI",
    # Software / internet platforms
    "NFLX","CRM","ORCL","ADBE","NOW","UBER","SHOP","PLTR","PANW","CRWD",
    # Health care / biotech
    "JNJ","UNH","ABBV","MRK","PFE","TMO","ISRG","VRTX",
    # Consumer + defensives
    "WMT","COST","HD","MCD","NKE","SBUX","KO","PEP","PG","PM",
    # Industrials / energy / cyclicals
    "XOM","CVX","SLB","CAT","GE","RTX","BA","DE","ETN","HON",
]

# Gorengan / Scalping Universe — high beta, small-mid cap, volatile
# ⚠️ HIGH RISK — suitable for scalp/momentum plays only
GORENGAN_UNIVERSE = [
    "BUMI.JK","BRPT.JK","ENRG.JK","WINS.JK","DEWA.JK",
    "BULL.JK","NFCX.JK","CUAN.JK","ARTO.JK","BELI.JK",
    "INET.JK","NICL.JK","TEBE.JK","WTON.JK","WIFI.JK",
    "PANI.JK","ARMY.JK","NICE.JK","COAL.JK","LAJU.JK",
    "MSIN.JK","BPTR.JK","KIOS.JK","PICO.JK","SMKL.JK",
    "ZONE.JK","STRK.JK","FOLK.JK","FITT.JK","DOOH.JK",
    "RATU.JK","BHAT.JK","CASH.JK","SICO.JK","MSJA.JK",
    "CLEO.JK","MAPA.JK","SBMA.JK","GULA.JK","WMUU.JK",
]

def resolve_universe(universe_name: str = "IDX") -> list:
    """
    Returns list of tickers based on selected universe.
    Options: 'LQ45', 'IDX', 'US', 'Gorengan', 'All', 'Global Mix'
    """
    name = str(universe_name).upper().strip()
    if name == "LQ45":
        return LQ45_UNIVERSE
    elif name in ("IDX", "IDX80"):
        return IDX_UNIVERSE
    elif name == "US":
        return US_UNIVERSE
    elif name == "GORENGAN":
        return GORENGAN_UNIVERSE
    elif name == "ALL":
        combined = list(set(IDX_UNIVERSE + GORENGAN_UNIVERSE))
        return sorted(combined)
    elif name in ("GLOBAL MIX", "GLOBAL", "MIX"):
        combined = list(set(IDX_UNIVERSE + US_UNIVERSE))
        return sorted(combined)
    else:
        return IDX_UNIVERSE  # safe fallback

SECTORS = {
    "Energy":      {"T1":["ADRO.JK","ITMG.JK","PTBA.JK"],"T2":["MEDC.JK","INDY.JK"],"T3":["HRUM.JK","BYAN.JK"]},
    "Banks":       {"T1":["BBCA.JK","BBRI.JK"],           "T2":["BMRI.JK"],           "T3":["BBNI.JK"]},
    "Commodities": {"T1":["ANTM.JK","INCO.JK"],           "T2":["MDKA.JK"],           "T3":["MBMA.JK"]},
    "Defensive":   {"T1":["ICBP.JK"],                     "T2":["UNVR.JK"],           "T3":["MYOR.JK"]},
    "Tech":        {"T1":["GOTO.JK"],                     "T2":["EMTK.JK"],           "T3":["BUKA.JK"]},
}

COMPANY_NAMES = {
    "ADRO.JK":["Adaro","ADRO"],"AGRO.JK":["Bank Agro","AGRO"],
    "AKRA.JK":["AKR","Corporindo"],"AMRT.JK":["Alfamart","AMRT"],
    "ANTM.JK":["Antam","ANTM"],"ASII.JK":["Astra","ASII"],
    "BBCA.JK":["BCA","Bank Central Asia","BBCA"],"BBNI.JK":["BNI","Bank Negara","BBNI"],
    "BBRI.JK":["BRI","Bank Rakyat","BBRI"],"BBTN.JK":["BTN","Bank Tabungan","BBTN"],
    "BMRI.JK":["Mandiri","BMRI"],"BRIS.JK":["BRI Syariah","BRIS"],
    "BUKA.JK":["Bukalapak","BUKA"],"CPIN.JK":["Charoen","Pokphand","CPIN"],
    "EMTK.JK":["Elang Mahkota","EMTK"],"EXCL.JK":["XL Axiata","EXCL"],
    "GOTO.JK":["Gojek","GoTo","Tokopedia","GOTO"],"HRUM.JK":["Harum Energy","HRUM"],
    "ICBP.JK":["Indofood CBP","ICBP"],"INCO.JK":["Vale Indonesia","INCO"],
    "INDF.JK":["Indofood","INDF"],"INDY.JK":["Indika","INDY"],
    "ITMG.JK":["Indo Tambangraya","ITMG"],"KLBF.JK":["Kalbe Farma","KLBF"],
    "MBMA.JK":["Merdeka Battery","MBMA"],"MDKA.JK":["Merdeka Copper","MDKA"],
    "MEDC.JK":["Medco","MEDC"],"MIKA.JK":["Mitra Keluarga","MIKA"],
    "MYOR.JK":["Mayora","MYOR"],"PGAS.JK":["PGN","Perusahaan Gas","PGAS"],
    "PTBA.JK":["Bukit Asam","PTBA"],"SIDO.JK":["Sido Muncul","SIDO"],
    "SMGR.JK":["Semen Indonesia","SMGR"],"TINS.JK":["Timah","TINS"],
    "TLKM.JK":["Telkom","TLKM"],"TOWR.JK":["Sarana Menara","TOWR"],
    "UNTR.JK":["United Tractors","UNTR"],"UNVR.JK":["Unilever","UNVR"],
    "ESSA.JK":["Surya Esa","ESSA"],"BYAN.JK":["Bayan Resources","BYAN"],
    "CTRA.JK":["Citraland","Ciputra","CTRA"],"PWON.JK":["Pakuwon","PWON"],
    "SMRA.JK":["Summarecon","SMRA"],"JPFA.JK":["Japfa","JPFA"],
    "HEAL.JK":["Hermina","HEAL"],"BJTM.JK":["Bank Jatim","BJTM"],
    "BJBR.JK":["Bank BJB","BJBR"],"INKP.JK":["Indah Kiat","INKP"],
    "INTP.JK":["Indocement","INTP"],"ISAT.JK":["Indosat","Ooredoo","ISAT"],
}

SHARIA_COMPLIANT = {
    "ADRO.JK":True,"AGRO.JK":False,"AKRA.JK":True,"AMRT.JK":True,
    "ANTM.JK":True,"ASII.JK":True,"BBCA.JK":False,"BBNI.JK":False,
    "BBRI.JK":False,"BBTN.JK":False,"BMRI.JK":False,"BRIS.JK":True,
    "BUKA.JK":True,"CPIN.JK":True,"EMTK.JK":True,"EXCL.JK":True,
    "GOTO.JK":True,"HRUM.JK":True,"ICBP.JK":True,"INCO.JK":True,
    "INDF.JK":True,"INDY.JK":True,"ITMG.JK":True,"KLBF.JK":True,
    "MBMA.JK":True,"MDKA.JK":True,"MEDC.JK":True,"MIKA.JK":True,
    "MYOR.JK":True,"PGAS.JK":True,"PTBA.JK":True,"SIDO.JK":True,
    "SMGR.JK":True,"TINS.JK":True,"TLKM.JK":True,"TOWR.JK":True,
    "UNTR.JK":True,"UNVR.JK":True,"ESSA.JK":True,"BYAN.JK":True,
    "CTRA.JK":True,"PWON.JK":True,"SMRA.JK":True,"JPFA.JK":True,
    "HEAL.JK":True,"BJTM.JK":False,"BJBR.JK":False,"INKP.JK":True,
    "INTP.JK":True,"ISAT.JK":True,
}

REGIME_ALLOCATIONS = {
    "CRISIS":     {"Defensive":0.30,"Cash":0.70},
    "RISK_OFF":   {"Defensive":0.30,"Banks":0.10,"Cash":0.60},
    "TIGHTENING": {"Energy":0.35,"Commodities":0.30,"Cash":0.35},
    "INFLATION":  {"Energy":0.50,"Commodities":0.30,"Cash":0.20},
    "RISK_ON":    {"Tech":0.35,"Banks":0.30,"Commodities":0.20,"Cash":0.15},
    "NEUTRAL":    {"Banks":0.35,"Defensive":0.20,"Commodities":0.15,"Cash":0.30},
}

REGIME_COLORS = {
    "CRISIS":"🔴","RISK_OFF":"🟠","TIGHTENING":"🟡",
    "INFLATION":"🟡","RISK_ON":"🟢","NEUTRAL":"⚪",
}

# Gen 4: Trade allocation caps
TRADE_ALLOCATION_CAPS = {"POSITION":0.50,"SWING":0.30,"SCALP":0.20}

TRADE_TYPES = {
    "SCALP":    {"sl_atr":0.75,"tp_atr":1.00,"hold_days":1, "size_mult":0.50,"emoji":"⚡"},
    "SWING":    {"sl_atr":1.50,"tp_atr":3.00,"hold_days":5, "size_mult":1.00,"emoji":"🌊"},
    "POSITION": {"sl_atr":2.50,"tp_atr":6.00,"hold_days":20,"size_mult":1.50,"emoji":"🏦"},
}

DEFAULT_WEIGHTS = {"macro":0.30,"technical":0.30,"sentiment":0.20,"fundamental":0.20}

# ═══════════════════════════════════════════════════════════
# MACRO SIGNAL ENGINE
# ═══════════════════════════════════════════════════════════
MACRO_TICKERS = {
    "Nasdaq": (["^IXIC", "QQQ"], True),
    "DXY":    (["DX-Y.NYB", "UUP"], False),
    "US10Y":  (["^TNX"], False),
    "VIX":    (["^VIX"], False),
}
COMMODITY_TICKERS = {
    "Crude Oil":"BZ=F","Gold":"GC=F","USD/IDR":"USDIDR=X",
    "Nickel":"VALE","CPO":"PALM.L",
}

def _download_close(tickers, period="6mo"):
    """Return first non-empty close series from a ticker or fallback ticker list."""
    if isinstance(tickers, str):
        tickers = [tickers]
    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(period=period, auto_adjust=False)
            if hist is None or hist.empty:
                continue
            if "Close" in hist.columns:
                close = hist["Close"].dropna()
            elif "close" in hist.columns:
                close = hist["close"].dropna()
            else:
                continue
            if len(close) >= 22:
                return close
        except Exception:
            continue
    return pd.Series(dtype=float)

def _weighted_signal(close, up_is_good=True):
    if len(close)<22: return 0.0,{},"N/A","N/A"
    ret_1d =float((close.iloc[-1]-close.iloc[-2]) /close.iloc[-2])
    ret_5d =float((close.iloc[-1]-close.iloc[-6]) /close.iloc[-6])
    ret_20d=float((close.iloc[-1]-close.iloc[-21])/close.iloc[-21])
    vol_20d=max(float(close.pct_change().rolling(20).std().iloc[-1]),0.001)
    s1d =np.clip(ret_1d /vol_20d,              -2,2)/2
    s5d =np.clip(ret_5d /(vol_20d*np.sqrt(5)), -2,2)/2
    s20d=np.clip(ret_20d/(vol_20d*np.sqrt(20)),-2,2)/2
    raw =s1d*0.50+s5d*0.30+s20d*0.20
    score=round(float(np.clip(raw if up_is_good else -raw,-1,1)),3)
    trend={"1d":f"{'▲' if ret_1d>0 else '▼'} {abs(ret_1d*100):.2f}%",
           "5d":f"{'▲' if ret_5d>0 else '▼'} {abs(ret_5d*100):.2f}%",
           "20d":f"{'▲' if ret_20d>0 else '▼'} {abs(ret_20d*100):.2f}%",
           "strength":score}
    return score,trend,f"{float(close.iloc[-1]):.2f}",f"{'▲' if ret_1d>0 else '▼'} {abs(ret_1d*100):.2f}%"

def get_macro_score():
    scores,details={},{}
    for name,(tickers,up_is_good) in MACRO_TICKERS.items():
        close = _download_close(tickers, period="6mo")
        if len(close) >= 22:
            score,trend,value,change=_weighted_signal(close,up_is_good)
            scores[name]=score
            details[name]={"score":score,"value":value,"change":change,"trend":trend}
        else:
            scores[name]=0.0
            details[name]={"score":0.0,"value":"N/A","change":"N/A","trend":{}}
    total=round(sum(scores.values()),3)
    if   total>=1.5: stance="AGGRESSIVE"
    elif total>=0.5: stance="MODERATE"
    elif total>=-0.5:stance="CAUTION"
    else:            stance="DEFENSIVE"
    return total,stance,scores,details

def get_macro_alignment(score_map, period="1mo"):
    """
    Build multi-horizon macro alignment:
    - daily: current weighted macro score
    - 1m rolling trend
    - ytd trend
    Returns combined score, confidence level, and per-horizon breakdown.
    """
    weights = {"daily": 0.5, "1m": 0.3, "ytd": 0.2}
    horizons = {"daily": 1, "1m": 21}
    horizon_scores = {"daily": 0.0, "1m": 0.0, "ytd": 0.0}

    for name, (tickers, up_is_good) in MACRO_TICKERS.items():
        close = _download_close(tickers, period="1y")
        if len(close) < 30:
            continue
        vol_20d = max(float(close.pct_change().rolling(20).std().iloc[-1]), 0.001)

        # Daily and 1-month rolling trend
        for h_name, lb in horizons.items():
            if len(close) > lb:
                ret = float((close.iloc[-1] - close.iloc[-(lb + 1)]) / close.iloc[-(lb + 1)])
                z = np.clip(ret / (vol_20d * np.sqrt(max(lb, 1))), -2, 2) / 2
                horizon_scores[h_name] += float(z if up_is_good else -z)

        # YTD trend
        try:
            ytd_close = close[close.index.year == pd.Timestamp.utcnow().year]
            if len(ytd_close) >= 2:
                ytd_ret = float((ytd_close.iloc[-1] - ytd_close.iloc[0]) / ytd_close.iloc[0])
                z_ytd = np.clip(ytd_ret / max(vol_20d * np.sqrt(len(ytd_close)), 0.001), -2, 2) / 2
                horizon_scores["ytd"] += float(z_ytd if up_is_good else -z_ytd)
        except Exception:
            pass

    # Normalize by number of macro inputs
    denom = max(len(MACRO_TICKERS), 1)
    for k in horizon_scores:
        horizon_scores[k] = round(float(np.clip(horizon_scores[k] / denom, -1, 1)), 3)

    combined = round(
        horizon_scores["daily"] * weights["daily"] +
        horizon_scores["1m"] * weights["1m"] +
        horizon_scores["ytd"] * weights["ytd"], 3
    )
    signs = [np.sign(horizon_scores["daily"]), np.sign(horizon_scores["1m"]), np.sign(horizon_scores["ytd"])]
    agreement = abs(sum(signs)) / 3 if len(signs) else 0
    mag = abs(combined)
    if agreement >= 0.9 and mag >= 0.25:
        conf = "HIGH"
    elif agreement >= 0.5 and mag >= 0.10:
        conf = "MEDIUM"
    else:
        conf = "LOW"

    return combined, conf, horizon_scores

def get_commodity_context():
    context={}
    for name,ticker in COMMODITY_TICKERS.items():
        close = _download_close(ticker, period="6mo")
        if len(close) >= 22:
            score,trend,value,change=_weighted_signal(close,up_is_good=True)
            context[name]={"value":value,"change":change,
                           "direction":1 if score>0 else(-1 if score<0 else 0),
                           "score":score,"trend":trend}
        else:
            context[name]={"value":"N/A","change":"N/A","direction":0,"score":0,"trend":{}}
    return context

# ═══════════════════════════════════════════════════════════
# GEN 4 NEW: MACRO MONEY FLOW TRACKER
# ═══════════════════════════════════════════════════════════
SECTOR_ETFS = {
    "Energy":      ["ADRO.JK","ITMG.JK","PTBA.JK"],
    "Banks":       ["BBCA.JK","BBRI.JK","BMRI.JK"],
    "Commodities": ["ANTM.JK","INCO.JK","MDKA.JK"],
    "Defensive":   ["ICBP.JK","UNVR.JK","MYOR.JK"],
    "Tech":        ["GOTO.JK","EMTK.JK","BUKA.JK"],
}

def get_money_flow(lookback_days=20):
    """
    Track capital rotation across sectors.
    Returns sector performance + flow direction over 5d and 20d.
    Identifies where money is moving TO.
    """
    flow = {}
    for sector, tickers in SECTOR_ETFS.items():
        returns_5d, returns_20d = [], []
        for ticker in tickers:
            try:
                hist  = yf.Ticker(ticker).history(period="60d")["Close"].dropna()
                if len(hist) >= 21:
                    r5  = float((hist.iloc[-1] - hist.iloc[-6])  / hist.iloc[-6]  * 100)
                    r20 = float((hist.iloc[-1] - hist.iloc[-21]) / hist.iloc[-21] * 100)
                    returns_5d.append(r5)
                    returns_20d.append(r20)
            except:
                continue
        if returns_5d:
            avg5  = round(float(np.mean(returns_5d)),  2)
            avg20 = round(float(np.mean(returns_20d)), 2)
            # Momentum: accelerating inflow if 5d > 20d trend
            momentum = "INFLOW ↑"  if avg5 > avg20 * 0.5 and avg5 > 0 else \
                       "OUTFLOW ↓" if avg5 < 0 else "STABLE →"
            flow[sector] = {
                "5d_return":  f"{avg5:+.2f}%",
                "20d_return": f"{avg20:+.2f}%",
                "momentum":   momentum,
                "score":      round(avg5 * 0.6 + avg20 * 0.4, 2),
            }
    # Rank sectors by score
    ranked = sorted(flow.items(), key=lambda x: x[1]["score"], reverse=True)
    top_sector = ranked[0][0] if ranked else "Unknown"
    bot_sector = ranked[-1][0] if ranked else "Unknown"

    # Rotation narrative
    if len(ranked) >= 2:
        narrative = (f"Capital flowing: {ranked[-1][0]} → {ranked[0][0]}. "
                     f"{ranked[0][0]} leads with {ranked[0][1]['5d_return']} (5d).")
    else:
        narrative = "Insufficient data for rotation analysis."

    return flow, top_sector, bot_sector, narrative

# ═══════════════════════════════════════════════════════════
# REGIME + ALLOCATION
# ═══════════════════════════════════════════════════════════
def detect_regime(scores, details, commodity_context):
    vix_score   =scores.get("VIX",   0.0)
    dxy_score   =scores.get("DXY",   0.0)
    yield_score =scores.get("US10Y", 0.0)
    nasdaq_score=scores.get("Nasdaq",0.0)
    oil_score   =commodity_context.get("Crude Oil",{}).get("score",0.0)
    vix_close = _download_close(["^VIX"], period="1mo")
    vix_raw = float(vix_close.iloc[-1]) if len(vix_close) else 20.0
    if   vix_raw>30:
        return "CRISIS",   round(min(1.0,(vix_raw-30)/20)*100,1),f"VIX={vix_raw:.1f} — extreme fear."
    elif vix_score<-0.4 and nasdaq_score<-0.3:
        return "RISK_OFF", round(abs((vix_score+nasdaq_score)/2)*100,1),"VIX rising + Nasdaq falling."
    elif dxy_score<-0.3 and yield_score<-0.3:
        return "TIGHTENING",round(abs((dxy_score+yield_score)/2)*100,1),"Dollar + yields rising."
    elif oil_score>0.3 and yield_score<-0.2:
        return "INFLATION",round((oil_score+abs(yield_score))/2*100,1),"Oil + yields rising — inflation."
    elif nasdaq_score>0.3 and vix_score>0.2:
        return "RISK_ON",  round((nasdaq_score+vix_score)/2*100,1),"Nasdaq strong + VIX falling."
    else:
        return "NEUTRAL",  50.0,"No dominant regime signal."

def auto_threshold(regime, macro_score):
    base={"RISK_ON":0.15,"NEUTRAL":0.25,"INFLATION":0.30,
          "TIGHTENING":0.35,"RISK_OFF":0.40,"CRISIS":0.99}.get(regime,0.25)
    if macro_score<0: base+=0.05
    return round(base,2)

def get_allocation(regime, macro_total):
    base=REGIME_ALLOCATIONS.get(regime,REGIME_ALLOCATIONS["NEUTRAL"]).copy()
    if macro_total<0:
        extra=min(0.20,abs(macro_total)*0.10)
        base["Cash"]=base.get("Cash",0)+extra
        non_cash={k:v for k,v in base.items() if k!="Cash"}
        total_nc=sum(non_cash.values())
        if total_nc>0:
            for k in non_cash: base[k]=round(base[k]/total_nc*(1-base["Cash"]),3)
    total=sum(base.values())
    return {k:round(v/total,3) for k,v in base.items()}

def recommend_sector(macro_score, scores, commodity_context):
    nasdaq_up=scores.get("Nasdaq",0.0)>0.1
    yield_up =scores.get("US10Y", 0.0)<-0.1
    oil_up   =commodity_context.get("Crude Oil",{}).get("score",0.0)>0.1
    nickel_up=commodity_context.get("Nickel",   {}).get("score",0.0)>0.1
    signals={"nasdaq_up":nasdaq_up,"yield_up":yield_up,
             "oil_up":oil_up,"nickel_up":nickel_up,
             "vix_down":scores.get("VIX",0.0)>0.1,
             "dxy_up":scores.get("DXY",0.0)<-0.1}
    if   macro_score<=-0.5:          return "Defensive",  "Macro negative — protect capital.",signals
    elif oil_up and yield_up:        return "Energy",     "Oil + yields rising — inflation.",signals
    elif nickel_up:                  return "Commodities","Nickel strength — ANTM, INCO, MDKA.",signals
    elif macro_score>=1.0 and nasdaq_up: return "Tech",  "Strong risk-on + Nasdaq — Tech.",signals
    elif macro_score>=0.5:           return "Banks",      "Stable positive macro — Banks lead.",signals
    else:                            return "Defensive",  "Mixed signals — sit out.",signals

def allocate_trades_by_sector(allocation, portfolio_value, high_beta_pct=0.15):
    cash_pct     =allocation.get("Cash",0.30)
    investable   =portfolio_value*(1-cash_pct)
    high_beta_idr=investable*high_beta_pct
    core_idr     =investable*(1-high_beta_pct)
    non_cash_total=sum(v for k,v in allocation.items() if k!="Cash")
    sector_budgets={}
    for sector,pct in allocation.items():
        if sector=="Cash": continue
        weight=pct/non_cash_total if non_cash_total>0 else 0
        sector_budgets[sector]=round(core_idr*weight,0)
    return {"sector_budgets":sector_budgets,
            "high_beta_budget":round(high_beta_idr,0),
            "cash_reserve":round(portfolio_value*cash_pct,0),
            "investable":round(investable,0)}

# ═══════════════════════════════════════════════════════════
# GEN 4 NEW: SECTOR → ACTION TRANSLATOR
# ═══════════════════════════════════════════════════════════
def sector_action_translator(sector, macro_score, scores, commodity_context, money_flow):
    """
    Converts "Energy is bullish" into specific execution instructions.
    Returns: breakout names, pullback names, ignore names, strategy text.
    """
    tier_data = SECTORS.get(sector, {})
    t1 = tier_data.get("T1", [])
    t2 = tier_data.get("T2", [])
    t3 = tier_data.get("T3", [])

    flow = money_flow.get(sector, {})
    flow_score = flow.get("score", 0)

    # Determine sector strength
    if flow_score > 2:
        strength = "STRONG"
    elif flow_score > 0:
        strength = "MODERATE"
    elif flow_score > -2:
        strength = "WEAK"
    else:
        strength = "AVOID"

    # Generate action map
    if strength == "STRONG":
        strategy = (
            f"**{sector} is in STRONG inflow ({flow.get('5d_return','?')}).**\n\n"
            f"🚀 **Breakouts → BUY NOW:** Focus {', '.join(t1[:2])} — lead names with volume.\n"
            f"🎯 **Pullbacks → LIMIT ORDER:** {', '.join(t2[:1])} on dips to MA20.\n"
            f"⚠️ **Speculative only:** {', '.join(t3[:1])} — smaller size, tighter SL.\n\n"
            f"**Entry type:** Buy stop above yesterday's high on T1 names."
        )
        breakouts = t1
        pullbacks = t2
        ignore    = []

    elif strength == "MODERATE":
        strategy = (
            f"**{sector} is MODERATE ({flow.get('5d_return','?')}).**\n\n"
            f"🎯 **Pullbacks only → LIMIT ORDER:** {', '.join(t1[:2])} on MA20 touch.\n"
            f"👀 **Watch:** {', '.join(t2[:1])} — wait for volume confirmation.\n"
            f"🚫 **Ignore:** {', '.join(t3)} — too risky in moderate flow.\n\n"
            f"**Entry type:** Limit orders only. Don't chase breakouts."
        )
        breakouts = []
        pullbacks = t1
        ignore    = t3

    elif strength == "WEAK":
        strategy = (
            f"**{sector} is WEAK ({flow.get('5d_return','?')}). Don't force trades.**\n\n"
            f"👀 **Watch only:** {', '.join(t1[:1])} — wait for regime shift.\n"
            f"🚫 **Ignore:** {', '.join(t2 + t3)} — outflows active.\n\n"
            f"**Entry type:** No new entries. Protect existing positions."
        )
        breakouts = []
        pullbacks = []
        ignore    = t2 + t3

    else:  # AVOID
        strategy = (
            f"**{sector} is in OUTFLOW. AVOID entirely.**\n\n"
            f"🚫 All {sector} names: IGNORE until regime improves.\n\n"
            f"**Entry type:** NONE. Wait for next regime change."
        )
        breakouts = []
        pullbacks = []
        ignore    = t1 + t2 + t3

    return {
        "strength":   strength,
        "strategy":   strategy,
        "breakouts":  breakouts,
        "pullbacks":  pullbacks,
        "ignore":     ignore,
        "flow_score": flow_score,
    }

# ═══════════════════════════════════════════════════════════
# MOMENTUM SCREEN
# ═══════════════════════════════════════════════════════════
def _rsi_series(close, period=14):
    delta=close.diff()
    gain =delta.clip(lower=0).rolling(period).mean()
    loss =(-delta.clip(upper=0)).rolling(period).mean()
    rs   =gain/loss.replace(0,np.nan)
    return 100-(100/(1+rs))

def fast_momentum_screen(universe=None, top_n=15):
    if universe is None: universe=IDX_UNIVERSE
    print(f"Downloading {len(universe)} stocks...")
    raw=yf.download(universe,period="1y",group_by="ticker",
                    auto_adjust=True,progress=False,threads=True)
    try:
        ihsg    =yf.Ticker("^JKSE").history(period="1mo")["Close"].dropna()
        ihsg_ret=float((ihsg.iloc[-1]-ihsg.iloc[0])/ihsg.iloc[0])
    except:
        ihsg_ret=0.0
    try:
        ihsg_full=yf.Ticker("^JKSE").history(period="1y")["Close"].dropna().pct_change().dropna()
    except:
        ihsg_full=None

    rows,raw_data_dict=[],{}
    for ticker in universe:
        try:
            df=(raw[ticker] if len(universe)>1 else raw).dropna()
            if len(df)<30: continue
            raw_data_dict[ticker]=df
            close,volume=df["Close"],df["Volume"]
            price=float(close.iloc[-1])

            rsi      =float(_rsi_series(close).iloc[-1])
            rsi_score=(1.0 if 50<rsi<70 else 0.5 if 40<rsi<=50 else -0.5 if rsi>=75 else -1.0)
            vol_avg  =float(volume.rolling(20).mean().iloc[-1])
            vol_today=float(volume.iloc[-1])
            vol_ratio=vol_today/vol_avg if vol_avg>0 else 1.0
            vol_score=float(np.clip(vol_ratio-1,-1,1))
            high_52w =float(close.rolling(252,min_periods=20).max().iloc[-1])
            proximity=price/high_52w if high_52w>0 else 0.5
            prox_score=float(np.clip((proximity-0.7)/0.3,-1,1))
            lookback =min(20,len(close)-1)
            mom_20   =float((close.iloc[-1]-close.iloc[-lookback])/close.iloc[-lookback])
            mom_score=float(np.clip(mom_20*5,-1,1))
            rs_score =float(np.clip((mom_20-ihsg_ret)*5,-1,1))

            beta=1.0
            if ihsg_full is not None:
                sr=close.pct_change().dropna()
                common=sr.index.intersection(ihsg_full.index)
                if len(common)>30:
                    sv=sr.loc[common].values; mv=ihsg_full.loc[common].values
                    beta=round(float(np.cov(sv,mv)[0,1]/np.var(mv)) if np.var(mv)>0 else 1.0,2)

            adr=float(((df["High"]-df["Low"])/df["Close"]).rolling(5).mean().iloc[-1])*100
            momentum=round(rsi_score*0.20+vol_score*0.20+
                           prox_score*0.25+mom_score*0.20+rs_score*0.15,3)
            rows.append({
                "ticker":ticker,"price":round(price,0),"momentum":momentum,
                "rsi":round(rsi,1),"vol_ratio":round(vol_ratio,2),
                "52w_prox":f"{proximity*100:.1f}%","20d_return":f"{mom_20*100:.1f}%",
                "vs_ihsg":f"{(mom_20-ihsg_ret)*100:.1f}%","beta":beta,
                "high_beta":beta>1.2,"sharia":SHARIA_COMPLIANT.get(ticker,True),
                "_rsi":rsi,"_vol_ratio":vol_ratio,"_mom_20":mom_20,"_adr":adr,"_beta":beta,
            })
        except: continue

    result=pd.DataFrame(rows).sort_values("momentum",ascending=False).reset_index(drop=True)
    return result,result.head(top_n)["ticker"].tolist(),raw_data_dict

def get_technical_score(ticker, raw_data=None):
    try:
        if raw_data and ticker in raw_data:
            df=raw_data[ticker].copy(); df.columns=[c.lower() for c in df.columns]
        else:
            df=yf.Ticker(ticker).history(period="6mo"); df.columns=[c.lower() for c in df.columns]
        if len(df)<30: return 0,{}
        rsi_s    =_ta_rsi(df["close"],14)
        rsi      =float(rsi_s.iloc[-1]) if rsi_s is not None else 50.0
        rsi_score=float(np.clip((rsi-50)/50,-1,1))
        macd_line,signal_line=_ta_macd(df["close"])
        macd_score=1.0 if float(macd_line.iloc[-1])>float(signal_line.iloc[-1]) else -1.0
        ma50 =float(df["close"].rolling(50, min_periods=10).mean().iloc[-1])
        ma200=float(df["close"].rolling(200,min_periods=50).mean().iloc[-1]) if len(df)>=50 else None
        price=float(df["close"].iloc[-1])
        if ma200 and not np.isnan(ma200):
            if   price>ma50>ma200: ma_score=1.0
            elif price>ma50:       ma_score=0.5
            elif price<ma50<ma200: ma_score=-1.0
            else:                  ma_score=-0.5
        else:
            ma_score=1.0 if price>ma50 else -1.0
        vol_avg  =float(df["volume"].rolling(20,min_periods=5).mean().iloc[-1])
        vol_today=float(df["volume"].iloc[-1])
        vol_score=float(np.clip((vol_today/vol_avg-1),-1,1)) if vol_avg>0 else 0.0

        # Gen 4: volatility sub-score
        vol_20d   = float(df["close"].pct_change().rolling(20).std().iloc[-1]) * 100
        vol_regime= "LOW" if vol_20d < 1.5 else "NORMAL" if vol_20d < 3 else "HIGH"

        details={
            "RSI":round(rsi,1),"RSI Score":round(rsi_score,2),
            "MACD Score":round(macd_score,2),"MA Score":round(ma_score,2),
            "Volume Score":round(vol_score,2),
            "Price":round(price,0),"MA50":round(ma50,0),
            "above_ma50":price>ma50,
            "above_ma200":(price>ma200) if ma200 else None,
            "Volatility":f"{vol_20d:.1f}%","Vol Regime":vol_regime,
        }
        return round(float(np.mean([rsi_score,macd_score,ma_score,vol_score])),2),details
    except Exception as e:
        return 0,{"error":str(e)}

def _score_pe(v):
    if v is None or np.isnan(v) or v<=0: return 0
    return 1.0 if v<10 else 0.5 if v<15 else 0.0 if v<25 else -0.5 if v<40 else -1.0
def _score_pb(v):
    if v is None or np.isnan(v) or v<=0: return 0
    return 1.0 if v<1 else 0.5 if v<2 else 0.0 if v<3 else -0.5
def _score_roe(v):
    if v is None or np.isnan(v): return 0
    return 1.0 if v>0.20 else 0.5 if v>0.10 else 0.0 if v>0 else -1.0
def _score_de(v):
    if v is None or np.isnan(v) or v<0: return 0
    return 1.0 if v<0.5 else 0.5 if v<1.0 else 0.0 if v<2.0 else -0.5

ANALYST_MAP={"strong_buy":1.0,"buy":0.75,"hold":0.0,"underperform":-0.75,"sell":-1.0}

def get_fundamental_score(ticker):
    fresh = _cache_get(_FUNDAMENTAL_CACHE, ticker, _FUNDAMENTAL_TTL_SECONDS, allow_stale=False)
    if fresh is not None:
        return fresh
    stale = _cache_get(_FUNDAMENTAL_CACHE, ticker, _FUNDAMENTAL_TTL_SECONDS, allow_stale=True)

    try:
        info   =yf.Ticker(ticker).info
        pe,pb  =info.get("trailingPE"),info.get("priceToBook")
        roe,de =info.get("returnOnEquity"),info.get("debtToEquity")
        analyst=info.get("recommendationKey","hold")
        if de: de=float(de)/100
        def _s(v): return float(v) if v is not None else None
        sc=[_score_pe(_s(pe)),_score_pb(_s(pb)),_score_roe(_s(roe)),
            _score_de(_s(de)),ANALYST_MAP.get(str(analyst).lower(),0.0)]
        details={"P/E":round(float(pe),2) if pe else "N/A",
                 "P/B":round(float(pb),2) if pb else "N/A",
                 "ROE":f"{float(roe)*100:.1f}%" if roe else "N/A",
                 "D/E":round(float(de),2) if de else "N/A","Analyst":analyst,
                 "PE Score":sc[0],"PB Score":sc[1],"ROE Score":sc[2],
                 "DE Score":sc[3],"Analyst Score":sc[4]}
        valid=[s for s in sc if s!=0]
        result = (round(float(np.mean(valid)) if valid else 0.0,2), details)
        _cache_set(_FUNDAMENTAL_CACHE, ticker, result)
        return result
    except Exception as e:
        if stale is not None and _is_rate_limited_error(e):
            cached_score, cached_details = stale
            fallback_details = dict(cached_details)
            fallback_details["cache_note"] = (
                "Using cached fundamentals temporarily due to upstream rate limits."
            )
            return cached_score, fallback_details
        return 0,{"error":str(e)}

def is_relevant(headline, ticker):
    names=COMPANY_NAMES.get(ticker,[ticker.replace(".JK","")])
    return any(n.lower() in headline.lower() for n in names)

def get_headlines(ticker, min_headlines=5):
    fresh = _cache_get(_HEADLINES_CACHE, ticker, _HEADLINES_TTL_SECONDS, allow_stale=False)
    if fresh is not None:
        return fresh
    stale = _cache_get(_HEADLINES_CACHE, ticker, _HEADLINES_TTL_SECONDS, allow_stale=True)

    headlines=[]
    try:
        for item in yf.Ticker(ticker).news:
            try:
                h=item["content"]["title"]
                if is_relevant(h,ticker):
                    headlines.append({"ticker":ticker,"headline":h})
            except KeyError: continue
    except: pass
    if len(headlines)<min_headlines:
        try:
            q  =COMPANY_NAMES.get(ticker,[ticker.replace(".JK","")])[0]
            url=f"https://news.google.com/rss/search?q={quote(q)}+stock&hl=en&gl=US&ceid=US:en"
            for e in feedparser.parse(url).entries[:20]:
                if is_relevant(e.title,ticker):
                    headlines.append({"ticker":ticker,"headline":e.title})
                if len(headlines)>=15: break
        except: pass
    if headlines:
        _cache_set(_HEADLINES_CACHE, ticker, headlines)
        return headlines
    if stale is not None:
        return stale
    return headlines

SCORE_MAP={"positive":1,"neutral":0,"negative":-1}

def get_sentiment_score(ticker):
    headlines=get_headlines(ticker)
    if not headlines: return 0.0,[]
    df=pd.DataFrame(headlines)
    df["sentiment"]=_get_sentiment_labels(headlines)
    df["score"]=df["sentiment"].map(SCORE_MAP)
    return round(float(df["score"].mean()),2),df.to_dict("records")

# ═══════════════════════════════════════════════════════════
# GEN 4 NEW: COMPOSITE SCORE BREAKDOWN (EXPLAINABILITY)
# ═══════════════════════════════════════════════════════════
def build_score_breakdown(ticker, macro_norm, tech_s, sent_s, fund_s,
                          tech_details, screen_row, weights):
    """
    Decomposes the composite score into sub-factors with plain-English explanations.
    Returns a breakdown dict the UI can render directly.
    """
    rsi       = tech_details.get("RSI", 50)
    vol_score = tech_details.get("Volume Score", 0)
    ma_score  = tech_details.get("MA Score", 0)
    macd_score= tech_details.get("MACD Score", 0)
    vol_regime= tech_details.get("Vol Regime", "NORMAL")

    mom_20    = screen_row.get("_mom_20", 0) if screen_row else 0
    vol_ratio = screen_row.get("_vol_ratio", 1) if screen_row else 1
    beta      = screen_row.get("_beta", 1) if screen_row else 1

    breakdown = {
        "Momentum":   {
            "score": round(np.clip(mom_20 * 5, -1, 1), 2),
            "raw":   f"{mom_20*100:.1f}% (20d)",
            "signal": "🟢 Strong uptrend" if mom_20 > 0.05 else
                      "🟡 Mild move" if mom_20 > 0 else "🔴 Downtrend",
        },
        "Volume":     {
            "score": round(vol_score, 2),
            "raw":   f"{vol_ratio:.1f}x avg",
            "signal": "🟢 Volume spike — conviction" if vol_ratio > 1.5 else
                      "🟡 Normal volume" if vol_ratio > 0.8 else "🔴 Low volume — weak",
        },
        "Trend":      {
            "score": round(ma_score, 2),
            "raw":   f"MA Score: {ma_score:+.2f}",
            "signal": "🟢 Price above MA50 + MA200" if ma_score >= 1.0 else
                      "🟡 Price above MA50" if ma_score >= 0.5 else "🔴 Below MA50",
        },
        "Volatility": {
            "score": round(np.clip(1 - beta * 0.3, -1, 1), 2),
            "raw":   f"β={beta:.2f}, Vol={vol_regime}",
            "signal": "🟢 Low volatility — stable" if vol_regime == "LOW" else
                      "🟡 Normal volatility" if vol_regime == "NORMAL" else "🔴 High volatility — risky",
        },
        "Sentiment":  {
            "score": round(sent_s, 2),
            "raw":   f"News score: {sent_s:+.2f}",
            "signal": "🟢 Positive news flow" if sent_s > 0.2 else
                      "🟡 Mixed/neutral news" if sent_s >= 0 else "🔴 Negative sentiment",
        },
        "Fundamental":{
            "score": round(fund_s, 2),
            "raw":   f"Fund score: {fund_s:+.2f}",
            "signal": "🟢 Strong fundamentals" if fund_s > 0.4 else
                      "🟡 Fair value" if fund_s >= 0 else "🔴 Weak fundamentals",
        },
        "Macro":      {
            "score": round(macro_norm, 2),
            "raw":   f"Macro norm: {macro_norm:+.2f}",
            "signal": "🟢 Macro tailwind" if macro_norm > 0.2 else
                      "🟡 Neutral macro" if macro_norm >= 0 else "🔴 Macro headwind",
        },
    }

    # Top 2 reasons this triggered
    scored = sorted(breakdown.items(), key=lambda x: abs(x[1]["score"]), reverse=True)
    top_reasons = [f"{k}: {v['signal'].split(' ',1)[1]}" for k, v in scored[:2]]
    why_triggered = " + ".join(top_reasons)

    return breakdown, why_triggered

def compute_trade_confidence(result, macro, sector_flow):
    """
    Produces 0-100 confidence score + grade bucket for a trade candidate.
    Inputs can be raw dicts from analysis and flow map.
    """
    comp = float(result.get("composite", 0))
    tech = float(result.get("technical", 0))
    sent = float(result.get("sentiment", 0))
    fund = float(result.get("fundamental", 0))
    macro_score = float(macro if isinstance(macro, (int, float)) else macro.get("combined", 0))
    sector_name = ticker_to_sector.get(result.get("ticker"), "Unknown")
    flow_score = 0.0
    if isinstance(sector_flow, dict):
        flow_score = float(sector_flow.get(sector_name, {}).get("score", 0))

    base = (
        np.clip((comp + 1) / 2, 0, 1) * 45 +
        np.clip((tech + 1) / 2, 0, 1) * 20 +
        np.clip((macro_score + 1) / 2, 0, 1) * 15 +
        np.clip((flow_score + 5) / 10, 0, 1) * 10 +
        np.clip((sent + 1) / 2, 0, 1) * 5 +
        np.clip((fund + 1) / 2, 0, 1) * 5
    )
    score = int(round(float(np.clip(base, 0, 100))))
    if score >= 90:
        grade = "A+"
    elif score >= 80:
        grade = "A"
    elif score >= 70:
        grade = "B"
    else:
        grade = "C"
    return score, grade

def explain_rejection(result, checks, macro_score):
    reasons = []
    if not checks.get("Macro positive", True):
        reasons.append("Macro mismatch / weak market backdrop")
    if not checks.get("Technical positive", True):
        reasons.append("Weak momentum / trend structure")
    if not checks.get("Sentiment positive", True):
        reasons.append("Negative or mixed news sentiment")
    if not checks.get("Fundamentals ok", True):
        reasons.append("Fundamental quality below threshold")
    if not checks.get("Playbook not AVOID", True):
        reasons.append("Playbook flagged AVOID for current regime")
    if not checks.get("Regime allows trading", True):
        reasons.append("Risk-off regime; capital preservation mode")

    vol_score = result.get("tech_details", {}).get("Volume Score", 0)
    if vol_score < 0:
        reasons.append("Low volume confirmation")
    if result.get("composite", 0) <= 0.25:
        reasons.append("Composite score below entry threshold")
    if not reasons:
        reasons.append("Relative ranking too weak versus better setups")
    return "; ".join(reasons[:3])

# ═══════════════════════════════════════════════════════════
# PLAYBOOK ENGINE
# ═══════════════════════════════════════════════════════════
def playbook_engine(ticker, composite, tech_details, macro_score, regime, trade_type):
    rsi         = tech_details.get("RSI", 50)
    above_ma50  = tech_details.get("above_ma50", True)
    extended_rsi= rsi > 70
    pullback_zone=45 < rsi < 58
    breakout_zone=60 < rsi < 72 and above_ma50
    strong_macro = macro_score > 1.0
    positive_macro=macro_score > 0

    if regime in ["CRISIS","RISK_OFF"]:
        action,strategy,reason="AVOID","Sit out","Regime too dangerous."
    elif strong_macro and breakout_zone:
        action,strategy,reason=(
            "BUY BREAKOUT","Enter now — buy stop above yesterday's high",
            f"Strong macro ({macro_score:+.2f}) + RSI {rsi:.0f} in breakout zone. Volume confirming.")
    elif positive_macro and extended_rsi:
        action,strategy,reason=(
            "WAIT PULLBACK",f"Set limit at MA50 (Rp {tech_details.get('MA50','?'):,})",
            f"RSI {rsi:.0f} extended — price stretched. Better R:R on pullback.")
    elif positive_macro and pullback_zone and above_ma50:
        action,strategy,reason=(
            "BUY PULLBACK","Enter limit order now — dip in uptrend",
            f"RSI {rsi:.0f} healthy pullback. Price holding MA50. Strong R:R.")
    elif positive_macro and composite > 0.4:
        action,strategy,reason=(
            "ACCUMULATE","Starter position — add on confirmation",
            f"Composite {composite:+.3f} solid but no trigger yet. Build gradually.")
    elif not positive_macro:
        action,strategy,reason="AVOID","Macro too weak","Wait for macro improvement."
    else:
        action,strategy,reason="WATCH","Monitor daily","Potential setup, no trigger yet."

    return {"action":action,"strategy":strategy,"reason":reason,
            "emoji":{"BUY BREAKOUT":"🚀","BUY PULLBACK":"🎯","WAIT PULLBACK":"⏳",
                     "ACCUMULATE":"📦","WATCH":"👀","AVOID":"🚫"}.get(action,"❓")}

def classify_trade_type(ticker, screen_row=None, raw_data=None):
    try:
        if screen_row is not None:
            rsi=screen_row.get("_rsi",50); vol_ratio=screen_row.get("_vol_ratio",1.0)
            mom_20=screen_row.get("_mom_20",0.0); adr=screen_row.get("_adr",2.0)
            beta=screen_row.get("_beta",1.0)
        else:
            if raw_data and ticker in raw_data:
                df=raw_data[ticker].copy(); df.columns=[c.lower() for c in df.columns]
            else:
                df=yf.Ticker(ticker).history(period="3mo"); df.columns=[c.lower() for c in df.columns]
            close=df["close"]; volume=df["volume"]
            rsi=float(_rsi_series(close).iloc[-1])
            vol_avg=float(volume.rolling(20).mean().iloc[-1])
            vol_today=float(volume.iloc[-1])
            vol_ratio=vol_today/vol_avg if vol_avg>0 else 1.0
            lookback=min(20,len(close)-1)
            mom_20=float((close.iloc[-1]-close.iloc[-lookback])/close.iloc[-lookback])
            adr=float(((df["high"]-df["low"])/df["close"]).rolling(5).mean().iloc[-1])*100
            beta=1.0
        if vol_ratio>2.0 and adr>3.0:
            return "SCALP",   f"Vol spike {vol_ratio:.1f}x + ADR {adr:.1f}%"
        elif rsi>55 and vol_ratio>1.3 and mom_20>0.03:
            return "SWING",   f"RSI {rsi:.0f} + momentum {mom_20*100:.1f}%"
        else:
            return "POSITION",f"Steady trend, β={beta:.2f}"
    except:
        return "SWING","Default"

def get_high_beta_plays(raw_data_dict, top_n=3):
    plays=[]
    for ticker,df in raw_data_dict.items():
        try:
            df=df.copy(); df.columns=[c.lower() for c in df.columns]
            close,volume=df["close"],df["volume"]
            price=float(close.iloc[-1])
            vol_avg=float(volume.rolling(20).mean().iloc[-1])
            vol_today=float(volume.iloc[-1])
            vol_ratio=vol_today/vol_avg if vol_avg>0 else 1.0
            mom_5=float((close.iloc[-1]-close.iloc[-5])/close.iloc[-5])
            adr=float(((df["high"]-df["low"])/df["close"]).rolling(5).mean().iloc[-1])*100
            rsi=float(_rsi_series(close).iloc[-1])
            beta_score=(np.clip(vol_ratio/3,0,1)*0.40+np.clip(mom_5*10,0,1)*0.30+
                        np.clip(adr/5,0,1)*0.20+(1 if 50<rsi<75 else 0)*0.10)
            plays.append({"ticker":ticker,"beta_score":round(beta_score,3),
                          "price":round(price,0),"vol_ratio":round(vol_ratio,2),
                          "5d_mom":f"{mom_5*100:.1f}%","adr":f"{adr:.1f}%",
                          "rsi":round(rsi,1),"sharia":SHARIA_COMPLIANT.get(ticker,True)})
        except: continue
    return sorted(plays,key=lambda x:x["beta_score"],reverse=True)[:top_n]

# ═══════════════════════════════════════════════════════════
# GEN 4 NEW: RISK ENGINE (1% capital risk rule)
# ═══════════════════════════════════════════════════════════
def risk_based_sizing(entry_price, stop_loss_price, portfolio_value,
                      risk_pct=0.01, regime="NEUTRAL", trade_type="SWING",
                      max_position_pct=0.05, is_fractional=False, currency_symbol="Rp ", lot_size=100):
    """
    1% risk rule with TWO hard caps:
    1. Max single position = max_position_pct of portfolio (default 5%)
    2. Minimum risk per share check to prevent absurd lot counts
    """
    if entry_price <= 0 or stop_loss_price <= 0 or entry_price <= stop_loss_price:
        return {"lots":0,"amount_idr":"—","risk_idr":"—","risk_pct":"—","pct_raw":0.0,"label":"Invalid SL"}

    regime_mult = {"RISK_ON":1.0,"NEUTRAL":0.8,"TIGHTENING":0.6,
                   "INFLATION":0.7,"RISK_OFF":0.4,"CRISIS":0.0}.get(regime, 0.8)
    type_mult   = TRADE_TYPES.get(trade_type, TRADE_TYPES["SWING"])["size_mult"]

    risk_budget    = portfolio_value * risk_pct * regime_mult * type_mult
    risk_per_share = entry_price - stop_loss_price
    unit_size = 1.0 if is_fractional else float(lot_size)
    risk_per_lot   = risk_per_share * unit_size

    if risk_per_lot <= 0:
        return {"lots":0,"amount_idr":"—","risk_idr":"—","risk_pct":"—","pct_raw":0.0,"label":"Zero risk/lot"}

    # Cap 1: from risk rule
    lots_risk = (risk_budget / risk_per_lot) if is_fractional else int(risk_budget // risk_per_lot)

    # Cap 2: hard position size cap — never more than max_position_pct of portfolio
    max_value  = portfolio_value * max_position_pct
    lots_cap   = (max_value / (entry_price * unit_size)) if is_fractional else int(max_value // (entry_price * unit_size))

    lots = min(lots_risk, lots_cap)
    if is_fractional:
        lots = round(float(lots), 4)
    if lots <= 0:
        return {"lots":0,"amount_idr":"—","risk_idr":f"{currency_symbol}{risk_budget:,.2f}",
                "risk_pct":"—","pct_raw":0.0,
                "label":f"Min 1 unit = {currency_symbol}{entry_price*unit_size:,.2f}"}

    actual_cost = lots * entry_price * unit_size
    actual_risk = lots * risk_per_lot
    capped      = lots < lots_risk  # was the cap triggered?

    return {
        "lots":       lots,
        "amount_idr": f"{currency_symbol}{actual_cost:,.2f}",
        "risk_idr":   f"{currency_symbol}{actual_risk:,.2f}",
        "risk_pct":   f"{actual_risk/portfolio_value*100:.2f}% of portfolio",
        "label":      f"{'⚠️ Size-capped at {:.0f}%'.format(max_position_pct*100) if capped else 'Risk-sized'}",
        "pct_raw":    actual_cost / portfolio_value,
        "was_capped": capped,
    }

def get_trade_setup(ticker, rr_ratio=2.0, trade_type="SWING", raw_data=None):
    try:
        cfg=TRADE_TYPES.get(trade_type,TRADE_TYPES["SWING"])
        if raw_data and ticker in raw_data:
            df=raw_data[ticker].copy(); df.columns=[c.lower() for c in df.columns]
        else:
            df=yf.Ticker(ticker).history(period="3mo"); df.columns=[c.lower() for c in df.columns]
        price=float(df["close"].iloc[-1])
        atr  =float(_ta_atr(df["high"],df["low"],df["close"],14).iloc[-1])
        if np.isnan(atr): atr=price*0.02
        sl_mult=cfg["sl_atr"]; tp_mult=cfg["tp_atr"]*rr_ratio/2.0
        entry_market=round(price,0)
        entry_limit =round(price-0.5*atr,0)
        stop_loss   =round(price-sl_mult*atr,0)
        take_profit =round(price+tp_mult*atr,0)
        stop_pct=round((price-stop_loss)/price*100,2)
        tp_pct  =round((take_profit-price)/price*100,2)
        # Gen 4: order expiry
        hold_days=cfg["hold_days"]
        expiry=pd.Timestamp.now()+pd.Timedelta(days=min(hold_days,3))
        return {"price":price,"entry_market":entry_market,"entry_limit":entry_limit,
                "stop_loss":stop_loss,"take_profit":take_profit,
                "stop_pct":f"-{stop_pct}%","tp_pct":f"+{tp_pct}%",
                "atr":round(atr,0),"rr_ratio":rr_ratio,"hold_days":hold_days,
                "resistance":round(float(df["high"].rolling(20).max().iloc[-1]),0),
                "support":   round(float(df["low"].rolling(20).min().iloc[-1]),0),
                "trade_type":trade_type,
                "order_expiry":expiry.strftime("%Y-%m-%d"),
                "entry_type": "BUY STOP" if trade_type=="SCALP" else "LIMIT ORDER"}
    except: return None

def calculate_lots(price, budget_idr):
    if price<=0 or budget_idr<=0: return 0,0,price*100
    lot_cost=price*100; lots=int(budget_idr//lot_cost)
    return lots,lots*lot_cost,lot_cost

def get_composite(ticker, macro_norm, regime, raw_data=None,
                  screen_rows=None, learned_weights=None):
    weights = learned_weights or DEFAULT_WEIGHTS
    tech_s, tech_d    = get_technical_score(ticker, raw_data)
    fund_s, fund_d    = get_fundamental_score(ticker)
    sent_s, headlines = get_sentiment_score(ticker)

    screen_row = None
    if screen_rows is not None:
        matches = screen_rows[screen_rows["ticker"] == ticker]
        if not matches.empty: screen_row = matches.iloc[0].to_dict()

    trade_type, type_reason = classify_trade_type(ticker, screen_row, raw_data)
    composite = round(
        weights["macro"]       * macro_norm +
        weights["technical"]   * tech_s     +
        weights["sentiment"]   * sent_s     +
        weights["fundamental"] * fund_s, 3
    )
    playbook  = playbook_engine(ticker, composite, tech_d, macro_norm*4, regime, trade_type)
    breakdown, why_triggered = build_score_breakdown(
        ticker, macro_norm, tech_s, sent_s, fund_s, tech_d, screen_row, weights
    )
    return {
        "ticker":ticker,"composite":composite,"macro":round(macro_norm,3),
        "technical":tech_s,"sentiment":sent_s,"fundamental":fund_s,
        "tech_details":tech_d,"fund_details":fund_d,"headlines":headlines,
        "regime":regime,"trade_type":trade_type,"type_reason":type_reason,
        "playbook":playbook,"sharia":SHARIA_COMPLIANT.get(ticker,True),
        "high_beta":screen_row.get("high_beta",False) if screen_row else False,
        "breakdown":breakdown,"why_triggered":why_triggered,
    }

def run_full_analysis(tickers, macro_score, regime, raw_data=None,
                      screen_rows=None, learned_weights=None):
    macro_norm = macro_score/4; results=[]
    weights    = learned_weights or DEFAULT_WEIGHTS
    for ticker in tickers:
        print(f"  Analyzing {ticker}...")
        results.append(get_composite(ticker, macro_norm, regime, raw_data, screen_rows, weights))
    return sorted(results, key=lambda x: x["composite"], reverse=True)

def trade_checklist(result, macro_score, regime, threshold=0.25):
    checks={
        "Macro positive":        macro_score>0,
        "Regime allows trading": regime not in ["CRISIS","RISK_OFF"],
        "Technical positive":    result["technical"]>0,
        "Sentiment positive":    result["sentiment"]>0,
        "Fundamentals ok":       result["fundamental"]>=0,
        f"Composite>{threshold}":result["composite"]>threshold,
        "Playbook not AVOID":    result.get("playbook",{}).get("action","")!="AVOID",
    }
    return checks,all(checks.values())

ticker_to_sector={}
for _s,_t in SECTORS.items():
    for _tk in _t["T1"]+_t["T2"]+_t["T3"]: ticker_to_sector[_tk]=_s

def build_execution_plan(results, macro_score, regime, allocation,
                         portfolio_value, rr_ratio, raw_data,
                         high_beta_plays, threshold=0.25,
                         screen_rows=None, risk_pct=0.01, sector_flow=None,
                         macro_alignment=None):

    budget_map       = allocate_trades_by_sector(allocation, portfolio_value)
    sector_budgets   = budget_map["sector_budgets"]
    high_beta_budget = budget_map["high_beta_budget"]
    investable       = budget_map["investable"]

    # Hard cap: total deployed can never exceed investable capital
    MAX_TOTAL_DEPLOY = investable
    total_deployed   = 0.0

    # Trade type caps (% of investable)
    type_caps = {k: investable * v for k, v in TRADE_ALLOCATION_CAPS.items()}
    type_used = {"POSITION": 0.0, "SWING": 0.0, "SCALP": 0.0}

    plan = {"POSITION":[],"SWING":[],"SCALP":[],"HIGH_BETA":[]}

    for r in results:
        # Stop adding trades if portfolio is full
        if total_deployed >= MAX_TOTAL_DEPLOY:
            break

        checks, passed = trade_checklist(r, macro_score, regime, threshold)
        if not passed:
            continue

        ticker     = r["ticker"]
        trade_type = r["trade_type"]

        # Check type cap
        if type_used.get(trade_type, 0) >= type_caps.get(trade_type, investable):
            continue

        sector     = ticker_to_sector.get(ticker, "Unknown")
        trade      = get_trade_setup(ticker, rr_ratio, trade_type, raw_data)
        if not trade:
            continue

        price      = trade["price"]
        stop_loss  = trade["stop_loss"]

        # Remaining budget available
        remaining = MAX_TOTAL_DEPLOY - total_deployed
        sec_budget = min(sector_budgets.get(sector, investable * 0.1), remaining)

        is_us = not ticker.endswith(".JK")
        unit_size = 1 if is_us else 100
        ccy = "$" if is_us else "Rp "
        sizing = risk_based_sizing(price, stop_loss, portfolio_value,
                                   risk_pct, regime, trade_type,
                                   max_position_pct=0.05,
                                   is_fractional=is_us,
                                   currency_symbol=ccy,
                                   lot_size=unit_size)
        if sizing["lots"] == 0:
            continue

        # Check actual cost fits in remaining budget
        actual_cost = sizing["lots"] * price * unit_size
        if actual_cost > remaining:
            # Try to fit fewer lots
            affordable_lots = (remaining / (price * unit_size)) if is_us else int(remaining // (price * unit_size))
            if affordable_lots <= 0:
                continue
            actual_cost = affordable_lots * price * unit_size
            actual_risk = affordable_lots * (price - stop_loss) * unit_size
            sizing = {
                "lots":       affordable_lots,
                "amount_idr": f"{ccy}{actual_cost:,.2f}",
                "risk_idr":   f"{ccy}{actual_risk:,.2f}",
                "risk_pct":   f"{actual_risk/portfolio_value*100:.2f}% of portfolio",
                "label":      "⚠️ Reduced — near portfolio cap",
                "pct_raw":    actual_cost / portfolio_value,
                "was_capped": True,
            }

        playbook = r.get("playbook", {})
        conf_score, conf_label = compute_trade_confidence(
            r,
            {"combined": macro_alignment if macro_alignment is not None else macro_score / 4},
            sector_flow or {}
        )
        why = (f"Macro:{r['macro']:+.2f} | Tech:{r['technical']:+.2f} | "
               f"Sent:{r['sentiment']:+.2f} | Fund:{r['fundamental']:+.2f}\n"
               f"{playbook.get('emoji','')} {playbook.get('action','')} — {playbook.get('reason','')}")

        entry_obj = {
            "ticker":ticker,"sector":sector,"trade_type":trade_type,
            "composite":r["composite"],"why":why,"breakdown":r.get("breakdown",{}),
            "action":playbook.get("action",""),"strategy":playbook.get("strategy",""),
            "entry":f"{ccy}{trade['entry_limit']:,.2f}",
            "entry_type":trade.get("entry_type","LIMIT ORDER"),
            "stop_loss":f"{ccy}{trade['stop_loss']:,.2f} ({trade['stop_pct']})",
            "take_profit":f"{ccy}{trade['take_profit']:,.2f} ({trade['tp_pct']})",
            "hold_days":trade["hold_days"],
            "order_expiry":trade.get("order_expiry",""),
            "lots":sizing["lots"],"amount":sizing["amount_idr"],
            "risk":sizing.get("risk_idr","—"),"risk_pct_str":sizing.get("risk_pct","—"),
            "pct_raw":sizing["pct_raw"],
            "confidence_score": conf_score,
            "confidence_label": conf_label,
            "sharia":r.get("sharia",True),"high_beta":r.get("high_beta",False),
        }
        plan[trade_type].append(entry_obj)
        type_used[trade_type] = type_used.get(trade_type, 0) + actual_cost
        total_deployed += actual_cost

    # High-beta — only if budget remaining
    per_hb = high_beta_budget / max(len(high_beta_plays), 1)
    for hb in high_beta_plays:
        if total_deployed >= MAX_TOTAL_DEPLOY:
            break
        trade = get_trade_setup(hb["ticker"], rr_ratio, "SCALP", raw_data)
        if not trade:
            continue
        remaining = MAX_TOTAL_DEPLOY - total_deployed
        budget    = min(per_hb, remaining)
        is_us = not hb["ticker"].endswith(".JK")
        unit_size = 1 if is_us else 100
        ccy = "$" if is_us else "Rp "
        sizing    = risk_based_sizing(trade["price"], trade["stop_loss"],
                                      portfolio_value, risk_pct, regime, "SCALP",
                                      max_position_pct=0.05,
                                      is_fractional=is_us,
                                      currency_symbol=ccy,
                                      lot_size=unit_size)
        if sizing["lots"] == 0:
            continue
        actual_cost = sizing["lots"] * trade["price"] * unit_size
        if actual_cost > remaining:
            affordable = (remaining / (trade["price"] * unit_size)) if is_us else int(remaining // (trade["price"] * unit_size))
            if affordable <= 0:
                continue
            actual_cost = affordable * trade["price"] * unit_size
            sizing["lots"] = affordable
            sizing["amount_idr"] = f"{ccy}{actual_cost:,.2f}"
            sizing["pct_raw"]    = actual_cost / portfolio_value

        plan["HIGH_BETA"].append({
            "ticker":hb["ticker"],"sector":ticker_to_sector.get(hb["ticker"],"Unknown"),
            "trade_type":"SCALP","composite":hb["beta_score"],
            "why":f"Vol {hb['vol_ratio']}x | ADR {hb['adr']} | 5d {hb['5d_mom']}",
            "action":"BUY BREAKOUT","strategy":"Momentum chase — tight SL",
            "entry":f"{ccy}{trade['entry_limit']:,.2f}",
            "entry_type":"BUY STOP",
            "stop_loss":f"{ccy}{trade['stop_loss']:,.2f} ({trade['stop_pct']})",
            "take_profit":f"{ccy}{trade['take_profit']:,.2f} ({trade['tp_pct']})",
            "hold_days":1,"order_expiry":trade.get("order_expiry",""),
            "lots":sizing["lots"],"amount":sizing["amount_idr"],
            "risk":sizing.get("risk_idr","—"),"risk_pct_str":sizing.get("risk_pct","—"),
            "pct_raw":sizing["pct_raw"],
            "sharia":hb.get("sharia",True),"high_beta":True,
        })
        total_deployed += actual_cost

    plan["_summary"] = {
        "total_deployed": round(total_deployed, 0),
        "pct_deployed":   round(total_deployed / portfolio_value * 100, 1),
        "cash_reserve":   round(budget_map["cash_reserve"], 0),
        "trade_count":    sum(len(v) for k, v in plan.items() if not k.startswith("_")),
        "type_allocation":{k: round(v/investable*100,1) for k,v in type_used.items() if investable>0},
    }
    return plan

# ═══════════════════════════════════════════════════════════
# GEN 4: BACKTEST V2 (Sharpe + per-regime + per-type)
# ═══════════════════════════════════════════════════════════
def run_backtest(raw_data_dict, universe, threshold=0.3,
                 start_capital=100_000_000, fee_pct=0.002,
                 max_positions=8, alloc_per_trade=0.12):
    all_trades = []
    errors = []
    for ticker in universe:
        if ticker not in raw_data_dict: continue
        try:
            df=raw_data_dict[ticker].copy(); df.columns=[c.lower() for c in df.columns]
            if len(df)<252*2:
                try:
                    df = yf.Ticker(ticker).history(period="3y")
                    df.columns=[c.lower() for c in df.columns]
                except Exception:
                    pass
            if len(df)<60: continue
            close,volume=df["close"],df["volume"]
            rsi    =_rsi_series(close)
            rsi_sc =(rsi-50)/50
            mom    =close.pct_change(20)
            mom_sc =mom.clip(-0.2,0.2)*5
            vol_avg=volume.rolling(20).mean()
            vol_sc =((volume/vol_avg)-1).clip(-1,1)
            ma50   =close.rolling(50,min_periods=10).mean()
            ma_sc  =pd.Series(np.where(close>ma50,0.5,-0.5),index=close.index)
            signal =(rsi_sc*0.30+mom_sc*0.30+vol_sc*0.20+ma_sc*0.20).fillna(0)
            vol_ratio_s=(volume/vol_avg).fillna(1)
            adr_s=((df["high"]-df["low"])/df["close"]).rolling(5).mean().fillna(0.02)*100

            in_trade,entry_px,entry_day,hold,ttype=False,0.0,0,5,"SWING"
            stop_pct, planned_exit_day = 0.08, 0
            for i in range(60,len(close)):
                if not in_trade:
                    if i + 1 >= len(close):
                        continue
                    if float(signal.iloc[i])>threshold:
                        vr=float(vol_ratio_s.iloc[i]); adr=float(adr_s.iloc[i]); rs=float(rsi.iloc[i])
                        if vr>2.0 and adr>3.0:  ttype,hold="SCALP",1
                        elif rs>55 and vr>1.3:  ttype,hold="SWING",5
                        else:                   ttype,hold="POSITION",20
                        stop_pct = {"SCALP": 0.03, "SWING": 0.06, "POSITION": 0.10}.get(ttype, 0.08)
                        entry_day = i + 1  # avoid look-ahead bias
                        entry_px = float(close.iloc[entry_day])
                        planned_exit_day = min(entry_day + hold, len(close) - 1)
                        in_trade = True
                elif i >= entry_day:
                    exit_now = False
                    exit_reason = "time"
                    stop_px = entry_px * (1 - stop_pct)
                    if float(close.iloc[i]) <= stop_px:
                        exit_now = True
                        exit_reason = "stop"
                    elif i >= planned_exit_day:
                        exit_now = True
                    if not exit_now:
                        continue
                    exit_px=float(close.iloc[i])
                    net=(exit_px-entry_px)/entry_px-fee_pct
                    hold_days = max(1, i - entry_day)
                    daily_ret = (1 + net) ** (1 / hold_days) - 1 if (1 + net) > 0 else -1.0
                    all_trades.append({"ticker":ticker,"trade_type":ttype,
                        "entry_date":pd.Timestamp(close.index[entry_day]),"exit_date":pd.Timestamp(close.index[i]),
                        "entry_px":round(entry_px,0),"exit_px":round(exit_px,0),
                        "hold_days":hold_days,"daily_ret":round(daily_ret,6),
                        "return_net":round(net,4),"win":net>0,"exit_reason":exit_reason})
                    in_trade=False
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")
            continue

    if not all_trades: return pd.DataFrame(),{},{}
    df_t=pd.DataFrame(all_trades).sort_values("entry_date").reset_index(drop=True)
    df_t["entry_date"] = pd.to_datetime(df_t["entry_date"])
    df_t["exit_date"] = pd.to_datetime(df_t["exit_date"])

    # Overlap-aware portfolio simulation with fixed per-trade allocation and position cap.
    alloc_per_trade = float(np.clip(alloc_per_trade, 0.01, 1.0))
    max_positions = int(max(1, max_positions))
    equity = float(start_capital)
    open_positions = []
    accepted_idx = []
    event_rows = []
    for idx, row in df_t.sort_values("entry_date").iterrows():
        current_day = row["entry_date"]
        still_open = []
        for pos in open_positions:
            if pos["exit_date"] <= current_day:
                pnl = pos["notional"] * pos["return_net"]
                equity += pnl
                event_rows.append((pos["exit_date"], equity))
            else:
                still_open.append(pos)
        open_positions = still_open
        if len(open_positions) >= max_positions:
            continue
        notional = start_capital * alloc_per_trade
        open_positions.append({
            "exit_date": row["exit_date"],
            "notional": notional,
            "return_net": float(row["return_net"]),
        })
        accepted_idx.append(idx)
    for pos in sorted(open_positions, key=lambda x: x["exit_date"]):
        pnl = pos["notional"] * pos["return_net"]
        equity += pnl
        event_rows.append((pos["exit_date"], equity))

    df_t = df_t.loc[accepted_idx].copy().sort_values("entry_date").reset_index(drop=True)
    if df_t.empty:
        return pd.DataFrame(), {}, {}
    if event_rows:
        event_df = pd.DataFrame(event_rows, columns=["exit_date", "equity"]).sort_values("exit_date")
        df_t = df_t.merge(event_df.drop_duplicates(subset=["exit_date"], keep="last"), on="exit_date", how="left")
        df_t["equity"] = df_t["equity"].ffill().fillna(start_capital)
    else:
        df_t["equity"] = start_capital
    equity_curve=df_t["equity"].values
    peak=np.maximum.accumulate(equity_curve)
    dd=(equity_curve-peak)/peak

    # Sharpe based on per-trade daily-equivalent returns.
    daily_rets = df_t["daily_ret"].astype(float).values
    sharpe = float(np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)) if np.std(daily_rets) > 0 else 0

    # Calmar ratio
    max_dd = float(dd.min())
    elapsed_days = len(pd.bdate_range(df_t["entry_date"].min(), df_t["exit_date"].max()))
    elapsed_days = max(1, int(elapsed_days))
    ann_ret= float((df_t["equity"].iloc[-1]/start_capital) ** (252/elapsed_days) - 1)
    calmar = abs(ann_ret / max_dd) if max_dd != 0 else 0

    # Expectancy
    wins   = df_t[df_t["win"]]["return_net"]
    losses = df_t[~df_t["win"]]["return_net"]
    win_rate=len(wins)/len(df_t) if len(df_t)>0 else 0
    expectancy = (win_rate * wins.mean() - (1-win_rate) * abs(losses.mean())) if len(losses)>0 else 0

    type_stats={}
    for tt in ["SCALP","SWING","POSITION"]:
        sub=df_t[df_t["trade_type"]==tt]
        if len(sub)==0: continue
        sub_rets=sub["daily_ret"].astype(float).values
        sub_sharpe=float(np.mean(sub_rets)/np.std(sub_rets)*np.sqrt(252)) if np.std(sub_rets)>0 else 0
        type_stats[tt]={
            "trades":len(sub),"win_rate":f"{sub['win'].mean()*100:.1f}%",
            "avg_ret":f"{sub['return_net'].mean()*100:.2f}%",
            "sharpe":f"{sub_sharpe:.2f}",
            "best":f"{sub['return_net'].max()*100:.2f}%",
            "worst":f"{sub['return_net'].min()*100:.2f}%",
        }

    stats={
        "Total Trades":  len(df_t),
        "Win Rate":      f"{win_rate*100:.1f}%",
        "Avg Return":    f"{df_t['return_net'].mean()*100:.2f}%",
        "Max Drawdown":  f"{max_dd*100:.2f}%",
        "Sharpe Ratio":  f"{sharpe:.2f}",
        "Calmar Ratio":  f"{calmar:.2f}",
        "Expectancy":    f"{expectancy*100:.2f}%",
        "Final Capital": f"Rp {df_t['equity'].iloc[-1]:,.0f}",
        "Total Return":  f"{(df_t['equity'].iloc[-1]/start_capital-1)*100:.1f}%",
        "Elapsed Days":  elapsed_days,
        "Signals Skipped (Cap)": int(len(all_trades) - len(df_t)),
        "Ticker Errors": int(len(errors)),
    }
    return df_t, stats, type_stats

# ═══════════════════════════════════════════════════════════
# TELEGRAM + IMAGE
# ═══════════════════════════════════════════════════════════
def send_telegram(token, chat_id, message):
    import requests
    try:
        r=requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                        data={"chat_id":chat_id,"text":message,"parse_mode":"Markdown"},timeout=10)
        return r.status_code==200
    except: return False

def send_telegram_photo(token, chat_id, image_path, caption=""):
    import requests
    try:
        with open(image_path,"rb") as f:
            r=requests.post(f"https://api.telegram.org/bot{token}/sendPhoto",
                            data={"chat_id":chat_id,"caption":caption,"parse_mode":"Markdown"},
                            files={"photo":f},timeout=30)
        return r.status_code==200
    except: return False

def build_morning_message(macro_score, stance, regime, regime_conf,
                          rec_sector, allocation, plan, scores,
                          commodity_context, portfolio_value,
                          money_flow_narrative=""):
    today  = datetime.now().strftime("%a, %d %b %Y")
    emoji  = REGIME_COLORS.get(regime,"⚪")
    summary= plan.get("_summary",{})

    nasdaq_up=scores.get("Nasdaq",0)>0.1
    vix_down =scores.get("VIX",  0)>0.1
    yield_up =scores.get("US10Y",0)<-0.1
    oil_up   =commodity_context.get("Crude Oil",{}).get("direction",0)>0

    interp={"RISK_ON":"→ Risk-on. Buy cyclicals. Chase breakouts.",
            "INFLATION":"→ Inflation trade. Energy + commodities lead.",
            "TIGHTENING":"→ Tightening. Real assets. Be selective.",
            "RISK_OFF":"→ Risk-off. Reduce exposure. More cash.",
            "CRISIS":"→ CRISIS. Capital preservation only.",
            "NEUTRAL":"→ Mixed signals. Smaller sizes."}.get(regime,"→ No clear signal.")

    lines=[
        f"📊 *IDX Morning Briefing*",f"_{today}_",f"",
        f"━━━━━━━━━━━━━━━━━━━━━",
        f"🌍 *MACRO RECAP*",
        f"  {'🟢 Nasdaq ↑' if nasdaq_up else '🔴 Nasdaq ↓'}  {'🟢 VIX ↓' if vix_down else '🔴 VIX ↑'}",
        f"  {'🔴 Yields ↑' if yield_up else '🟢 Yields ↓'}  {'🟢 Oil ↑' if oil_up else '⚪ Oil flat'}",
        f"",f"🧠 *INTERPRETATION*",f"  {interp}",
    ]
    if money_flow_narrative:
        lines+=[f"",f"💸 *MONEY FLOW*",f"  {money_flow_narrative}"]

    lines+=[f"",f"🎯 *STRATEGY TODAY*",
            f"  {emoji} {regime} ({regime_conf:.0f}%) | {stance} ({macro_score:+.2f})",
            f"  Focus: *{rec_sector}*",
            f"",f"💼 *ALLOCATION*"]
    for s,p in allocation.items():
        bar="█"*int(p*12)
        lines.append(f"  {s}: {p*100:.0f}% `{bar}`")
    lines+=["","━━━━━━━━━━━━━━━━━━━━━"]

    any_trades=False
    for bucket,(icon,label) in [("POSITION",("🏦","POSITION — 20d")),
                                  ("SWING",   ("🌊","SWING — 5d")),
                                  ("SCALP",   ("⚡","SCALP — 1d")),
                                  ("HIGH_BETA",("🔥","HIGH BETA"))]:
        trades=plan.get(bucket,[])
        if not trades: continue
        any_trades=True
        lines.append(f"{icon} *{label}*")
        for t in trades:
            pb_em={"BUY BREAKOUT":"🚀","BUY PULLBACK":"🎯","WAIT PULLBACK":"⏳",
                   "ACCUMULATE":"📦"}.get(t.get("action",""),"")
            halal="☪️" if t.get("sharia") else ""
            lines.append(
                f"  {pb_em} *{t['ticker']}* {halal} {t['lots']}lot {t['amount']}\n"
                f"    {t['strategy']} | {t.get('entry_type','LIMIT')}\n"
                f"    E:{t['entry']} SL:{t['stop_loss']} TP:{t['take_profit']}\n"
                f"    Risk: {t.get('risk_pct_str','?')} | Exp: {t.get('order_expiry','?')}")
        lines.append("")

    if not any_trades:
        lines+=["⚠️ _No trades today. Sit on hands._",""]
    lines+=["━━━━━━━━━━━━━━━━━━━━━",
            f"💰 Deployed: Rp {summary.get('total_deployed',0):,.0f} ({summary.get('pct_deployed',0):.1f}%)",
            f"💵 Cash: Rp {summary.get('cash_reserve',0):,.0f}",
            f"📊 Type mix: "+" | ".join(f"{k}:{v}%" for k,v in summary.get("type_allocation",{}).items())]
    return "\n".join(lines)

def generate_brief_image(macro_score, stance, regime, rec_sector,
                         allocation, plan, portfolio_value, save_path="brief.png"):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig=plt.figure(figsize=(14,10),facecolor="#0f1117")
    fig.suptitle(f"IDX Morning Briefing — {datetime.now().strftime('%d %b %Y')}",
                 color="white",fontsize=16,fontweight="bold",y=0.98)
    gs=fig.add_gridspec(3,3,hspace=0.5,wspace=0.4,left=0.05,right=0.95,top=0.90,bottom=0.05)

    # Regime banner
    ax0=fig.add_subplot(gs[0,:])
    ax0.set_facecolor("#1e2130"); ax0.set_xlim(0,1); ax0.set_ylim(0,1); ax0.axis("off")
    rc={"RISK_ON":"#4CAF50","NEUTRAL":"#9E9E9E","INFLATION":"#FF9800",
        "TIGHTENING":"#FF9800","RISK_OFF":"#F44336","CRISIS":"#B71C1C"}.get(regime,"#9E9E9E")
    ax0.add_patch(mpatches.FancyBboxPatch((0.01,0.1),0.98,0.8,
                  boxstyle="round,pad=0.02",fc=rc,alpha=0.3,ec=rc))
    ax0.text(0.5,0.65,f"{REGIME_COLORS.get(regime,'')} {regime}",
             ha="center",va="center",color=rc,fontsize=20,fontweight="bold",transform=ax0.transAxes)
    ax0.text(0.5,0.25,f"{stance} ({macro_score:+.2f}) | Focus: {rec_sector}",
             ha="center",va="center",color="white",fontsize=12,transform=ax0.transAxes)

    # Allocation pie
    ax1=fig.add_subplot(gs[1,0]); ax1.set_facecolor("#1e2130")
    colors=["#4CAF50","#2196F3","#FF9800","#9C27B0","#F44336","#9E9E9E"]
    labels=list(allocation.keys()); sizes=list(allocation.values())
    wedges,texts,autos=ax1.pie(sizes,colors=colors[:len(labels)],autopct="%1.0f%%",
                                startangle=90,wedgeprops={"edgecolor":"#0f1117","linewidth":2})
    for a in autos: a.set_color("white"); a.set_fontsize(9)
    ax1.legend(wedges,labels,loc="lower center",bbox_to_anchor=(0.5,-0.18),
               ncol=2,fontsize=7,labelcolor="white",facecolor="#1e2130",edgecolor="none")
    ax1.set_title("Allocation",color="white",fontsize=10,pad=8)

    # Trades table
    ax2=fig.add_subplot(gs[1,1:]); ax2.set_facecolor("#1e2130"); ax2.axis("off")
    all_trades=[]
    for bucket in ["POSITION","SWING","SCALP","HIGH_BETA"]:
        for t in plan.get(bucket,[]):
            all_trades.append([
                f"{TRADE_TYPES.get(t['trade_type'],{}).get('emoji','?')} {t['ticker']}",
                t["trade_type"],str(t["lots"]),t["amount"],
                t["entry"],t["stop_loss"][:12],t["take_profit"][:12],
                t.get("risk_pct_str","?")[:12],
            ])
    if all_trades:
        cols=["Stock","Type","Lots","Amount","Entry","SL","TP","Risk%"]
        tbl=ax2.table(cellText=all_trades,colLabels=cols,cellLoc="center",loc="center",bbox=[0,0,1,1])
        tbl.auto_set_font_size(False); tbl.set_fontsize(7.5)
        for (r,c),cell in tbl.get_celld().items():
            cell.set_facecolor("#1e2130" if r>0 else "#2d3250")
            cell.set_text_props(color="white"); cell.set_edgecolor("#3d4270")
    else:
        ax2.text(0.5,0.5,"No trades pass checklist today.",
                 ha="center",va="center",color="#9E9E9E",fontsize=12,transform=ax2.transAxes)
    ax2.set_title("Today's Trades",color="white",fontsize=10,pad=8)

    # Summary
    ax3=fig.add_subplot(gs[2,:]); ax3.set_facecolor("#1e2130"); ax3.axis("off")
    summary=plan.get("_summary",{})
    type_mix=" | ".join(f"{k}:{v}%" for k,v in summary.get("type_allocation",{}).items())
    ax3.text(0.5,0.6,
             f"Deployed: Rp {summary.get('total_deployed',0):,.0f} ({summary.get('pct_deployed',0):.1f}%)   "
             f"Cash: Rp {summary.get('cash_reserve',0):,.0f}   Trades: {summary.get('trade_count',0)}",
             ha="center",va="center",color="white",fontsize=10,transform=ax3.transAxes)
    ax3.text(0.5,0.25,f"Type Mix: {type_mix}",
             ha="center",va="center",color="#9E9E9E",fontsize=9,transform=ax3.transAxes)

    plt.savefig(save_path,dpi=150,bbox_inches="tight",facecolor="#0f1117")
    plt.close()
    return save_path

# ═══════════════════════════════════════════════════════════
# GEN 5 NEW: FAIR VALUE GAP (FVG) SCANNER
# ═══════════════════════════════════════════════════════════
def detect_fvg(df):
    if len(df) < 3: return df
        
    df['Bull_FVG'] = False
    df['Bear_FVG'] = False
    df['FVG_Size'] = 0.0

    bull_fvg = df['low'] > df['high'].shift(2)
    bear_fvg = df['high'] < df['low'].shift(2)

    df.loc[bull_fvg, 'Bull_FVG'] = True
    df.loc[bull_fvg, 'FVG_Size'] = df['low'] - df['high'].shift(2)

    df.loc[bear_fvg, 'Bear_FVG'] = True
    df.loc[bear_fvg, 'FVG_Size'] = df['low'].shift(2) - df['high']

    return df

def scan_market_for_fvg(raw_data_dict):
    active_fvgs = []
    for ticker, df in raw_data_dict.items():
        try:
            df = df.copy()
            df.columns = [c.lower() for c in df.columns]
            if len(df) < 3: continue
                
            fvg_df = detect_fvg(df)
            last_row = fvg_df.iloc[-1]
            price = float(last_row['close'])
            
            if last_row['Bull_FVG']:
                active_fvgs.append({"Ticker": ticker, "Type": "Bullish FVG 🟢", "Size": last_row['FVG_Size'], "Price": price})
            elif last_row['Bear_FVG']:
                active_fvgs.append({"Ticker": ticker, "Type": "Bearish FVG 🔴", "Size": last_row['FVG_Size'], "Price": price})
        except:
            continue
            
    return pd.DataFrame(active_fvgs)
