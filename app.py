import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mplfinance as mpf
import pandas as pd
import numpy as np
import yfinance as yf
import engine as eng
from utils.fred_client import fetch_fred, parse_latest_value, parse_change_arrow
from utils.supabase_client import (
    get_supabase_client,
    safe_select,
    safe_insert,
    compute_portfolio_heat,
)
from utils.regime_detector import classify_ihsg_regime, classify_vix_bucket, detect_operating_mode
from utils.dg_checklist import build_dg_metrics, score_dg
from engine import (
    get_macro_score, get_commodity_context,
    detect_regime, auto_threshold, get_allocation, recommend_sector,
    allocate_trades_by_sector, get_money_flow, sector_action_translator,
    fast_momentum_screen, run_full_analysis, run_backtest,
    build_execution_plan, get_high_beta_plays,
    trade_checklist, risk_based_sizing, get_trade_setup,
    build_morning_message, generate_brief_image,
    scan_market_for_fvg,
    get_macro_alignment, compute_trade_confidence, explain_rejection,
    send_telegram, send_telegram_photo,
    resolve_universe,
    SECTORS, IDX_UNIVERSE, TRADE_TYPES, REGIME_COLORS,
    REGIME_ALLOCATIONS, SHARIA_COMPLIANT, DEFAULT_WEIGHTS,
)
from trade_journal import (
    load_journal, log_trade, close_trade, expire_stale_trades,
    get_learned_weights, get_journal_df, get_journal_stats, get_open_trades,
)
from signal_tracker import (
    register_signals_from_plan, update_signal_statuses, compute_signal_performance,
)

if not hasattr(st, "_orig_line_chart_fn"):
    st._orig_line_chart_fn = st.line_chart
_orig_line_chart = st._orig_line_chart_fn

def _safe_line_chart(data=None, *args, **kwargs):
    try:
        if isinstance(data, pd.DataFrame):
            safe_df = data.copy()
            if isinstance(safe_df.columns, pd.MultiIndex):
                safe_df.columns = safe_df.columns.get_level_values(0)
            safe_df.columns = [str(c) for c in safe_df.columns]
            return _orig_line_chart(safe_df, *args, **kwargs)
        return _orig_line_chart(data, *args, **kwargs)
    except Exception as exc:
        st.warning(f"line_chart fallback engaged: {exc}")
        if isinstance(data, pd.DataFrame) and not data.empty:
            fig, ax = plt.subplots(figsize=(8, 2.4))
            for col in data.columns[:2]:
                try:
                    y = pd.Series(data[col]).astype(float).values
                    ax.plot(range(len(y)), y, label=str(col))
                except Exception:
                    continue
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(alpha=0.2)
            st.pyplot(fig)
            plt.close(fig)
        return None

if st.line_chart is not _safe_line_chart:
    st.line_chart = _safe_line_chart

def _safe_df(rows):
    """Convert all values to strings to prevent Arrow type errors."""
    df = pd.DataFrame(rows)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str)
    return df

def _fallback_macro_alignment(_scores):
    total = round(float(sum(_scores.values())), 3) if isinstance(_scores, dict) else 0.0
    if abs(total) >= 1.0:
        conf = "HIGH"
    elif abs(total) >= 0.4:
        conf = "MEDIUM"
    else:
        conf = "LOW"
    return total, conf, {"daily": total, "1m": total, "ytd": total}

def _fallback_trade_confidence(result, macro, sector_flow):
    _ = sector_flow
    comp = float(result.get("composite", 0))
    macro_score = float(macro.get("combined", 0)) if isinstance(macro, dict) else float(macro or 0)
    score = int(np.clip(round(((comp + 1) / 2) * 80 + ((macro_score + 1) / 2) * 20), 0, 100))
    label = "A+" if score >= 90 else "A" if score >= 80 else "B" if score >= 70 else "C"
    return score, label

def _fallback_explain_rejection(result, checks, macro_score):
    _ = macro_score
    fails = [k for k, v in checks.items() if not v]
    if fails:
        return " | ".join(fails[:3])
    if float(result.get("composite", 0)) <= 0.25:
        return "Composite below threshold"
    return "Relative ranking below selected setups"

def get_timing_model_signal(details, scores, flow_data, context, cpi_yoy=3.0, credit_spread=4.0, accounting_risk=False):
    _ = details, scores, flow_data, context, accounting_risk
    signal = "NEUTRAL"
    confidence = "MEDIUM"
    reason = "Macro conditions are mixed"
    if float(credit_spread or 0) > 8.0:
        signal = "CAUTION"
        confidence = "HIGH"
        reason = "Credit spread elevated"
    elif float(cpi_yoy or 0) > 3.5:
        signal = "CAUTION"
        confidence = "MEDIUM"
        reason = "Inflation too high for Fed to act"
    elif float(credit_spread or 0) < 5.0 and float(cpi_yoy or 0) < 3.0:
        signal = "BUY"
        confidence = "HIGH"
        reason = "Credit spread and inflation are supportive"
    return {"signal": signal, "confidence": confidence, "reason": reason}

def _fallback_timing_model_signal(*args, **kwargs):
    """Backwards-compatible fallback used by older merge paths."""
    _details = args[0] if len(args) > 0 else {}
    _scores = args[1] if len(args) > 1 else {}
    _flow_data = args[2] if len(args) > 2 else {}
    _context = args[3] if len(args) > 3 else None
    _ = _details, _scores, _flow_data, _context
    cpi_yoy = kwargs.get("cpi_yoy", 3.0)
    credit_spread = kwargs.get("credit_spread", 4.0)
    signal = "NEUTRAL"
    confidence = "MEDIUM"
    reason = "Macro conditions are mixed"
    if float(credit_spread or 0) > 8.0:
        signal = "CAUTION"
        confidence = "HIGH"
        reason = "Credit spread elevated"
    elif float(cpi_yoy or 0) > 3.5:
        signal = "CAUTION"
        confidence = "MEDIUM"
        reason = "Inflation too high for Fed to act"
    elif float(credit_spread or 0) < 5.0 and float(cpi_yoy or 0) < 3.0:
        signal = "BUY"
        confidence = "HIGH"
        reason = "Credit spread and inflation are supportive"
    return {"signal": signal, "confidence": confidence, "reason": reason}

get_macro_alignment = getattr(eng, "get_macro_alignment", _fallback_macro_alignment)
compute_trade_confidence = getattr(eng, "compute_trade_confidence", _fallback_trade_confidence)
explain_rejection = getattr(eng, "explain_rejection", _fallback_explain_rejection)
    
st.set_page_config(page_title="IDX Trading Dashboard — Gen 5", layout="wide")
st.markdown("""
<style>
.stApp { background-color: #0f1117; color: #e6e6e6; }
[data-testid="stMetricValue"], [data-testid="stMetricDelta"], .stDataFrame, table {
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
}
.orange-accent { color: #FF9900; font-weight: 700; }
.stTabs [role="tab"] { padding: 0.35rem 0.7rem; }
</style>
""", unsafe_allow_html=True)
st.title("📊 IDX Macro Trading Dashboard — Gen 7")
st.caption("Build: gen7-no-linechart-jkse-v3")

@st.cache_data(ttl=900, show_spinner=False)
def load_yf_data(symbol, period="6mo", interval="1d"):
    try:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            # yfinance may return MultiIndex columns even for a single ticker.
            # Keep the first level (Open/High/Low/Close/Volume) for Streamlit chart compatibility.
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def load_fred_bundle(api_key):
    if not api_key:
        return {}
    series_map = {
        "hy_spread": "BAMLH0A0HYM2",
        "fed_funds": "FEDFUNDS",
        "cpi": "CPIAUCSL",
        "us10y": "DGS10",
    }
    out = {}
    for key, series in series_map.items():
        try:
            out[key] = fetch_fred(series, api_key, limit=24 if key == "cpi" else 5)
        except Exception:
            out[key] = []
    return out

# ── SIDEBAR (GEN 7 REFACTOR) ─────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
with st.sidebar.expander("Universe and filters", expanded=True):
    portfolio_raw = st.text_input(
        "Portfolio Value (IDR)",
        value="100,000,000",
        help="Enter any amount, e.g. 75,000,000"
    )
    try:
        portfolio_value = int(portfolio_raw.replace(",","").replace(".","").strip())
    except Exception:
        portfolio_value = 100_000_000
        st.warning("Invalid input — using Rp 100,000,000")
    st.caption(f"= Rp {portfolio_value:,.0f}")

    risk_pct_input = st.slider("Risk per trade (%)", 0.5, 3.0, 1.0, 0.25) / 100
    top_n = st.slider("Top N candidates", 5, 20, 10)
    use_sector = st.checkbox("Limit screen to recommended sector", value=False)
    rr_ratio = st.slider("Risk/Reward Ratio", 1.0, 4.0, 2.0, 0.5)
    high_beta_pct = st.slider("High-Beta capital %", 0, 30, 15) / 100
    sharia_only = st.checkbox("☪️ Sharia compliant only", value=False)
    selected_universe = st.radio(
        "Universe",
        options=["IDX", "LQ45", "Gorengan", "All"],
        index=0,
        horizontal=True,
        help="IDX: ~50 blue chips | LQ45: top 45 liquid | Gorengan: high-beta small caps ⚠️ | All: everything"
    )
    use_auto = st.checkbox("Use auto threshold", value=True)

with st.sidebar.expander("Model settings", expanded=False):
    st.caption("Manual override — use if FRED API unavailable")
    cpi_yoy_override = st.number_input("Manual US CPI YoY (%)", min_value=0.0, max_value=15.0, value=3.3, step=0.1)
    credit_spread_override = st.number_input("Manual HY Credit Spread (%)", min_value=0.0, max_value=25.0, value=4.0, step=0.1)
    manual_threshold_input = st.slider("Manual threshold", 0.10, 0.60, 0.30, 0.05)
    accounting_risk_input = st.checkbox("Accounting trust risk", value=False)
    use_learned = st.checkbox("Use learned weights", value=True)

with st.sidebar.expander("Alerts and automation", expanded=False):
    auto_send_summary = st.checkbox("Auto-send morning summary", value=False)
    st.caption("Telegram credentials can come from Streamlit secrets.")
    secret_tg_token = (
        st.secrets.get("telegram_bot_token")
        or st.secrets.get("TG_BOT_TOKEN")
        or st.secrets.get("bot_token")
        or st.secrets.get("telegram", {}).get("bot_token")
    )
    secret_tg_chat_id = (
        st.secrets.get("telegram_chat_id")
        or st.secrets.get("TG_CHAT_ID")
        or st.secrets.get("chat_id")
        or st.secrets.get("telegram", {}).get("chat_id")
    )
    tg_token = st.text_input("Bot Token", type="password", value=secret_tg_token or "")
    tg_chat_id = st.text_input("Chat ID", value=secret_tg_chat_id or "")

# Learned weights (kept, but clean messaging)
learned_weights = get_learned_weights()
_, journal_stats = get_journal_stats()
if journal_stats:
    st.sidebar.caption(f"Trades logged: {journal_stats.get('total_trades','?')} | Win rate: {journal_stats.get('win_rate','?')}")
else:
    st.sidebar.caption("No trades logged yet — learning weights defaulting to base")
if not journal_stats:
    use_learned = False
active_weights = learned_weights if (use_learned and journal_stats) else DEFAULT_WEIGHTS

# ── CACHED MACRO ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_macro():    return get_macro_score()

@st.cache_data(ttl=3600, show_spinner=False)
def load_commodities(): return get_commodity_context()

@st.cache_data(ttl=1800, show_spinner=False)
def load_money_flow():  return get_money_flow()

supabase_client = get_supabase_client(st.secrets)
supabase_configured = supabase_client is not None
if not supabase_configured:
    st.info("Supabase is not configured. Margin debt, earnings log, and portfolio heat will use local fallback mode.")
    raw_supabase_url = str(st.secrets.get("SUPABASE_URL", st.secrets.get("supabase_url", ""))).strip()
    if "/rest/v1" in raw_supabase_url:
        st.caption("Tip: use project URL without `/rest/v1` in `SUPABASE_URL`.")
fred_api_key = str(st.secrets.get("FRED_API_KEY", "")).strip()
fred_bundle = load_fred_bundle(fred_api_key)

fred_unavailable = not bool(fred_bundle) or all(not v for v in fred_bundle.values())
if fred_unavailable:
    st.warning("FRED API unavailable — using manual inputs.")

hy_spread_live = parse_latest_value(fred_bundle.get("hy_spread", [])) if not fred_unavailable else None
fed_funds_live = parse_latest_value(fred_bundle.get("fed_funds", [])) if not fred_unavailable else None
us10y_live = parse_latest_value(fred_bundle.get("us10y", [])) if not fred_unavailable else None

def _calc_cpi_yoy(cpi_obs):
    vals = []
    for obs in cpi_obs:
        try:
            vals.append(float(obs.get("value")))
        except Exception:
            continue
    if len(vals) < 13:
        return None
    return ((vals[0] / vals[12]) - 1) * 100

cpi_yoy_live = _calc_cpi_yoy(fred_bundle.get("cpi", [])) if not fred_unavailable else None
credit_spread_input = hy_spread_live if hy_spread_live is not None else credit_spread_override
cpi_yoy_input = cpi_yoy_live if cpi_yoy_live is not None else cpi_yoy_override

fed_stance = "HOLDING"
fed_values = [parse_latest_value([o]) for o in fred_bundle.get("fed_funds", [])[:3]]
fed_values = [v for v in fed_values if v is not None]
if len(fed_values) >= 3:
    if fed_values[0] > fed_values[2]:
        fed_stance = "HIKING"
    elif fed_values[0] < fed_values[2]:
        fed_stance = "CUTTING"

macro_score, stance, scores, details = load_macro()
macro_alignment_score, macro_conf_level, macro_breakdown = get_macro_alignment(scores)
commodity_context                    = load_commodities()
regime, regime_conf, regime_reason   = detect_regime(scores, details, commodity_context)
allocation                           = get_allocation(regime, macro_score)
rec_sector, rec_reason, signals      = recommend_sector(macro_score, scores, commodity_context)
emoji                                = REGIME_COLORS.get(regime,"⚪")
smart_threshold                      = auto_threshold(regime, macro_score)
threshold = smart_threshold if use_auto else manual_threshold_input
budget_map  = allocate_trades_by_sector(allocation, portfolio_value, high_beta_pct)
timing_model = get_timing_model_signal(
    details, scores, {}, None,
    cpi_yoy=cpi_yoy_input,
    credit_spread=credit_spread_input,
    accounting_risk=accounting_risk_input,
)

# Auto-expire stale trades
expired = expire_stale_trades()
if expired:
    st.warning(f"⏰ {len(expired)} trade(s) auto-expired today: {', '.join(t['ticker'] for t in expired)}")

# SECTION 0 — Operating mode banner
ihsg_df = load_yf_data("^JKSE", period="6mo")
vix_df = load_yf_data("^VIX", period="3mo")
ihsg_close = float(ihsg_df["Close"].dropna().iloc[-1]) if not ihsg_df.empty else 0.0
ihsg_ema50 = float(ihsg_df["Close"].ewm(span=50, adjust=False).mean().dropna().iloc[-1]) if not ihsg_df.empty else 0.0
ihsg_regime = classify_ihsg_regime(ihsg_close, ihsg_ema50) if ihsg_close else "NEUTRAL"
vix_now = float(vix_df["Close"].dropna().iloc[-1]) if not vix_df.empty else 0.0
vix_bucket = classify_vix_bucket(vix_now)

margin_rows, margin_err = safe_select(supabase_client, "margin_debt", columns="month,debit_balance_millions,recorded_at", order_by="recorded_at", limit=2)
margin_direction = "UNKNOWN"
if len(margin_rows) >= 2:
    current_md = float(margin_rows[0].get("debit_balance_millions", 0) or 0)
    prev_md = float(margin_rows[1].get("debit_balance_millions", 0) or 0)
    margin_direction = "DELEVERAGING" if current_md < prev_md else "LEVERAGING UP"

earnings_rows, _ = safe_select(supabase_client, "earnings_log", columns="ticker,eps_actual,eps_estimate,revenue_beat,recorded_at", order_by="recorded_at", limit=200)
earnings_df = pd.DataFrame(earnings_rows) if earnings_rows else pd.DataFrame(columns=["ticker", "eps_actual", "eps_estimate", "revenue_beat"])
if not earnings_df.empty:
    earnings_df["beat"] = earnings_df.apply(lambda r: (r.get("eps_actual", 0) or 0) > (r.get("eps_estimate", 0) or 0), axis=1)
leading_tickers = set((SECTORS.get(rec_sector, {}) or {}).get("T1", [])[:3])
earnings_beats = int(earnings_df[earnings_df["ticker"].isin(leading_tickers)]["beat"].sum()) if (not earnings_df.empty and "ticker" in earnings_df.columns) else 0

sector_leader_text = f"{rec_sector} leader"
dg_metrics = build_dg_metrics(vix_now, fed_stance, margin_direction, sector_leader_text, earnings_beats)
dg_score, dg_verdict = score_dg(dg_metrics)
guardrails = {
    "credit_spread>8": float(credit_spread_input or 0) > 8,
    "cpi>3": float(cpi_yoy_input or 0) > 3,
    "margin_rising": margin_direction == "LEVERAGING UP",
}
mode_payload = detect_operating_mode(vix_now, dg_score, guardrails)

color_map = {"blue": "#1E88E5", "amber": "#F9A825", "green": "#2E7D32", "red": "#C62828"}
st.markdown(
    f"""
<div style="padding:16px;border-radius:12px;background:{color_map.get(mode_payload['color'], '#1E88E5')};color:white;margin-bottom:14px;">
<b>{mode_payload['mode']}</b> — {mode_payload['message']}<br/>
VIX: {vix_now:.2f} ({vix_bucket}) | DG score: {dg_score}/5 | Sizing: {mode_payload['risk_guidance']}<br/>
Guardrails: Credit>{'ON' if guardrails['credit_spread>8'] else 'OFF'}, CPI>{'ON' if guardrails['cpi>3'] else 'OFF'}, Margin rising={'ON' if guardrails['margin_rising'] else 'OFF'}
</div>
""",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════
# 1. REGIME + MACRO
# ═══════════════════════════════════════════════════════════
st.header("1. Market Regime + Macro")

cmap={"CRISIS":"error","RISK_OFF":"error","TIGHTENING":"warning",
      "INFLATION":"warning","RISK_ON":"success","NEUTRAL":"info"}
getattr(st,cmap[regime])(f"{emoji} **{regime}** — {regime_conf}% | {regime_reason}")

macro_c1, macro_c2, macro_c3 = st.columns(3)
macro_c1.metric("IHSG Regime Gate", ihsg_regime)
macro_c2.metric("VIX Alarm", f"{vix_now:.2f}", vix_bucket)
macro_c3.metric("Operating Mode", mode_payload["mode"], mode_payload["risk_guidance"])

if (not ihsg_df.empty) and ("Close" in ihsg_df.columns):
    ihsg_close_series = ihsg_df["Close"].astype(float)
    spark = pd.DataFrame({
        "Close": ihsg_close_series.tail(30),
        "EMA50": ihsg_close_series.ewm(span=50, adjust=False).mean().tail(30),
    }).dropna()
    if isinstance(spark.columns, pd.MultiIndex):
        spark.columns = spark.columns.get_level_values(0)
    spark.columns = [str(c) for c in spark.columns]
    if not spark.empty:
        try:
            fig_spark, ax_spark = plt.subplots(figsize=(8, 2.4))
            ax_spark.plot(spark.index, spark["Close"].values, label="IHSG", linewidth=1.8)
            ax_spark.plot(spark.index, spark["EMA50"].values, label="EMA50", linewidth=1.2, linestyle="--")
            ax_spark.legend(loc="upper left", fontsize=8)
            ax_spark.grid(alpha=0.2)
            ax_spark.set_ylabel("Price")
            st.pyplot(fig_spark)
            plt.close(fig_spark)
        except Exception as exc:
            st.warning(f"IHSG sparkline unavailable: {exc}")

fred_cols = st.columns(4)
fred_cols[0].metric("HY Credit Spread", f"{float(credit_spread_input or 0):.2f}%", parse_change_arrow(fred_bundle.get("hy_spread", [])))
fred_cols[1].metric("Fed Funds", f"{float(fed_funds_live or 0):.2f}%", f"{fed_stance}")
fred_cols[2].metric("CPI YoY", f"{float(cpi_yoy_input or 0):.2f}%", parse_change_arrow(fred_bundle.get("cpi", [])))
fred_cols[3].metric("10Y Treasury", f"{float(us10y_live or 0):.2f}%", parse_change_arrow(fred_bundle.get("us10y", [])))

st.caption("FINRA margin debt data is typically delayed by ~3–4 weeks.")
md_col1, md_col2 = st.columns([2, 1])
with md_col1:
    if margin_rows:
        st.write(f"Latest FINRA update: {margin_rows[0].get('month', 'N/A')} | Direction: **{margin_direction}**")
    if margin_err and supabase_configured:
        st.warning(f"Margin debt table unavailable: {margin_err}")
with md_col2:
    new_month = st.text_input("FINRA month", value="")
    new_margin = st.number_input("Debit balance ($M)", min_value=0.0, value=0.0, step=100.0)
    if st.button("Save margin debt", disabled=not supabase_configured):
        err = safe_insert(supabase_client, "margin_debt", {"month": new_month, "debit_balance_millions": float(new_margin)})
        if err:
            st.warning(f"Failed to save margin debt: {err}")
        else:
            st.success("Margin debt record saved.")

with st.expander("🌍 Macro Recap + Interpretation", expanded=True):
    rc1,rc2=st.columns(2)
    with rc1:
        st.markdown("**Signal Summary:**")
        nasdaq_up=scores.get("Nasdaq",0)>0.1; vix_down=scores.get("VIX",0)>0.1
        yield_up=scores.get("US10Y",0)<-0.1; oil_up=commodity_context.get("Crude Oil",{}).get("direction",0)>0
        for lbl,pos,good,bad in [
            ("Nasdaq",nasdaq_up,"↑ Risk-on appetite","↓ Risk-off pressure"),
            ("VIX",vix_down,"↓ Fear falling","↑ Fear rising"),
            ("US10Y",not yield_up,"↓ Yields supportive","↑ Yields tightening"),
            ("Crude Oil",oil_up,"↑ Energy/inflation signal","↓ Oil weak"),
        ]:
            st.write(f"{'🟢' if pos else '🔴'} **{lbl}:** {good if pos else bad}")
    with rc2:
        st.markdown("**Interpretation:**")
        interp={"RISK_ON":("🟢","Risk-on mode.","Buy cyclicals + growth. Chase breakouts."),
                "INFLATION":("🟡","Inflation trade.","Energy + commodities outperform."),
                "TIGHTENING":("🟡","Monetary tightening.","Real assets > growth."),
                "RISK_OFF":("🟠","Risk-off.","Reduce exposure. Favor cash."),
                "CRISIS":("🔴","CRISIS.","Capital preservation only."),
                "NEUTRAL":("⚪","Mixed signals.","No strong edge. Smaller sizes.")}.get(regime,("⚪","",""))
        st.markdown(f"**{interp[0]} {interp[1]}**\n\n{interp[2]}")
        st.divider()
        st.markdown(f"**Today:** Focus **{rec_sector}** — {rec_reason}")
        st.caption(f"Threshold: **{threshold}** ({'auto' if use_auto else 'manual'}) | "
                   f"Risk/trade: **{risk_pct_input*100:.1f}%**")
        st.markdown(
            f"<span class='orange-accent'>Macro Alignment Score:</span> {macro_alignment_score:+.2f} | "
            f"<span class='orange-accent'>Confidence:</span> {macro_conf_level}",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Breakdown — Daily: {macro_breakdown.get('daily',0):+.2f} | "
            f"1M: {macro_breakdown.get('1m',0):+.2f} | YTD: {macro_breakdown.get('ytd',0):+.2f}"
        )

cols=st.columns(5)
for i,name in enumerate(["Nasdaq","DXY","US10Y","VIX"]):
    d,t=details[name],details[name].get("trend",{})
    with cols[i]:
        st.metric(name,d["value"],d["change"])
        if t:
            st.caption(f"5d:{t.get('5d','N/A')} 20d:{t.get('20d','N/A')}")
            s=d["score"]
            st.caption(f"{'🟢' if s>0 else '🔴'} {s:+.3f} {'█'*int(abs(s)*10)}")
cols[4].metric("TOTAL",f"{macro_score:+.2f}",stance)

st.subheader("Commodity Context")
ccols=st.columns(len(commodity_context))
for i,(name,data) in enumerate(commodity_context.items()):
    icon="🟢" if data["direction"]>0 else("🔴" if data["direction"]<0 else "⚪")
    with ccols[i]:
        st.metric(f"{icon} {name}",data["value"],data["change"])
        t=data.get("trend",{})
        if t: st.caption(f"5d:{t.get('5d','N/A')} 20d:{t.get('20d','N/A')}")

show_dg_direct = mode_payload["mode"] in ("ELEVATED RISK", "CRISIS ENTRY")
dg_container = st.container() if show_dg_direct else st.expander("DG crisis checklist (inactive — VIX too low).", expanded=False)
with dg_container:
    st.subheader("DG CRISIS ENTRY CHECKLIST")
    for metric_name, payload in dg_metrics.items():
        marker = "✓" if payload["ok"] else "✗"
        color = "🟢" if payload["ok"] else "⚪"
        st.write(f"[{marker}] {metric_name} — {payload['detail']} {color}")
    st.markdown(f"**SCORE:** {dg_score} / 5")
    st.markdown(f"**VERDICT:** {dg_verdict}")

    earnings_editor = earnings_df.copy()
    if earnings_editor.empty:
        earnings_editor = pd.DataFrame([{"ticker": "", "period": "", "eps_actual": 0.0, "eps_estimate": 0.0, "revenue_beat": False, "beat": False}])
    editable = st.data_editor(
        earnings_editor[["ticker", "period", "eps_actual", "eps_estimate", "revenue_beat", "beat"]] if "period" in earnings_editor.columns else earnings_editor,
        num_rows="dynamic",
        width="stretch",
        key="earnings_editor",
    )
    if st.button("Save earnings rows"):
        saved = 0
        for _, row in editable.iterrows():
            if not str(row.get("ticker", "")).strip():
                continue
            payload = {
                "ticker": str(row.get("ticker", "")).strip(),
                "period": str(row.get("period", "")).strip(),
                "eps_actual": float(row.get("eps_actual", 0) or 0),
                "eps_estimate": float(row.get("eps_estimate", 0) or 0),
                "revenue_beat": bool(row.get("revenue_beat", False)),
            }
            err = safe_insert(supabase_client, "earnings_log", payload)
            if not err:
                saved += 1
        st.success(f"Saved {saved} earnings row(s).")

# ═══════════════════════════════════════════════════════════
# 2. ALLOCATION + MONEY FLOW + SECTOR ACTION
# ═══════════════════════════════════════════════════════════
st.header("2. Portfolio Allocation + Money Flow")

open_trade_rows, open_trade_err = safe_select(supabase_client, "trades", columns="risk_pct,status", limit=500)
open_trade_rows = [r for r in open_trade_rows if str(r.get("status", "OPEN")).upper() == "OPEN"]
heat_pct, open_count = compute_portfolio_heat(open_trade_rows)
heat_norm = min(1.0, heat_pct / 6.0)
st.caption(f"Portfolio heat: {heat_pct:.2f}% capital at risk across {open_count} open trades")
st.progress(heat_norm)
if open_trade_err and supabase_configured:
    st.warning(f"Portfolio heat unavailable from Supabase: {open_trade_err}")

with st.spinner("Loading sector money flow..."):
    try:
        flow_data, top_sector, bot_sector, flow_narrative = load_money_flow()
    except:
        flow_data, top_sector, bot_sector, flow_narrative = {}, rec_sector, "", "Flow data unavailable."
timing_model = get_timing_model_signal(
    details, scores, flow_data, None,
    cpi_yoy=cpi_yoy_input,
    credit_spread=credit_spread_input,
    accounting_risk=accounting_risk_input,
)

cl,cr=st.columns([1,1])
with cl:
    st.subheader(f"{emoji} {regime}")
    for sector,pct in allocation.items():
        amount=portfolio_value*pct; bar="█"*int(pct*40)
        st.markdown(f"{'🟢' if sector!='Cash' else '⚪'} **{sector}**: {pct*100:.0f}%  `{bar}`")
        st.caption(f"→ Rp {amount:,.0f}")
    st.divider()
    st.caption(f"🔥 High-Beta: Rp {budget_map['high_beta_budget']:,.0f}")
    st.caption(f"💵 Cash: Rp {budget_map['cash_reserve']:,.0f}")

with cr:
    fig,ax=plt.subplots(figsize=(5,4)); ax.set_facecolor("#0f1117")
    fig.patch.set_facecolor("#0f1117")
    labels=list(allocation.keys()); sizes=list(allocation.values())
    custom_color_map = {"Banks":"#1D9E75","Defensive":"#7F77DD","Commodities":"#EF9F27","Cash":"#888780"}
    colors=[custom_color_map.get(lbl, "#4CAF50") for lbl in labels]
    wedges,texts,autos=ax.pie(sizes,colors=colors,autopct="%1.0f%%",startangle=90,
                               wedgeprops={"edgecolor":"#0f1117","linewidth":2})
    for a in autos: a.set_color("white"); a.set_fontsize(9)
    ax.legend(wedges,labels,loc="lower center",bbox_to_anchor=(0.5,-0.15),
              ncol=2,fontsize=7,labelcolor="white",facecolor="#0f1117",edgecolor="none")
    ax.set_title(f"{regime} Allocation",color="white",fontsize=10)
    st.pyplot(fig)

# Money flow table
if flow_data:
    st.subheader("💸 Sector Money Flow")
    st.caption(flow_narrative)
    flow_df=pd.DataFrame([
        {"Sector":s,"5D":d["5d_return"],"20D":d["20d_return"],
         "Flow":d["momentum"],"Score":d["score"]}
        for s,d in flow_data.items()
    ]).sort_values("Score",ascending=False)
    st.dataframe(
        flow_df.style.background_gradient(subset=["Score"],cmap="RdYlGn"),
        width="stretch"
    )

# Sector action translator
tier_data=SECTORS[rec_sector]
st.subheader(f"🎯 Sector Action: {rec_sector}")
sector_action=sector_action_translator(rec_sector,macro_score,scores,commodity_context,flow_data)
action_color={"STRONG":"success","MODERATE":"info","WEAK":"warning","AVOID":"error"}.get(
    sector_action["strength"],"info")
getattr(st,action_color)(f"**{sector_action['strength']}** — Flow score: {sector_action['flow_score']:+.2f}")
st.markdown(sector_action["strategy"])
c1,c2,c3=st.columns(3)
c1.markdown("🥇 **Tier 1**\n\n"+"\n\n".join(tier_data["T1"]))
c2.markdown("🥈 **Tier 2**\n\n"+"\n\n".join(tier_data["T2"]))
c3.markdown("🥉 **Tier 3**\n\n"+"\n\n".join(tier_data["T3"]))

# ═══════════════════════════════════════════════════════════
# 3. MOMENTUM SCREEN
# ═══════════════════════════════════════════════════════════
st.header("3. Universe Scanner")
_base_universe = resolve_universe(selected_universe)
scan_universe = (
    tier_data["T1"] + tier_data["T2"] + tier_data["T3"]
    if use_sector
    else _base_universe
)
# Always ensure recommended sector T1 stocks are in the scan universe
# so they have data available for the execution plan
_sector_stocks = tier_data["T1"] + tier_data["T2"]
for _s in _sector_stocks:
    if _s not in scan_universe:
        scan_universe = list(scan_universe) + [_s]

universe_label = f"{'sector only' if use_sector else selected_universe} universe"
st.info(f"🌐 Scanning **{universe_label}** — {len(scan_universe)} stocks | Sector: **{rec_sector}** ({sector_action['strength']})")

if st.button("🔍 Run Momentum Screen"):
    with st.spinner("Downloading and scoring..."):
        all_scores,top_candidates,raw_data=fast_momentum_screen(scan_universe,top_n=top_n)
        hb_plays=get_high_beta_plays(raw_data,top_n=3)
        st.session_state.update({"all_scores":all_scores,"top_candidates":top_candidates,
                                  "raw_data":raw_data,"hb_plays":hb_plays})

if "all_scores" not in st.session_state:
    st.info("👆 Click **Run Momentum Screen** to start.")
    all_scores = pd.DataFrame(columns=["ticker","price","momentum","rsi","vol_ratio","52w_prox","20d_return","vs_ihsg","beta","sharia","high_beta","sector","avg_volume"])
    top_candidates = []
    raw_data = {}
    hb_plays = []
else:
    all_scores    =st.session_state["all_scores"]
    top_candidates=st.session_state["top_candidates"]
    raw_data      =st.session_state.get("raw_data",{})
    hb_plays      =st.session_state.get("hb_plays",[])

disp=all_scores.copy()
if sharia_only: disp=disp[disp["sharia"]==True]
if "52w_prox" in disp.columns:
    disp = disp.rename(columns={"52w_prox": "52w_hi_pct"})
if "avg_volume" not in disp.columns:
    disp["avg_volume"] = np.nan
disp["ARB_risk"] = disp.apply(
    lambda r: "HIGH" if (float(r.get("price", 0) or 0) < 200 or float(r.get("avg_volume", 0) or 0) < 5_000_000) else "LOW",
    axis=1
)
disp["sector_warning"] = disp.get("sector", "").apply(
    lambda s: "⚠ AVOID SECTOR" if sector_action.get("strength") == "AVOID" and str(s).strip() else ""
) if "sector" in disp.columns else ""
disp["☪️"]=disp["sharia"].apply(lambda x:"☪️" if x else "")
disp["🚀"]=disp["high_beta"].apply(lambda x:"🚀" if x else "")
for col in disp.select_dtypes(include=[np.number]).columns:
    disp[col] = disp[col].round(2)
st.dataframe(
    disp.head(top_n)[["ticker","price","momentum","rsi","vol_ratio",
                       "52w_hi_pct","20d_return","vs_ihsg","beta","ARB_risk","sector_warning","☪️","🚀"]]
    .style.background_gradient(subset=["momentum"],cmap="RdYlGn"),
    width="stretch"
)

if hb_plays:
    st.subheader("🔥 High-Beta Plays")
    hb_df=pd.DataFrame(hb_plays)
    hb_df["☪️"]=hb_df["sharia"].apply(lambda x:"☪️" if x else "")
    st.dataframe(hb_df[["ticker","beta_score","price","vol_ratio","5d_mom","adr","rsi","☪️"]]
                 .style.background_gradient(subset=["beta_score"],cmap="Oranges"),
                 width="stretch")

if sharia_only: top_candidates=[t for t in top_candidates if SHARIA_COMPLIANT.get(t,True)]
default_candidates = [t for t in top_candidates if t in scan_universe]
custom=st.multiselect("Override candidates:",options=scan_universe,default=default_candidates)
final_candidates=custom if custom else top_candidates
if sharia_only: final_candidates=[t for t in final_candidates if SHARIA_COMPLIANT.get(t,True)]

# ── SECTOR INJECTION: always include recommended sector T1/T2 in execution analysis ──
# This fixes the "sector recommends Commodities but Execution tab is empty" bug
if sector_action["strength"] in ("STRONG", "MODERATE") and not use_sector:
    all_screened_tickers = set(all_scores["ticker"].tolist())
    sector_must_include  = tier_data["T1"] + tier_data["T2"]
    injected = []
    for s in sector_must_include:
        if s not in final_candidates:
            # Stock was in scan → data available → inject
            if s in all_screened_tickers:
                final_candidates = list(final_candidates) + [s]
                injected.append(s)
            else:
                # Stock wasn't scanned (e.g. Gorengan universe selected) — note it
                pass
    if injected:
        st.info(f"🎯 **Sector injection ({rec_sector} {sector_action['strength']}):** "
                f"Added {', '.join(injected)} to analysis — they're the recommended plays today.")

# ═══════════════════════════════════════════════════════════
# 4. DEEP ANALYSIS
# ═══════════════════════════════════════════════════════════
st.header("4. Deep Signal Analysis")
st.caption(f"Threshold: {threshold} | Risk/trade: {risk_pct_input*100:.1f}% | "
           f"Weights: {'🧠 Learned' if use_learned and journal_stats else '📐 Default'}")
st.caption(f"Analysis context: **{mode_payload['mode']}** — sizing guidance: **{mode_payload['risk_guidance']}**")

if all_scores.empty:
    st.info(
        "Preview output: Conviction score | Entry zone | Stop loss | Target | Risk/reward | "
        "FVG detected | ARB risk | Sector status"
    )

if st.button(f"⚡ Run Deep Analysis on {len(final_candidates)} stocks"):
    with st.spinner("Running full analysis — grab a coffee ☕"):
        results=run_full_analysis(final_candidates,macro_score,regime,
                                   raw_data,all_scores,active_weights)
        plan=build_execution_plan(results,macro_score,regime,allocation,
                                   portfolio_value,rr_ratio,raw_data,
                                   hb_plays,threshold,all_scores,risk_pct_input,
                                   flow_data,macro_alignment_score)
        created_signals = register_signals_from_plan(plan)
        if created_signals:
            st.success(f"📡 Signal tracker updated: {created_signals} new signal(s) persisted.")
        fvg_results = scan_market_for_fvg(raw_data) if raw_data else pd.DataFrame()
        analysis_inputs = {
            "portfolio_value": portfolio_value,
            "risk_pct_input": risk_pct_input,
            "rr_ratio": rr_ratio,
            "threshold": threshold,
            "high_beta_pct": high_beta_pct,
        }
        st.session_state.update({"results":results,"plan":plan,
                                  "macro_score":macro_score,"regime":regime,
                                  "fvg_results":fvg_results,
                                  "analysis_inputs":analysis_inputs})
        if tg_token and tg_chat_id:
            msg=build_morning_message(macro_score,stance,regime,regime_conf,
                                       rec_sector,allocation,plan,scores,
                                       commodity_context,portfolio_value,flow_narrative)
            img_path=generate_brief_image(macro_score,stance,regime,rec_sector,
                                           allocation,plan,portfolio_value)
            if send_telegram(tg_token,tg_chat_id,msg):
                st.success("📱 Telegram text sent!")
            if send_telegram_photo(tg_token,tg_chat_id,img_path):
                st.success("🖼️ Telegram image sent!")

if "results" not in st.session_state:
    st.info("👆 Run momentum screen first, then **Run Deep Analysis**.")
    st.stop()

results    =st.session_state["results"]
plan       =st.session_state["plan"]
macro_score=st.session_state["macro_score"]
regime     =st.session_state["regime"]

saved_inputs = st.session_state.get("analysis_inputs", {})
current_inputs = {
    "portfolio_value": portfolio_value,
    "risk_pct_input": risk_pct_input,
    "rr_ratio": rr_ratio,
    "threshold": threshold,
    "high_beta_pct": high_beta_pct,
}
if saved_inputs and saved_inputs != current_inputs:
    old_port = saved_inputs.get("portfolio_value", 0)
    st.warning(
        "⚠️ Settings changed after the last Deep Analysis. "
        "Sizing numbers below may reflect old inputs."
    )
    st.caption(
        f"Last analysis portfolio: Rp {old_port:,.0f} | "
        f"Current portfolio: Rp {portfolio_value:,.0f}. "
        "Click **Run Deep Analysis** again to refresh the plan."
    )

analysis_portfolio_value = saved_inputs.get("portfolio_value", portfolio_value)
analysis_risk_pct_input = saved_inputs.get("risk_pct_input", risk_pct_input)

# ═══════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════
tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10=st.tabs([
    "🎯 Execution Plan","📊 Rankings","🕯 Chart + Playbook",
    "✅ Checklist","💰 Sizing","📖 Journal","📈 Backtest","📱 Telegram", "🧲 FVG Scanner", "📡 Signal Tracker"
])
# ── TAB 1: EXECUTION PLAN ─────────────────────────────────
with tab1:
    st.subheader("🎯 Today's Execution Plan")
    summary=plan.get("_summary",{})
    fvg_df = st.session_state.get("fvg_results", pd.DataFrame())
    fvg_map = {}
    if not fvg_df.empty:
        fvg_map = {row["Ticker"]: row for _, row in fvg_df.iterrows()}
        bullish_count = int(fvg_df["Type"].str.contains("Bullish").sum())
        bearish_count = int(fvg_df["Type"].str.contains("Bearish").sum())
        st.caption(
            f"🧲 SMC Overlay active: {len(fvg_df)} fresh FVGs "
            f"({bullish_count} bullish / {bearish_count} bearish). "
            "Execution rows now show ticker-level FVG context."
        )

    mc1,mc2,mc3,mc4=st.columns(4)
    mc1.metric("Trades",         summary.get("trade_count",0))
    mc2.metric("Capital Deployed",f"Rp {summary.get('total_deployed',0):,.0f}")
    mc3.metric("% Deployed",     f"{summary.get('pct_deployed',0):.1f}%")
    mc4.metric("Cash Reserve",   f"Rp {summary.get('cash_reserve',0):,.0f}")

    # Type allocation bar
    type_alloc=summary.get("type_allocation",{})
    if type_alloc:
        st.subheader("Type Mix")
        ta_cols=st.columns(len(TRADE_TYPES))
        for i,(tt,cfg) in enumerate(TRADE_TYPES.items()):
            used=type_alloc.get(tt,0)
            cap ={"POSITION":50,"SWING":30,"SCALP":20}.get(tt,100)
            with ta_cols[i]:
                st.metric(f"{cfg['emoji']} {tt}",f"{used:.1f}%",f"Cap: {cap}%")
                bar_fill=int(used/cap*10) if cap>0 else 0
                st.caption(f"{'█'*bar_fill}{'░'*(10-bar_fill)}")
    st.divider()

    for bucket,(icon,title,desc) in [
        ("POSITION",("🏦","POSITION TRADES","Hold ~20 days — core conviction")),
        ("SWING",   ("🌊","SWING TRADES",   "Hold ~5 days — momentum plays")),
        ("SCALP",   ("⚡","SCALP TRADES",   "Hold 1 day — fast in/out")),
        ("HIGH_BETA",("🔥","HIGH-BETA",     "Vol spike — tight SL required")),
    ]:
        trades=plan.get(bucket,[])
        if not trades: continue
        st.subheader(f"{icon} {title}"); st.caption(desc)
        for trade_idx, t in enumerate(trades):
            halal="☪️" if t.get("sharia") else ""
            hb="🚀" if t.get("high_beta") else ""
            pb_em={"BUY BREAKOUT":"🚀","BUY PULLBACK":"🎯","WAIT PULLBACK":"⏳",
                   "ACCUMULATE":"📦"}.get(t.get("action",""),"")
            with st.expander(
                f"{icon} **{t['ticker']}** {halal}{hb} — {t['lots']} lots | {t['amount']} | "
                f"{pb_em} {t['action']} | {t.get('entry_type','LIMIT')}"
            ):
                c1,c2,c3,c4,c5=st.columns(5)
                c1.metric("Entry",      t["entry"])
                c2.metric("Stop Loss",  t["stop_loss"])
                c3.metric("Take Profit",t["take_profit"])
                c4.metric("Hold",       f"{t['hold_days']}d")
                c5.metric("Expiry",     t.get("order_expiry",""))
                cc1,cc2 = st.columns(2)
                cc1.metric("Confidence", f"{t.get('confidence_score',0)} / 100")
                cc2.metric("Grade", t.get("confidence_label", "C"))
                st.info(f"📋 **Strategy:** {t['strategy']}")
                st.caption(f"**Why:** {t['why']}")
                st.caption(f"**Risk:** {t.get('risk','')} ({t.get('risk_pct_str','')})")
                fvg_info = fvg_map.get(t["ticker"])
                if fvg_info is not None:
                    fvg_type = fvg_info.get("Type", "")
                    fvg_size = float(fvg_info.get("Size", 0))
                    fvg_price = fvg_info.get("Price", "")
                    if "Bullish" in fvg_type:
                        st.success(f"🧲 FVG: {fvg_type} | Size: {fvg_size:.0f} | Ref: Rp {fvg_price:,.0f}")
                    else:
                        st.warning(f"🧲 FVG: {fvg_type} | Size: {fvg_size:.0f} | Ref: Rp {fvg_price:,.0f} (counter-trend risk)")
                else:
                    st.caption("🧲 FVG: No fresh gap signal today for this ticker.")
                # Score breakdown mini-chart
                breakdown=t.get("breakdown",{})
                if breakdown:
                    bd_cols=st.columns(len(breakdown))
                    for i,(factor,data) in enumerate(breakdown.items()):
                        with bd_cols[i]:
                            sc=data["score"]
                            color="🟢" if sc>0.2 else "🔴" if sc<-0.2 else "🟡"
                            st.caption(f"{color} **{factor[:4]}**\n{sc:+.2f}")

# Log trade button
                trade_obj=get_trade_setup(t["ticker"],rr_ratio,t["trade_type"],raw_data)
                # Use bucket + index to guarantee uniqueness across repeated ticker/type combos.
                log_btn_key = f"log_{bucket}_{trade_idx}_{t['ticker']}_{t['trade_type']}"
                if trade_obj and st.button("📝 Log this trade", key=log_btn_key):
                    tid=log_trade(
                        t["ticker"],t["trade_type"],t["composite"],
                        t.get("macro_score",0),0,0,0,
                        trade_obj["entry_limit"],trade_obj["stop_loss"],trade_obj["take_profit"],
                        t["lots"],regime,t["sector"],t["action"]
                    )
                    st.success(f"✅ Trade #{tid} logged!")

    if summary.get("trade_count",0)==0:
        st.warning(f"{emoji} No trades pass threshold ({threshold}) today. Sit on hands. 🙌")

# ── TAB 2: RANKINGS ──────────────────────────────────────
with tab2:
    filtered=results if not sharia_only else [r for r in results if r.get("sharia",True)]
    show_aplus = st.checkbox("Show only A+ setups", value=False)
    min_conf = st.slider("Show only signals with confidence > X", 0, 100, 70, 5)
    scored_filtered = []
    for r in filtered:
        conf_score, conf_label = compute_trade_confidence(r, {"combined": macro_alignment_score}, flow_data)
        if show_aplus and conf_label != "A+":
            continue
        if conf_score < min_conf:
            continue
        r2 = dict(r)
        r2["confidence_score"] = conf_score
        r2["confidence_label"] = conf_label
        scored_filtered.append(r2)
    filtered = scored_filtered
    display_df=pd.DataFrame([{
        "Ticker":r["ticker"],"☪️":"☪️" if r.get("sharia") else "",
        "🚀":"🚀" if r.get("high_beta") else "",
        "Type":f"{TRADE_TYPES.get(r['trade_type'],{}).get('emoji','?')} {r['trade_type']}",
        "Playbook":f"{r.get('playbook',{}).get('emoji','')} {r.get('playbook',{}).get('action','')}",
        "Why":r.get("why_triggered",""),
        "🎓 Conf":f"{r.get('confidence_score',0)} ({r.get('confidence_label','C')})",
        "⚡ Composite":r["composite"],"📈 Technical":r["technical"],
        "📰 Sentiment":r["sentiment"],"🏗 Fundamental":r["fundamental"],
    } for r in filtered])
    if not display_df.empty:
        st.dataframe(
            display_df.style.background_gradient(
                subset=["⚡ Composite","📈 Technical","📰 Sentiment","🏗 Fundamental"],cmap="RdYlGn"),
            width="stretch")
    else:
        st.info("No setups match the selected confidence filters.")

    fig,ax=plt.subplots(figsize=(12,4))
    tlist=[r["ticker"] for r in filtered]; x,w=np.arange(len(tlist)),0.2
    ax.bar(x-1.5*w,[r["technical"]   for r in filtered],w,label="Technical",  color="#2196F3")
    ax.bar(x-0.5*w,[r["sentiment"]   for r in filtered],w,label="Sentiment",  color="#FF9800")
    ax.bar(x+0.5*w,[r["fundamental"] for r in filtered],w,label="Fundamental",color="#9C27B0")
    ax.bar(x+1.5*w,[r["composite"]   for r in filtered],w,label="Composite",  color="#4CAF50")
    ax.set_xticks(x); ax.set_xticklabels(tlist,rotation=45)
    ax.axhline(0,color="black",linewidth=0.8,linestyle="--")
    ax.axhline(threshold,color="green",linewidth=1,linestyle=":",label=f"Threshold {threshold}")
    ax.legend(); plt.tight_layout(); st.pyplot(fig)

# ── TAB 3: CHART + PLAYBOOK ───────────────────────────────
with tab3:
    choice    =st.selectbox("Select stock:",[r["ticker"] for r in results])
    r         =next(x for x in results if x["ticker"]==choice)
    trade_type=r["trade_type"]
    trade     =get_trade_setup(choice,rr_ratio,trade_type,raw_data)
    playbook  =r.get("playbook",{})

    # Playbook banner
    pb_color={"BUY BREAKOUT":"success","BUY PULLBACK":"success",
               "ACCUMULATE":"info","WAIT PULLBACK":"warning",
               "WATCH":"info","AVOID":"error"}.get(playbook.get("action",""),"info")
    getattr(st,pb_color)(
        f"{playbook.get('emoji','')} **{playbook.get('action','')}** — {playbook.get('strategy','')}\n\n"
        f"{playbook.get('reason','')}")

    # Score breakdown radar
    breakdown=r.get("breakdown",{})
    if breakdown:
        st.subheader("📊 Score Breakdown — Why this stock triggered")
        bd_cols=st.columns(len(breakdown))
        for i,(factor,data) in enumerate(breakdown.items()):
            sc=data["score"]
            color="🟢" if sc>0.2 else "🔴" if sc<-0.2 else "🟡"
            with bd_cols[i]:
                st.metric(factor,f"{sc:+.2f}")
                st.caption(data["signal"])
                st.caption(data["raw"])
        st.info(f"**WHY triggered:** {r.get('why_triggered','')}")

    halal_str="☪️ Sharia" if r.get("sharia") else "❌ Non-Sharia"
    hb_str   =" | 🚀 High Beta" if r.get("high_beta") else ""
    st.caption(f"{halal_str}{hb_str} | {TRADE_TYPES.get(trade_type,{}).get('emoji','')} {trade_type}")

    try:
        hist=yf.Ticker(choice).history(period="3mo")
        hist.index=pd.to_datetime(hist.index)
        hist=hist[["Open","High","Low","Close","Volume"]]
        add_lines=[]
        if trade:
            for val,col in [(trade["entry_limit"],"blue"),(trade["stop_loss"],"red"),(trade["take_profit"],"green")]:
                add_lines.append(mpf.make_addplot([val]*len(hist),color=col,linestyle="--",width=1.5))
        fig_c,_=mpf.plot(hist,type="candle",style="charles",addplot=add_lines if add_lines else [],
                          volume=True,returnfig=True,
                          title=f"{choice} | {trade_type} | R:R 1:{rr_ratio}",figsize=(12,6))
        st.pyplot(fig_c)
    except Exception as e:
        st.warning(f"Chart: {e}")

    if trade:
        c1,c2,c3,c4,c5,c6=st.columns(6)
        c1.metric("Price",    f"Rp {trade['price']:,.0f}")
        c2.metric("🔵 Entry", f"Rp {trade['entry_limit']:,.0f}")
        c3.metric("🔴 SL",    f"Rp {trade['stop_loss']:,.0f}",trade["stop_pct"])
        c4.metric("🟢 TP",    f"Rp {trade['take_profit']:,.0f}",trade["tp_pct"])
        c5.metric("R:R",      f"1:{rr_ratio}")
        c6.metric("Expiry",   trade.get("order_expiry",""))
        st.caption(f"Entry type: **{trade.get('entry_type','LIMIT ORDER')}** | "
                   f"ATR: Rp {trade['atr']:,.0f} | Support: {trade['support']:,.0f} | Resist: {trade['resistance']:,.0f}")

    st.divider()
    col1,col2=st.columns(2)
    with col1:
        st.markdown("**📈 Technical**")
        td=r["tech_details"]
        if "error" not in td:
            st.metric("RSI",         td.get("RSI","N/A"),   f"Score: {td.get('RSI Score',0)}")
            st.metric("Price/MA50",  f"{td.get('Price','N/A')} / {td.get('MA50','N/A')}",f"Score: {td.get('MA Score',0)}")
            st.metric("MACD Score",  td.get("MACD Score","N/A"))
            st.metric("Vol Score",   td.get("Volume Score","N/A"))
            st.metric("Volatility",  td.get("Volatility","N/A"),td.get("Vol Regime",""))
        else: st.warning(td["error"])
    with col2:
        st.markdown("**🏗 Fundamental**")
        fd=r["fund_details"]
        if "error" not in fd:
            if fd.get("cache_note"):
                st.caption(f"🕒 {fd['cache_note']}")
            for lbl,k,sk in [("P/E","P/E","PE Score"),("P/B","P/B","PB Score"),
                              ("ROE","ROE","ROE Score"),("D/E","D/E","DE Score"),
                              ("Analyst","Analyst","Analyst Score")]:
                st.metric(lbl,fd.get(k,"N/A"),f"Score: {fd.get(sk,0)}")
        else:
            err_msg = fd["error"]
            if "too many requests" in err_msg.lower() or "rate limit" in err_msg.lower() or "429" in err_msg.lower():
                st.info("Fundamental endpoint is temporarily rate-limited. The app will auto-retry and use cached data when available.")
            else:
                st.warning(err_msg)

    st.markdown("**📰 Headlines**")
    for h in r["headlines"][:8]:
        s=h.get("sentiment","neutral")
        icon="🟢" if s=="positive" else "🔴" if s=="negative" else "⚪"
        st.write(f"{icon} {h['headline']}")

# ── TAB 4: CHECKLIST ─────────────────────────────────────
with tab4:
    st.subheader(f"Trade Checklist — Threshold: {threshold}")
    passed_list=[]
    for r in results:
        if sharia_only and not r.get("sharia",True): continue
        checks,passed=trade_checklist(r,macro_score,regime,threshold)
        if passed: passed_list.append(r["ticker"])
        emoji_t=TRADE_TYPES.get(r["trade_type"],{}).get("emoji","?")
        pb=r.get("playbook",{})
        halal="☪️" if r.get("sharia") else ""
        with st.expander(
            f"{'✅' if passed else '❌'} {emoji_t} {r['ticker']} {halal} [{r['trade_type']}] "
            f"— {pb.get('emoji','')} {pb.get('action','')} | {r['composite']:+.3f}"
        ):
            for check,ok in checks.items():
                st.write(f"{'✅' if ok else '❌'} {check}")
            st.info(f"📋 {pb.get('strategy','')} — {pb.get('reason','')}")
            if passed: st.success("All checks passed → valid setup")
            else: st.error(f"Failed: {', '.join(k for k,v in checks.items() if not v)}")
    st.divider()
    if passed_list:
        st.success(f"✅ Today's setups: **{', '.join(passed_list)}**")
    else:
        st.warning(f"{emoji} No stocks pass threshold {threshold} today.")

# ── TAB 5: SIZING ────────────────────────────────────────
with tab5:
    st.subheader("💰 Risk-Based Position Sizing (1% Rule)")
    st.caption(
        f"Risk per trade: **{analysis_risk_pct_input*100:.1f}%** = Rp {analysis_portfolio_value*analysis_risk_pct_input:,.0f} | "
        f"1 lot = 100 shares | Max single position: 5% | Total cap: investable capital"
    )

    # Pull ALL trades directly from the plan (already correctly capped)
    active_rows    = []
    watch_rows     = []
    total_deployed = 0.0

    # Active trades from POSITION/SWING/SCALP buckets
    for bucket in ["POSITION","SWING","SCALP"]:
        for t in plan.get(bucket, []):
            active_rows.append({
                "Ticker":      f"{t['ticker']} {'☪️' if t.get('sharia') else ''}",
                "Type":        f"{TRADE_TYPES.get(t['trade_type'],{}).get('emoji','?')} {t['trade_type']}",
                "Action":      t.get("action",""),
                "Score":       f"{t['composite']:+.3f}",
                "Lots":        str(t["lots"]), # FIX: Forces column to be string for PyArrow
                "Amount":      t["amount"],
                "Risk":        t.get("risk","—"),
                "Risk%":       t.get("risk_pct_str","—"),
                "Entry":       t["entry"],
                "Entry Type":  t.get("entry_type",""),
                "SL":          t["stop_loss"],
                "TP":          t["take_profit"],
                "Expiry":      t.get("order_expiry",""),
                "Note":        "⚠️ Capped" if t.get("was_capped") else "",
            })
            amount_num = float(str(t["amount"]).replace("Rp","").replace(",","").strip() or 0)
            total_deployed += amount_num

    # High-beta trades
    for t in plan.get("HIGH_BETA", []):
        active_rows.append({
            "Ticker":     f"🔥 {t['ticker']} {'☪️' if t.get('sharia') else ''}",
            "Type":       "⚡ SCALP",
            "Action":     "🚀 BUY BREAKOUT",
            "Score":      f"{t['composite']:.3f}",
            "Lots":       str(t["lots"]), # FIX: Forces column to be string for PyArrow
            "Amount":     t["amount"],
            "Risk":       t.get("risk","—"),
            "Risk%":      t.get("risk_pct_str","—"),
            "Entry":      t["entry"],
            "Entry Type": "BUY STOP",
            "SL":         t["stop_loss"],
            "TP":         t["take_profit"],
            "Expiry":     t.get("order_expiry",""),
            "Note":       "HIGH BETA",
        })
        amount_num = float(str(t["amount"]).replace("Rp","").replace(",","").strip() or 0)
        total_deployed += amount_num

    # Watchlist — stocks that did NOT make the plan
    plan_tickers = {t["ticker"] for bucket in ["POSITION","SWING","SCALP","HIGH_BETA"]
                    for t in plan.get(bucket,[])}
    for r in results:
        if r["ticker"] in plan_tickers:
            continue
        if sharia_only and not r.get("sharia", True):
            continue
        trade = get_trade_setup(r["ticker"], rr_ratio, r["trade_type"], raw_data)
        pb    = r.get("playbook", {})
        watch_rows.append({
            "Ticker": f"{r['ticker']} {'☪️' if r.get('sharia') else ''}",
            "Type":   f"{TRADE_TYPES.get(r['trade_type'],{}).get('emoji','?')} {r['trade_type']}",
            "Action": f"{pb.get('emoji','')} {pb.get('action','')}",
            "Score":  f"{r['composite']:+.3f}",
            "Entry":  f"Rp {trade['entry_limit']:,.0f}" if trade else "N/A",
            "SL":     f"Rp {trade['stop_loss']:,.0f}" if trade else "N/A",
            "TP":     f"Rp {trade['take_profit']:,.0f}" if trade else "N/A",
            "Why not": explain_rejection(r, trade_checklist(r, macro_score, regime, threshold)[0], macro_score),
        })

    summary = plan.get("_summary", {})

    # Active trades
    st.subheader(f"✅ Active Trades — {len(active_rows)}")
    if active_rows:
        st.dataframe(_safe_df(active_rows), width="stretch")
    else:
        st.warning("No stocks passed checklist today. Sit on hands 🙌")

    st.divider()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Portfolio",      f"Rp {analysis_portfolio_value:,.0f}")
    c2.metric("Total Deployed", f"Rp {summary.get('total_deployed',0):,.0f}")
    c3.metric("% Deployed",     f"{summary.get('pct_deployed',0):.1f}%")
    c4.metric("Max risk/trade", f"Rp {analysis_portfolio_value*analysis_risk_pct_input:,.0f}")

    # Sanity check — should never exceed 100% now
    pct = summary.get("pct_deployed", 0)
    if pct > 95:
        st.error(f"⚠️ {pct}% deployed — near or over portfolio limit. Check position caps.")
    elif pct > 70:
        st.warning(f"🟡 {pct}% deployed — concentrated. Make sure you have liquidity.")

    # Watchlist (collapsed)
    with st.expander(f"👀 Watchlist — {len(watch_rows)} stocks (did not pass checklist)"):
        st.caption("These stocks were analyzed but did not make the execution plan.")
        if watch_rows:
            st.dataframe(
                pd.DataFrame(watch_rows)[["Ticker","Type","Action","Score","Entry","SL","TP","Why not"]]
                .style.background_gradient(subset=["Score"], cmap="RdYlGn"),
                width="stretch"
            )

# ── TAB 6: JOURNAL ────────────────────────────────────────
with tab6:
    st.subheader("📖 Trade Journal + Learning Loop")

    # Open trades
    open_trades=get_open_trades()
    if open_trades:
        st.subheader(f"🟡 Open Trades ({len(open_trades)})")
        for t in open_trades:
            with st.expander(f"#{t['id']} {t['ticker']} [{t['trade_type']}] — opened {t['date']} | exp: {t.get('expiry_date','')}"):
                c1,c2,c3,c4=st.columns(4)
                c1.metric("Entry",     f"Rp {t['entry_price']:,.0f}")
                c2.metric("Stop Loss", f"Rp {t['stop_loss']:,.0f}")
                c3.metric("Take Profit",f"Rp {t['take_profit']:,.0f}")
                c4.metric("Lots",      t['lots'])

                exit_price=st.number_input(f"Exit price for #{t['id']}:",
                                            min_value=0.0,value=float(t["entry_price"]),
                                            key=f"exit_{t['id']}")
                if st.button(f"✅ Close trade #{t['id']}",key=f"close_{t['id']}"):
                    close_trade(t["id"],exit_price)
                    st.success("Trade closed and weights updated!")
                    st.rerun()
    else:
        st.info("No open trades. Log trades from the Execution Plan tab.")

    st.divider()

    # Journal stats
    stats,weights=get_journal_stats()
    if stats:
        st.subheader("📊 Performance Stats")
        s_cols=st.columns(4)
        s_cols[0].metric("Total Trades",stats.get("total_trades",""))
        s_cols[1].metric("Win Rate",    stats.get("win_rate",""))
        s_cols[2].metric("Avg Win",     stats.get("avg_win_pct",""))
        s_cols[3].metric("Avg Loss",    stats.get("avg_loss_pct",""))

        st.subheader("🧠 Learned Weights")
        w_cols=st.columns(4)
        for i,(k,v) in enumerate(weights.items()):
            default=DEFAULT_WEIGHTS.get(k,0.25)
            delta=round(v-default,3)
            w_cols[i].metric(k.title(),f"{v:.3f}",f"{delta:+.3f} vs default")
        st.caption(f"Last updated: {stats.get('last_updated','Never')}")

    # Full journal table
    jdf=get_journal_df()
    if not jdf.empty:
        st.subheader("All Trades")
        st.dataframe(
            jdf[["id","date","ticker","trade_type","regime","composite",
                 "entry_price","exit_price","pnl_pct","result","status"]]
            .style.background_gradient(subset=["pnl_pct"],cmap="RdYlGn"),
            width="stretch"
        )
        # PnL chart
        closed=jdf[jdf["status"]=="CLOSED"].copy()
        if len(closed)>0:
            st.subheader("P&L Distribution")
            fig,ax=plt.subplots(figsize=(10,3))
            rets=closed["pnl_pct"].values
            ax.bar(range(len(rets)),rets,
                   color=["#4CAF50" if r>0 else "#F44336" for r in rets],width=0.8)
            ax.axhline(0,color="black",linewidth=0.8)
            ax.set_xlabel("Trade #"); ax.set_ylabel("P&L %")
            plt.tight_layout(); st.pyplot(fig)
    else:
        st.info("No trades logged yet.")

# ── TAB 7: BACKTEST V2 ───────────────────────────────────
with tab7:
    st.subheader("📈 Backtest V2 — Sharpe + Per-Type Analytics")
    bc1,bc2=st.columns(2)
    bt_threshold=bc1.slider("Entry threshold",0.10,0.80,0.30,0.05)
    bt_capital  =bc2.number_input("Start capital (IDR)",value=100_000_000,
                                   min_value=10_000_000,step=10_000_000,format="%d")
    if st.button("▶️ Run Backtest"):
        if not raw_data: st.warning("Run Momentum Screen first.")
        else:
            with st.spinner("Simulating trades..."):
                trades_df,stats,type_stats=run_backtest(raw_data,final_candidates,
                                                         threshold=bt_threshold,start_capital=bt_capital)
                st.session_state.update({"bt_trades":trades_df,"bt_stats":stats,"bt_type_stats":type_stats})

    if "bt_stats" in st.session_state:
        stats=st.session_state["bt_stats"]
        trades_df=st.session_state["bt_trades"]
        type_stats=st.session_state.get("bt_type_stats",{})

        # Overall stats — now includes Sharpe, Calmar, Expectancy
        st.subheader("Overall Results")
        scols=st.columns(5)
        for i,(k,v) in enumerate(list(stats.items())[:10]):
            scols[i%5].metric(k,v)

        if type_stats:
            st.subheader("By Trade Type")
            tcols=st.columns(len(type_stats))
            for i,(tt,ts) in enumerate(type_stats.items()):
                with tcols[i]:
                    st.markdown(f"**{TRADE_TYPES.get(tt,{}).get('emoji','?')} {tt}**")
                    for k,v in ts.items(): st.metric(k,v)

        if not trades_df.empty:
            st.subheader("Equity Curve")
            fig,axes=plt.subplots(2,1,figsize=(12,7),gridspec_kw={"height_ratios":[3,1]})
            colors_t={"SCALP":"#FF9800","SWING":"#2196F3","POSITION":"#4CAF50"}
            for tt in trades_df["trade_type"].unique():
                sub=trades_df[trades_df["trade_type"]==tt]
                axes[0].plot(sub.index,sub["equity"],label=tt,color=colors_t.get(tt,"gray"),linewidth=1.5)
            axes[0].axhline(bt_capital,color="gray",linestyle="--",linewidth=1,label="Start")
            axes[0].set_ylabel("Portfolio (IDR)"); axes[0].legend()
            equity=trades_df["equity"].values; peak=np.maximum.accumulate(equity)
            dd=(equity-peak)/peak*100
            axes[1].fill_between(range(len(dd)),dd,0,color="red",alpha=0.4)
            axes[1].set_ylabel("Drawdown %"); axes[1].set_xlabel("Trade #")
            plt.tight_layout(); st.pyplot(fig)

            with st.expander("All Trades"):
                st.dataframe(
                    trades_df[["ticker","trade_type","entry_date","exit_date",
                               "entry_px","exit_px","return_net","win"]]
                    .style.background_gradient(subset=["return_net"],cmap="RdYlGn"),
                    width="stretch")

# ── TAB 8: TELEGRAM ─────────────────────────────────────
with tab8:
    st.subheader("📱 Telegram Morning Briefing")
    st.markdown("""
    **Setup:**
    1. Telegram → `@BotFather` → `/newbot` → copy token
    2. Message your bot → `https://api.telegram.org/bot<TOKEN>/getUpdates`
    3. Copy `chat_id` → paste in sidebar
    """)
    st.caption(
        "You can also store credentials in Streamlit secrets using keys: "
        "`telegram_bot_token` + `telegram_chat_id` (or `[telegram] bot_token/chat_id`)."
    )
    msg=build_morning_message(macro_score,stance,regime,regime_conf,
                               rec_sector,allocation,plan,scores,
                               commodity_context,portfolio_value,flow_narrative)
    st.subheader("Preview Text")
    st.code(msg,language=None)

    st.subheader("Preview Image")
    if st.button("🖼️ Generate Image Preview", key="preview_telegram_image"):
        st.session_state["telegram_preview_img"] = generate_brief_image(
            macro_score, stance, regime, rec_sector, allocation, plan, portfolio_value
        )

    preview_img = st.session_state.get("telegram_preview_img")
    if preview_img:
        st.image(preview_img, caption="Telegram brief image preview", width="stretch")
    else:
        st.caption("Click **Generate Image Preview** to render the latest briefing image.")

    if not (tg_token and tg_chat_id):
        st.info("You can preview text/image without credentials. Add Bot Token + Chat ID only when ready to send.")

    col1,col2=st.columns(2)
    if col1.button("📤 Send Text Brief", disabled=not (tg_token and tg_chat_id), key="send_telegram_text"):
        if send_telegram(tg_token,tg_chat_id,msg): st.success("✅ Sent!")
        else: st.error("❌ Failed.")
    if col2.button("📤 Send Image Brief", disabled=not (tg_token and tg_chat_id), key="send_telegram_image"):
        img=preview_img or generate_brief_image(macro_score,stance,regime,rec_sector,
                                                allocation,plan,portfolio_value)
        if send_telegram_photo(tg_token,tg_chat_id,img):
            st.success("✅ Image sent!")
        else: st.error("❌ Failed.")

# ── TAB 9: FVG SCANNER ──────────────────────────────────
with tab9:
    st.subheader("🧲 Smart Money Concepts: Fair Value Gap (FVG) Scanner")
    st.markdown("Scans the IDX universe for institutional imbalances. A fresh FVG indicates where Smart Money just aggressively stepped in.")
    
    if "raw_data" not in st.session_state or not st.session_state["raw_data"]:
        st.warning("⚠️ Please run the **Momentum Screen** (in section 3 above) first to download market data.")
    else:
        if st.button("Scan for Fresh FVGs Today"):
            with st.spinner("Hunting for imbalances..."):
                fvg_results = scan_market_for_fvg(st.session_state["raw_data"])
                st.session_state["fvg_results"] = fvg_results
                
                if not fvg_results.empty:
                    # Sort by the largest FVG size
                    fvg_results = fvg_results.sort_values(by="Size", ascending=False).reset_index(drop=True)
                    st.success(f"Found {len(fvg_results)} active Fair Value Gaps today!")
                    
                    st.dataframe(
                        fvg_results.style.format({"Size": "{:.0f}", "Price": "Rp {:,.0f}"}),
                        width="stretch"
                    )
                else:
                    st.info("No fresh Fair Value Gaps detected on the daily timeframe today. Market is balanced.")

        cached_fvg = st.session_state.get("fvg_results", pd.DataFrame())
        if not cached_fvg.empty:
            st.divider()
            st.caption("Latest cached FVG scan (also used by Execution Plan overlay):")
            st.dataframe(
                cached_fvg.sort_values(by="Size", ascending=False).reset_index(drop=True)
                .style.format({"Size": "{:.0f}", "Price": "Rp {:,.0f}"}),
                width="stretch"
            )

with tab10:
    st.subheader("📡 Signal Tracker")
    all_signals = update_signal_statuses()
    perf = compute_signal_performance(all_signals)
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Win Rate (7d)", perf["win_rate_7d"])
    mc2.metric("Win Rate (30d)", perf["win_rate_30d"])
    mc3.metric("Win Rate (All)", perf["win_rate_all"])
    mc4.metric("Avg Return", perf["avg_return"])
    pc1, pc2 = st.columns(2)
    pc1.metric("Expectancy", perf["expectancy"])
    pc2.metric("Max Drawdown", perf["max_drawdown"])

    if all_signals:
        sdf = pd.DataFrame(all_signals).sort_values("timestamp", ascending=False)
        sdf["age_decay"] = sdf["days_since_signal"].apply(lambda d: max(0, min(25, d * 2)))
        sdf["effective_confidence"] = (
            pd.to_numeric(sdf.get("initial_confidence"), errors="coerce").fillna(0) - sdf["age_decay"]
        ).clip(lower=0)
        sdf["stale"] = sdf["days_since_signal"] >= 7
        sdf["status"] = np.where(sdf["stale"] & (sdf["status"] == "ACTIVE"), "ACTIVE ⚠️ STALE", sdf["status"])
        st.dataframe(
            sdf[[
                "ticker","entry","stop_loss","take_profit","timestamp","expiry_date",
                "current_price","current_return_pct","days_since_signal",
                "effective_confidence","status"
            ]],
            width="stretch"
        )
        active_tape = sdf[sdf["status"].str.contains("ACTIVE", na=False)].head(12)
        if not active_tape.empty:
            tape = " | ".join([
                f"{r['ticker']} {float(r.get('current_return_pct',0) or 0):+.2f}% ({int(r.get('effective_confidence',0))})"
                for _, r in active_tape.iterrows()
            ])
            st.markdown(
                f"<marquee behavior='scroll' direction='left' scrollamount='8'>"
                f"<span class='orange-accent'>Ticker Tape:</span> {tape}</marquee>",
                unsafe_allow_html=True
            )
    else:
        st.info("No signals tracked yet. Run Deep Analysis to generate fresh signals.")
