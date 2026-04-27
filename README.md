# IDX Macro Trading Dashboard — Gen 7

Professional Streamlit dashboard for macro-aware IDX trading with operating modes (Daily Grind / Elevated Risk / Crisis Entry), DG 5-box checklist, and journaling support.

## Requirements

- Python **3.11**
- Streamlit Cloud or local environment with internet access
- API credentials in Streamlit secrets (no hardcoded keys)

## Installation

```bash
pip install -r requirements.txt
```

## Run locally

```bash
streamlit run app.py
```

Open the URL printed by Streamlit (normally `http://localhost:8501`).

## Streamlit secrets

Add these keys in `.streamlit/secrets.toml` (local) or Streamlit Cloud Secrets:

```toml
FRED_API_KEY = "your_fred_key"
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_KEY = "your_supabase_anon_or_service_key"
telegram_bot_token = "123:abc"
telegram_chat_id = "123456789"
```

Supported aliases in app:
- Telegram: `telegram_bot_token` / `TG_BOT_TOKEN`, `telegram_chat_id` / `TG_CHAT_ID`
- Supabase: `SUPABASE_URL`/`supabase_url`, `SUPABASE_KEY`/`supabase_key`

## How to use (simple)

1. Configure **Universe and filters** in sidebar.
2. Check the top **Operating Mode Banner** for today’s risk guidance.
3. Click **Run Momentum Screen**.
4. (Optional) edit **Override candidates**.
5. Click **Run Deep Analysis**.
6. Review execution plan, ARB risk flags, and journal entries.

## Supabase tables

The app expects these tables:

- `trades` (existing)
- `margin_debt` (`month` text, `debit_balance_millions` float, `recorded_at` timestamptz)
- `earnings_log` (`ticker` text, `period` text, `eps_actual` float, `eps_estimate` float, `revenue_beat` boolean, `recorded_at` timestamptz)

## Notes

- IDX lot convention is enforced in sizing logic (1 lot = 100 shares, round down).
- ARB risk is flagged when price < 200 IDR or average volume < 5M shares/day.
- If FRED API is unavailable, use manual overrides from the sidebar fallback expander.
