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
# Optional: app falls back to public FRED CSV when omitted.
FRED_API_KEY = "your_fred_key"
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_KEY = "your_supabase_anon_or_service_key"
telegram_bot_token = "123:abc"
telegram_chat_id = "123456789"
```

Supported aliases in app:
- Telegram: `telegram_bot_token` / `TG_BOT_TOKEN`, `telegram_chat_id` / `TG_CHAT_ID`
- Supabase: `SUPABASE_URL`/`supabase_url`, `SUPABASE_KEY`/`supabase_key`
- FRED (optional): `FRED_API_KEY` / `fred_api_key`; without it, the app uses FRED's public CSV feed.

Nested sections are also supported:

```toml
[supabase]
url = "https://xxxx.supabase.co"
key = "your_supabase_key"

[fred]
api_key = "your_fred_key"

[telegram]
bot_token = "123:abc"
chat_id = "123456789"
```

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
- `FRED_API_KEY` is optional. The app first uses the keyed FRED API when configured, then falls back to FRED's public CSV feed, and finally to manual overrides if live macro data is unavailable.

## Streamlit file watcher in containers

This repo includes `.streamlit/config.toml` with `server.fileWatcherType = "none"` so Streamlit does not allocate Linux inotify watches in hosted/container deployments. This prevents noisy startup errors such as `OSError: [Errno 28] inotify watch limit reached` while keeping the app runnable. If you want local hot-reload during development, override it with `STREAMLIT_SERVER_FILE_WATCHER_TYPE=poll streamlit run app.py`.
