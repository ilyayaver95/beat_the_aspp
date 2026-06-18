# Beat the ASPP — Multi-Agent AI Stock Evaluator

> A production-style, multi-agent LLM application that produces institutional-grade equity research reports for any US-listed ticker. Built end-to-end in Python: data ingestion, three specialized analyst agents, a synthesizing portfolio strategist, structured outputs, cost tracking, per-user accounts, portfolio tracking, scheduled scans, Telegram alerts, and an interactive Streamlit dashboard.

---

## Why this project

Most "ChatGPT-asks-the-market" demos are a single prompt wrapping a price feed. This project does what a real research desk does — it **decomposes the problem across specialists**, each with their own tools, prompts, and structured outputs, then has a senior strategist agent reconcile their views into a single defensible verdict.

It is intentionally built to demonstrate the engineering disciplines that matter when shipping LLM systems in production:

- Multi-agent orchestration with parallel execution
- Schema-enforced structured outputs (Pydantic)
- Provider abstraction across paid + free LLM backends (Anthropic / Groq / Ollama)
- Prompt caching, adaptive thinking, streaming
- Token & cost telemetry per call
- Per-user authentication (password + Google OAuth) and isolated state
- Multi-tenant persistence (Postgres on Cloud, SQLite locally) via SQLAlchemy
- Portfolio tracking + baseline-and-forward P&L
- Scheduled scans and Telegram alert delivery

---

## System architecture

```
                 ┌──────────────────────────────┐
                 │        ORCHESTRATOR          │
                 │  ThreadPoolExecutor(max=3)   │
                 │  (max=1 on Groq free tier)   │
                 └──────┬────────┬────────┬─────┘
                        │        │        │
                ┌───────▼──┐ ┌───▼───┐ ┌──▼──────┐
                │Technical │ │Fund.  │ │Sentiment│
                │  Agent   │ │ Agent │ │  Agent  │
                └───────┬──┘ └───┬───┘ └──┬──────┘
       yfinance OHLCV  │  yfinance financials  │   yfinance + RSS
       indicators     │  10-K/10-Q derivations │   multi-source news
                        │        │              │
                ┌───────▼────────▼──────────────▼─────┐
                │          SYNTHESIZER                │
                │  Claude Opus 4.6                    │
                │  + Adaptive Thinking, effort: high  │
                │  Persona: Sr. Equity Research Lead  │
                │  Weights: 45% Fund / 35% Tech / 20% │
                └────────────────┬────────────────────┘
                                 ▼
                ┌────────────────────────────────────┐
                │  Final Report (Pydantic schema)    │
                │  · Composite score & verdict       │
                │  · Confidence & price target       │
                │  · Risks, opportunities, thesis    │
                │  · Interactive HTML + Plotly chart  │
                └────────────────────────────────────┘
                                 │
                                 ▼
                ┌────────────────────────────────────┐
                │  Persistence (per user, SQL)       │
                │  · users / favorites / telegram    │
                │  · trades (Portfolio Tracker)      │
                │  · baseline + transfers + trades   │
                │    (Portfolio Baseline)            │
                │  + JSON snapshot for Scanner       │
                └────────────────────────────────────┘
                                 │
                                 ▼
                ┌────────────────────────────────────┐
                │  Scanner + Alerts                  │
                │  · Re-runs stale favorites         │
                │  · Detects buy / sell zone hits    │
                │  · Pushes Telegram (HTML formatted)│
                └────────────────────────────────────┘
```

Each agent owns its own data pipeline → deterministic feature extraction → LLM evaluation → Pydantic-validated report. The orchestrator is the only component aware of all three. Everything below the synthesizer (persistence, scanning, alerts) runs against the same per-user identity established by `auth.require_login()`.

---

## What I built — and what each piece demonstrates

| Component | What it does | What it demonstrates |
|---|---|---|
| `agents/technical_agent.py` | Computes EMA/SMA stack (20/50/100/150/200), RSI, ATR, candlestick patterns, support/resistance, golden/death cross. LLM scores the setup. | Hybrid system: deterministic signals + LLM judgment. The LLM never invents numbers — it only interprets them. |
| `agents/fundamental_agent.py` | Pulls financials (revenue growth, net margin, P/E, EPS growth, ROE, D/E, FCF) and grades each metric A–F before scoring. | Constraining the LLM to a rubric reduces hallucination and makes outputs comparable across tickers. |
| `agents/sentiment_agent.py` | Scrapes 5+ news sources, extracts themes, classifies tone, identifies upcoming catalysts. | Multi-source aggregation + LLM as an information-extraction layer (not a content generator). |
| `orchestrator.py` | Runs all three agents in parallel via `ThreadPoolExecutor`, then calls Claude Opus 4.6 with **adaptive thinking** and `effort: high` to synthesize. Streams the analyst narrative live, then runs a cheap Sonnet pass to extract structured fields. Auto-degrades to sequential on Groq (free-tier rate limits). | Two-stage synthesis pattern: expensive model writes the prose, cheap model parses it into JSON. ~5× cost reduction with no quality loss. |
| `llm_client.py` | Provider abstraction (Anthropic / Groq / Ollama) behind a unified `client.messages.parse()` and `client.messages.stream()` interface — duck-typed to match the Anthropic SDK exactly. Thread-local progress callbacks surface retries in the UI. | Clean ports-and-adapters design. Agents are provider-agnostic. Switching backends is a single radio button. |
| `cost_tracker.py` | Wraps every Anthropic call to record tokens, model, cache hits, and dollar cost per agent call. Transparently injects `cache_control` blocks into system prompts. Exposes `predict_analysis_cost()` for the UI sidebar. | Production-grade observability. Prompt caching cuts repeat-call system-prompt cost by ~90%. |
| `models/report.py` | Pydantic schemas for every agent output and the final report. | Schema-first LLM engineering — guarantees downstream code can rely on field shapes. |
| `report_generator.py` | Renders a self-contained HTML report with a Plotly candlestick chart, scored tables, and embedded news quotes. | The output is a deliverable a portfolio manager could actually read. |
| `auth.py` + `auth_db.py` | Streamlit login gate. Username/password (bcrypt) **or** Google OAuth via `st.login()`. Per-user favorites and Telegram credentials live on the user row. | Real multi-user app: every downstream feature is scoped by `user_id`. |
| `db.py` | Single SQLAlchemy engine for the whole app. Uses Postgres when `DATABASE_URL` is set (Streamlit Cloud / Neon / Supabase / Render), falls back to `data/local.db` SQLite locally. Normalises legacy `postgres://` URLs. | One code path, two dialects — no `if cloud:` branches in app code. |
| `portfolio_db.py` + `pages/2_Portfolio_Tracker.py` | Trade ledger with realized + unrealized P&L. Every row scoped by `user_id`. | Multi-tenant data done right: shared DB, hard-scoped queries. |
| `portfolio_baseline_db.py` + `pages/3_Portfolio_Baseline.py` | Snapshot the current portfolio state as a baseline, then track forward BUY/SELL trades and cash transfers against it. | The "I don't have my full trade history" workflow — value+return baselines that roll forward cleanly. |
| `migrate_to_db.py` | One-shot, idempotent migration from the old per-file SQLite layout to the unified DB. | Real-world data migrations: schema-aware, dry-run-able. |
| `app.py` + `pages/` | Streamlit dashboard: auth, ticker input, favorites with per-favorite scanning toggle, cached past analyses, BYO-API-key sidebar, Telegram setup. | Full UX layer on top of the agent pipeline. |
| `scanner.py` + `alerts/` | Re-analyzes favorited tickers during US market hours and pushes Telegram / WhatsApp alerts when price hits the buy/sell zone defined by the latest report's support/resistance. Per-user creds resolved via the DB. | Closes the loop from analysis → action. Useful, not just a demo. |

---

## Claude API techniques used

Each of these is a deliberate choice, not boilerplate:

| Technique | Where | Why |
|---|---|---|
| **Adaptive thinking + effort: high** | Synthesis step (Opus 4.6) | Multi-factor reconciliation needs deep reasoning; the synthesizer is the one place where extra latency is worth it. |
| **Prompt caching** | All system prompts via `TrackedMessages` | System prompts are large and identical across tickers. Caching makes repeat scans cheap. |
| **Structured output (`output_format=`)** | All three agents | Guaranteed Pydantic objects → no fragile regex parsing. |
| **Streaming** | Synthesis + UI | Long reports otherwise hang for 30–60s. Streaming makes the UX feel like a human typing. |
| **Two-stage write-then-extract** | `_synthesize` → `_parse_streamed_report` | Opus writes the narrative, Sonnet extracts the JSON. Same quality, ~5× cheaper. |
| **Parallel agent execution** | `ThreadPoolExecutor(max_workers=3)` | Three independent LLM calls run concurrently → ~3× faster wall-clock time. Auto-falls back to sequential on Groq (6K tok/min rate cap). |
| **Persona system prompts** | Every agent + synthesizer | "Senior Equity Research Analyst at a top-tier bank" produces measurably more decisive, specific output than a generic prompt. |
| **Graceful provider abstraction** | `llm_client.py` | Anthropic-specific kwargs (`thinking`, `output_config`) are silently dropped when running against Groq/Ollama, so the same agent code works everywhere. |
| **Thread-local progress callbacks** | `llm_client.set_progress_callback` | Retries inside parallel agents stream live status to the Streamlit UI without leaking thread state. |

---

## Tech stack

- **LLMs**: Anthropic Claude (Opus 4.6 / Sonnet 4.6), Groq (Llama 3.1 / 3.3 / Llama-4 Maverick / Qwen3), Ollama (local)
- **Data**: yfinance, RSS feeds, BeautifulSoup, feedparser
- **Validation**: Pydantic v2
- **Concurrency**: `concurrent.futures.ThreadPoolExecutor` with shared thread-local progress callback
- **UI**: Streamlit (multi-page), Plotly, Rich (CLI)
- **Auth**: bcrypt password hashing, Streamlit-native Google OAuth (`st.login`)
- **Persistence**: SQLAlchemy 2 engine — Postgres (`DATABASE_URL`) on Streamlit Cloud, SQLite (`data/local.db`) locally
- **Notifications**: Telegram Bot API (per-user creds), WhatsApp via Twilio

---

## Running it

```bash
# 1. Install
pip install -r requirements.txt

# 2. Add an API key (any one works)
cp .env.example .env
#   ANTHROPIC_API_KEY=...   (paid, best quality)
#   GROQ_API_KEY=...        (free tier, very fast — default in the web UI)
#   or run --llm ollama for fully local

# 3a. CLI (Anthropic or Ollama; Groq is web-UI only)
python main.py --ticker NVDA
python main.py --ticker MSFT --period 6mo --llm ollama --ollama-model llama3.2

# 3b. Web UI (Streamlit, multi-page: Analyzer / Portfolio Tracker / Portfolio Baseline)
streamlit run app.py
# - log in with username+password, or Google if [auth] is configured in
#   .streamlit/secrets.toml
# - pick Groq / Anthropic / Ollama from the provider radio
# - paste your own API key in the sidebar to bypass the deployer's quota
```

Sample output: a streamed analyst narrative, a scored breakdown across the three dimensions, a verdict (`STRONG BUY` → `STRONG SELL`), confidence %, price target, key risks/opportunities, and a saved interactive HTML report — plus, in the web app, automatic JSON snapshots the Scanner uses to alert you when price hits the buy zone.

### Running on Streamlit Cloud

1. Push to GitHub and connect the repo on Streamlit Cloud.
2. In **App settings → Secrets**, add a `DATABASE_URL` (Neon / Supabase / Render — plain `postgres://...` URLs are auto-normalised) and any of `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `NEWS_API_KEY`.
3. Optional: add an `[auth]` block with a Google provider to enable single-click Google sign-in via `st.login`.
4. First run will auto-create all tables (`db.init_all()`).

### Migrating an existing local install

```bash
python migrate_to_db.py --dry-run    # show what would move
python migrate_to_db.py              # apply (idempotent)
```

This sweeps `data/users.db`, legacy per-user `portfolio_u<N>.db` / `portfolio_<sha>.db` files, baseline DBs, and the old global `favorites.json` into the unified schema.

---

## What I'd build next

- Backtesting harness: replay agents against historical data and measure verdict accuracy vs forward returns.
- Sector-relative scoring: a `STRONG BUY` semiconductor in a falling sector should be flagged.
- Anthropic Batches API for overnight portfolio-wide scans at 50% cost.
- Tool use / agentic loops so agents can request additional data (e.g. peer comparisons) instead of receiving a fixed feature set.
- Eval harness: a regression set of tickers with expected verdict ranges, run on every commit.

---

## What this project is *not*

Investment advice. The verdict is a synthesis of public data by an LLM. Treat it as a research starting point, not a trade signal.
