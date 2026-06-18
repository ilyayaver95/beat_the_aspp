# Beat the ASPP — Project Flowchart & Status Tracker

> Update this file as each component is built.

## Legend
- ✅ **Complete** — Built and tested
- 🔄 **In Progress** — Currently building
- 📋 **Planned** — On the roadmap
- ⚠️ **Needs Enhancement** — Works but can be improved

---

## System Architecture (Full Data Flow)

```
┌─────────────────────────────────────────────────────────────────────┐
│  ENTRY                                                              │
│  · CLI:  main.py --ticker NVDA                                      │
│  · Web:  streamlit run app.py  →  auth.require_login()              │
│                                  Per-user identity from now on.     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR (orchestrator.py)                 │
│  - Spawns 3 agents in parallel (ThreadPoolExecutor, max_workers=3)  │
│    · auto-degrades to max_workers=1 for Groq (6K tok/min cap)       │
│  - Collects structured reports from each                            │
│  - Calls Claude (Opus 4.6 + Adaptive Thinking) for synthesis        │
│  - Streams retries via thread-local progress callback               │
└───────┬────────────────────┬───────────────────────┬───────────────┘
        │                    │                        │
        ▼                    ▼                        ▼
┌───────────────┐   ┌────────────────┐   ┌───────────────────────┐
│   AGENT 1     │   │    AGENT 2     │   │       AGENT 3         │
│  Technical    │   │  Fundamental   │   │      Sentiment        │
│  Analysis     │   │   Analysis     │   │      Analysis         │
└───────┬───────┘   └───────┬────────┘   └───────────┬───────────┘
        │                   │                         │
        ▼                   ▼                         ▼
┌───────────────┐   ┌────────────────┐   ┌───────────────────────┐
│  market_data  │   │ financial_data │   │    news_scraper       │
│   (yfinance)  │   │  (yfinance)    │   │ (yfinance + RSS feeds)│
└───────┬───────┘   └───────┬────────┘   └───────────┬───────────┘
        │                   │                         │
        ▼                   ▼                         ▼
┌───────────────┐   ┌────────────────┐   ┌───────────────────────┐
│  Processing:  │   │  Processing:   │   │   Processing:         │
│ • Candle      │   │ • Revenue      │   │ • Scrape 5 sources    │
│   anatomy     │   │   Growth YoY   │   │ • NLP sentiment       │
│ • Patterns    │   │ • Net Income   │   │ • Key themes          │
│   (Hammer,    │   │ • P/E Ratio    │   │ • Upcoming catalysts  │
│   Doji, etc.) │   │ • EPS Growth   │   │ • Analyst tone        │
│ • Support &   │   │ • ROE / D/E    │   │                       │
│   Resistance  │   │ • Free Cash    │   │                       │
│ • MA 20/50/   │   │   Flow         │   │                       │
│   100/150/200 │   │ • Margins      │   │                       │
└───────┬───────┘   └───────┬────────┘   └───────────┬───────────┘
        │                   │                         │
        ▼                   ▼                         ▼
┌───────────────┐   ┌────────────────┐   ┌───────────────────────┐
│  LLM Call     │   │  LLM Call      │   │    LLM Call           │
│  (Sonnet 4.6  │   │  (Sonnet 4.6   │   │    (Sonnet 4.6        │
│  / Groq /     │   │  / Groq /      │   │    / Groq /           │
│  Ollama)      │   │  Ollama)       │   │    Ollama)            │
│  Score: /10   │   │  Score: /10    │   │    Score: /10         │
└───────┬───────┘   └───────┬────────┘   └───────────┬───────────┘
        └─────────────────── ┤ ──────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    SYNTHESIZER  (orchestrator.py)                   │
│  Step A — WRITE: claude-opus-4-6 + adaptive thinking + effort:high  │
│           streamed analyst narrative                                │
│  Step B — EXTRACT: claude-sonnet-4-6 parses narrative → JSON        │
│  Persona: Senior Equity Research Analyst                            │
│  Weights: 35% Technical · 45% Fundamental · 20% Sentiment           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         FINAL REPORT                                │
│  • Composite Score (weighted)                                       │
│  • Verdict: STRONG BUY / BUY / HOLD / SELL / STRONG SELL            │
│  • Confidence %                                                     │
│  • Price Target (if determinable)                                   │
│  • Key Risks & Opportunities                                        │
│  • Analyst Thesis (narrative)                                       │
│  • Recommended Time Horizon                                         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PERSIST                                                            │
│  · analysis_store.save_analysis() → data/analyses/{TICKER}_latest   │
│  · report_generator.generate_html_report() → reports/{T}_{date}.html│
│  · cost_tracker → data/usage_log.jsonl (tokens, $, cache hits)      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SCANNER + ALERTS  (scanner.py, alerts/telegram.py)                 │
│  · Iterates favorites scoped to the logged-in user                  │
│  · Re-runs analysis if snapshot >3 days old                         │
│  · Detects in-buy-zone / below-support / in-sell-zone / above-res   │
│  · Per-user Telegram creds from auth_db → HTML-formatted alert      │
└─────────────────────────────────────────────────────────────────────┘
```

### Auth + Persistence layer (transverse to the pipeline)

```
┌───────────────────────────────────────────────────────────────────┐
│  auth.require_login()                                             │
│    ├── username + password (bcrypt) ──────► users table           │
│    └── Google OAuth via st.login()  ──────► users.google_sub      │
└───────────────────────┬───────────────────────────────────────────┘
                        │ user_id
        ┌───────────────┼─────────────────────────────────┐
        ▼               ▼                                 ▼
┌──────────────┐ ┌───────────────────┐ ┌───────────────────────────┐
│  favorites   │ │  trades           │ │  baseline_meta            │
│  (per user)  │ │  (Portfolio       │ │  baseline_positions       │
│              │ │   Tracker page)   │ │  baseline_trades          │
│              │ │                   │ │  baseline_transfers       │
│              │ │                   │ │  (Portfolio Baseline page)│
└──────────────┘ └───────────────────┘ └───────────────────────────┘
                          ▲
                          │ db.py — single SQLAlchemy engine
                          │   · Postgres when DATABASE_URL is set
                          │   · SQLite (data/local.db) otherwise
                          │   · postgres:// → postgresql+psycopg:// normalised
```

---

## Component Status

### Phase 1 — Foundation
| File | Status | Description |
|------|--------|-------------|
| `FLOWCHART.md` | ✅ Complete | This document |
| `requirements.txt` | ✅ Complete | All Python dependencies |
| `.env.example` | ✅ Complete | API key template |
| `models/report.py` | ✅ Complete | Pydantic data models for all agent outputs |

### Phase 2 — Data Layer (Tools)
| File | Status | Description |
|------|--------|-------------|
| `tools/market_data.py` | ✅ Complete | yfinance OHLCV + pandas indicators, patterns, S/R |
| `tools/financial_data.py` | ✅ Complete | yfinance fundamentals (income, balance, cashflow) |
| `tools/news_scraper.py` | ✅ Complete | Multi-source news fetcher (yfinance + RSS) |

### Phase 3 — Agents (Intelligence Layer)
| File | Status | Description |
|------|--------|-------------|
| `agents/technical_agent.py` | ✅ Complete | Candlestick patterns, MAs, S&R, trend analysis |
| `agents/fundamental_agent.py` | ✅ Complete | Revenue, P/E, EPS, ROE, D/E, FCF grading |
| `agents/sentiment_agent.py` | ✅ Complete | News scraping + LLM sentiment analysis |

### Phase 4 — Orchestration
| File | Status | Description |
|------|--------|-------------|
| `orchestrator.py` | ✅ Complete | Parallel agent runner + two-stage Claude synthesis |
| `llm_client.py` | ✅ Complete | Anthropic / Groq / Ollama duck-typed clients + progress callback |
| `cost_tracker.py` | ✅ Complete | Per-call tokens & cost; auto prompt caching; `predict_analysis_cost` |
| `main.py` | ✅ Complete | CLI entry (Anthropic + Ollama; Groq is web-UI only) |
| `report_generator.py` | ✅ Complete | HTML report with Plotly chart, tables, news quotes |
| `analysis_store.py` | ✅ Complete | JSON snapshot used by scanner |

### Phase 5 — Web UI & Multi-User
| File | Status | Description |
|------|--------|-------------|
| `app.py` | ✅ Complete | Streamlit Analyzer page with BYO-API-key + Telegram setup |
| `pages/2_Portfolio_Tracker.py` | ✅ Complete | Trade ledger with realized + unrealized P&L |
| `pages/3_Portfolio_Baseline.py` | ✅ Complete | Baseline-then-forward trades + cash transfers |
| `auth.py` | ✅ Complete | Login gate: password (bcrypt) + Google OAuth via `st.login` |
| `auth_db.py` | ✅ Complete | `users`, `favorites`, per-user Telegram credentials |
| `db.py` | ✅ Complete | Single engine — Postgres via `DATABASE_URL`, SQLite locally |
| `portfolio_db.py` | ✅ Complete | `trades` table, user_id-scoped CRUD + P&L math |
| `portfolio_baseline_db.py` | ✅ Complete | Baseline meta + positions + trades + transfers |
| `migrate_to_db.py` | ✅ Complete | Idempotent migration from legacy per-file SQLite layout |

### Phase 6 — Scanner & Alerts
| File | Status | Description |
|------|--------|-------------|
| `scanner.py` | ✅ Complete | Buy/sell zone detection, staleness re-run, market-hours check |
| `alerts/telegram.py` | ✅ Complete | Telegram Bot API; per-user creds or env-var fallback |
| `alerts/whatsapp.py` | ✅ Complete | Twilio WhatsApp sender |

### Phase 7 — Roadmap
| Feature | Status | Description |
|---------|--------|-------------|
| Eval harness for prompt quality | 📋 Planned | Regression set of tickers + expected verdict ranges |
| Backtesting | 📋 Planned | Test agent signals against historical returns |
| Sector Comparison | 📋 Planned | Compare vs sector peers, sector-aware weighting |
| Anthropic Batches API | 📋 Planned | Overnight portfolio-wide scans at 50% cost |
| Tool use / agentic loops | 📋 Planned | Agents request additional data on demand |
| Pytest suite | 📋 Planned | Mocked agents + provider-parity smoke test |

---

## Claude API Techniques Used

| Technique | Where Used | Why |
|-----------|-----------|-----|
| `adaptive thinking` | Synthesizer in `orchestrator.py` | Deep reasoning for multi-factor analysis |
| `effort: high` | Synthesizer | Ensures thorough equity analysis |
| `streaming` | Synthesis (both CLI & Streamlit) | Real-time output, avoids timeout on long reports |
| `Pydantic structured output` | All 3 agents + extraction stage of synthesis | Guaranteed JSON schema from Claude |
| `system prompt persona` | All agents + synthesizer | Controls analytical style and tone |
| `parallel execution` | `orchestrator.py` | All 3 agents run simultaneously (3× faster); auto-sequential on Groq |
| `prompt caching` | `cost_tracker.TrackedMessages` injects `cache_control: ephemeral` | ~90% input-token discount on repeat calls within 5 min |
| `two-stage write-then-extract` | Opus writes narrative → Sonnet extracts JSON | ~5× cheaper than asking Opus for structured output directly |
| `thread-local progress callback` | `llm_client.set_progress_callback` | Surface SDK retries live to the Streamlit UI |

---

## Example Run

```bash
# CLI (Anthropic or Ollama)
python main.py --ticker DRS                     # Leonardo DRS Inc
python main.py --ticker AAPL                    # Apple
python main.py --ticker MSFT --period 6mo       # Microsoft, 6-month candles
python main.py --ticker NVDA --llm ollama --ollama-model llama3.2

# Web UI (Groq / Anthropic / Ollama all selectable; multi-page)
streamlit run app.py

# One-shot data migration from the legacy per-file SQLite layout
python migrate_to_db.py --dry-run
python migrate_to_db.py
```
