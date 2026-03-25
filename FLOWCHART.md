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
│                         USER  (main.py)                             │
│                     Inputs: Ticker Symbol                           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR (orchestrator.py)                 │
│  - Spawns 3 agents in parallel (ThreadPoolExecutor)                 │
│  - Collects structured reports from each                            │
│  - Calls Claude (Opus 4.6 + Adaptive Thinking) for synthesis        │
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
│  Claude LLM   │   │  Claude LLM    │   │    Claude LLM         │
│  Technical    │   │  Fundamental   │   │    Sentiment          │
│  Evaluator    │   │  Evaluator     │   │    Evaluator          │
│  Score: /10   │   │  Score: /10    │   │    Score: /10         │
└───────┬───────┘   └───────┬────────┘   └───────────┬───────────┘
        └─────────────────── ┤ ──────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    SYNTHESIZER  (orchestrator.py)                   │
│         Claude Opus 4.6 + Adaptive Thinking + Effort: High          │
│         Persona: Senior Equity Research Analyst                     │
│         Weights: 35% Technical · 45% Fundamental · 20% Sentiment   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         FINAL REPORT                                │
│  • Composite Score (weighted)                                       │
│  • Verdict: STRONG BUY / BUY / HOLD / SELL / STRONG SELL           │
│  • Confidence %                                                     │
│  • Price Target (if determinable)                                   │
│  • Key Risks & Opportunities                                        │
│  • Analyst Thesis (narrative)                                       │
│  • Recommended Time Horizon                                         │
└─────────────────────────────────────────────────────────────────────┘
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
| `tools/market_data.py` | ✅ Complete | yfinance OHLCV candle fetcher |
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
| `orchestrator.py` | ✅ Complete | Parallel agent runner + Claude synthesis |
| `main.py` | ✅ Complete | CLI entry point with streaming output |
| `report_generator.py` | ✅ Complete | HTML report with Plotly chart, tables, news quotes |

### Phase 5 — Enhancements (Roadmap)
| Feature | Status | Description |
|---------|--------|-------------|
| Streamlit Web UI | 📋 Planned | Visual dashboard for reports |
| Portfolio Mode | 📋 Planned | Analyze multiple tickers at once |
| Chart Generation | 📋 Planned | matplotlib candlestick + indicator charts |
| Historical Storage | 📋 Planned | SQLite/JSON to track reports over time |
| Alert System | 📋 Planned | Email/Telegram alerts on rating changes |
| Backtesting | 📋 Planned | Test agent signals against historical returns |
| Sector Comparison | 📋 Planned | Compare vs sector peers |

---

## Claude API Techniques Used

| Technique | Where Used | Why |
|-----------|-----------|-----|
| `adaptive thinking` | Synthesizer in orchestrator.py | Deep reasoning for multi-factor analysis |
| `effort: high` | Synthesizer | Ensures thorough equity analysis |
| `streaming` | main.py | Real-time output, avoids timeout on long reports |
| `Pydantic structured output` | All 3 agents | Guaranteed JSON schema from Claude |
| `system prompt persona` | All agents + synthesizer | Controls analytical style and tone |
| `parallel execution` | orchestrator.py | All 3 agents run simultaneously (3x faster) |
| `prompt caching` | 📋 Planned | Cache large system prompts to reduce cost |

---

## Example Run
```bash
python main.py --ticker DRS                     # Leonardo DRS Inc
python main.py --ticker AAPL                    # Apple
python main.py --ticker MSFT --period 6mo       # Microsoft, 6-month candles
```
