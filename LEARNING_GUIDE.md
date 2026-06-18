# Learning Guide — Beat the ASPP

> A guide written *for you*, not for recruiters. Goal: by the end of this document you should be able to explain every design decision, defend every line in a code review, and know what you'd do differently next time.

---

## How to use this document

Read it once top to bottom. Then, for each module:
1. Open the actual `.py` file next to this guide.
2. Read the section here, then the file.
3. Try to predict what the next function does *before* reading it.

If you can't predict it, you don't yet understand the abstraction the file is built on. Re-read.

---

## Part 1 — The 30-second mental model

Strip everything away and the project is **one function** that does this:

```
ticker  ──►  3 specialists analyze in parallel  ──►  1 strategist reconciles  ──►  report
```

That's it. Every other file exists to serve that one sentence.

When you find yourself lost in the code, come back here and ask: *which step am I in?* You're always in one of these four:

| Step | File responsible | What it owns |
|---|---|---|
| Input | `main.py` (CLI) or `app.py` + `pages/*.py` (Streamlit, gated by `auth.require_login`) | Parse arguments, load env, identify the user, hand off |
| Parallel agents | `orchestrator.py` (the dispatch) + `agents/*` (the workers) | Run 3 analyses simultaneously |
| Reconciliation | `orchestrator.py` (`_synthesize`) | Call Claude Opus to write the verdict |
| Output | `report_generator.py`, `analysis_store.py`, + per-user SQL writes (`auth_db`, `portfolio_db`, `portfolio_baseline_db`) | Write HTML + JSON to disk, scoped state to the DB |

Everything else is **infrastructure**: `llm_client.py` (which LLM to call), `cost_tracker.py` (how much did it cost), `tools/` (where data comes from), `models/` (what shape data has), `db.py` (which SQL engine), `scanner.py` + `alerts/` (closing the loop after analysis).

---

## Part 2 — The big picture, visualized

```
┌──────────────────────────────────────────────────────────────────┐
│  USER                                                            │
│  $ python main.py --ticker NVDA                                  │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  main.py                                                         │
│  • parse args   • load .env   • choose LLM provider              │
│  • call orchestrator.run_analysis()                              │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  orchestrator.py :: run_analysis()                               │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐    │
│   │  ThreadPoolExecutor(max_workers=3)                      │    │
│   │  ┌────────────┐  ┌─────────────┐  ┌─────────────────┐   │    │
│   │  │ Technical  │  │ Fundamental │  │   Sentiment     │   │    │
│   │  │   agent    │  │    agent    │  │     agent       │   │    │
│   │  └─────┬──────┘  └──────┬──────┘  └────────┬────────┘   │    │
│   │        │                │                   │            │    │
│   │   yfinance OHLCV    yfinance financials   yfinance + RSS │    │
│   │        │                │                   │            │    │
│   │   pandas indicators  Python formatting   BeautifulSoup   │    │
│   │        │                │                   │            │    │
│   │   Claude Sonnet     Claude Sonnet        Claude Sonnet   │    │
│   │   (score+narrative) (score+narrative)    (score+themes)  │    │
│   └────────┼────────────────┼───────────────────┼────────────┘    │
│            └────────────────┼───────────────────┘                 │
│                             │ (3 Pydantic reports, all done)      │
│                             ▼                                     │
│   ┌─────────────────────────────────────────────────────────┐     │
│   │  _synthesize()                                          │     │
│   │  Step 1: Claude Opus 4.6 (adaptive thinking, effort=hi) │     │
│   │          → writes the analyst NARRATIVE (streamed)      │     │
│   │  Step 2: Claude Sonnet 4.6 reads the narrative          │     │
│   │          → returns a structured FinalReport JSON        │     │
│   └────────────────────────┬────────────────────────────────┘     │
│                            │                                      │
│                            ▼                                      │
│   ┌─────────────────────────────────────────────────────────┐     │
│   │  analysis_store.save_analysis() → JSON snapshot         │     │
│   │  report_generator.generate_html_report() → HTML+Plotly  │     │
│   └─────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────┘
```

Every LLM call passes through `cost_tracker.TrackedMessages`, which:
- Injects prompt-cache directives into the system prompt automatically.
- Records tokens & dollars in a thread-safe singleton + `data/usage_log.jsonl`.

---

## Part 3 — Walk a single request through the code

Open four files side-by-side: `main.py`, `orchestrator.py`, `agents/technical_agent.py`, `llm_client.py`. Trace this:

### Step 1 — `main.py` parses args
- Reads `--ticker`, `--llm`, etc.
- Loads `.env` so `ANTHROPIC_API_KEY` is available.
- Calls `run_analysis(ticker, period, stream_output, llm_provider, llm_model)`.

**Why this design?** `main.py` knows *nothing* about agents, LLMs, or pandas. It's a thin shell. This is the **separation of concerns** principle: input handling lives at the edge.

### Step 2 — `orchestrator.run_analysis()` creates the LLM client
```python
client = create_llm_client(llm_provider, llm_model)
```
This returns one of three concrete clients (`AnthropicLLMClient`, `GroqLLMClient`, `OllamaLLMClient`) — but all three expose the **same** interface:
```python
client.messages.parse(...)   # structured output
client.messages.stream(...)  # streaming text
client.get_model_name()
```

This is **duck typing** in service of **the strategy pattern**. The orchestrator never has an `if provider == "groq":` branch; it just calls `.parse()` and trusts the right thing happens.

### Step 3 — Three agents launched in parallel
```python
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = {
        executor.submit(run_technical_agent, ticker, period, client): "technical",
        executor.submit(run_fundamental_agent, ticker, client): "fundamental",
        executor.submit(run_sentiment_agent, ticker, None, client): "sentiment",
    }
    for future in as_completed(futures):
        ...
```

**Visualizing concurrency:**
```
Time  ────────────────────────────►
         ┌───── Technical ────┐
T0   ───►├───── Fundamental ──┤── all three finish
         └───── Sentiment ────┘
                              │
                              ▼
                         Synthesizer starts
```
Without parallelism this is ~3× slower. The agents don't share state, so threading is safe.

> **Code review trap**: "Why threads, not asyncio?" Honest answer: the SDK calls are blocking I/O wrapped in synchronous methods. Threads work, are simple, and avoid colorful `async`/`await` plumbing. For 3 workers it's the right tool. (For 100 concurrent requests you'd switch to async.)

### Step 4 — Inside `run_technical_agent`
The agent itself is a 4-step recipe (read it in `agents/technical_agent.py`):

```
1. data = get_market_data(ticker)      ← yfinance + pandas; ALL math here
2. system_prompt + user_prompt         ← short, pre-digested
3. client.messages.parse(              ← Claude Sonnet
       output_format=_ClaudeOutput     ← Pydantic schema enforces shape
   )
4. report = TechnicalReport(...)       ← merge Claude's score with computed fields
   report = _fill_computed_fields(...)
```

The single most important design decision in this agent:

> **Claude does NOT compute indicators.** Python computes RSI, MAs, support, gaps. Claude only sees the *summary* and writes a score + a sentence. This is the **separation of facts from interpretation**.

Why does this matter?
- Cheaper (small prompt, small output).
- Deterministic (RSI = 47.3 every time; an LLM might drift).
- Auditable (you can re-derive every number without an API call).

### Step 5 — Synthesis (`_synthesize` in `orchestrator.py`)
Two sub-steps that look like one:

```
┌──────────────────────────────────────────────────────────────┐
│  Sub-step A: WRITE                                           │
│  Model: claude-opus-4-6                                      │
│  Why Opus? Reconciling 3 conflicting signals needs depth.    │
│  Mode: streaming, adaptive thinking, effort=high             │
│  Output: free-form analyst narrative (printed to terminal)   │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼  full_text (string)
┌──────────────────────────────────────────────────────────────┐
│  Sub-step B: EXTRACT                                         │
│  Model: claude-sonnet-4-6                                    │
│  Why Sonnet? It's a parsing job, not a reasoning job.        │
│  Mode: parse() with output_format=FinalReport                │
│  Output: validated Pydantic FinalReport                      │
└──────────────────────────────────────────────────────────────┘
```

**This is the most important pattern in the whole project.** Memorize it:

> **Two-stage write-then-extract** — let the expensive model produce prose, let the cheap model turn that prose into JSON. Same final quality, ~5× cheaper than asking Opus to produce structured output directly.

### Step 6 — Persist
- `analysis_store.save_analysis()` writes JSON snapshot — used later by the scanner.
- `report_generator.generate_html_report()` builds an HTML file with a Plotly candlestick chart.

---

## Part 4 — Module-by-module deep dive

### `models/report.py` — The contract

Pydantic models are **the data contract** between every layer. If you understand these models, you understand 50% of the project, because every function either *consumes* or *produces* one of them.

```
┌─────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐
│  TechnicalReport    │  │ FundamentalReport    │  │ SentimentReport  │
│  ─ score: float     │  │ ─ score: float       │  │ ─ score: float   │
│  ─ moving_averages  │  │ ─ revenue_growth_yoy │  │ ─ sentiment_score│
│  ─ rsi_value        │  │   (FundamentalMetric)│  │ ─ key_themes     │
│  ─ support_levels   │  │ ─ pe_ratio (...)     │  │ ─ catalysts      │
│  ─ ...              │  │ ─ ...                │  │ ─ ...            │
└──────────┬──────────┘  └──────────┬───────────┘  └────────┬─────────┘
           │                        │                       │
           └────────────┬───────────┴───────────────────────┘
                        ▼
                ┌────────────────┐
                │  FinalReport   │
                │  ─ verdict     │   "STRONG BUY" / "BUY" / ...
                │  ─ composite_score
                │  ─ confidence_pct
                │  ─ analyst_thesis
                │  ─ key_risks   │
                └────────────────┘
```

**Code review questions you'll be asked:**
- *"Why Pydantic and not dataclasses?"* → Pydantic validates at runtime, integrates with the Anthropic SDK's `output_format=` argument, and gives JSON schemas for free. Dataclasses do none of this.
- *"What happens if Claude returns a score of 11?"* → `Field(..., ge=0, le=10)` raises a validation error, which the agent's caller catches.

### `llm_client.py` — Provider abstraction (the strategy pattern)

This is the **most architecturally interesting** file. Read it twice.

```
        ┌────────────────────────────────────────┐
        │  agents call client.messages.parse(...) │
        └────────────────────┬───────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
   AnthropicLLMClient   GroqLLMClient   OllamaLLMClient
   uses TrackedMessages uses _GroqMessages uses _OllamaMessages
   (real Anthropic SDK)  (translates to    (translates to local
                          Groq's OpenAI-    HTTP /api/chat)
                          compatible API)
```

All three expose the same `.messages.parse()` and `.messages.stream()` methods. The orchestrator and agents never know which one they're using.

The trick that makes this work — for non-Anthropic providers, structured output is *faked*:
1. Build a flat example JSON from the Pydantic model (`_build_example_json`).
2. Inject it into the system prompt with strict instructions.
3. Use the provider's JSON mode (`response_format={"type": "json_object"}` for Groq, `format="json"` for Ollama).
4. Parse the returned string and validate with `model.model_validate()`.

**Code review traps:**
- *"What if Ollama returns malformed JSON?"* → `_clean_ollama_response` strips schema metadata that smaller models echo back, then validates. If still invalid, raises `ValueError` with the raw response truncated to 300 chars (so the user can see what went wrong).
- *"Why ignore `thinking=` and `output_config=` for Groq?"* → They're Anthropic-specific. The wrapper silently drops them via `**kwargs` so the same agent code works against any backend. Defensible — the alternative would be `if isinstance(client, AnthropicLLMClient)` branches in every agent.

### `cost_tracker.py` — Wrapper / decorator pattern

`TrackedMessages` is a **wrapper** around the Anthropic SDK's `messages` resource. Every call goes through it; it does two jobs:

1. **Inject prompt caching automatically:**
   ```python
   kwargs["system"] = [{
       "type": "text",
       "text": system,
       "cache_control": {"type": "ephemeral"},
   }]
   ```
   First call writes the cache (+25% cost). Subsequent calls within 5 min read the cache (−90% cost). Net win once you make 2+ calls with the same system prompt.

2. **Record token usage & cost:**
   - Thread-local context (`set_context(ticker, op)`) tags each call with which agent made it.
   - One JSON line per call appended to `data/usage_log.jsonl`.
   - In-memory accumulators give instant session totals to the Streamlit sidebar.

```
┌──────────────────────────────────────────────────────────┐
│  set_context("AAPL", "technical_agent")                  │
│         ↓ (thread-local)                                 │
│  client.messages.parse(...)                              │
│         ↓                                                │
│  TrackedMessages.parse                                   │
│         ↓ inject cache_control                           │
│  Anthropic SDK call                                      │
│         ↓ response with .usage                           │
│  tracker.record(ticker="AAPL", op="technical_agent",     │
│                 input=403, output=187, cache_read=312)   │
│         ↓                                                │
│  data/usage_log.jsonl  +=  one line                      │
└──────────────────────────────────────────────────────────┘
```

**Code review traps:**
- *"Why thread-local context instead of passing it as an argument?"* → Because three agents run in parallel and each wants to tag its own calls. A module-level global would mix them up. `threading.local()` gives each thread its own slot.
- *"What if recording fails?"* → Wrapped in `try/except: pass`. Tracking must never break the analysis. (You should be able to defend this — observability is secondary to functionality.)

### `tools/market_data.py` — Pure pandas, zero LLM

This file is a Python textbook on technical analysis:
- EMA (exponential weighting via `.ewm`)
- SMA (rolling mean via `.rolling`)
- RSI (gain/loss decomposition + smoothed average)
- ATR (true range, then rolling mean)
- Volume regime (green vs red day comparison)
- Pattern detection (Cup & Handle, VCP, Flat Base — windowed scans)
- Support/resistance (swing detection + volume-weighted clustering)

**It's worth reading slowly.** Even if you never touch trading again, the patterns generalize:
- **Windowed scans** for local extrema → useful for any time-series outlier detection.
- **Clustering with thresholds** → applies to any "deduplicate near-duplicate values" problem.
- **Decision rules built from booleans** → `_compute_buy_sell_zones` is a tiny rule engine.

**Code review trap**: *"This is rule-based — couldn't an ML model do better?"* Yes, possibly. But: rules are explainable, debuggable, free, and a 200-line `_compute_buy_sell_zones` is faster to maintain than a model with a training pipeline. **For an MVP, rules win.** Acknowledge the trade-off; don't over-defend.

### Agents — Same pattern, different specialty

All three agents share the same shape:

```
┌────────────────────────────────────────┐
│  set_context(ticker, agent_name)       │  → for cost tracking
│  data = tools.get_*_data(ticker)       │  → fetch & pre-process
│  prompt = build_prompt(data)           │  → compact, pre-digested
│  response = client.messages.parse(     │  → structured Claude call
│      output_format=PydanticModel       │
│  )                                     │
│  return _fill_computed_fields(...)     │  → merge Claude + Python
└────────────────────────────────────────┘
```

The **only thing each agent owns** is its prompt + its data tool. Everything else is shared. That's why adding a 4th agent (e.g. macroeconomic) would take ~50 lines.

**Subtle but important:** the technical agent uses `_ClaudeOutput`, a *minimal* Pydantic model with just 5 fields, instead of `output_format=TechnicalReport`. Why?
- The full `TechnicalReport` has 20+ fields, most computed in Python.
- Asking Claude to produce all 20 wastes tokens and risks hallucination.
- We give Claude only the fields it should actually decide (score, summary, outlook), then merge with Python-computed fields.

This is a small detail you should be ready to defend in a review — it's a deliberate optimization.

### `report_generator.py` & `analysis_store.py` — Outputs

- `analysis_store.py` saves a JSON snapshot of every analysis, keyed by ticker + date. The scanner reads these later to detect price-at-support events.
- `report_generator.py` builds an HTML report (Plotly chart, scored tables, news quotes). Self-contained — no server needed.

### `scanner.py` + `alerts/` — Closing the loop

This is what turns a one-off analysis into a **system**: re-checks favorited tickers during US market hours and pushes Telegram/WhatsApp alerts when price hits the buy zone defined by the latest analysis.

The buy-zone logic is dead simple — within 1.5% above primary support → alert. The interesting part is *staleness handling*: if the last analysis is older than 3 days, re-run before scanning. The Telegram sender in `alerts/telegram.py` looks at **per-user credentials first** (resolved through `auth_db.get_telegram_credentials`) and falls back to env vars only for the CLI path — so each logged-in user gets alerts only on their own bot/chat.

### `auth.py` + `auth_db.py` + `db.py` — The multi-user layer

Originally this app stored everything in flat JSON files (`data/favorites.json`) or per-user SQLite files. That worked for one local user. When the app moved to Streamlit Cloud, two problems showed up: (1) Streamlit Cloud's filesystem is ephemeral and shared, and (2) we wanted real accounts so users couldn't see each other's favorites and Telegram tokens. The whole `auth.py` / `auth_db.py` / `db.py` triangle exists to fix that.

```
            ┌──────────────────────────────────┐
            │  auth.require_login() — every    │
            │  Streamlit page imports this and │
            │  calls it before any data load.  │
            └──────────────────────────────────┘
                            │
        ┌───────────────────┴────────────────────┐
        ▼                                        ▼
┌──────────────────────────┐         ┌────────────────────────────┐
│ username + password      │         │ Google OAuth via           │
│ (bcrypt hash in `users`) │         │ st.login("google")         │
│                          │         │   ↳ upsert_google_user()   │
└──────────────────────────┘         └────────────────────────────┘
                            │
                            ▼
                ┌───────────────────────────┐
                │  user dict in session     │
                │  state. Every downstream  │
                │  query is keyed on        │
                │  user_id from now on.     │
                └───────────────────────────┘
```

**`db.py` is the only place SQLAlchemy lives.** Exactly one `Engine` per process, lazily created and thread-safe. The selection rule is:

```python
if DATABASE_URL is set (env or st.secrets):
    engine = create_engine(postgres_url, pool_pre_ping=True, …)
else:
    engine = create_engine("sqlite:///data/local.db", …)
```

The URL normaliser handles the common gotcha: providers like Neon, Supabase, and Render hand out `postgres://...` URLs, but SQLAlchemy 2.x requires `postgresql+psycopg://...`. We rewrite it transparently so deploys "just work" against any of them. There's one shared `metadata` object — every `*_db.py` module registers its tables on it, so `init_all()` does a single `create_all()` and you never have to remember to run a migration.

**`auth_db.py`** owns the `users` and `favorites` tables. Two kinds of users live in `users`:
- Password users — `password_hash` is set, `google_sub` is NULL.
- Google users — `google_sub` is set (the Google subject ID, stable across email changes), `password_hash` is NULL.

The `upsert_google_user()` helper deserves a quick read. It:
1. Looks up by `google_sub` (not email — emails can change).
2. If found, updates `last_login` and refreshes the email from Google.
3. If not, derives a unique username from the email local part and inserts.

**Per-user Telegram credentials live on the user row** (`telegram_bot_token`, `telegram_chat_id`). When the scanner sends an alert it passes those to `alerts.telegram.send_alert` — env-var creds become the CLI fallback only.

### `portfolio_db.py` + `pages/2_Portfolio_Tracker.py` — Trade ledger

Add a `trades` table; **every row has a `user_id`**. The CRUD functions (`add_trade`, `update_trade`, `delete_trade`, `get_all_trades`, …) **all take `user_id` and AND it into the WHERE clause**. This is the single most important security pattern in the multi-user layer:

> *Hard-scope every query by `user_id` at the SQL boundary. Never trust higher layers to filter for you.*

The pure-Python P&L math (`compute_position`, `enrich_with_price`) doesn't touch the DB at all — it's a function of a `list[dict]`. That makes it trivially testable and means the same code drives both the ticker drill-down and the portfolio summary.

### `portfolio_baseline_db.py` + `pages/3_Portfolio_Baseline.py` — Baseline-then-forward

A second portfolio model for the realistic case: *"I don't have my full trade history; I just know what I hold today."* You enter a baseline (one row per ticker with current value $ and total return %), plus free cash and an as-of date. From there you log forward BUY/SELL trades and cash transfers. The dashboard rolls baseline + trades + live prices into value, unrealized P&L, realized P&L, and total return %.

Four tables, all scoped by `user_id`:
- `baseline_meta` — free_cash and as_of_date (key/value).
- `baseline_positions` — one row per ticker at the baseline.
- `baseline_trades` — forward trades since the baseline.
- `baseline_transfers` — cash IN / OUT events.

### `migrate_to_db.py` — One-shot data migration

When the storage backend changed from "JSON file + per-user SQLite" to "one shared DB scoped by `user_id`", existing local users had legacy data. `migrate_to_db.py` sweeps every known legacy shape (`users.db`, `portfolio_u<N>.db`, `portfolio_<sha20>.db`, `portfolio_baseline*.db`, `favorites.json`) and re-inserts them into the unified schema.

Two design choices worth reading:
- **Idempotent.** Re-running won't duplicate rows. The script SELECTs by natural key (username, (user_id, ticker), trade signature) and skips the INSERT if a match exists. Steady-state writes inside the app (`auth_db.add_favorite`, etc.) use proper `ON CONFLICT DO NOTHING` instead.
- **`--dry-run`.** Walks the same code paths but never commits. Always run this first.

This is a generally good shape for any one-off data migration script you'll write again in your career.

---

## Part 5 — The design patterns at play (named, with locations)

Memorize this table. These are the words to use in a code review.

| Pattern | Where in this code | Why it's used here |
|---|---|---|
| **Strategy** | `llm_client.create_llm_client()` returns one of three concrete clients with the same interface | Lets the orchestrator switch LLM providers with no `if/else` branches |
| **Wrapper / Decorator** | `cost_tracker.TrackedMessages` wraps `anthropic.Anthropic().messages` | Adds tracking + caching transparently — agents don't know it's there |
| **Singleton** | `cost_tracker.tracker` (module-level) | One shared usage tracker across all imports |
| **Duck typing** | `_GroqStream` and `_OllamaStream` mimic Anthropic's stream event shape | The orchestrator's streaming loop works for all three providers without adapter layers |
| **Thread-local storage** | `cost_tracker._ctx = threading.local()` | Three parallel agents each tag their own LLM calls without clobbering each other |
| **Schema-first / Contract-first** | `models/report.py` Pydantic classes | Every layer relies on validated shapes — fewer runtime surprises |
| **Two-stage prompting** | `_synthesize` (Opus writes) → `_parse_streamed_report` (Sonnet extracts) | Cheap-model JSON parsing of expensive-model prose |
| **Separation of facts from interpretation** | All math in `tools/`, only scoring + narrative in agents | Deterministic numbers, LLM only judges |
| **Fallback / graceful degradation** | `_fallback_technical/_fundamental/_sentiment` in orchestrator | A failing data source returns a neutral report instead of crashing the pipeline |
| **Gate / decorator (semantic)** | `auth.require_login()` at the top of every Streamlit page | One call short-circuits unauthenticated requests before any data is loaded |
| **Multi-tenant row-level scoping** | `user_id` column on `favorites`, `trades`, `baseline_*` | Hard-scope every query by `user_id` at the SQL boundary; never trust higher layers to filter |
| **Adapter (URL normaliser)** | `db._normalise_url()` | Plain `postgres://...` strings from Neon / Supabase / Render are rewritten to the SQLAlchemy 2 driver form, so deploys "just work" |
| **Shared MetaData registry** | One `metadata` in `db.py`, every `*_db.py` registers its tables on it | A single `create_all()` migrates the whole app; no "did you remember to import?" foot-guns |
| **Per-request credential lookup** | `alerts.telegram.send_alert(token=…, chat_id=…)` falls back to env vars when not provided | Each user's alerts go to their own bot; CLI keeps working without code changes |
| **Idempotent migration** | `migrate_to_db.py` SELECTs by natural key before each INSERT, plus `--dry-run` | Safe to re-run; safe to *preview* before running |

---

## Part 6 — Code review prep: questions you WILL be asked

For each, the question, the right framing, and the trap to avoid.

### "Walk me through what happens when I run `python main.py --ticker NVDA`."
- **Right framing:** Use Part 3 above. Tell the four steps; mention parallelism; mention the two-stage synthesis.
- **Trap:** Don't dive into `_compute_rsi`. Stay at the architecture level unless asked.

### "Why three agents instead of one big prompt?"
- **Answer:** Specialization, parallelism, isolated failure. A monolithic prompt would be ~3× larger, sequential, and a single hallucination would corrupt the whole verdict.
- **Counter-question to ask back:** "Have you found that one large prompt works better than decomposition for similar problems?" → shows curiosity.

### "Why two LLM calls in synthesis instead of one?"
- **Answer:** Cost. Opus produces the prose (worth the price), Sonnet extracts JSON (5× cheaper, equally good at extraction). Asking Opus for structured output also constrains its reasoning.
- **Trap:** Don't claim it's "more accurate" — it's mostly a cost optimization that doesn't sacrifice quality.

### "Is threading really safe with the Anthropic SDK?"
- **Answer:** Yes — the official SDK is thread-safe; each `.messages.parse()` is an independent HTTP call. We share one client across threads, which is the recommended usage.

### "What happens if the technical agent fails?"
- **Answer:** The future raises, the orchestrator catches it, and `_fallback_technical()` substitutes a neutral report (score 5.0). Synthesis still runs. The user sees a complete (but degraded) report.
- **Honest weakness:** the synthesis prompt isn't *told* a fallback was used. A reviewer will catch this. Acknowledge it as a known improvement.

### "What if the same ticker is analyzed twice in 5 minutes?"
- **Answer:** Prompt caching kicks in for system prompts. User-level result caching is *not* implemented — each call is fresh. (Improvement: add a TTL cache keyed on `ticker+period`.)

### "How do you know your prompts are good?"
- **Answer:** Honest answer — there's no eval harness yet. Spot-checks across ~10 tickers with known characters (NVDA bullish, weak biotech, etc.) and reading the output. **This is the project's biggest weakness.** Acknowledge it.

### "Why Pydantic v2?"
- **Answer:** v2 is faster (Rust core), has stricter validation, and is what the Anthropic SDK's `output_format=` expects.

### "What's `cache_control: ephemeral`?"
- **Answer:** Anthropic's prompt-cache directive. The first call within 5 minutes writes a cache entry (charged at 1.25× input rate). Subsequent calls within 5 minutes read it (charged at 0.1× input rate). System prompts here are 200–800 tokens — caching pays off after just 2 calls.

### "How would you scale this to 1000 tickers a night?"
- **Answer:** Switch synthesis to Anthropic's Batches API (50% cost, 24h SLA). The persistence layer is already on Postgres in production (`db.py`). Add a job queue (Celery / RQ). Make the news scraper async with per-source rate limits.

### "Why one shared DB engine instead of per-request connections?"
- **Answer:** `db.py` builds a single SQLAlchemy `Engine` lazily on first use, behind a lock. SQLAlchemy's engine *is* a pool — the right abstraction. `pool_pre_ping=True` survives stale connections after Streamlit Cloud sleeps the app. We use `engine.begin()` per write — explicit transactions, no implicit autocommit surprises.

### "How do users stay isolated? What stops user A from seeing user B's trades?"
- **Answer:** Every per-user table (`favorites`, `trades`, `baseline_*`) has a `user_id` column. Every CRUD function takes `user_id` and includes it in the SQL WHERE clause. The Streamlit page resolves `user_id` exactly once, from `auth.require_login()`, before any data load. There's no "current user" global — it has to be passed in, which prevents accidental cross-user reads.

### "Why both bcrypt and Google OAuth? Isn't that twice the complexity?"
- **Answer:** Both write into the same `users` table — password users have a `password_hash`, Google users have a `google_sub`. The downstream app treats them identically (it only cares about `user_id`). `auth.py` is the only place that knows the difference, which keeps the cost of the second auth method to ~30 lines.

### "What if the same email logs in via Google after first creating a password account?"
- **Answer:** Currently they'd become two separate users — `google_sub` is the lookup key for OAuth, not email. Honest weakness. Fix: an explicit "link Google" flow in the user settings page that updates an existing user row instead of creating a new one.

### "Why is the `users` row read fresh from the DB on every request?"
- **Answer:** `get_current_user()` re-reads from the DB so an admin change (e.g. clearing Telegram credentials) is visible immediately, not just after re-login. The cost is one indexed PK lookup — negligible. The alternative (trust session state) creates stale-data bugs that are very hard to diagnose.

### "Why did you migrate off per-file SQLite?"
- **Answer:** Streamlit Cloud's filesystem is ephemeral *and* shared across users. A file at `data/portfolio_u3.db` would either disappear on restart or, worse, be visible to other users if they guessed the path. The single shared DB on `DATABASE_URL` (Postgres in prod, one SQLite file locally) fixes both — and the unified schema is easier to reason about.

---

## Part 7 — Be honest about the weaknesses

A reviewer respects honesty more than polish. Practice saying these out loud:

| Weakness | Why it exists | What you'd do |
|---|---|---|
| No automated tests | Built solo for learning; no CI yet | Add `pytest` covering each agent with mocked data + Claude responses |
| No evaluation harness for prompt quality | "Looks good to me" is not a methodology | Build a regression set: 20 tickers + expected verdict ranges, fail CI on drift |
| `_fallback_*` reports aren't flagged in synthesis | Oversight | Add a `data_quality` field to each report; synthesis prompt should weight low-quality reports less |
| News scraping is best-effort, no rate limiting | Started simple, never hardened | Per-source backoff; cache responses for ~6 hours |
| Cost tracker `try/except: pass` swallows everything | Defensive but blind | At minimum, log at WARN level so silent failures are visible |
| `output_format=FundamentalReport` directly on the full schema | Inconsistent with the technical agent's minimal-output approach | Refactor fundamental agent to mirror the technical agent's split |
| Hardcoded weights (45/35/20) in synthesis | Reflects one analyst philosophy | Could be configurable per-user, or sector-aware |
| Three providers but no unit tests proving parity | Risk of provider drift | Run a "provider parity" smoke test occasionally |
| No CSRF protection on login forms | Relying on Streamlit's session model + same-origin | Move sensitive actions behind explicit POST-token forms if exposed publicly |
| No "link Google to existing account" flow | Same email via password + Google creates two rows | Add a settings-page action that merges by verified email |
| No rate limiting on the login form | Vulnerable to credential stuffing | Add per-IP attempt counters in front of `login_with_password` |
| `pages/` Streamlit files repeat the auth + `init_db()` preamble | Cost of multi-page Streamlit's design | Could be extracted into a single `page_setup()` helper |
| Migration script paths are hard-coded to `./data/` | Fine for local, awkward for a manual cloud import | Accept `--data-dir` flag for arbitrary roots |

If the reviewer points one of these out, **don't get defensive**. Say "yes, you're right, and here's how I'd fix it" — that's the answer that wins the review.

---

## Part 8 — Mini glossary

- **EMA / SMA**: Exponential / Simple Moving Average. EMA weights recent prices more.
- **RSI**: Relative Strength Index (0–100). >70 = overbought, <30 = oversold.
- **ATR**: Average True Range. A volatility measure in dollars.
- **Golden / Death Cross**: EMA50 crossing above (golden) or below (death) SMA200.
- **Support / Resistance**: Price levels where buying / selling has historically clustered.
- **Adaptive thinking**: Claude API parameter that lets the model decide how long to "think" before answering.
- **Prompt caching**: Anthropic feature that caches a marked portion of the prompt for 5 minutes; subsequent calls pay ~10% of original cost for that portion.
- **Structured output (`output_format=`)**: Anthropic SDK feature that forces Claude's response to validate against a Pydantic model.
- **Streaming**: Receiving the response token-by-token instead of all at once.
- **Two-stage write-then-extract**: Pattern in this code where one model writes prose and a cheaper model extracts JSON from it.

---

## Part 9 — Suggested learning exercises

Do these in order. They get progressively harder.

1. **Add a print statement** in `cost_tracker.TrackedMessages.parse` that logs `[$cost] [model] [op]` after every call. Run an analysis. Watch the costs accumulate.
2. **Change the synthesis weights** from 45/35/20 to 33/33/33. Re-run on the same ticker. Compare verdicts. Read why the verdict changed.
3. **Add a 4th agent** — `agents/macro_agent.py` — that fetches the VIX from yfinance and rates "market regime" (calm / nervous / panic). Wire it into `orchestrator.py`. This forces you to touch every layer.
4. **Write one pytest test** for `_compute_rsi` using a hand-crafted DataFrame where you know the answer. Now you have a test suite.
5. **Add a `--cache` flag** that skips the analysis if a JSON snapshot from today already exists. Decide where the cache check belongs (orchestrator? main?). Defend your choice.
6. **Profile a full run** with `cProfile`. Find the slowest non-LLM function. Make it faster.

If you complete all six, you're no longer "the person who built this" — you're the person who *understands* it.

---

## Closing note

The point of this document is not to memorize answers. It's to make sure that when somebody asks you "why did you do X?", you can answer: *"Because of Y, but in retrospect Z would have been better."* That answer impresses every interviewer. Polished certainty does not.

Now go re-read `orchestrator.py` with this guide open. You'll see things you missed the first time.
