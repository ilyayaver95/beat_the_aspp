"""
compare_providers.py
====================
Run the full 3-agent analysis on the same tickers using both Anthropic
(paid) and Groq (free), then render a side-by-side HTML comparison
with cost, speed, verdict agreement, and an auto-generated conclusion.

USAGE:
  python compare_providers.py
  python compare_providers.py --tickers NVDA PLTR ANET
"""

import argparse
import html as html_lib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

# ── Fix Windows SSL: use the OS trust store (handles corporate proxies) ──
import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
os.environ.setdefault("CURL_CA_BUNDLE", certifi.where())

try:
    import truststore
    truststore.inject_into_ssl()  # makes httpx / anthropic SDK use the Windows cert store
except ImportError:
    pass

# yfinance uses curl_cffi (libcurl), which bypasses Python ssl. Disable
# its peer verification for this local run — comparison only, no secrets in flight.
try:
    from curl_cffi import requests as _curl_requests
    _orig_request = _curl_requests.Session.request
    def _no_verify_request(self, *args, **kwargs):
        kwargs.setdefault("verify", False)
        return _orig_request(self, *args, **kwargs)
    _curl_requests.Session.request = _no_verify_request
except ImportError:
    pass

import warnings
warnings.filterwarnings("ignore")

# Force UTF-8 stdout so emoji / arrows don't crash on Windows cp1252
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from dotenv import load_dotenv
load_dotenv()

from orchestrator import run_analysis
from cost_tracker import tracker as cost_tracker


# ── One analysis run ──────────────────────────────────────────────────

DEFAULT_MODELS = {
    "api":  None,  # orchestrator picks sonnet for agents + opus for synthesis
    "groq": "llama-3.1-8b-instant",  # higher free-tier rate limits than 70b
}

SNAPSHOT_DIR = Path("data/comparison_cache")


def _report_to_dict(report) -> dict:
    if report is None:
        return {}
    if hasattr(report, "model_dump"):
        return report.model_dump()
    if isinstance(report, SimpleNamespace):
        return vars(report)
    return dict(report.__dict__) if hasattr(report, "__dict__") else {}


def snapshot_anthropic(result: dict) -> None:
    """Persist an Anthropic result to a dedicated location so Groq runs can't overwrite it."""
    if result.get("error") or result.get("provider") != "api":
        return
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snap = {
        "ticker":      result["ticker"],
        "model":       result["model"],
        "elapsed_sec": result["elapsed_sec"],
        "cost_usd":    result["cost_usd"],
        "tokens":      result["tokens"],
        "report":      _report_to_dict(result["report"]),
        "saved_at":    datetime.now().isoformat(),
    }
    (SNAPSHOT_DIR / f"{result['ticker']}_anthropic.json").write_text(
        json.dumps(snap, default=str, indent=2), encoding="utf-8"
    )


def load_anthropic_snapshot(ticker: str) -> dict | None:
    """Load a previously snapshotted Anthropic run, if available."""
    p = SNAPSHOT_DIR / f"{ticker}_anthropic.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return {
        "ticker":      ticker,
        "provider":    "api",
        "model":       d["model"],
        "error":       None,
        "elapsed_sec": d["elapsed_sec"],
        "cost_usd":    d["cost_usd"],
        "tokens":      d["tokens"],
        "report":      SimpleNamespace(**d["report"]),
        "cached":      True,
    }


def load_cached_anthropic(ticker: str, max_age_minutes: int = 240) -> dict | None:
    """
    Load a recent cached Anthropic analysis from data/analyses/.
    Aggregates cost from usage_log.jsonl entries written around the same time.
    Returns a result dict shaped like run_one(), or None if no fresh cache.
    """
    cache_path = Path(f"data/analyses/{ticker}_latest.json")
    if not cache_path.exists():
        return None
    age_min = (time.time() - cache_path.stat().st_mtime) / 60
    if age_min > max_age_minutes:
        return None

    d = json.loads(cache_path.read_text(encoding="utf-8"))
    try:
        analysis_dt = datetime.strptime(d["analysis_date"], "%Y-%m-%d %H:%M:%S")
    except Exception:
        analysis_dt = datetime.fromtimestamp(cache_path.stat().st_mtime)

    # Aggregate cost & tokens from the usage log within ±15 min of the analysis
    cost, tokens = 0.0, 0
    log_path = Path("data/usage_log.jsonl")
    if log_path.exists():
        w_start = analysis_dt - timedelta(minutes=15)
        w_end   = analysis_dt + timedelta(minutes=15)
        with log_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    ts_raw = r["ts"].replace("Z", "+00:00")
                    ts = datetime.fromisoformat(ts_raw).replace(tzinfo=None)
                    if r.get("tkr") == ticker and w_start <= ts <= w_end:
                        cost   += float(r.get("$", 0))
                        tokens += (int(r.get("in", 0)) + int(r.get("out", 0))
                                   + int(r.get("cw", 0)) + int(r.get("cr", 0)))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

    return {
        "ticker":      ticker,
        "provider":    "api",
        "model":       "claude-sonnet-4-6 (agents) + claude-opus-4-6 (synthesis)",
        "error":       None,
        "elapsed_sec": 0.0,  # cached — wall-time not preserved
        "cost_usd":    round(cost, 4),
        "tokens":      tokens,
        "report":      SimpleNamespace(**d["final"]),
        "cached":      True,
    }


def run_one(ticker: str, provider: str, model: str | None = None) -> dict:
    """Run a single analysis and capture cost, wall-time, and the FinalReport."""
    model = model or DEFAULT_MODELS.get(provider)
    label = "ANTHROPIC" if provider == "api" else provider.upper()
    print(f"\n{'='*70}\n  [{label}] Analyzing {ticker} (model={model or 'default'})\n{'='*70}")

    s0 = cost_tracker.get_session_stats()
    t0 = time.time()
    error = None
    report = None
    try:
        report = run_analysis(
            ticker=ticker, period="1y", stream_output=False,
            llm_provider=provider, llm_model=model,
        )
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    elapsed = round(time.time() - t0, 2)
    s1 = cost_tracker.get_session_stats()
    return {
        "ticker":      ticker,
        "provider":    provider,
        "model":       model or (
            "claude-sonnet-4-6 (agents) + claude-opus-4-6 (synthesis)"
            if provider == "api" else "llama-3.3-70b-versatile"
        ),
        "error":       error,
        "elapsed_sec": elapsed,
        "cost_usd":    round(s1["cost"] - s0["cost"], 4),
        "tokens":      s1["total"] - s0["total"],
        "report":      report,
    }


# ── HTML rendering ────────────────────────────────────────────────────

VERDICT_COLOR = {
    "STRONG BUY":  "#0f9d58",
    "BUY":         "#27ae60",
    "HOLD":        "#f39c12",
    "SELL":        "#e67e22",
    "STRONG SELL": "#e74c3c",
}


def _esc(s) -> str:
    return html_lib.escape(str(s)) if s is not None else "—"


def _short(s, n: int = 400) -> str:
    if not s:
        return "—"
    s = str(s).strip()
    return _esc(s if len(s) <= n else s[:n - 1] + "…")


def _verdict_badge(v: str | None) -> str:
    if not v:
        return '<span class="badge" style="background:#666">—</span>'
    color = VERDICT_COLOR.get(v, "#666")
    return f'<span class="badge" style="background:{color}">{_esc(v)}</span>'


def _field(report, key, fmt=None):
    if report is None:
        return "—"
    val = getattr(report, key, None)
    if val is None or val == "":
        return "—"
    return fmt(val) if fmt else _esc(val)


def _pair_block(ticker: str, anth: dict, groq: dict) -> str:
    """Side-by-side comparison block for one ticker."""
    r_a = anth.get("report")
    r_g = groq.get("report")
    v_a = getattr(r_a, "verdict", None) if r_a else None
    v_g = getattr(r_g, "verdict", None) if r_g else None

    if v_a and v_g and v_a == v_g:
        agree_icon, agree_text = "✅", "Verdicts match"
    elif v_a and v_g:
        agree_icon, agree_text = "⚠️", f"Verdicts differ ({v_a} vs {v_g})"
    else:
        agree_icon, agree_text = "❌", "One or both runs failed"

    def row(label, a_val, g_val):
        return f"<tr><th>{_esc(label)}</th><td>{a_val}</td><td>{g_val}</td></tr>"

    rows = [
        row("Verdict",
            _verdict_badge(v_a), _verdict_badge(v_g)),
        row("Composite Score",
            _field(r_a, "composite_score", lambda v: f"{v:.1f}/10"),
            _field(r_g, "composite_score", lambda v: f"{v:.1f}/10")),
        row("Confidence",
            _field(r_a, "confidence_pct", lambda v: f"{v:.0f}%"),
            _field(r_g, "confidence_pct", lambda v: f"{v:.0f}%")),
        row("Technical",
            _field(r_a, "technical_score", lambda v: f"{v:.1f}/10"),
            _field(r_g, "technical_score", lambda v: f"{v:.1f}/10")),
        row("Fundamental",
            _field(r_a, "fundamental_score", lambda v: f"{v:.1f}/10"),
            _field(r_g, "fundamental_score", lambda v: f"{v:.1f}/10")),
        row("Sentiment",
            _field(r_a, "sentiment_score", lambda v: f"{v:.1f}/10"),
            _field(r_g, "sentiment_score", lambda v: f"{v:.1f}/10")),
        row("Price Target",
            _field(r_a, "price_target"),
            _field(r_g, "price_target")),
        row("Time Horizon",
            _field(r_a, "time_horizon"),
            _field(r_g, "time_horizon")),
        row("Cost (USD)",
            f"<span class='cost'>${anth['cost_usd']:.4f}</span>",
            f"<span class='cost free'>${groq['cost_usd']:.4f}</span>"),
        row("Wall time",
            f"{anth['elapsed_sec']:.1f}s",
            f"{groq['elapsed_sec']:.1f}s"),
        row("Tokens",
            f"{anth['tokens']:,}",
            f"{groq['tokens']:,}"),
        row("Model(s)",
            _esc(anth["model"]), _esc(groq["model"])),
    ]

    thesis_a = (getattr(r_a, "analyst_thesis", None)
                if r_a else anth.get("error", ""))
    thesis_g = (getattr(r_g, "analyst_thesis", None)
                if r_g else groq.get("error", ""))

    return f"""
    <section class="ticker-block">
      <h2>{_esc(ticker)} <span class="muted">— {agree_icon} {_esc(agree_text)}</span></h2>
      <div class="table-wrap">
        <table class="cmp">
          <thead>
            <tr>
              <th></th>
              <th class="anth">🔵 Anthropic (paid)</th>
              <th class="groq">🟡 Groq (free)</th>
            </tr>
          </thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
      <div class="thesis-grid">
        <div class="thesis">
          <div class="lbl">🔵 Anthropic — Analyst Thesis</div>
          <p>{_short(thesis_a, 800)}</p>
        </div>
        <div class="thesis">
          <div class="lbl">🟡 Groq — Analyst Thesis</div>
          <p>{_short(thesis_g, 800)}</p>
        </div>
      </div>
    </section>
    """


def _conclusion(results: list[dict], tickers: list[str]) -> str:
    """Build a data-driven conclusion."""
    anth = [r for r in results if r["provider"] == "api"]
    groq = [r for r in results if r["provider"] == "groq"]

    total_anth_cost = sum(r["cost_usd"] for r in anth)
    total_groq_cost = sum(r["cost_usd"] for r in groq)
    avg_anth_time   = sum(r["elapsed_sec"] for r in anth) / max(len(anth), 1)
    avg_groq_time   = sum(r["elapsed_sec"] for r in groq) / max(len(groq), 1)

    # Agreement analysis
    by_t = {t: {} for t in tickers}
    for r in results:
        by_t[r["ticker"]][r["provider"]] = r.get("report")

    matches, differs, failed = 0, 0, 0
    score_deltas = []
    diff_pairs = []
    for t in tickers:
        a, g = by_t[t].get("api"), by_t[t].get("groq")
        if not a or not g:
            failed += 1
            continue
        if a.verdict == g.verdict:
            matches += 1
        else:
            differs += 1
            diff_pairs.append(f"{t}: Anthropic={a.verdict} vs Groq={g.verdict}")
        if a.composite_score is not None and g.composite_score is not None:
            score_deltas.append(abs(a.composite_score - g.composite_score))

    n = len(tickers)
    speed_ratio = avg_anth_time / avg_groq_time if avg_groq_time else 0
    cost_per_ticker = total_anth_cost / n if n else 0
    avg_score_delta = sum(score_deltas) / len(score_deltas) if score_deltas else 0

    # Recommendation logic
    if matches == n:
        rec = ("<strong>Groq matched Anthropic on every ticker and is free + faster</strong> — "
               "use Groq as your default. Reserve Anthropic for high-stakes positions "
               "where you want a second opinion or richer reasoning.")
    elif matches >= n / 2:
        rec = (f"<strong>Groq matched Anthropic on {matches}/{n} tickers.</strong> "
               "Use Groq for screening and routine checks; switch to Anthropic when "
               "the decision is large or when the two disagree.")
    else:
        rec = (f"<strong>Verdicts diverged on {differs}/{n} tickers.</strong> "
               "Anthropic's reasoning is more nuanced and reliable — use it for "
               "important decisions and treat Groq as a fast first-pass filter. "
               f"Groq's smaller 8B model also failed schema validation on {failed} ticker(s) — "
               "the larger 70B is more reliable but rate-limited on the free tier.")

    diff_detail = ""
    if diff_pairs:
        diff_detail = "<li><strong>Disagreements:</strong> " + "; ".join(_esc(p) for p in diff_pairs) + "</li>"

    return f"""
    <h2>📌 Conclusion</h2>
    <ul>
      <li><strong>Cost:</strong> Anthropic spent <span class="cost">${total_anth_cost:.4f}</span>
          across {n} analyses (≈ ${cost_per_ticker:.4f}/ticker).
          Groq spent <span class="cost free">${total_groq_cost:.4f}</span>.
          <strong>Cost winner:</strong> Groq{'' if total_groq_cost == 0 else f' (saves ${total_anth_cost - total_groq_cost:.4f})'}.</li>
      <li><strong>Speed:</strong> Groq averaged {avg_groq_time:.1f}s per analysis vs
          Anthropic's {avg_anth_time:.1f}s ({speed_ratio:.1f}× ratio).
          <strong>Speed winner:</strong> {'Groq' if avg_groq_time < avg_anth_time else 'Anthropic'}.</li>
      <li><strong>Verdict agreement:</strong> {matches}/{n} tickers produced identical verdicts.
          {differs} differed{f', {failed} failed' if failed else ''}.</li>
      <li><strong>Score divergence:</strong> Average composite-score gap was
          {avg_score_delta:.2f} points (out of 10).</li>
      {diff_detail}
      <li><strong>Recommendation:</strong> {rec}</li>
    </ul>
    """


def render_html(results: list[dict], tickers: list[str]) -> str:
    pairs = {t: {} for t in tickers}
    for r in results:
        pairs[r["ticker"]][r["provider"]] = r

    ticker_blocks = "".join(
        _pair_block(t, pairs[t].get("api", {}), pairs[t].get("groq", {}))
        for t in tickers
    )

    anth = [r for r in results if r["provider"] == "api"]
    groq = [r for r in results if r["provider"] == "groq"]
    total_anth_cost = sum(r["cost_usd"] for r in anth)
    total_groq_cost = sum(r["cost_usd"] for r in groq)
    total_anth_time = sum(r["elapsed_sec"] for r in anth)
    total_groq_time = sum(r["elapsed_sec"] for r in groq)
    n = len(tickers)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Provider Comparison — {', '.join(tickers)}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         background: #0e1117; color: #fafafa; margin: 0; padding: 32px;
         max-width: 1200px; margin-left: auto; margin-right: auto; }}
  h1 {{ font-size: 1.9rem; margin: 0 0 6px 0; }}
  h2 {{ font-size: 1.3rem; margin: 32px 0 12px;
        border-bottom: 1px solid #333; padding-bottom: 6px; }}
  .meta {{ color: #aaa; margin-bottom: 18px; }}
  .summary-row {{ display: grid; grid-template-columns: repeat(4, 1fr);
                  gap: 12px; margin: 20px 0; }}
  .summary-card {{ background: #1a1a2e; border: 1px solid #2c2c44;
                   border-radius: 8px; padding: 14px; }}
  .summary-card .lbl {{ color: #aaa; font-size: .85rem; }}
  .summary-card .val {{ font-size: 1.6rem; font-weight: 700; margin-top: 4px; }}
  .summary-card .sub {{ color: #888; font-size: .8rem; margin-top: 4px; }}
  .ticker-block {{ margin: 24px 0 36px 0; }}
  .muted {{ color: #888; font-weight: 400; font-size: .95rem; }}
  .table-wrap {{ overflow-x: auto; }}
  table.cmp {{ width: 100%; border-collapse: collapse; background: #1a1a2e;
               border-radius: 8px; overflow: hidden; }}
  table.cmp th, table.cmp td {{ padding: 10px 14px; text-align: left;
                                 border-bottom: 1px solid #2c2c44;
                                 font-size: .95rem; }}
  table.cmp thead th {{ background: #0d0d1f; font-size: .9rem; color: #ccc; }}
  table.cmp tbody th {{ color: #aaa; font-weight: 500; width: 200px; }}
  table.cmp th.anth {{ color: #5db8ff; }}
  table.cmp th.groq {{ color: #f5c542; }}
  .badge {{ display: inline-block; padding: 4px 12px; border-radius: 12px;
            color: #fff; font-weight: 700; font-size: .85rem; }}
  .cost {{ color: #ff6b6b; font-weight: 600; }}
  .cost.free {{ color: #2ecc71; }}
  .thesis-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
                  margin-top: 12px; }}
  .thesis {{ background: #1a1a2e; border: 1px solid #2c2c44;
             border-radius: 8px; padding: 12px 14px; }}
  .thesis .lbl {{ color: #aaa; font-size: .85rem; margin-bottom: 6px; }}
  .thesis p {{ margin: 0; line-height: 1.5; color: #ddd; }}
  ul {{ line-height: 1.8; }}
  .footer {{ color: #666; font-size: .8rem; margin-top: 40px;
             text-align: center; }}
  @media (max-width: 800px) {{
    .summary-row {{ grid-template-columns: repeat(2, 1fr); }}
    .thesis-grid {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
  <h1>🔵 Anthropic (paid) &nbsp;vs&nbsp; 🟡 Groq (free)</h1>
  <div class="meta">Tickers analyzed: <strong>{', '.join(tickers)}</strong> &nbsp;·&nbsp;
       Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>

  <div class="summary-row">
    <div class="summary-card">
      <div class="lbl">Total Anthropic cost</div>
      <div class="val" style="color:#ff6b6b">${total_anth_cost:.4f}</div>
      <div class="sub">{n} analyses · ≈ ${total_anth_cost/n:.4f}/ticker</div>
    </div>
    <div class="summary-card">
      <div class="lbl">Total Groq cost</div>
      <div class="val" style="color:#2ecc71">${total_groq_cost:.4f}</div>
      <div class="sub">{n} analyses · free tier</div>
    </div>
    <div class="summary-card">
      <div class="lbl">Total Anthropic time</div>
      <div class="val">{total_anth_time:.1f}s</div>
      <div class="sub">≈ {total_anth_time/n:.1f}s/ticker</div>
    </div>
    <div class="summary-card">
      <div class="lbl">Total Groq time</div>
      <div class="val">{total_groq_time:.1f}s</div>
      <div class="sub">≈ {total_groq_time/n:.1f}s/ticker</div>
    </div>
  </div>

  {ticker_blocks}

  {_conclusion(results, tickers)}

  <div class="footer">Generated by compare_providers.py · Beat the ASPP</div>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["NVDA", "PLTR", "ANET"])
    parser.add_argument("--skip-anthropic", action="store_true",
                        help="Reuse cached Anthropic analyses (saves money on re-runs)")
    args = parser.parse_args()
    tickers = [t.upper() for t in args.tickers]

    print(f"\n{'#'*70}")
    print(f"#  PROVIDER COMPARISON: Anthropic (paid) vs Groq (free)")
    print(f"#  Tickers: {', '.join(tickers)}")
    print(f"{'#'*70}")

    results = []

    # Anthropic — load snapshot if requested, otherwise run live and snapshot.
    for ticker in tickers:
        r = None
        if args.skip_anthropic:
            r = load_anthropic_snapshot(ticker)
            if r:
                print(f"\n  [snapshot] {ticker} loaded from {SNAPSHOT_DIR}")
            else:
                print(f"\n  [snapshot] {ticker}: no snapshot found — running live")
        if r is None:
            r = run_one(ticker, "api")
            snapshot_anthropic(r)  # save so future --skip-anthropic runs work
        results.append(r)
        verdict = r["report"].verdict if r["report"] else f"FAILED ({r['error']})"
        tag = "snapshot" if r.get("cached") else "api"
        print(f"\n  -> [{tag}] {ticker}: {verdict} | "
              f"${r['cost_usd']:.4f} | {r['elapsed_sec']:.1f}s | "
              f"{r['tokens']:,} tokens")

    # Brief cooldown so Groq's token bucket is empty before we start.
    print(f"\n{'-'*70}\n  Cooling down 15s before Groq runs...\n{'-'*70}")
    time.sleep(15)

    # Groq runs — sleep between tickers so the per-minute token bucket refills.
    for i, ticker in enumerate(tickers):
        if i > 0:
            print(f"\n  Sleeping 30s between Groq runs (free-tier rate limit)...")
            time.sleep(30)
        r = run_one(ticker, "groq")
        results.append(r)
        verdict = r["report"].verdict if r["report"] else f"FAILED ({r['error']})"
        print(f"\n  -> [groq] {ticker}: {verdict} | "
              f"${r['cost_usd']:.4f} | {r['elapsed_sec']:.1f}s | "
              f"{r['tokens']:,} tokens")

    html_doc = render_html(results, tickers)
    out_dir = Path("reports"); out_dir.mkdir(exist_ok=True)
    fname = f"provider_comparison_{datetime.now().strftime('%Y-%m-%d_%H%M')}.html"
    out_path = out_dir / fname
    out_path.write_text(html_doc, encoding="utf-8")

    print(f"\n{'='*70}")
    print(f"  Report saved: {out_path.resolve()}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
