"""
compare.py
==========
Side-by-side comparison of four stock evaluation tools on AAPL and NVDA.

TOOLS COMPARED:
  1. Beat-the-ASPP  — your 3-agent system (technical + fundamental + sentiment + Claude synthesis)
  2. Claude Baseline — raw Claude API call, no live data, no agents (just LLM knowledge)
  3. Danelfin        — AI Score 1-10 per dimension (enter from danelfin.com)
  4. Seeking Alpha   — Quant rating + Factor Grades (enter from seekingalpha.com)

USAGE:
  python compare.py
  python compare.py --tickers AAPL NVDA          # default
  python compare.py --tickers MSFT TSLA          # custom pair
  python compare.py --skip-external              # skip manual entry, mark as N/A
  python compare.py --no-browser                 # don't auto-open the report

OUTPUT:
  reports/comparison_AAPL_NVDA_{date}.html
"""

import argparse
import json
import os
import sys
import webbrowser
from datetime import datetime

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich import box

load_dotenv()
console = Console()


# ── Dataclass-like dicts for external tool results ─────────────────────────

def _empty_danelfin():
    return {
        "ai_score": None,       # 1-10
        "technical": None,      # 1-10
        "fundamental": None,    # 1-10
        "sentiment": None,      # 1-10
        "signal": None,         # Strong Buy / Buy / Neutral / Sell / Strong Sell
    }


def _empty_seeking_alpha():
    return {
        "quant_rating": None,   # Strong Buy / Buy / Hold / Sell / Strong Sell
        "valuation": None,      # A+ / A / B / C / D / F
        "growth": None,
        "profitability": None,
        "momentum": None,
        "eps_revisions": None,
        "price_target": None,   # optional, from Wall St consensus
    }


# ── Claude Baseline (single call, no live data) ────────────────────────────

def run_claude_baseline(ticker: str, client) -> dict:
    """
    Ask Claude to evaluate a stock using ONLY its training knowledge.
    No yfinance, no news scraping, no agents — just the raw LLM.

    This is the 'Anthropic Tool for Stock Market Evaluation' baseline:
    what you get when you skip the multi-agent framework and just ask Claude.
    """
    console.print(f"  [dim]Claude baseline: evaluating {ticker} from internal knowledge...[/dim]")

    try:
        response = client._client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            system=(
                "You are a financial analyst. You will evaluate a stock based solely on your "
                "training knowledge — no live data or tools are available. Be honest about "
                "knowledge cutoff limitations. Return ONLY a valid JSON object, no markdown."
            ),
            messages=[{
                "role": "user",
                "content": (
                    f"Evaluate {ticker} stock as of your knowledge cutoff. "
                    f"Return JSON with exactly these fields:\n"
                    f'{{"verdict": "STRONG BUY|BUY|HOLD|SELL|STRONG SELL", '
                    f'"score_overall": <float 0-10>, '
                    f'"technical_score": <float 0-10>, '
                    f'"fundamental_score": <float 0-10>, '
                    f'"sentiment_score": <float 0-10>, '
                    f'"confidence_pct": <float 0-100>, '
                    f'"price_target": "<string or null>", '
                    f'"thesis": "<2 sentences max>", '
                    f'"limitation": "<1 sentence on what live data would change>"}}'
                ),
            }],
        )
        raw = response.content[0].text.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
        return json.loads(raw)
    except Exception as e:
        console.print(f"  [yellow]Claude baseline failed for {ticker}: {e}[/yellow]")
        return {
            "verdict": "N/A",
            "score_overall": None,
            "technical_score": None,
            "fundamental_score": None,
            "sentiment_score": None,
            "confidence_pct": None,
            "price_target": None,
            "thesis": "Claude baseline call failed.",
            "limitation": str(e),
        }


# ── Interactive prompts for external tool data ─────────────────────────────

def _prompt_score(label: str, lo: float = 1, hi: float = 10) -> float | None:
    raw = Prompt.ask(f"    {label} [{lo}-{hi}, or Enter to skip]", default="")
    if not raw.strip():
        return None
    try:
        v = float(raw.strip())
        return max(lo, min(hi, v))
    except ValueError:
        return None


def _prompt_text(label: str) -> str | None:
    raw = Prompt.ask(f"    {label} [or Enter to skip]", default="")
    return raw.strip() or None


def collect_danelfin(ticker: str) -> dict:
    console.print(f"\n[bold cyan]Danelfin — {ticker}[/bold cyan]")
    console.print(f"  [dim]→ Go to danelfin.com → search {ticker} → copy scores below[/dim]")
    d = _empty_danelfin()
    d["ai_score"]    = _prompt_score("AI Score (overall)", 1, 10)
    d["technical"]   = _prompt_score("Technical Score", 1, 10)
    d["fundamental"] = _prompt_score("Fundamental Score", 1, 10)
    d["sentiment"]   = _prompt_score("Sentiment Score", 1, 10)
    d["signal"]      = _prompt_text("Signal (e.g. Strong Buy, Buy, Neutral, Sell)")
    return d


def collect_seeking_alpha(ticker: str) -> dict:
    console.print(f"\n[bold green]Seeking Alpha Quant — {ticker}[/bold green]")
    console.print(f"  [dim]→ Go to seekingalpha.com/{ticker} → Quant Rating + Factor Grades[/dim]")
    s = _empty_seeking_alpha()
    s["quant_rating"]   = _prompt_text("Quant Rating (e.g. Strong Buy, Hold)")
    s["valuation"]      = _prompt_text("Valuation Grade (A+ … F)")
    s["growth"]         = _prompt_text("Growth Grade")
    s["profitability"]  = _prompt_text("Profitability Grade")
    s["momentum"]       = _prompt_text("Momentum Grade")
    s["eps_revisions"]  = _prompt_text("EPS Revisions Grade")
    s["price_target"]   = _prompt_text("Wall St Price Target (e.g. $245)")
    return s


# ── HTML report generator ──────────────────────────────────────────────────

GRADE_COLORS = {
    "A+": "#22c55e", "A": "#4ade80", "B": "#86efac",
    "C": "#fbbf24", "D": "#f97316", "F": "#ef4444",
}

VERDICT_COLORS = {
    "STRONG BUY":  "#16a34a",
    "BUY":         "#22c55e",
    "HOLD":        "#d97706",
    "SELL":        "#ef4444",
    "STRONG SELL": "#991b1b",
    "N/A":         "#6b7280",
}

SIGNAL_TO_VERDICT = {
    "strong buy": "STRONG BUY",
    "buy": "BUY",
    "neutral": "HOLD",
    "sell": "SELL",
    "strong sell": "STRONG SELL",
}


def _verdict_badge(verdict: str | None) -> str:
    if not verdict:
        return '<span style="color:#6b7280">N/A</span>'
    key = verdict.upper().strip()
    normalized = SIGNAL_TO_VERDICT.get(key.lower(), key)
    color = VERDICT_COLORS.get(normalized, "#6b7280")
    return (
        f'<span style="background:{color};color:white;padding:3px 10px;'
        f'border-radius:4px;font-weight:bold;font-size:0.85em">{verdict}</span>'
    )


def _score_bar(score: float | None, max_score: float = 10) -> str:
    if score is None:
        return '<span style="color:#6b7280">N/A</span>'
    pct = int((score / max_score) * 100)
    if pct >= 70:
        color = "#22c55e"
    elif pct >= 50:
        color = "#d97706"
    else:
        color = "#ef4444"
    return (
        f'<div style="display:flex;align-items:center;gap:8px">'
        f'<div style="flex:1;background:#1f2937;border-radius:4px;height:8px">'
        f'<div style="width:{pct}%;background:{color};border-radius:4px;height:8px"></div>'
        f'</div>'
        f'<span style="min-width:32px;font-weight:bold;color:{color}">{score:.1f}</span>'
        f'</div>'
    )


def _grade_chip(grade: str | None) -> str:
    if not grade:
        return '<span style="color:#6b7280">N/A</span>'
    color = GRADE_COLORS.get(grade.upper(), "#6b7280")
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:4px;font-weight:bold;font-size:0.8em">{grade}</span>'
    )


def _row(label: str, *cells: str) -> str:
    cells_html = "".join(f'<td style="padding:10px 14px;border-bottom:1px solid #1f2937">{c}</td>' for c in cells)
    return f'<tr><td style="padding:10px 14px;border-bottom:1px solid #1f2937;color:#9ca3af;font-size:0.85em">{label}</td>{cells_html}</tr>'


def _na(val) -> str:
    return str(val) if val is not None else '<span style="color:#6b7280">N/A</span>'


def _score_cell(val) -> str:
    return _score_bar(val) if val is not None else '<span style="color:#6b7280">N/A</span>'


def generate_comparison_html(results: dict, tickers: list[str], date_str: str) -> str:
    """Build the full comparison HTML page."""

    ticker_sections = ""
    for ticker in tickers:
        r = results[ticker]
        final   = r["final_report"]       # FinalReport pydantic model
        baseline = r["claude_baseline"]   # dict
        danelfin = r["danelfin"]          # dict
        sa       = r["seeking_alpha"]     # dict

        # Tool headers
        headers = (
            "<th>Dimension</th>"
            "<th>Beat-the-ASPP<br><small style='color:#9ca3af;font-weight:normal'>"
            "3-Agent + Claude Synthesis</small></th>"
            "<th>Claude Baseline<br><small style='color:#9ca3af;font-weight:normal'>"
            "Raw LLM, No Live Data</small></th>"
            "<th>Danelfin<br><small style='color:#9ca3af;font-weight:normal'>"
            "AI Score Platform</small></th>"
            "<th>Seeking Alpha Quant<br><small style='color:#9ca3af;font-weight:normal'>"
            "Factor-Grade System</small></th>"
        )

        # Score rows
        def sa_factor_row(label, grade):
            return _row(label,
                        "—",
                        "—",
                        "—",
                        _grade_chip(grade))

        rows = (
            _row("Overall Score",
                 _score_bar(final.composite_score),
                 _score_cell(baseline.get("score_overall")),
                 _score_bar(danelfin.get("ai_score")),
                 _na(sa.get("quant_rating")),
            ) +
            _row("Technical",
                 _score_bar(final.technical_score),
                 _score_cell(baseline.get("technical_score")),
                 _score_bar(danelfin.get("technical")),
                 _grade_chip(sa.get("momentum")),
            ) +
            _row("Fundamental",
                 _score_bar(final.fundamental_score),
                 _score_cell(baseline.get("fundamental_score")),
                 _score_bar(danelfin.get("fundamental")),
                 f'{_grade_chip(sa.get("growth"))} Growth / {_grade_chip(sa.get("profitability"))} Profit',
            ) +
            _row("Sentiment",
                 _score_bar(final.sentiment_score),
                 _score_cell(baseline.get("sentiment_score")),
                 _score_bar(danelfin.get("sentiment")),
                 _grade_chip(sa.get("eps_revisions")),
            ) +
            _row("Valuation",
                 "—",
                 "—",
                 "—",
                 _grade_chip(sa.get("valuation")),
            ) +
            _row("Verdict",
                 _verdict_badge(final.verdict),
                 _verdict_badge(baseline.get("verdict")),
                 _verdict_badge(danelfin.get("signal")),
                 _verdict_badge(sa.get("quant_rating")),
            ) +
            _row("Confidence",
                 f'{final.confidence_pct:.0f}%',
                 _na(f'{baseline["confidence_pct"]:.0f}%' if baseline.get("confidence_pct") else None),
                 "—",
                 "—",
            ) +
            _row("Price Target",
                 _na(final.price_target),
                 _na(baseline.get("price_target")),
                 "—",
                 _na(sa.get("price_target")),
            ) +
            _row("Live Data Used",
                 '<span style="color:#22c55e">✓ yfinance + News</span>',
                 '<span style="color:#ef4444">✗ Training Only</span>',
                 '<span style="color:#22c55e">✓ Proprietary Feed</span>',
                 '<span style="color:#22c55e">✓ Proprietary Feed</span>',
            ) +
            _row("Methodology",
                 "3 parallel agents + adaptive thinking synthesis",
                 "Single LLM call, no tools",
                 "ML model on price/fundamental/news data",
                 "Quantitative factor scoring",
            )
        )

        # Analyst thesis block
        thesis_html = ""
        if final.analyst_thesis:
            thesis_html = f"""
            <div style="margin-top:16px;padding:16px;background:#1e3a5f;border-radius:8px;border-left:4px solid #3b82f6">
              <div style="font-size:0.75em;color:#93c5fd;margin-bottom:6px;text-transform:uppercase;letter-spacing:1px">
                Beat-the-ASPP Analyst Thesis
              </div>
              <div style="color:#e2e8f0;line-height:1.6">{final.analyst_thesis}</div>
            </div>"""

        baseline_limitation = baseline.get("limitation", "")
        limitation_html = ""
        if baseline_limitation and baseline_limitation != "N/A":
            limitation_html = f"""
            <div style="margin-top:12px;padding:12px;background:#2d1b1b;border-radius:8px;border-left:4px solid #f87171">
              <div style="font-size:0.75em;color:#fca5a5;margin-bottom:4px;text-transform:uppercase;letter-spacing:1px">
                Claude Baseline Limitation
              </div>
              <div style="color:#e2e8f0;font-size:0.9em">{baseline_limitation}</div>
            </div>"""

        baseline_thesis = baseline.get("thesis", "")
        baseline_thesis_html = ""
        if baseline_thesis and baseline_thesis != "Claude baseline call failed.":
            baseline_thesis_html = f"""
            <div style="margin-top:12px;padding:12px;background:#1c2a1c;border-radius:8px;border-left:4px solid #86efac">
              <div style="font-size:0.75em;color:#86efac;margin-bottom:4px;text-transform:uppercase;letter-spacing:1px">
                Claude Baseline Thesis
              </div>
              <div style="color:#e2e8f0;font-size:0.9em">{baseline_thesis}</div>
            </div>"""

        # Key differences box
        diffs = _compute_differences(final, baseline, danelfin, sa)
        diff_items = "".join(f'<li style="margin-bottom:6px">{d}</li>' for d in diffs)

        ticker_sections += f"""
        <section style="margin-bottom:48px">
          <h2 style="color:#f8fafc;font-size:1.4em;margin-bottom:4px">
            {ticker}
            <span style="font-size:0.65em;color:#9ca3af;font-weight:normal;margin-left:12px">
              Current Price: ${final.current_price:.2f} &nbsp;|&nbsp;
              Report Date: {final.report_date}
            </span>
          </h2>

          <div style="overflow-x:auto">
            <table style="width:100%;border-collapse:collapse;background:#111827;border-radius:8px;overflow:hidden">
              <thead>
                <tr style="background:#1f2937;color:#e2e8f0;text-align:left">
                  {headers}
                </tr>
              </thead>
              <tbody>{rows}</tbody>
            </table>
          </div>

          {thesis_html}
          {baseline_thesis_html}
          {limitation_html}

          <div style="margin-top:16px;padding:16px;background:#1f2937;border-radius:8px">
            <div style="font-size:0.8em;color:#9ca3af;margin-bottom:8px;text-transform:uppercase;letter-spacing:1px">
              Key Differences Observed
            </div>
            <ul style="color:#e2e8f0;margin:0;padding-left:20px;font-size:0.9em;line-height:1.8">
              {diff_items}
            </ul>
          </div>
        </section>"""

    # Summary methodology table
    methodology = """
    <section style="margin-bottom:48px">
      <h2 style="color:#f8fafc;font-size:1.4em;margin-bottom:16px">Methodology Comparison</h2>
      <table style="width:100%;border-collapse:collapse;background:#111827;border-radius:8px;overflow:hidden">
        <thead>
          <tr style="background:#1f2937;color:#e2e8f0;text-align:left">
            <th style="padding:12px 14px">Attribute</th>
            <th style="padding:12px 14px">Beat-the-ASPP</th>
            <th style="padding:12px 14px">Claude Baseline</th>
            <th style="padding:12px 14px">Danelfin</th>
            <th style="padding:12px 14px">Seeking Alpha Quant</th>
          </tr>
        </thead>
        <tbody>
    """ + (
        _row("Architecture",      "3 parallel agents",          "Single LLM call",       "ML ensemble",             "Quantitative scoring") +
        _row("Live Market Data",  "yfinance (free, real-time)", "None (training cutoff)", "Proprietary data feed",   "Proprietary data feed") +
        _row("News/Sentiment",    "Scraped news + NLP",         "Training data only",     "NLP on news feed",        "EPS Revisions proxy") +
        _row("Reasoning",         "Adaptive thinking synthesis","Single-pass generation", "Black-box ML",            "Formula-based grades") +
        _row("Explainability",    "Full analyst thesis",         "Thesis + caveats",       "Score only",              "Grade breakdown") +
        _row("Customizable",      "Yes — weights, prompts",      "Yes — any prompt",       "No",                      "No") +
        _row("Cost",              "Claude API tokens",           "Claude API tokens",      "Freemium",                "Freemium / Premium") +
        _row("Sector-aware",      "Yes — CFA-style grading",    "Partially",              "Partially",               "Universal benchmarks") +
        _row("Scoring scale",     "0–10 per dimension",          "0–10 per dimension",     "1–10 per dimension",      "A+ → F per factor")
    ) + """
        </tbody>
      </table>
    </section>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Stock Evaluation Tool Comparison — {' vs '.join(tickers)}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #0f172a; color: #e2e8f0; margin: 0; padding: 24px; }}
    h1 {{ color: #f8fafc; font-size: 1.8em; margin-bottom: 4px; }}
    h2 {{ border-bottom: 1px solid #1f2937; padding-bottom: 8px; }}
    th {{ padding: 12px 14px; font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.5px; }}
    small {{ display: block; margin-top: 2px; }}
    a {{ color: #60a5fa; }}
  </style>
</head>
<body>
  <header style="margin-bottom:36px">
    <h1>Stock Evaluation Tool Comparison</h1>
    <div style="color:#9ca3af;margin-bottom:8px">
      Tickers: <strong style="color:#f8fafc">{' | '.join(tickers)}</strong>
      &nbsp;·&nbsp; Generated: <strong style="color:#f8fafc">{date_str}</strong>
    </div>
    <div style="padding:12px 16px;background:#1e293b;border-radius:8px;font-size:0.85em;color:#94a3b8;max-width:800px">
      <strong style="color:#e2e8f0">Tools:</strong>
      &nbsp;<span style="color:#60a5fa">Beat-the-ASPP</span> (your 3-agent system) ·
      &nbsp;<span style="color:#86efac">Claude Baseline</span> (raw Claude API, no live data) ·
      &nbsp;<span style="color:#fbbf24">Danelfin</span> (danelfin.com, AI score 1–10) ·
      &nbsp;<span style="color:#f472b6">Seeking Alpha Quant</span> (factor grades A–F)
    </div>
  </header>

  {ticker_sections}
  {methodology}

  <footer style="margin-top:48px;padding-top:16px;border-top:1px solid #1f2937;color:#6b7280;font-size:0.8em">
    Generated by Beat-the-ASPP compare.py · Model: claude-opus-4-6 ·
    Danelfin and Seeking Alpha data entered manually from their respective platforms.
    This report is for educational/research purposes only — not financial advice.
  </footer>
</body>
</html>"""


def _compute_differences(final, baseline, danelfin, sa) -> list[str]:
    """Generate a plain-language list of notable differences between the four tools."""
    diffs = []

    # Verdict alignment
    ba_verdict = (final.verdict or "").upper()
    bl_verdict = (baseline.get("verdict") or "").upper()
    da_signal  = (danelfin.get("signal") or "").upper()
    sa_rating  = (sa.get("quant_rating") or "").upper()

    verdicts = [v for v in [ba_verdict, bl_verdict, da_signal, sa_rating] if v and v != "N/A"]
    if verdicts:
        buys  = sum(1 for v in verdicts if "BUY" in v)
        sells = sum(1 for v in verdicts if "SELL" in v)
        holds = sum(1 for v in verdicts if "HOLD" in v or "NEUTRAL" in v)
        if buys == len(verdicts):
            diffs.append("All four tools agree: <strong>bullish consensus</strong>.")
        elif sells == len(verdicts):
            diffs.append("All four tools agree: <strong>bearish consensus</strong>.")
        elif ba_verdict != bl_verdict and bl_verdict:
            diffs.append(
                f"<strong>Beat-the-ASPP</strong> says <em>{ba_verdict}</em> vs "
                f"<strong>Claude Baseline</strong> says <em>{bl_verdict}</em> — "
                "live data shifts the verdict."
            )

    # Score gap: multi-agent vs baseline
    if final.composite_score is not None and baseline.get("score_overall") is not None:
        gap = round(final.composite_score - baseline["score_overall"], 1)
        if abs(gap) >= 1.0:
            direction = "higher" if gap > 0 else "lower"
            diffs.append(
                f"Beat-the-ASPP scores <strong>{abs(gap)} points {direction}</strong> than Claude Baseline "
                f"({final.composite_score:.1f} vs {baseline['score_overall']:.1f}) — "
                "live fundamental/technical data changes the picture."
            )

    # Danelfin comparison
    if danelfin.get("ai_score") is not None and final.composite_score is not None:
        gap = round(final.composite_score - danelfin["ai_score"], 1)
        if abs(gap) >= 1.5:
            direction = "more bullish" if gap > 0 else "more conservative"
            diffs.append(
                f"Beat-the-ASPP is <strong>{direction}</strong> than Danelfin "
                f"({final.composite_score:.1f} vs {danelfin['ai_score']:.1f}/10)."
            )
        else:
            diffs.append(
                f"Beat-the-ASPP and Danelfin are broadly aligned "
                f"({final.composite_score:.1f} vs {danelfin['ai_score']:.1f}/10)."
            )

    # Seeking Alpha valuation note
    val_grade = sa.get("valuation")
    if val_grade and val_grade.upper() in ("D", "D+", "D-", "F"):
        diffs.append(
            f"Seeking Alpha flags <strong>valuation concern</strong> (grade: {val_grade}) — "
            "Beat-the-ASPP weights this inside the fundamental score."
        )

    # Explainability note
    diffs.append(
        "Beat-the-ASPP provides a full <strong>analyst thesis</strong> explaining the verdict; "
        "Danelfin and Seeking Alpha return scores/grades with limited narrative."
    )

    # Live data note
    diffs.append(
        "Claude Baseline uses <strong>training data only</strong> (knowledge cutoff) — "
        "no real-time prices, earnings, or news. The other three tools use live feeds."
    )

    return diffs


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare stock evaluation tools side-by-side")
    parser.add_argument("--tickers", nargs=2, default=["AAPL", "NVDA"], metavar="TICKER",
                        help="Two tickers to compare (default: AAPL NVDA)")
    parser.add_argument("--period", default="1y", choices=["3mo", "6mo", "1y", "2y"],
                        help="Historical period for technical analysis (default: 1y)")
    parser.add_argument("--skip-external", action="store_true",
                        help="Skip manual entry of Danelfin + Seeking Alpha data")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not auto-open the report in a browser")
    args = parser.parse_args()

    tickers = [t.upper() for t in args.tickers]

    if not os.getenv("ANTHROPIC_API_KEY"):
        console.print(
            "[bold red]ERROR:[/bold red] ANTHROPIC_API_KEY not found in .env",
            style="red"
        )
        sys.exit(1)

    from llm_client import AnthropicLLMClient
    client = AnthropicLLMClient()

    console.print(Panel.fit(
        f"[bold cyan]Beat-the-ASPP Tool Comparison[/bold cyan]\n"
        f"[dim]4-way analysis: your tool · Claude baseline · Danelfin · Seeking Alpha[/dim]\n\n"
        f"Tickers: [bold yellow]{' | '.join(tickers)}[/bold yellow]",
        border_style="cyan"
    ))

    results = {}

    # ── Step 1: Run full 3-agent analysis for each ticker ─────────────
    from orchestrator import run_analysis
    for ticker in tickers:
        console.print(f"\n[bold]{'='*60}[/bold]")
        console.print(f"[bold cyan]  Running Beat-the-ASPP → {ticker}[/bold cyan]")
        console.print(f"[bold]{'='*60}[/bold]")
        final_report = run_analysis(
            ticker=ticker,
            period=args.period,
            stream_output=True,
            llm_provider="api",
        )
        results[ticker] = {"final_report": final_report}

    # ── Step 2: Run Claude baseline for each ticker ───────────────────
    console.print(f"\n[bold]{'='*60}[/bold]")
    console.print("[bold green]  Running Claude Baseline (no live data)[/bold green]")
    console.print(f"[bold]{'='*60}[/bold]\n")
    for ticker in tickers:
        results[ticker]["claude_baseline"] = run_claude_baseline(ticker, client)
        bl = results[ticker]["claude_baseline"]
        console.print(
            f"  [green]✓[/green] {ticker} baseline: "
            f"[bold]{bl.get('verdict', 'N/A')}[/bold] | "
            f"Score: {bl.get('score_overall', 'N/A')}/10"
        )

    # ── Step 3: Collect external tool data ────────────────────────────
    if args.skip_external:
        for ticker in tickers:
            results[ticker]["danelfin"]      = _empty_danelfin()
            results[ticker]["seeking_alpha"] = _empty_seeking_alpha()
    else:
        console.print(f"\n[bold]{'='*60}[/bold]")
        console.print("[bold yellow]  External Tool Data Entry[/bold yellow]")
        console.print(f"[bold]{'='*60}[/bold]")
        console.print("[dim]  Enter scores from each platform. Press Enter to skip any field.[/dim]\n")

        for ticker in tickers:
            results[ticker]["danelfin"]      = collect_danelfin(ticker)
            results[ticker]["seeking_alpha"] = collect_seeking_alpha(ticker)

    # ── Step 4: Generate comparison report ────────────────────────────
    date_str  = datetime.now().strftime("%Y-%m-%d")
    os.makedirs("reports", exist_ok=True)
    filename  = f"reports/comparison_{'_'.join(tickers)}_{date_str}.html"

    html = generate_comparison_html(results, tickers, date_str)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)

    console.print(f"\n[bold green]✓ Comparison report saved:[/bold green] {filename}")

    if not args.no_browser:
        webbrowser.open(f"file:///{os.path.abspath(filename)}")

    # ── Step 5: Print quick console summary ───────────────────────────
    console.print()
    for ticker in tickers:
        final   = results[ticker]["final_report"]
        bl      = results[ticker]["claude_baseline"]
        danelf  = results[ticker]["danelfin"]
        sa_data = results[ticker]["seeking_alpha"]

        t = Table(
            title=f"[bold]{ticker}[/bold] — Quick Comparison",
            box=box.ROUNDED, header_style="bold cyan",
        )
        t.add_column("Tool",         style="dim",   width=26)
        t.add_column("Score",        justify="center", width=10)
        t.add_column("Verdict",      justify="center", width=16)

        t.add_row(
            "Beat-the-ASPP",
            f"{final.composite_score:.1f}/10",
            final.verdict or "—",
            style="cyan",
        )
        t.add_row(
            "Claude Baseline",
            f"{bl.get('score_overall') or '—'}/10" if bl.get("score_overall") else "—",
            bl.get("verdict") or "—",
            style="green",
        )
        t.add_row(
            "Danelfin",
            f"{danelf.get('ai_score') or '—'}/10" if danelf.get("ai_score") else "—",
            danelf.get("signal") or "—",
            style="yellow",
        )
        t.add_row(
            "Seeking Alpha Quant",
            "—",
            sa_data.get("quant_rating") or "—",
            style="magenta",
        )
        console.print(t)
        console.print()


if __name__ == "__main__":
    main()