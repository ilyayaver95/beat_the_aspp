"""
report_generator.py
====================
Generates a self-contained HTML report with an interactive Plotly chart,
scores summary, fundamental metrics table, and news section.

Called by orchestrator.py after all three agents complete.
"""

import os
from datetime import datetime


def generate_html_report(
    ticker: str,
    tech_report,
    fund_report,
    sent_report,
    final_report,
    market_data: dict = None,
    news_articles: list = None,
    model_name: str = "claude-opus-4-6",
) -> str:
    """
    Generate an HTML report and save it to reports/<TICKER>_<timestamp>.html.

    Returns the file path, or None if generation fails.
    """
    try:
        os.makedirs("reports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reports/{ticker}_{timestamp}.html"

        chart_html = _build_chart(ticker, market_data, tech_report) if market_data else ""
        html = _build_html(
            ticker=ticker,
            tech=tech_report,
            fund=fund_report,
            sent=sent_report,
            final=final_report,
            chart_html=chart_html,
            news=news_articles or [],
            model_name=model_name,
        )

        with open(filename, "w", encoding="utf-8") as f:
            f.write(html)

        return filename

    except Exception as e:
        print(f"  [report_generator] Warning: HTML report failed: {e}")
        return None


# ── Chart ─────────────────────────────────────────────────────────────

def _build_chart(ticker: str, data: dict, tech_report) -> str:
    """Build a Plotly candlestick + volume chart with MA overlays."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import pandas as pd

        df = data.get("df")
        if df is None or df.empty:
            return ""

        # Use last 6 months for readability
        df = df.tail(126).copy()

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25],
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=ticker,
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ), row=1, col=1)

        # Moving averages
        ma_colors = {
            "ema20": ("#42a5f5", "EMA20"),
            "ema50": ("#ff9800", "EMA50"),
            "sma150": ("#ab47bc", "SMA150"),
            "sma200": ("#ef5350", "SMA200"),
        }
        emas = data.get("emas", {})
        smas = data.get("smas", {})
        all_ma = {**emas, **smas}

        for key, (color, label) in ma_colors.items():
            series = all_ma.get(key)
            if series is not None:
                series_aligned = series.reindex(df.index)
                fig.add_trace(go.Scatter(
                    x=df.index, y=series_aligned,
                    name=label, line=dict(color=color, width=1.5),
                    opacity=0.8,
                ), row=1, col=1)

        # Support & Resistance lines
        current_price = data.get("current_price", 0)
        for level in data.get("support_levels", []):
            fig.add_hline(
                y=level, line_dash="dot", line_color="#26a69a",
                line_width=1, opacity=0.6,
                annotation_text=f"S ${level:.2f}",
                annotation_position="left",
                row=1, col=1,
            )
        for level in data.get("resistance_levels", []):
            fig.add_hline(
                y=level, line_dash="dot", line_color="#ef5350",
                line_width=1, opacity=0.6,
                annotation_text=f"R ${level:.2f}",
                annotation_position="left",
                row=1, col=1,
            )

        # Volume bars
        colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"],
            name="Volume", marker_color=colors, opacity=0.7,
        ), row=2, col=1)

        fig.update_layout(
            title=dict(
                text=f"{fund_report_name(tech_report)} ({ticker}) — Technical Chart",
                font=dict(size=16),
            ),
            template="plotly_dark",
            height=600,
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=50, t=80, b=30),
        )

        fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    except Exception as e:
        print(f"  [report_generator] Chart error: {e}")
        return ""


def fund_report_name(tech_report) -> str:
    try:
        return tech_report.ticker
    except Exception:
        return ""


# ── HTML Template ─────────────────────────────────────────────────────

def _build_html(ticker, tech, fund, sent, final, chart_html, news, model_name):
    verdict_colors = {
        "STRONG BUY": "#00e676",
        "BUY": "#69f0ae",
        "HOLD": "#ffd740",
        "SELL": "#ff6d00",
        "STRONG SELL": "#f44336",
    }
    verdict_color = verdict_colors.get(final.verdict, "#ffffff")

    # Score bar helper
    def score_bar(score, color):
        pct = score * 10
        return f"""
        <div class="score-bar-bg">
          <div class="score-bar-fill" style="width:{pct}%;background:{color}"></div>
        </div>"""

    # Fundamental metrics rows
    fund_rows = ""
    metrics = [
        ("Revenue Growth YoY", fund.revenue_growth_yoy),
        ("Net Income Margin", fund.net_income_margin),
        ("P/E Ratio", fund.pe_ratio),
        ("EPS Growth", fund.eps_growth),
        ("Return on Equity", fund.return_on_equity),
        ("Debt / Equity", fund.debt_to_equity),
        ("Free Cash Flow", fund.free_cash_flow),
    ]
    grade_colors = {
        "A+": "#00e676", "A": "#69f0ae", "B": "#b2ff59",
        "C": "#ffd740", "D": "#ff9100", "F": "#f44336",
    }
    for name, metric in metrics:
        gc = grade_colors.get(metric.grade, "#ffffff")
        val_str = f"{metric.value:.2f}" if metric.value is not None else "—"
        fund_rows += f"""
        <tr>
          <td>{name}</td>
          <td>{val_str}</td>
          <td><span class="grade-badge" style="background:{gc}">{metric.grade}</span></td>
          <td class="muted">{metric.comment}</td>
        </tr>"""

    # Opportunities and risks
    opps = "".join(f'<li class="opp">✓ {o}</li>' for o in final.key_opportunities[:4])
    risks = "".join(f'<li class="risk">✗ {r}</li>' for r in final.key_risks[:4])
    watch = "".join(f'<li>→ {w}</li>' for w in final.watch_for[:4])

    # News section
    news_html = ""
    for article in news[:8]:
        news_html += f"""
        <div class="news-item">
          <div class="news-date">{article.get('date', '')} · {article.get('publisher', '')}</div>
          <div class="news-title">{article.get('title', '')}</div>
          {f'<div class="news-summary muted">{article["summary"][:200]}…</div>' if article.get('summary') else ''}
        </div>"""

    if not news_html:
        news_html = '<p class="muted">No news articles available.</p>'

    patterns_html = "".join(f"<li>{p}</li>" for p in tech.key_patterns) or "<li>None detected</li>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Beat the ASPP — {ticker} Analysis</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0d1117; color: #e6edf3; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 24px 16px; }}
  h1 {{ font-size: 1.6rem; color: #58a6ff; margin-bottom: 4px; }}
  h2 {{ font-size: 1.1rem; color: #8b949e; margin: 24px 0 12px; text-transform: uppercase; letter-spacing: .05em; }}
  .meta {{ color: #8b949e; font-size: 0.85rem; margin-bottom: 24px; }}
  .grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 12px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }}
  .card-title {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: .06em; color: #8b949e; margin-bottom: 8px; }}
  .card-value {{ font-size: 1.8rem; font-weight: 700; }}
  .card-sub {{ font-size: 0.8rem; color: #8b949e; margin-top: 4px; }}
  .verdict-card {{ background: #161b22; border: 2px solid {verdict_color}; border-radius: 8px; padding: 20px; text-align: center; }}
  .verdict-label {{ font-size: 2rem; font-weight: 800; color: {verdict_color}; }}
  .score-bar-bg {{ background: #30363d; border-radius: 4px; height: 6px; margin-top: 6px; }}
  .score-bar-fill {{ height: 6px; border-radius: 4px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
  th {{ text-align: left; padding: 8px 12px; color: #8b949e; border-bottom: 1px solid #30363d; font-weight: 500; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #21262d; }}
  .grade-badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-weight: 700; color: #000; font-size: 0.8rem; }}
  .muted {{ color: #8b949e; font-size: 0.82rem; }}
  ul {{ list-style: none; padding: 0; }}
  li {{ padding: 4px 0; font-size: 0.88rem; }}
  .opp {{ color: #69f0ae; }}
  .risk {{ color: #ff7043; }}
  .news-item {{ border-bottom: 1px solid #21262d; padding: 10px 0; }}
  .news-date {{ font-size: 0.75rem; color: #8b949e; margin-bottom: 2px; }}
  .news-title {{ font-size: 0.9rem; font-weight: 500; }}
  .news-summary {{ font-size: 0.8rem; margin-top: 3px; }}
  .thesis {{ line-height: 1.65; font-size: 0.92rem; color: #c9d1d9; }}
  .chart-container {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 8px; margin-bottom: 12px; }}
</style>
</head>
<body>
<div class="container">

  <h1>Beat the ASPP — {fund.company_name} ({ticker})</h1>
  <div class="meta">
    Generated {final.report_date} · Model: {model_name} ·
    Price: ${final.current_price:.2f}
    {f'· Target: {final.price_target}' if final.price_target else ''}
  </div>

  <!-- Verdict + Scores -->
  <div class="grid-2">
    <div class="verdict-card">
      <div class="card-title">Analyst Verdict</div>
      <div class="verdict-label">{final.verdict}</div>
      <div class="card-sub">
        Confidence: {final.confidence_pct:.0f}% · Horizon: {final.time_horizon}
      </div>
      <div class="card-sub" style="margin-top:8px">
        Composite: <strong>{final.composite_score:.1f}/10</strong>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Score Breakdown</div>
      <table style="font-size:0.85rem">
        <tr>
          <td style="color:#42a5f5">Technical</td>
          <td style="text-align:right">{final.technical_score:.1f}/10</td>
          <td style="width:40%;padding-left:8px">{score_bar(final.technical_score, '#42a5f5')}</td>
          <td class="muted" style="font-size:0.75rem">35%</td>
        </tr>
        <tr>
          <td style="color:#66bb6a">Fundamental</td>
          <td style="text-align:right">{final.fundamental_score:.1f}/10</td>
          <td style="width:40%;padding-left:8px">{score_bar(final.fundamental_score, '#66bb6a')}</td>
          <td class="muted" style="font-size:0.75rem">45%</td>
        </tr>
        <tr>
          <td style="color:#ab47bc">Sentiment</td>
          <td style="text-align:right">{final.sentiment_score:.1f}/10</td>
          <td style="width:40%;padding-left:8px">{score_bar(final.sentiment_score, '#ab47bc')}</td>
          <td class="muted" style="font-size:0.75rem">20%</td>
        </tr>
      </table>
    </div>
  </div>

  <!-- Chart -->
  {f'<div class="chart-container">{chart_html}</div>' if chart_html else ''}

  <!-- Analyst Thesis -->
  <h2>Analyst Thesis</h2>
  <div class="card">
    <p class="thesis">{final.analyst_thesis}</p>
    <div style="margin-top:12px;display:grid;grid-template-columns:1fr 1fr;gap:12px">
      <div>
        <div class="card-title" style="color:#69f0ae">Bull Case</div>
        <p class="muted" style="font-size:0.85rem">{final.bull_case}</p>
      </div>
      <div>
        <div class="card-title" style="color:#ff7043">Bear Case</div>
        <p class="muted" style="font-size:0.85rem">{final.bear_case}</p>
      </div>
    </div>
  </div>

  <!-- Opportunities & Risks -->
  <div class="grid-2">
    <div class="card">
      <div class="card-title" style="color:#69f0ae">Key Opportunities</div>
      <ul>{opps}</ul>
    </div>
    <div class="card">
      <div class="card-title" style="color:#ff7043">Key Risks</div>
      <ul>{risks}</ul>
    </div>
  </div>

  <!-- Fundamental Metrics -->
  <h2>Fundamental Analysis — {fund.score:.1f}/10</h2>
  <div class="card">
    {f'<div class="muted" style="margin-bottom:8px">{fund.company_name} · {fund.sector} · {fund.industry}</div>' if fund.sector else ''}
    <table>
      <thead><tr><th>Metric</th><th>Value</th><th>Grade</th><th>Comment</th></tr></thead>
      <tbody>{fund_rows}</tbody>
    </table>
    <p style="margin-top:12px;font-size:0.88rem">{fund.summary}</p>
    <div style="margin-top:8px;display:grid;grid-template-columns:1fr 1fr;gap:8px">
      <div>
        <div class="card-title" style="color:#69f0ae;margin-top:8px">Strengths</div>
        <ul>{''.join(f'<li class="opp">✓ {s}</li>' for s in fund.key_strengths)}</ul>
      </div>
      <div>
        <div class="card-title" style="color:#ff7043;margin-top:8px">Concerns</div>
        <ul>{''.join(f'<li class="risk">✗ {c}</li>' for c in fund.key_concerns)}</ul>
      </div>
    </div>
  </div>

  <!-- Technical Analysis -->
  <h2>Technical Analysis — {tech.score:.1f}/10</h2>
  <div class="card">
    <div class="grid-3" style="margin-bottom:12px">
      <div>
        <div class="card-title">Trend</div>
        <div style="font-weight:600">{tech.trend} ({tech.trend_strength})</div>
      </div>
      <div>
        <div class="card-title">RSI(14)</div>
        <div style="font-weight:600">{f'{tech.rsi_value:.1f}' if tech.rsi_value else '—'} — {tech.rsi_condition or '—'}</div>
      </div>
      <div>
        <div class="card-title">Volume Signal</div>
        <div style="font-weight:600">{tech.volume_signal or '—'}</div>
      </div>
    </div>
    <p style="font-size:0.88rem;margin-bottom:10px">{tech.summary}</p>
    <p class="muted" style="font-size:0.85rem">{tech.short_term_outlook}</p>
    {f'<div style="margin-top:10px"><div class="card-title">Chart Patterns</div><ul>{"".join(f"<li>• {p}</li>" for p in tech.key_patterns)}</ul></div>' if tech.key_patterns else ''}
    <div style="margin-top:10px;display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:0.85rem">
      <div>
        <span class="muted">Buy Zone:</span>
        <strong> ${tech.buy_zone:.2f}</strong> — {', '.join(tech.buy_reasons[:2]) if tech.buy_reasons else '—'}
      </div>
      <div>
        <span class="muted">Sell Zone:</span>
        <strong> ${tech.sell_zone:.2f}</strong> — {', '.join(tech.sell_reasons[:2]) if tech.sell_reasons else '—'}
      </div>
    </div>
  </div>

  <!-- Sentiment -->
  <h2>Sentiment Analysis — {sent.score:.1f}/10</h2>
  <div class="card">
    <div style="margin-bottom:10px">
      <span style="font-weight:600">{sent.overall_sentiment}</span>
      <span class="muted"> · {sent.articles_analyzed} articles · {sent.date_range}</span>
    </div>
    <p style="font-size:0.88rem;margin-bottom:10px">{sent.summary}</p>
    {f'<div><div class="card-title">Key Themes</div><ul>{"".join(f"<li>• {t}</li>" for t in sent.key_themes[:5])}</ul></div>' if sent.key_themes else ''}
    {f'<div style="margin-top:8px"><div class="card-title" style="color:#ffd740">Upcoming Catalysts</div><ul>{"".join(f"<li>→ {c}</li>" for c in sent.upcoming_catalysts[:4])}</ul></div>' if sent.upcoming_catalysts else ''}
  </div>

  <!-- Watch For -->
  {f'<h2>Watch For</h2><div class="card"><ul>{watch}</ul></div>' if final.watch_for else ''}

  <!-- News -->
  <h2>Recent News ({len(news)} articles)</h2>
  <div class="card">{news_html}</div>

  <div class="meta" style="margin-top:24px;text-align:center">
    Beat the ASPP · AI Stock Evaluator · {final.report_date}
  </div>

</div>
</body>
</html>"""
