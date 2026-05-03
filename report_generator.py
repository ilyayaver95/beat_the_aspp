"""
report_generator.py
====================
Generates a professional, self-contained HTML report from all agent outputs.

WHAT'S INCLUDED IN THE REPORT:
  Section 1 — Technical Analysis
    • Interactive Plotly candlestick chart with MA lines (20/50/100/200)
    • Volume bars below the chart
    • Support (green dashed) and resistance (red dashed) horizontal lines
    • Detected patterns list

  Section 2 — Fundamental Analysis
    • Metrics table: each row has value, grade (color-coded A+→F), comment
    • Key strengths and concerns

  Section 3 — Sentiment Analysis
    • Top news article cards with title, source, date
    • Key themes chips
    • Upcoming catalysts list

  Section 4 — Final Analyst Report
    • Weighted score breakdown
    • Verdict badge (color = conviction)
    • Analyst thesis narrative
    • Bull case / Bear case side by side
    • What to watch list

OUTPUT:
  Saved to: reports/{ticker}_{date}.html
  Opens automatically in the default browser.
"""

import os
import re
import json
import webbrowser
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models.report import TechnicalReport, FundamentalReport, SentimentReport, FinalReport


def generate_html_report(
    ticker: str,
    tech_report: TechnicalReport,
    fund_report: FundamentalReport,
    sent_report: SentimentReport,
    final_report: FinalReport,
    market_data: dict | None,
    news_articles: list | None,
    model_name: str = "claude-opus-4-6",
) -> str:
    """
    Generate a complete HTML report and save it to disk.

    Args:
        ticker:        Stock symbol
        tech_report:   Technical Analysis output
        fund_report:   Fundamental Analysis output
        sent_report:   Sentiment Analysis output
        final_report:  Synthesized final verdict
        market_data:   Raw OHLCV data dict (from tools/market_data.py)
        news_articles: List of news article dicts (from tools/news_scraper.py)

    Returns:
        Path to the saved HTML file.
    """
    print(f"\n  [report_generator] Building HTML report...")

    # Ensure output directory exists
    os.makedirs("reports", exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"reports/{ticker}_{date_str}.html"

    # ── Build each section ─────────────────────────────────────────────
    chart_html = _build_chart(ticker, market_data, tech_report) if market_data else "<p>Chart data unavailable.</p>"
    fundamentals_html = _build_fundamentals_table(fund_report)
    sentiment_html = _build_sentiment_section(sent_report, news_articles or [])
    summary_html = _build_summary_section(final_report)
    scores_html = _build_score_cards(tech_report, fund_report, sent_report, final_report)

    # ── Assemble full page ─────────────────────────────────────────────
    html = _build_full_page(
        ticker=ticker,
        company_name=fund_report.company_name,
        final_report=final_report,
        scores_html=scores_html,
        chart_html=chart_html,
        tech_report=tech_report,
        fundamentals_html=fundamentals_html,
        sentiment_html=sentiment_html,
        summary_html=summary_html,
        date_str=date_str,
        fund_summary=fund_report.summary or "",
        sent_summary=sent_report.summary or "",
        model_name=model_name,
    )

    # ── Write file ─────────────────────────────────────────────────────
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)

    # Open in browser automatically
    abs_path = os.path.abspath(filename)
    webbrowser.open(f"file://{abs_path}")

    print(f"  [report_generator] Saved: {abs_path}")
    return abs_path


# ═══════════════════════════════════════════════════════════════════════
#  SECTION BUILDERS
# ═══════════════════════════════════════════════════════════════════════

def _build_chart(ticker: str, market_data: dict, tech: TechnicalReport) -> str:
    """
    Build a clean Micha Stocks style chart:
      - Candlesticks with clean dark background
      - EMA 20 (blue), EMA 50 (orange), SMA 150 (red — the main reference line)
      - Key swing high/low price labels (like Micha annotates pivots)
      - Max 1-2 horizontal S/R lines (white/gray, not colored)
      - Buy zone / Sell zone markers
      - Volume subplot with color-coded bars
      - Info overlay: ATR, Above/Below 150 MA

    NO: candlestick pattern labels, many colored horizontal lines
    """
    df: pd.DataFrame = market_data["df"].tail(120).copy()  # ~6 months

    # FIX: Plotly drops candlesticks when index has a timezone — strip it.
    df.index = df.index.tz_localize(None)

    close = df["Close"]
    support = tech.support_levels or []
    resistance = tech.resistance_levels or []

    # ── Build subplots: price (72%) + volume (28%) ─────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.72, 0.28],
        subplot_titles=(f"{ticker} — Daily", "Volume"),
    )

    # ── Candlestick ────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="OHLC",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a",
        decreasing_fillcolor="#ef5350",
    ), row=1, col=1)

    # ── Moving Averages — Micha style: EMA20, EMA50, SMA150 (main) ────
    # EMA 20 — thin blue line
    if len(df) >= 20:
        ema20 = close.ewm(span=20, adjust=False).mean()
        fig.add_trace(go.Scatter(
            x=df.index, y=ema20,
            name="EMA 20",
            line=dict(color="#2196F3", width=1.2),
            hovertemplate="EMA20: $%{y:.2f}<extra></extra>",
        ), row=1, col=1)

    # EMA 50 — thin orange line
    if len(df) >= 50:
        ema50 = close.ewm(span=50, adjust=False).mean()
        fig.add_trace(go.Scatter(
            x=df.index, y=ema50,
            name="EMA 50",
            line=dict(color="#FF9800", width=1.2),
            hovertemplate="EMA50: $%{y:.2f}<extra></extra>",
        ), row=1, col=1)

    # SMA 150 — THE main reference line (red, thicker — Micha's key MA)
    full_df = market_data["df"].copy()
    full_df.index = full_df.index.tz_localize(None)
    if len(full_df) >= 150:
        sma150_full = full_df["Close"].rolling(150).mean()
        # Only plot within our chart range
        sma150_plot = sma150_full.loc[df.index[0]:]
        fig.add_trace(go.Scatter(
            x=sma150_plot.index, y=sma150_plot,
            name="SMA 150",
            line=dict(color="#F44336", width=2.0),
            hovertemplate="SMA150: $%{y:.2f}<extra></extra>",
        ), row=1, col=1)

    # ── Key Support/Resistance — max 1-2 clean white horizontal lines ─
    # Only show the primary (nearest) support and resistance
    if support:
        level = support[0]
        fig.add_shape(
            type="line",
            x0=df.index[0], x1=df.index[-1],
            y0=level, y1=level,
            line=dict(color="rgba(255,255,255,0.5)", width=1, dash="solid"),
            layer="below", row=1, col=1,
        )
        fig.add_annotation(
            x=df.index[-1], y=level,
            text=f"${level:.2f}",
            xanchor="left", yanchor="middle",
            font=dict(color="rgba(255,255,255,0.7)", size=9),
            showarrow=False, xshift=5,
            row=1, col=1,
        )

    if resistance:
        level = resistance[0]
        fig.add_shape(
            type="line",
            x0=df.index[0], x1=df.index[-1],
            y0=level, y1=level,
            line=dict(color="rgba(255,255,255,0.5)", width=1, dash="solid"),
            layer="below", row=1, col=1,
        )
        fig.add_annotation(
            x=df.index[-1], y=level,
            text=f"${level:.2f}",
            xanchor="left", yanchor="middle",
            font=dict(color="rgba(255,255,255,0.7)", size=9),
            showarrow=False, xshift=5,
            row=1, col=1,
        )

    # ── Swing high/low price annotations (Micha style pivot labels) ───
    swing_annotations = market_data.get("swing_annotations", [])
    for ann in swing_annotations:
        try:
            date = ann["date"]
            if hasattr(date, 'tz_localize'):
                date = date.tz_localize(None)
            if date < df.index[0] or date > df.index[-1]:
                continue
            price = ann["price"]
            is_high = ann["type"] == "high"
            fig.add_annotation(
                x=date, y=price,
                text=f"{price:.2f}",
                xanchor="center",
                yanchor="bottom" if is_high else "top",
                yshift=8 if is_high else -8,
                font=dict(color="rgba(255,255,255,0.8)", size=9),
                showarrow=False,
                row=1, col=1,
            )
        except Exception:
            pass

    # ── Buy Zone marker ───────────────────────────────────────────────
    buy_zone = tech.buy_zone
    if buy_zone and buy_zone < float(df["High"].max()) and buy_zone > float(df["Low"].min()) * 0.8:
        fig.add_annotation(
            x=df.index[len(df) * 3 // 4],
            y=buy_zone,
            text=f"BUY ZONE ${buy_zone:.2f}",
            xanchor="center", yanchor="top",
            font=dict(color="#00E676", size=10, family="Segoe UI"),
            showarrow=True,
            arrowhead=3, arrowcolor="#00E676", arrowwidth=1.5,
            ax=0, ay=25,
            bgcolor="rgba(0,230,118,0.15)",
            bordercolor="#00E676", borderwidth=1, borderpad=4,
            row=1, col=1,
        )

    # ── Sell Zone marker ──────────────────────────────────────────────
    sell_zone = tech.sell_zone
    if sell_zone and sell_zone > float(df["Low"].min()) and sell_zone < float(df["High"].max()) * 1.2:
        fig.add_annotation(
            x=df.index[len(df) * 3 // 4],
            y=sell_zone,
            text=f"SELL ZONE ${sell_zone:.2f}",
            xanchor="center", yanchor="bottom",
            font=dict(color="#FF5252", size=10, family="Segoe UI"),
            showarrow=True,
            arrowhead=3, arrowcolor="#FF5252", arrowwidth=1.5,
            ax=0, ay=-25,
            bgcolor="rgba(255,82,82,0.15)",
            bordercolor="#FF5252", borderwidth=1, borderpad=4,
            row=1, col=1,
        )

    # ── Info overlay — top left (Micha style) ─────────────────────────
    atr = market_data.get("atr", {})
    ma_data = market_data.get("moving_averages", {})
    rsi = market_data.get("rsi", {})
    market_cap = market_data.get("market_cap")

    info_lines = [f"<b>{market_data.get('company_name', ticker)}</b>"]
    if market_cap:
        cap_str = f"${market_cap / 1e9:.2f}B" if market_cap >= 1e9 else f"${market_cap / 1e6:.0f}M"
        info_lines[0] += f" ({cap_str})"

    if atr.get("current"):
        atr_color = "#FF5252" if atr["pct"] > 5 else "#FFC107" if atr["pct"] > 3 else "#4CAF50"
        info_lines.append(f"ATR(14): {atr['current']:.2f} ({atr['pct']:.1f}%)")

    above_150 = ma_data.get("price_vs_ma150")
    if above_150:
        ma150_indicator = "Above 150 MA" if above_150 == "above" else "Below 150 MA"
        ma150_color = "#4CAF50" if above_150 == "above" else "#FF5252"
        info_lines.append(f"{ma150_indicator}")

    if rsi.get("current"):
        rsi_val = rsi["current"]
        rsi_label = f"RSI(14): {rsi_val:.1f}"
        if rsi_val >= 70:
            rsi_label += " (Overbought)"
        elif rsi_val <= 30:
            rsi_label += " (Oversold)"
        info_lines.append(rsi_label)

    vol_analysis = market_data.get("volume_analysis", {})
    if vol_analysis.get("signal") and vol_analysis["signal"] != "neutral":
        info_lines.append(f"Volume: {vol_analysis['signal'].title()}")

    fig.add_annotation(
        x=0.01, y=0.98,
        xref="paper", yref="paper",
        text="<br>".join(info_lines),
        xanchor="left", yanchor="top",
        font=dict(color="#e0e0e0", size=10, family="Segoe UI"),
        showarrow=False,
        bgcolor="rgba(0,0,0,0.6)",
        bordercolor="rgba(255,255,255,0.1)",
        borderwidth=1, borderpad=8,
        align="left",
    )

    # ── Volume bars ────────────────────────────────────────────────────
    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        name="Volume", marker_color=colors,
        opacity=0.7, showlegend=False,
    ), row=2, col=1)

    # ── 50-day average volume line on volume subplot ──────────────────
    if len(df) >= 50:
        avg_vol = df["Volume"].rolling(50).mean()
        fig.add_trace(go.Scatter(
            x=df.index, y=avg_vol,
            name="50d Avg Vol",
            line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash"),
            showlegend=False,
            hovertemplate="50d Avg: %{y:,.0f}<extra></extra>",
        ), row=2, col=1)

    # ── Layout — clean dark theme ─────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0", family="Segoe UI"),
        height=600,
        autosize=True,
        margin=dict(l=20, r=60, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0.3)",
            font=dict(size=10),
        ),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#2a2a4a", showgrid=True, tickformat="%b '%y")
    fig.update_yaxes(gridcolor="#2a2a4a", showgrid=True, tickprefix="$")

    return fig.to_html(full_html=False, include_plotlyjs='cdn', config={"responsive": True})


def _build_fundamentals_table(fund: FundamentalReport) -> str:
    """Build the color-coded metrics table for the Fundamental section."""

    grade_colors = {
        "A+": ("#1b5e20", "#a5d6a7"),  # dark-green bg, light-green text
        "A":  ("#1b5e20", "#c8e6c9"),
        "B":  ("#1a237e", "#bbdefb"),  # dark-blue bg, light-blue text
        "C":  ("#f57f17", "#fff176"),  # amber bg, yellow text
        "D":  ("#bf360c", "#ffccbc"),  # deep-orange bg
        "F":  ("#b71c1c", "#ef9a9a"),  # deep-red bg
    }

    def grade_cell(grade: str) -> str:
        bg, fg = grade_colors.get(grade, ("#333", "#fff"))
        return f'<span class="grade-badge" style="background:{bg};color:{fg}">{grade}</span>'

    def metric_row(label: str, metric, fmt_val: str = None) -> str:
        if metric is None:
            return ""
        val_str = fmt_val or (f"{metric.value:.2f}" if metric.value is not None else "N/A")
        return f"""
        <tr>
            <td class="metric-name">{label}</td>
            <td class="metric-value">{val_str}</td>
            <td>{grade_cell(metric.grade)}</td>
            <td class="metric-comment">{metric.comment}</td>
        </tr>"""

    # Format percentage values
    def pct(m):
        if m and m.value is not None:
            return f"{m.value * 100:.1f}%"
        return "N/A"

    def ratio(m):
        if m and m.value is not None:
            return f"{m.value:.2f}x"
        return "N/A"

    def billions(m):
        if m and m.value is not None:
            return f"${m.value / 1e9:.2f}B"
        return "N/A"

    rows = (
        metric_row("Revenue Growth YoY", fund.revenue_growth_yoy, pct(fund.revenue_growth_yoy)) +
        metric_row("Net Income Margin", fund.net_income_margin, pct(fund.net_income_margin)) +
        metric_row("P/E Ratio (trailing)", fund.pe_ratio, ratio(fund.pe_ratio)) +
        metric_row("EPS Growth", fund.eps_growth, pct(fund.eps_growth)) +
        metric_row("Return on Equity", fund.return_on_equity, pct(fund.return_on_equity)) +
        metric_row("Debt / Equity", fund.debt_to_equity, ratio(fund.debt_to_equity)) +
        metric_row("Free Cash Flow", fund.free_cash_flow, billions(fund.free_cash_flow))
    )

    strengths = "".join(f'<li>{s}</li>' for s in fund.key_strengths)
    concerns = "".join(f'<li>{c}</li>' for c in fund.key_concerns)

    return f"""
    <div class="table-wrapper">
      <table class="metrics-table">
        <thead>
          <tr>
            <th>Metric</th><th>Value</th><th>Grade</th><th>Assessment</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    <div class="two-col">
      <div class="card green-card">
        <h4>Key Strengths</h4>
        <ul>{strengths or "<li>—</li>"}</ul>
      </div>
      <div class="card red-card">
        <h4>Key Concerns</h4>
        <ul>{concerns or "<li>—</li>"}</ul>
      </div>
    </div>"""


def _build_sentiment_section(sent: SentimentReport, articles: list) -> str:
    """Build news article cards and themes for the Sentiment section."""

    # ── Sentiment gauge bar ────────────────────────────────────────────
    score = sent.sentiment_score  # -1 to +1
    pct = int((score + 1) / 2 * 100)  # Convert to 0-100%
    bar_color = "#4CAF50" if score > 0.2 else ("#F44336" if score < -0.2 else "#FF9800")
    gauge = f"""
    <div class="sentiment-gauge">
      <div class="gauge-labels">
        <span>Very Bearish</span>
        <span>Neutral</span>
        <span>Very Bullish</span>
      </div>
      <div class="gauge-track">
        <div class="gauge-fill" style="width:{pct}%;background:{bar_color}"></div>
        <div class="gauge-marker" style="left:{pct}%"></div>
      </div>
      <div class="gauge-value" style="color:{bar_color}">
        {sent.overall_sentiment} ({score:+.2f})
      </div>
    </div>"""

    # ── News article cards (top 5 with titles as "quotes") ────────────
    cards = ""
    shown = [a for a in articles if a.get("title") and len(a["title"]) > 15][:5]
    for article in shown:
        url = article.get("url", "")
        # Wrap the headline in a link if a URL is available
        if url:
            headline_html = (
                f'<a href="{url}" target="_blank" rel="noopener noreferrer">'
                f'"{article["title"]}"'
                f'</a>'
            )
        else:
            headline_html = f'"{article["title"]}"'

        cards += f"""
        <div class="news-card">
          <div class="news-quote">{headline_html}</div>
          <div class="news-meta">
            <span class="news-source">{article.get('source', 'Unknown')}</span>
            <span class="news-date">{article.get('date', '')}</span>
          </div>
        </div>"""

    if not cards:
        cards = "<p class='dim'>No news articles available for display.</p>"

    # ── Themes ────────────────────────────────────────────────────────
    themes = "".join(
        f'<span class="chip">{t}</span>' for t in sent.key_themes[:6]
    )

    # ── Catalysts ─────────────────────────────────────────────────────
    catalysts = "".join(f"<li>{c}</li>" for c in sent.upcoming_catalysts[:4])
    risks_mentioned = "".join(f"<li>{r}</li>" for r in sent.risks_mentioned[:4])

    return f"""
    {gauge}
    <div class="themes-row">
      <h4>Key Themes</h4>
      <div class="chips">{themes or "<span class='dim'>None identified</span>"}</div>
    </div>
    <h4 style="margin-top:1.5rem">Recent Headlines</h4>
    <div class="news-grid">{cards}</div>
    <div class="two-col" style="margin-top:1.5rem">
      <div class="card">
        <h4>Upcoming Catalysts</h4>
        <ul>{catalysts or "<li>None identified</li>"}</ul>
      </div>
      <div class="card">
        <h4>Risks Mentioned</h4>
        <ul>{risks_mentioned or "<li>None identified</li>"}</ul>
      </div>
    </div>"""


def _build_summary_section(final: FinalReport) -> str:
    """Build the final analyst thesis section."""

    watch = "".join(f"<li>{w}</li>" for w in final.watch_for[:5])

    return f"""
    <div class="thesis-box">
      <h4>Analyst Thesis</h4>
      <p>{final.analyst_thesis or "—"}</p>
    </div>
    <div class="two-col">
      <div class="card green-card">
        <h4>Bull Case</h4>
        <p>{final.bull_case or "—"}</p>
      </div>
      <div class="card red-card">
        <h4>Bear Case</h4>
        <p>{final.bear_case or "—"}</p>
      </div>
    </div>
    <div class="card" style="margin-top:1rem">
      <h4>What to Watch</h4>
      <ul>{watch or "<li>—</li>"}</ul>
    </div>"""


def _build_score_cards(tech, fund, sent, final) -> str:
    """Build the three score cards + composite score."""

    def score_bar(score, color):
        pct = int(score / 10 * 100)
        return f'<div class="score-bar-track"><div class="score-bar-fill" style="width:{pct}%;background:{color}"></div></div>'

    return f"""
    <div class="score-cards">
      <div class="score-card">
        <div class="sc-label">Technical</div>
        <div class="sc-score" style="color:#2196F3">{tech.score:.1f}<span>/10</span></div>
        {score_bar(tech.score, "#2196F3")}
        <div class="sc-sub">35% weight</div>
      </div>
      <div class="score-card">
        <div class="sc-label">Fundamental</div>
        <div class="sc-score" style="color:#9C27B0">{fund.score:.1f}<span>/10</span></div>
        {score_bar(fund.score, "#9C27B0")}
        <div class="sc-sub">45% weight</div>
      </div>
      <div class="score-card">
        <div class="sc-label">Sentiment</div>
        <div class="sc-score" style="color:#FF9800">{sent.score:.1f}<span>/10</span></div>
        {score_bar(sent.score, "#FF9800")}
        <div class="sc-sub">20% weight</div>
      </div>
      <div class="score-card composite">
        <div class="sc-label">Composite</div>
        <div class="sc-score" style="color:#00E5FF">{final.composite_score:.1f}<span>/10</span></div>
        {score_bar(final.composite_score, "#00E5FF")}
        <div class="sc-sub">Weighted average</div>
      </div>
    </div>"""


# ═══════════════════════════════════════════════════════════════════════
#  FULL PAGE ASSEMBLER
# ═══════════════════════════════════════════════════════════════════════

def _build_full_page(
    ticker, company_name, final_report, scores_html,
    chart_html, tech_report, fundamentals_html,
    sentiment_html, summary_html, date_str,
    fund_summary="", sent_summary="", model_name="claude-opus-4-6",
) -> str:
    """Assemble all sections into a complete HTML page."""

    verdict = final_report.verdict or "HOLD"
    verdict_colors = {
        "STRONG BUY":  "#00c853",
        "BUY":         "#69f0ae",
        "HOLD":        "#ffd740",
        "SELL":        "#ff6d00",
        "STRONG SELL": "#d50000",
    }
    verdict_color = verdict_colors.get(verdict, "#90a4ae")
    confidence = final_report.confidence_pct or 0
    price_target = final_report.price_target or "N/A"
    horizon = final_report.time_horizon or "N/A"

    ma = tech_report.moving_averages

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{ticker} — Stock Analysis Report</title>
  <!-- Plotly is bundled inline with the chart via include_plotlyjs='cdn' -->
  <style>
    :root {{
      --bg:       #0f0f1a;
      --surface:  #1a1a2e;
      --surface2: #16213e;
      --border:   #2a2a4a;
      --text:     #e0e0e0;
      --dim:      #888;
      --accent:   #00E5FF;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; font-size: 15px; line-height: 1.6; }}
    a {{ color: var(--accent); }}

    /* ── Header ── */
    .header {{ background: linear-gradient(135deg, #0d1b2a 0%, #1a1a3e 100%); padding: 2rem; border-bottom: 1px solid var(--border); }}
    .header-top {{ display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem; }}
    .company-info h1 {{ font-size: 2rem; font-weight: 700; color: #fff; }}
    .company-info h2 {{ font-size: 1rem; color: var(--dim); font-weight: 400; margin-top: 0.25rem; }}
    .verdict-block {{ text-align: right; }}
    .verdict-badge {{
      display: inline-block;
      padding: 0.6rem 1.5rem;
      border-radius: 8px;
      font-size: 1.3rem;
      font-weight: 800;
      letter-spacing: 1px;
      color: #000;
      background: {verdict_color};
      box-shadow: 0 0 20px {verdict_color}66;
    }}
    .verdict-meta {{ margin-top: 0.5rem; color: var(--dim); font-size: 0.85rem; }}
    .price-tag {{ font-size: 1.4rem; color: #fff; font-weight: 600; margin-top: 0.5rem; }}

    /* ── Score cards ── */
    .score-cards {{ display: flex; gap: 1rem; flex-wrap: wrap; margin: 1.5rem 0; }}
    .score-card {{
      flex: 1; min-width: 140px;
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1rem;
      text-align: center;
    }}
    .score-card.composite {{ border-color: #00E5FF44; background: #001a2244; }}
    .sc-label {{ font-size: 0.75rem; color: var(--dim); text-transform: uppercase; letter-spacing: 1px; }}
    .sc-score {{ font-size: 2rem; font-weight: 800; margin: 0.3rem 0; }}
    .sc-score span {{ font-size: 0.9rem; color: var(--dim); font-weight: 400; }}
    .sc-sub {{ font-size: 0.7rem; color: var(--dim); margin-top: 0.3rem; }}
    .score-bar-track {{ background: var(--border); border-radius: 4px; height: 6px; margin: 0.3rem 0; overflow: hidden; }}
    .score-bar-fill {{ height: 100%; border-radius: 4px; transition: width 1s ease; }}

    /* ── Main layout ── */
    .container {{ max-width: 1200px; margin: 0 auto; padding: 2rem; }}
    .section {{ margin-bottom: 2.5rem; }}
    .section-title {{
      font-size: 1.1rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 2px;
      color: var(--accent);
      border-bottom: 1px solid var(--border);
      padding-bottom: 0.5rem;
      margin-bottom: 1.2rem;
    }}
    .section-title .num {{ background: var(--accent); color: #000; padding: 0.1rem 0.5rem; border-radius: 4px; margin-right: 0.5rem; font-size: 0.85rem; }}

    /* ── Two-column layout ── */
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; }}
    @media(max-width: 700px) {{ .two-col {{ grid-template-columns: 1fr; }} }}

    /* ── Cards ── */
    .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.2rem; }}
    .card h4 {{ font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; color: var(--dim); margin-bottom: 0.7rem; }}
    .card ul {{ padding-left: 1.2rem; }}
    .card li {{ margin-bottom: 0.4rem; font-size: 0.9rem; }}
    .card p {{ font-size: 0.9rem; line-height: 1.7; }}
    .green-card {{ border-color: #2e7d3244; background: #1b5e2011; }}
    .green-card h4 {{ color: #66bb6a; }}
    .red-card {{ border-color: #b71c1c44; background: #7f00001a; }}
    .red-card h4 {{ color: #ef5350; }}

    /* ── Fundamentals table ── */
    .table-wrapper {{ overflow-x: auto; }}
    .metrics-table {{ width: 100%; border-collapse: collapse; margin-bottom: 1rem; }}
    .metrics-table th {{ background: var(--surface2); color: var(--dim); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; padding: 0.6rem 0.8rem; text-align: left; border-bottom: 1px solid var(--border); }}
    .metrics-table td {{ padding: 0.65rem 0.8rem; border-bottom: 1px solid var(--border); font-size: 0.9rem; }}
    .metrics-table tr:hover td {{ background: var(--surface2); }}
    .metric-name {{ font-weight: 600; color: #ccc; }}
    .metric-value {{ font-family: 'Courier New', monospace; color: var(--accent); }}
    .metric-comment {{ color: var(--dim); font-size: 0.82rem; }}
    .grade-badge {{ display: inline-block; padding: 0.15rem 0.55rem; border-radius: 4px; font-weight: 700; font-size: 0.8rem; }}

    /* ── Technical analysis ── */
    .tech-meta {{ display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; }}
    .cross-badge {{ display: inline-block; padding: 0.2rem 0.7rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }}
    .cross-badge.green {{ background: rgba(76,175,80,0.2); color: #4CAF50; border: 1px solid #4CAF5044; }}
    .cross-badge.red {{ background: rgba(244,67,54,0.2); color: #F44336; border: 1px solid #F4433644; }}
    .trend-badge {{ display: inline-block; padding: 0.2rem 0.7rem; border-radius: 20px; font-size: 0.8rem; color: var(--accent); background: rgba(0,229,255,0.08); border: 1px solid rgba(0,229,255,0.2); }}

    /* ── Sentiment ── */
    .sentiment-gauge {{ margin: 1rem 0 1.5rem; }}
    .gauge-labels {{ display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--dim); margin-bottom: 0.3rem; }}
    .gauge-track {{ position: relative; background: var(--border); border-radius: 6px; height: 12px; overflow: visible; }}
    .gauge-fill {{ height: 100%; border-radius: 6px; transition: width 1s ease; }}
    .gauge-marker {{ position: absolute; top: -4px; width: 4px; height: 20px; background: #fff; border-radius: 2px; transform: translateX(-50%); box-shadow: 0 0 6px rgba(255,255,255,0.6); }}
    .gauge-value {{ text-align: center; margin-top: 0.5rem; font-weight: 700; font-size: 1rem; }}
    .themes-row {{ margin-top: 1rem; }}
    .themes-row h4 {{ font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; color: var(--dim); margin-bottom: 0.5rem; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 0.5rem; }}
    .chip {{ background: var(--surface2); border: 1px solid var(--border); border-radius: 20px; padding: 0.2rem 0.8rem; font-size: 0.8rem; color: var(--accent); }}
    .news-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 1rem; margin-top: 0.5rem; }}
    .news-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; border-left: 3px solid var(--accent); }}
    .news-quote {{ font-size: 0.88rem; color: #ccc; line-height: 1.5; font-style: italic; margin-bottom: 0.6rem; }}
    .news-meta {{ display: flex; justify-content: space-between; font-size: 0.75rem; }}
    .news-source {{ color: var(--accent); font-weight: 600; }}
    .news-date {{ color: var(--dim); }}

    /* ── Analyst thesis ── */
    .thesis-box {{ background: linear-gradient(135deg, #0d2137, #12122e); border: 1px solid #00E5FF33; border-radius: 10px; padding: 1.5rem; margin-bottom: 1rem; }}
    .thesis-box h4 {{ color: var(--accent); margin-bottom: 0.8rem; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; }}
    .thesis-box p {{ font-size: 0.95rem; line-height: 1.8; color: #d0d0d0; }}

    /* ── Footer ── */
    .footer {{ text-align: center; color: var(--dim); font-size: 0.78rem; padding: 2rem; border-top: 1px solid var(--border); margin-top: 2rem; }}

    .dim {{ color: var(--dim); font-size: 0.85rem; }}

    /* ══ MOBILE RESPONSIVE ══════════════════════════════════════════ */

    /* Tablets & large phones (landscape) */
    @media (max-width: 768px) {{
      body {{ font-size: 14px; }}

      /* Header */
      .header {{ padding: 1.2rem 1rem; }}
      .header-top {{ flex-direction: column; gap: 0.8rem; }}
      .company-info h1 {{ font-size: 1.4rem; }}
      .company-info h2 {{ font-size: 0.8rem; }}
      .price-tag {{ font-size: 1.1rem; }}
      .verdict-block {{ text-align: left; }}
      .verdict-badge {{ font-size: 1.1rem; padding: 0.4rem 1.1rem; }}

      /* Score cards — 2×2 grid on tablets */
      .score-cards {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.6rem;
      }}
      .score-card {{ min-width: 0; padding: 0.8rem; }}
      .sc-score {{ font-size: 1.6rem; }}

      /* Content */
      .container {{ padding: 1rem; }}
      .section {{ margin-bottom: 1.8rem; }}

      /* News grid — single column */
      .news-grid {{ grid-template-columns: 1fr; }}

      /* Tech layout already collapses at 900px */
    }}

    /* Small phones (portrait) */
    @media (max-width: 480px) {{
      body {{ font-size: 13px; }}

      .header {{ padding: 1rem; }}
      .company-info h1 {{ font-size: 1.15rem; }}
      .company-info h2 {{ font-size: 0.75rem; }}
      .price-tag {{ font-size: 1rem; }}
      .verdict-badge {{ font-size: 0.95rem; padding: 0.35rem 0.9rem; letter-spacing: 0.5px; }}
      .verdict-meta {{ font-size: 0.75rem; }}

      /* Score cards stack 2×2 */
      .score-cards {{ gap: 0.4rem; }}
      .score-card {{ padding: 0.6rem 0.5rem; }}
      .sc-score {{ font-size: 1.3rem; }}
      .sc-label {{ font-size: 0.65rem; }}
      .sc-sub {{ font-size: 0.6rem; }}

      /* Section titles */
      .section-title {{ font-size: 0.85rem; letter-spacing: 1px; }}

      /* Container */
      .container {{ padding: 0.75rem; }}

      /* Cards */
      .card {{ padding: 0.75rem; }}
      .card h4 {{ font-size: 0.75rem; }}

      /* Reduce chart height on small phones via CSS */
      .chart-area > div:first-child > div {{ min-height: 300px !important; }}

      /* Tables: force scroll on very small screens */
      .table-wrapper {{
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
      }}
      .metrics-table {{ min-width: 500px; }}

      /* Gauge labels */
      .gauge-labels {{ font-size: 0.65rem; }}
      .gauge-value {{ font-size: 0.9rem; }}

      /* News cards */
      .news-card {{ padding: 0.75rem; }}
      .news-quote {{ font-size: 0.82rem; }}

      /* Thesis */
      .thesis-box {{ padding: 1rem; }}
      .thesis-box p {{ font-size: 0.88rem; }}

      /* Footer */
      .footer {{ padding: 1.2rem 0.75rem; font-size: 0.72rem; }}
    }}

    /* Prevent overflow on all screen sizes */
    img, table, iframe, video {{ max-width: 100%; }}
    pre {{ white-space: pre-wrap; word-break: break-word; }}
  </style>
</head>
<body>

<!-- ══ HEADER ══════════════════════════════════════════════════════ -->
<div class="header">
  <div class="header-top">
    <div class="company-info">
      <h1>{company_name}</h1>
      <h2>{ticker} &nbsp;·&nbsp; Generated {date_str} &nbsp;·&nbsp; Model: {model_name}</h2>
      <div class="price-tag">Current Price: ${final_report.current_price:.2f}
        {f'&nbsp;→&nbsp; Target: {price_target}' if price_target != 'N/A' else ''}
      </div>
    </div>
    <div class="verdict-block">
      <div class="verdict-badge">{verdict}</div>
      <div class="verdict-meta">
        Confidence: {confidence:.0f}% &nbsp;|&nbsp; Horizon: {horizon}
      </div>
    </div>
  </div>

  <!-- Score cards in header -->
  {scores_html}
</div>

<!-- ══ MAIN CONTENT ════════════════════════════════════════════════ -->
<div class="container">

  <!-- ── Section 1: Technical Analysis ── -->
  <div class="section">
    <div class="section-title"><span class="num">1</span>Technical Analysis</div>
    <div class="chart-area">
      {chart_html}
    </div>
    <div class="tech-meta" style="margin-top:0.8rem">
      {'<span class="cross-badge green">Golden Cross</span>' if ma.golden_cross else ''}
      {'<span class="cross-badge red">Death Cross</span>' if ma.death_cross else ''}
      <span class="trend-badge">Trend: {tech_report.trend} · {tech_report.trend_strength}</span>
      <span class="trend-badge">RSI: {tech_report.rsi_value or 'N/A'}{' · ' + tech_report.rsi_condition.replace('_', ' ').title() if tech_report.rsi_condition else ''}</span>
      <span class="trend-badge">Volume: {tech_report.volume_signal.title() if tech_report.volume_signal else 'N/A'}</span>
      {f'<span class="cross-badge red">RSI Divergence: {tech_report.rsi_divergence}</span>' if tech_report.rsi_divergence else ''}
    </div>
    <div class="card" style="margin-top:0.6rem">
      <p style="font-size:0.9rem">{tech_report.summary} {tech_report.short_term_outlook}</p>
      {f'<p style="font-size:0.85rem;color:#00E676;margin-top:0.5rem"><b>Action:</b> {tech_report.action_reason}</p>' if tech_report.action_reason else ''}
    </div>
  </div>

  <!-- ── Section 2: Fundamental Analysis ── -->
  <div class="section">
    <div class="section-title"><span class="num">2</span>Fundamental Analysis</div>
    {fundamentals_html}
    <div class="card" style="margin-top:1rem">
      <p style="font-size:0.9rem">{fund_summary}</p>
    </div>
  </div>

  <!-- ── Section 3: Sentiment Analysis ── -->
  <div class="section">
    <div class="section-title"><span class="num">3</span>Market Sentiment & News</div>
    {sentiment_html}
    <div class="card" style="margin-top:1rem">
      <p style="font-size:0.9rem">{sent_summary}</p>
    </div>
  </div>

  <!-- ── Section 4: Analyst Report ── -->
  <div class="section">
    <div class="section-title"><span class="num">4</span>Analyst Report</div>
    {summary_html}
  </div>

</div>

<!-- ══ FOOTER ════════════════════════════════════════════════════ -->
<div class="footer">
  Beat the ASPP &nbsp;·&nbsp; AI-Powered Stock Evaluator &nbsp;·&nbsp;
  Powered by {model_name} &nbsp;·&nbsp; Data: Yahoo Finance &nbsp;·&nbsp;
  {date_str} &nbsp;·&nbsp;
  <strong>Not financial advice. For educational purposes only.</strong>
</div>

</body>
</html>"""
