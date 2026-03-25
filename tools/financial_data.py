"""
tools/financial_data.py
=======================
Fetches fundamental financial data from Yahoo Finance using yfinance.

This is the data source for Agent 2 (Fundamental Analysis).

DATA SOURCES (all via yfinance — no API key required):
  - ticker.financials          → Income Statement (annual)
  - ticker.quarterly_financials → Income Statement (quarterly)
  - ticker.balance_sheet       → Balance Sheet (annual)
  - ticker.cashflow            → Cash Flow Statement (annual)
  - ticker.info                → Key ratios, market cap, P/E, etc.

GOLDEN STANDARDS used for grading (based on CFA/value investing principles):
  - Revenue Growth:    >15% = excellent, >8% = good, >0% = OK, <0% = concern
  - Net Margin:        >20% = excellent, >10% = good, >5% = acceptable
  - P/E Ratio:         <15 = value, <25 = fair, <40 = growth, >40 = expensive
  - EPS Growth:        >20% = excellent, >10% = good
  - ROE:               >20% = excellent, >15% = good, >10% = acceptable
  - Debt/Equity:       <0.5 = conservative, <1.0 = moderate, >2.0 = high risk
  - Free Cash Flow:    Positive = good, growing = excellent
"""

import pandas as pd
import numpy as np
from typing import Optional
import yfinance as yf


def get_financial_data(ticker: str) -> dict:
    """
    Fetch all fundamental data needed for the Fundamental Analysis Agent.

    Args:
        ticker: Stock ticker symbol (e.g., "DRS")

    Returns:
        Dictionary with all financial metrics, raw DataFrames, and
        a formatted text summary ready to paste into a Claude prompt.
    """
    print(f"  [financial_data] Fetching fundamentals for {ticker}...")

    tk = yf.Ticker(ticker)

    # ── Key Ratios & Company Info ──────────────────────────────────────
    info = tk.info
    company_name = info.get("longName", ticker)

    # ── Income Statement ───────────────────────────────────────────────
    income_stmt = tk.financials  # Annual, most recent year first
    quarterly_income = tk.quarterly_financials

    # ── Balance Sheet ──────────────────────────────────────────────────
    balance = tk.balance_sheet

    # ── Cash Flow Statement ────────────────────────────────────────────
    cashflow = tk.cashflow

    # ── Compute Key Metrics ────────────────────────────────────────────
    metrics = _compute_metrics(info, income_stmt, balance, cashflow)

    # ── Format for LLM ────────────────────────────────────────────────
    formatted_text = _format_for_llm(
        ticker, company_name, info, income_stmt, balance, cashflow, metrics
    )

    print(f"  [financial_data] Fundamentals ready for {company_name}")

    return {
        "company_name": company_name,
        "ticker": ticker,
        "info": info,
        "income_stmt": income_stmt,
        "balance_sheet": balance,
        "cashflow": cashflow,
        "metrics": metrics,
        "formatted_text": formatted_text,
    }


def _safe_get(df: pd.DataFrame, row_key: str, col_idx: int = 0) -> Optional[float]:
    """
    Safely extract a value from a yfinance financial DataFrame.

    yfinance DataFrames use row = metric name, column = fiscal year.
    This helper handles missing rows/columns gracefully.

    Args:
        df: yfinance financial DataFrame
        row_key: Exact row label (e.g., "Total Revenue")
        col_idx: Column index (0 = most recent year)

    Returns:
        Float value or None if not available.
    """
    try:
        if row_key in df.index and col_idx < len(df.columns):
            val = df.loc[row_key].iloc[col_idx]
            if pd.notna(val):
                return float(val)
    except Exception:
        pass
    return None


def _compute_metrics(
    info: dict,
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
) -> dict:
    """
    Compute key fundamental metrics from raw financial data.

    All values are extracted from yfinance DataFrames using safe lookup.
    Percentages are expressed as floats (e.g., 0.15 = 15%).

    Returns:
        Dict with all computed metrics as floats or None.
    """
    m = {}

    # ── Revenue ──────────────────────────────────────────────────────
    # income.columns: [most_recent_year, previous_year, ...]
    rev_current = _safe_get(income, "Total Revenue", 0)
    rev_prior    = _safe_get(income, "Total Revenue", 1)

    m["revenue_current"] = rev_current
    m["revenue_prior"] = rev_prior
    m["revenue_growth_yoy"] = (
        (rev_current - rev_prior) / abs(rev_prior)
        if rev_current and rev_prior and rev_prior != 0
        else None
    )

    # ── Net Income & Margins ──────────────────────────────────────────
    net_income = _safe_get(income, "Net Income", 0)
    net_income_prior = _safe_get(income, "Net Income", 1)
    m["net_income"] = net_income
    m["net_income_growth"] = (
        (net_income - net_income_prior) / abs(net_income_prior)
        if net_income and net_income_prior and net_income_prior != 0
        else None
    )
    m["net_margin"] = (
        net_income / rev_current
        if net_income and rev_current and rev_current != 0
        else None
    )

    # ── Gross Margin ──────────────────────────────────────────────────
    gross_profit = _safe_get(income, "Gross Profit", 0)
    m["gross_margin"] = (
        gross_profit / rev_current
        if gross_profit and rev_current and rev_current != 0
        else None
    )

    # ── EBITDA ────────────────────────────────────────────────────────
    m["ebitda"] = _safe_get(income, "EBITDA", 0) or info.get("ebitda")

    # ── EPS (from yfinance info — more reliable) ──────────────────────
    m["eps_trailing"] = info.get("trailingEps")
    m["eps_forward"]  = info.get("forwardEps")
    m["eps_growth"] = (
        (m["eps_forward"] - m["eps_trailing"]) / abs(m["eps_trailing"])
        if m["eps_forward"] and m["eps_trailing"] and m["eps_trailing"] != 0
        else None
    )

    # ── Valuation ─────────────────────────────────────────────────────
    m["pe_trailing"] = info.get("trailingPE")
    m["pe_forward"]  = info.get("forwardPE")
    m["price_to_book"] = info.get("priceToBook")
    m["price_to_sales"] = info.get("priceToSalesTrailing12Months")
    m["ev_to_ebitda"] = info.get("enterpriseToEbitda")
    m["market_cap"] = info.get("marketCap")

    # ── Balance Sheet — Debt & Equity ─────────────────────────────────
    total_debt = (
        _safe_get(balance, "Long Term Debt", 0) or
        _safe_get(balance, "Total Debt", 0) or
        info.get("totalDebt")
    )
    total_equity = (
        _safe_get(balance, "Total Stockholder Equity", 0) or
        _safe_get(balance, "Stockholders Equity", 0) or
        info.get("bookValue")
    )
    m["total_debt"] = total_debt
    m["total_equity"] = total_equity
    m["debt_to_equity"] = (
        total_debt / total_equity
        if total_debt is not None and total_equity and total_equity != 0
        else info.get("debtToEquity")
    )
    if m["debt_to_equity"] and m["debt_to_equity"] > 10:
        # yfinance sometimes returns D/E * 100; normalize
        m["debt_to_equity"] = m["debt_to_equity"] / 100

    # ── Return on Equity (ROE) ────────────────────────────────────────
    m["roe"] = (
        net_income / total_equity
        if net_income and total_equity and total_equity != 0
        else info.get("returnOnEquity")
    )

    # ── Return on Assets (ROA) ────────────────────────────────────────
    total_assets = _safe_get(balance, "Total Assets", 0)
    m["roa"] = (
        net_income / total_assets
        if net_income and total_assets and total_assets != 0
        else info.get("returnOnAssets")
    )

    # ── Free Cash Flow ────────────────────────────────────────────────
    operating_cf = (
        _safe_get(cashflow, "Operating Cash Flow", 0) or
        _safe_get(cashflow, "Total Cash From Operating Activities", 0)
    )
    capex = (
        _safe_get(cashflow, "Capital Expenditure", 0) or
        _safe_get(cashflow, "Capital Expenditures", 0)
    )
    if capex and capex > 0:
        capex = -capex  # capex is typically negative in yfinance; normalize

    m["operating_cashflow"] = operating_cf
    m["capex"] = capex
    m["free_cash_flow"] = (
        operating_cf + capex  # capex is negative, so this is OCF - |capex|
        if operating_cf is not None and capex is not None
        else info.get("freeCashflow")
    )

    # ── Dividend ──────────────────────────────────────────────────────
    m["dividend_yield"] = info.get("dividendYield")
    m["payout_ratio"] = info.get("payoutRatio")

    # ── Growth (analyst estimates from info) ─────────────────────────
    m["revenue_growth_estimate"] = info.get("revenueGrowth")
    m["earnings_growth_estimate"] = info.get("earningsGrowth")

    return m


def _format_for_llm(
    ticker: str,
    company_name: str,
    info: dict,
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
    metrics: dict,
) -> str:
    """
    Format all financial data into a structured text block for Claude.

    WHY FORMAT THIS WAY:
      Claude reads text, not DataFrames. By pre-formatting the data
      into a clean, labeled table, we:
        1. Reduce tokens (vs. dumping raw JSON)
        2. Make it easier for Claude to find specific numbers
        3. Highlight the most important metrics upfront
        4. Include our own golden standard benchmarks for context

    Returns:
        Multi-section text string ready to paste into a prompt.
    """

    def fmt_b(val):
        """Format value in billions with 2 decimal places."""
        if val is None:
            return "N/A"
        return f"${val / 1e9:.2f}B"

    def fmt_m(val):
        """Format value in millions."""
        if val is None:
            return "N/A"
        return f"${val / 1e6:.1f}M"

    def fmt_pct(val):
        """Format as percentage."""
        if val is None:
            return "N/A"
        return f"{val * 100:.1f}%"

    def fmt_ratio(val, decimals=2):
        """Format as plain number."""
        if val is None:
            return "N/A"
        return f"{val:.{decimals}f}x"

    lines = [
        f"=== FUNDAMENTAL DATA: {company_name} ({ticker}) ===",
        "",
        f"Sector:    {info.get('sector', 'N/A')}",
        f"Industry:  {info.get('industry', 'N/A')}",
        f"Market Cap: {fmt_b(metrics.get('market_cap'))}",
        f"Current Price: ${info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))}",
        "",
        "─── INCOME STATEMENT (Annual) ───────────────────────────",
        f"Revenue (TTM):       {fmt_b(metrics.get('revenue_current'))}",
        f"Revenue (Prior Yr):  {fmt_b(metrics.get('revenue_prior'))}",
        f"Revenue Growth YoY:  {fmt_pct(metrics.get('revenue_growth_yoy'))}  "
        f"[Benchmark: >15% = excellent, >8% = good]",
        "",
        f"Net Income:          {fmt_b(metrics.get('net_income'))}",
        f"Net Margin:          {fmt_pct(metrics.get('net_margin'))}  "
        f"[Benchmark: >20% = excellent, >10% = good]",
        f"Gross Margin:        {fmt_pct(metrics.get('gross_margin'))}",
        f"EBITDA:              {fmt_b(metrics.get('ebitda'))}",
        "",
        "─── EARNINGS PER SHARE ──────────────────────────────────",
        f"EPS (Trailing):      ${metrics.get('eps_trailing', 'N/A')}",
        f"EPS (Forward):       ${metrics.get('eps_forward', 'N/A')}",
        f"EPS Growth (est.):   {fmt_pct(metrics.get('eps_growth'))}  "
        f"[Benchmark: >20% = excellent, >10% = good]",
        "",
        "─── VALUATION ───────────────────────────────────────────",
        f"P/E (Trailing):      {fmt_ratio(metrics.get('pe_trailing'))}  "
        f"[Benchmark: <15 = value, <25 = fair, >40 = expensive]",
        f"P/E (Forward):       {fmt_ratio(metrics.get('pe_forward'))}",
        f"Price/Book:          {fmt_ratio(metrics.get('price_to_book'))}",
        f"Price/Sales:         {fmt_ratio(metrics.get('price_to_sales'))}",
        f"EV/EBITDA:           {fmt_ratio(metrics.get('ev_to_ebitda'))}",
        "",
        "─── BALANCE SHEET HEALTH ────────────────────────────────",
        f"Total Debt:          {fmt_b(metrics.get('total_debt'))}",
        f"Total Equity:        {fmt_b(metrics.get('total_equity'))}",
        f"Debt/Equity:         {fmt_ratio(metrics.get('debt_to_equity'))}  "
        f"[Benchmark: <0.5 = conservative, <1.0 = moderate, >2.0 = risky]",
        "",
        "─── PROFITABILITY ───────────────────────────────────────",
        f"ROE (Return on Equity): {fmt_pct(metrics.get('roe'))}  "
        f"[Benchmark: >20% = excellent, >15% = good]",
        f"ROA (Return on Assets): {fmt_pct(metrics.get('roa'))}",
        "",
        "─── CASH FLOW ───────────────────────────────────────────",
        f"Operating Cash Flow: {fmt_b(metrics.get('operating_cashflow'))}",
        f"CapEx:               {fmt_b(metrics.get('capex'))}",
        f"Free Cash Flow:      {fmt_b(metrics.get('free_cash_flow'))}  "
        f"[Benchmark: Positive = healthy, growing = excellent]",
        "",
        "─── GROWTH ESTIMATES ─────────────────────────────────────",
        f"Revenue Growth (est.):  {fmt_pct(metrics.get('revenue_growth_estimate'))}",
        f"Earnings Growth (est.): {fmt_pct(metrics.get('earnings_growth_estimate'))}",
        "",
        "─── DIVIDEND ────────────────────────────────────────────",
        f"Dividend Yield: {fmt_pct(metrics.get('dividend_yield'))}",
        f"Payout Ratio:   {fmt_pct(metrics.get('payout_ratio'))}",
    ]

    return "\n".join(lines)
