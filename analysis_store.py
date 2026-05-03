"""
analysis_store.py
=================
Persistence layer for analysis results.

Saves and loads structured analysis data (TechnicalReport + FinalReport)
as JSON so the scanner can read support levels and buy zones later.

Storage: data/analyses/{TICKER}_latest.json
"""

import json
import os
from datetime import datetime
from typing import Optional

from models.report import TechnicalReport, FundamentalReport, SentimentReport, FinalReport

ANALYSES_DIR = "data/analyses"


def save_analysis(
    ticker: str,
    tech_report: TechnicalReport,
    fund_report: FundamentalReport,
    sent_report: SentimentReport,
    final_report: FinalReport,
) -> str:
    """
    Save all agent reports as a single JSON file for later scanning.

    Returns:
        Path to the saved JSON file.
    """
    os.makedirs(ANALYSES_DIR, exist_ok=True)

    data = {
        "ticker": ticker,
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "technical": tech_report.model_dump(mode="json"),
        "fundamental": fund_report.model_dump(mode="json"),
        "sentiment": sent_report.model_dump(mode="json"),
        "final": final_report.model_dump(mode="json"),
    }

    path = os.path.join(ANALYSES_DIR, f"{ticker}_latest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    return path


def load_analysis(ticker: str) -> Optional[dict]:
    """
    Load the latest saved analysis for a ticker.

    Returns:
        Dict with keys: ticker, analysis_date, technical, fundamental,
        sentiment, final — or None if no analysis exists.
    """
    path = os.path.join(ANALYSES_DIR, f"{ticker}_latest.json")
    if not os.path.exists(path):
        return None

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_analysis_age_days(ticker: str) -> Optional[float]:
    """
    Return how many days old the latest analysis is, or None if none exists.
    """
    data = load_analysis(ticker)
    if not data or "analysis_date" not in data:
        return None

    try:
        analysis_dt = datetime.strptime(data["analysis_date"], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        # Fallback for date-only format
        analysis_dt = datetime.strptime(data["analysis_date"][:10], "%Y-%m-%d")

    age = datetime.now() - analysis_dt
    return age.total_seconds() / 86400


def list_all_analyses() -> list[dict]:
    """Return a summary of all saved analyses."""
    if not os.path.exists(ANALYSES_DIR):
        return []

    results = []
    for filename in os.listdir(ANALYSES_DIR):
        if filename.endswith("_latest.json"):
            ticker = filename.replace("_latest.json", "")
            data = load_analysis(ticker)
            if data:
                results.append({
                    "ticker": ticker,
                    "analysis_date": data.get("analysis_date", "Unknown"),
                    "verdict": data.get("final", {}).get("verdict", "N/A"),
                    "composite_score": data.get("final", {}).get("composite_score"),
                    "support_levels": data.get("technical", {}).get("support_levels", []),
                    "current_price": data.get("final", {}).get("current_price"),
                })
    return results
