"""
analysis_store.py
=================
Saves structured analysis results to JSON files for later review.
"""

import json
import os
from datetime import datetime

from models.report import TechnicalReport, FundamentalReport, SentimentReport, FinalReport


def save_analysis(
    ticker: str,
    tech_report: TechnicalReport,
    fund_report: FundamentalReport,
    sent_report: SentimentReport,
    final_report: FinalReport,
) -> str:
    """Save all agent reports to a JSON file. Returns the file path."""
    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reports/{ticker}_{timestamp}.json"

    data = {
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "technical": tech_report.model_dump(),
        "fundamental": fund_report.model_dump(),
        "sentiment": sent_report.model_dump(),
        "final": final_report.model_dump(),
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return filename
