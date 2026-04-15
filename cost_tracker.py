"""
cost_tracker.py
===============
Simple context tracker for the analysis pipeline.
Can be extended later for actual API cost monitoring.
"""

_current_context = {
    "ticker": None,
    "step": None,
}


def set_context(ticker: str, step: str) -> None:
    """Set the current analysis context."""
    _current_context["ticker"] = ticker
    _current_context["step"] = step


def get_context() -> dict:
    return _current_context.copy()
