"""
cost_tracker.py
===============
Tracks LLM API token usage and associated costs.

HOW IT WORKS:
  1. Agents call set_context(ticker, operation) at the start of each run.
     This stores the context in a thread-local variable so parallel agents
     don't overwrite each other's context.
  2. AnthropicLLMClient wraps its .messages with TrackedMessages.
  3. TrackedMessages intercepts parse() / stream() calls and records usage
     after each response is received from the Anthropic SDK.
  4. Records are appended to data/usage_log.jsonl (one JSON per line).
  5. app.py reads session / 24h / total stats from the tracker singleton.

PRICING (Claude API, USD per 1 million tokens, as of 2025):
  claude-opus-4-6:    $15 input / $75 output / $18.75 cache-write / $1.50 cache-read
  claude-sonnet-4-6:   $3 input / $15 output / $3.75  cache-write / $0.30 cache-read
"""

import json
import os
import threading
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

# ── Pricing table (USD per 1M tokens) ─────────────────────────────────────
_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-6": {
        "input":       15.00,
        "output":      75.00,
        "cache_write": 18.75,
        "cache_read":   1.50,
    },
    "claude-sonnet-4-6": {
        "input":        3.00,
        "output":      15.00,
        "cache_write":  3.75,
        "cache_read":   0.30,
    },
    "claude-haiku-4-5-20251001": {
        "input":        0.80,
        "output":        4.00,
        "cache_write":   1.00,
        "cache_read":    0.08,
    },
}
# Fallback — assume Opus pricing for unknown models (conservative / safe)
_DEFAULT_RATES = _PRICING["claude-opus-4-6"]

USAGE_LOG_FILE = "data/usage_log.jsonl"

# ── Thread-local context ───────────────────────────────────────────────────
_ctx = threading.local()


def set_context(ticker: str, operation: str) -> None:
    """
    Tag subsequent API calls on this thread with the given ticker and operation.
    Call this at the start of each agent function before any LLM call.

    Example:
        set_context("AAPL", "technical_agent")
    """
    _ctx.ticker = ticker
    _ctx.operation = operation


def _get_context() -> tuple[str, str]:
    return getattr(_ctx, "ticker", "unknown"), getattr(_ctx, "operation", "unknown")


# ── Cost calculation ───────────────────────────────────────────────────────

def compute_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Return the USD cost for one API call. Rounded to 6 decimal places."""
    rates = _PRICING.get(model, _DEFAULT_RATES)
    return round(
        input_tokens            * rates["input"]       / 1_000_000
        + output_tokens         * rates["output"]      / 1_000_000
        + cache_creation_tokens * rates["cache_write"] / 1_000_000
        + cache_read_tokens     * rates["cache_read"]  / 1_000_000,
        6,
    )


# ── UsageTracker ───────────────────────────────────────────────────────────

class UsageTracker:
    """
    Thread-safe singleton that records every Anthropic API call.

    Two layers of storage:
      - In-memory accumulators  → instant session stats (no disk I/O)
      - Persistent JSONL file   → 24h and all-time history across restarts
    """

    def __init__(self) -> None:
        self.session_id: str = str(uuid.uuid4())[:8]
        self._lock = threading.Lock()
        # Session accumulators (reset every time the Python process restarts)
        self._s_input: int = 0
        self._s_output: int = 0
        self._s_cache_write: int = 0
        self._s_cache_read: int = 0
        self._s_cost: float = 0.0
        self._s_calls: int = 0
        os.makedirs("data", exist_ok=True)

    # ── Public: recording ──────────────────────────────────────────────

    def record(
        self,
        ticker: str,
        operation: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> float:
        """
        Persist one API call record and update session accumulators.
        Returns the USD cost for that call.
        """
        cost = compute_cost(
            model, input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens
        )
        entry = {
            "ts":  datetime.now(timezone.utc).isoformat(),
            "sid": self.session_id,
            "tkr": ticker,
            "op":  operation,
            "mdl": model,
            "in":  input_tokens,
            "out": output_tokens,
            "cw":  cache_creation_tokens,
            "cr":  cache_read_tokens,
            "$":   cost,
        }
        with self._lock:
            with open(USAGE_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            self._s_input       += input_tokens
            self._s_output      += output_tokens
            self._s_cache_write += cache_creation_tokens
            self._s_cache_read  += cache_read_tokens
            self._s_cost        += cost
            self._s_calls       += 1
        return cost

    # ── Public: querying ───────────────────────────────────────────────

    def get_session_stats(self) -> dict:
        """Return in-memory stats for the current process session (fast, no I/O)."""
        with self._lock:
            return {
                "calls":       self._s_calls,
                "input":       self._s_input,
                "output":      self._s_output,
                "cache_write": self._s_cache_write,
                "cache_read":  self._s_cache_read,
                "total":       self._s_input + self._s_output + self._s_cache_write + self._s_cache_read,
                "cost":        round(self._s_cost, 4),
            }

    def get_24h_stats(self) -> dict:
        """Return aggregated stats for the past 24 hours (reads JSONL file)."""
        since = datetime.now(timezone.utc) - timedelta(hours=24)
        return self._aggregate(self._load(since_ts=since))

    def get_total_stats(self) -> dict:
        """Return aggregated stats for all time (reads JSONL file)."""
        return self._aggregate(self._load())

    # ── Private helpers ────────────────────────────────────────────────

    def _load(self, since_ts: Optional[datetime] = None) -> list[dict]:
        if not os.path.exists(USAGE_LOG_FILE):
            return []
        records = []
        try:
            with open(USAGE_LOG_FILE, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                        if since_ts:
                            ts = datetime.fromisoformat(r["ts"])
                            if ts.tzinfo is None:
                                ts = ts.replace(tzinfo=timezone.utc)
                            if ts < since_ts:
                                continue
                        records.append(r)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except OSError:
            pass
        return records

    @staticmethod
    def _aggregate(records: list[dict]) -> dict:
        inp  = sum(r.get("in",  0) for r in records)
        out  = sum(r.get("out", 0) for r in records)
        cw   = sum(r.get("cw",  0) for r in records)
        cr   = sum(r.get("cr",  0) for r in records)
        cost = sum(r.get("$",   0.0) for r in records)
        return {
            "calls":       len(records),
            "input":       inp,
            "output":      out,
            "cache_write": cw,
            "cache_read":  cr,
            "total":       inp + out + cw + cr,
            "cost":        round(cost, 4),
        }


# ── Module-level singleton (shared across all imports) ─────────────────────
tracker = UsageTracker()


# ── TrackedMessages — wraps Anthropic messages interface ───────────────────

class _TrackedStream:
    """
    Wraps Anthropic's MessageStreamManager context manager.
    Captures token usage from get_final_message() on context exit.
    """

    def __init__(self, mgr, tracker_ref: UsageTracker, model: str) -> None:
        self._mgr = mgr
        self._tracker = tracker_ref
        self._model = model
        self._stream = None

    def __enter__(self):
        self._stream = self._mgr.__enter__()
        return self

    def __iter__(self):
        return iter(self._stream)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Capture usage before tearing down the stream
        try:
            msg = self._stream.get_final_message()
            if msg and getattr(msg, "usage", None):
                u = msg.usage
                ticker, op = _get_context()
                self._tracker.record(
                    ticker=ticker,
                    operation=op,
                    model=self._model,
                    input_tokens=getattr(u, "input_tokens", 0) or 0,
                    output_tokens=getattr(u, "output_tokens", 0) or 0,
                    cache_creation_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
                    cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                )
        except Exception:
            pass  # Never let tracking errors break the analysis
        return self._mgr.__exit__(exc_type, exc_val, exc_tb)

    def __getattr__(self, name):
        """Forward unknown attributes to the real MessageStream."""
        return getattr(self._stream, name)


class TrackedMessages:
    """
    Drop-in wrapper for anthropic.resources.Messages.
    Intercepts parse() and stream() to record token usage via the tracker.

    COST OPTIMIZATION — Automatic Prompt Caching:
      Any string `system=` prompt is automatically converted to a cacheable
      content block (cache_control: ephemeral). This is transparent to all agents.

      How caching saves money:
        First call:  system tokens billed at cache_write rate (+25% vs normal)
        Next calls (within 5 min): system tokens billed at cache_read rate (-90%)

      For repeated analyses (common in a trading session), this cuts system-prompt
      costs by ~90%. The actual $ saving depends on system prompt size (~200-400
      tokens per agent), but every bit counts on long sessions.

    Usage:
        client.messages = TrackedMessages(client.messages, tracker)
    """

    def __init__(self, real_messages, tracker_ref: UsageTracker) -> None:
        self._real = real_messages
        self._tracker = tracker_ref

    @staticmethod
    def _inject_cache_control(kwargs: dict) -> dict:
        """
        Convert a plain string system prompt to a cacheable content block.
        Leaves list-style system prompts (already formatted) unchanged.
        """
        system = kwargs.get("system")
        if isinstance(system, str) and system:
            kwargs = dict(kwargs)  # shallow copy — don't mutate caller's dict
            kwargs["system"] = [{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }]
        return kwargs

    def parse(self, model: str = "claude-opus-4-6", **kwargs):
        kwargs = self._inject_cache_control(kwargs)
        result = self._real.parse(model=model, **kwargs)
        try:
            usage = getattr(result, "usage", None)
            if usage:
                ticker, op = _get_context()
                self._tracker.record(
                    ticker=ticker,
                    operation=op,
                    model=model,
                    input_tokens=getattr(usage, "input_tokens", 0) or 0,
                    output_tokens=getattr(usage, "output_tokens", 0) or 0,
                    cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
                    cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                )
        except Exception:
            pass
        return result

    def stream(self, model: str = "claude-opus-4-6", **kwargs):
        kwargs = self._inject_cache_control(kwargs)
        mgr = self._real.stream(model=model, **kwargs)
        return _TrackedStream(mgr, self._tracker, model)

    def __getattr__(self, name):
        """Forward .create() and any other methods to the real messages object."""
        return getattr(self._real, name)
