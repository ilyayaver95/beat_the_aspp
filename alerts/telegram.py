"""
alerts/telegram.py
==================
Telegram alert integration via Bot API (free, no third-party SDK needed).

SETUP (takes 2 minutes):
  1. Open Telegram → search @BotFather → /newbot → follow prompts
  2. Copy the bot token (looks like: 123456789:ABCdefGHIjklMNOpqrsTUVwxyz)
  3. Start a chat with your new bot (click the link BotFather gives you)
  4. Send any message to the bot (e.g. "hello")
  5. Get your chat_id:
       Open in browser: https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
       Look for: "chat":{"id":XXXXXXX  ← that number is your chat_id
  6. Add to your .env:
       TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
       TELEGRAM_CHAT_ID=your_chat_id
"""

import os
import requests


TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def _resolve(token: str | None, chat_id: str | None) -> tuple[str | None, str | None]:
    """Caller-supplied creds take precedence; fall back to env vars."""
    tok = (token or "").strip() or os.getenv("TELEGRAM_BOT_TOKEN")
    cid = (chat_id or "").strip() or os.getenv("TELEGRAM_CHAT_ID")
    return tok or None, cid or None


def is_configured(token: str | None = None, chat_id: str | None = None) -> bool:
    """Check if a working Telegram token + chat id is available."""
    tok, cid = _resolve(token, chat_id)
    return bool(tok and cid)


def get_config_status(token: str | None = None, chat_id: str | None = None) -> dict:
    """Return which credentials are set (for debugging in the UI)."""
    tok, cid = _resolve(token, chat_id)
    return {
        "TELEGRAM_BOT_TOKEN": bool(tok),
        "TELEGRAM_CHAT_ID": bool(cid),
    }


def send_alert(
    message: str,
    token: str | None = None,
    chat_id: str | None = None,
) -> dict:
    """
    Send a Telegram message via Bot API.

    Per-user credentials passed via `token` / `chat_id` take precedence over
    the TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID env vars (used by CLI scripts).

    Returns:
        {"success": True/False, "error": "..." if failed}
    """
    tok, cid = _resolve(token, chat_id)

    if not tok or not cid:
        missing = [k for k, v in get_config_status(token, chat_id).items() if not v]
        return {
            "success": False,
            "error": f"Telegram not configured. Missing: {', '.join(missing)}",
        }

    try:
        resp = requests.post(
            TELEGRAM_API.format(token=tok),
            json={
                "chat_id": cid,
                "text": message,
                "parse_mode": "HTML",
            },
            timeout=10,
        )
        data = resp.json()

        if data.get("ok"):
            return {"success": True, "message_id": data["result"]["message_id"]}
        else:
            return {"success": False, "error": data.get("description", "Unknown Telegram error")}

    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot reach Telegram API. Check your internet connection."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def send_zone_alert(
    scan_result,
    token: str | None = None,
    chat_id: str | None = None,
) -> dict:
    """
    Send a formatted buy/sell zone alert for a ScanResult.
    Uses HTML formatting for Telegram (bold, italic).
    """
    r = scan_result

    if r.below_support:
        header = f"🔴 <b>{r.ticker} BELOW SUPPORT</b>"
    elif r.in_buy_zone:
        header = f"🟢 <b>{r.ticker} IN BUY ZONE</b>"
    elif r.above_resistance:
        header = f"🟣 <b>{r.ticker} ABOVE RESISTANCE</b>"
    elif r.in_sell_zone:
        header = f"🟡 <b>{r.ticker} IN SELL ZONE</b>"
    else:
        header = f"📊 <b>{r.ticker} ZONE ALERT</b>"

    lines = [header, ""]
    lines.append(f"💰 Price: <b>${r.current_price:.2f}</b>")

    if r.primary_support:
        lines.append(f"🟢 Buy Zone: ${r.primary_support:.2f} ({r.distance_to_support_pct:+.1f}%)")
    if r.primary_resistance:
        lines.append(f"🔴 Sell Zone: ${r.primary_resistance:.2f} ({r.distance_to_resistance_pct:+.1f}%)")

    lines.append("")
    lines.append(f"📋 Verdict: <b>{r.verdict}</b>")
    if r.composite_score:
        lines.append(f"⭐ Score: {r.composite_score:.1f}/10")
    if r.price_target:
        lines.append(f"🎯 Target: {r.price_target}")

    lines.append("")
    lines.append(f"📅 Analysis: {r.analysis_date}")
    lines.append("📈 <i>Beat the ASPP Scanner</i>")

    return send_alert("\n".join(lines), token=token, chat_id=chat_id)
