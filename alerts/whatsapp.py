"""
alerts/whatsapp.py
==================
WhatsApp alert integration via Twilio API.

SETUP:
  1. Create a free Twilio account: https://www.twilio.com/try-twilio
  2. Enable the WhatsApp Sandbox: https://console.twilio.com/us1/develop/sms/try-it-out/whatsapp-learn
  3. Follow the sandbox instructions (send "join <your-sandbox-word>" from your phone)
  4. Add these to your .env file:
       TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
       TWILIO_AUTH_TOKEN=your_auth_token
       TWILIO_WHATSAPP_FROM=whatsapp:+14155238886    (Twilio sandbox number)
       ALERT_PHONE_NUMBER=whatsapp:+1YOURNUMBER      (your phone in whatsapp:+XXX format)

  5. Install twilio:  pip install twilio

USAGE:
  from alerts.whatsapp import send_alert, is_configured
  if is_configured():
      send_alert("AAPL is in the buy zone! $150.23 near support $148.00")
"""

import os
from typing import Optional


def is_configured() -> bool:
    """Check if all required Twilio WhatsApp environment variables are set."""
    required = [
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "TWILIO_WHATSAPP_FROM",
        "ALERT_PHONE_NUMBER",
    ]
    return all(os.getenv(var) for var in required)


def get_config_status() -> dict:
    """Return which env vars are set (for debugging)."""
    vars_to_check = [
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "TWILIO_WHATSAPP_FROM",
        "ALERT_PHONE_NUMBER",
    ]
    return {var: bool(os.getenv(var)) for var in vars_to_check}


def send_alert(message: str) -> dict:
    """
    Send a WhatsApp alert message via Twilio.

    Args:
        message: The alert text to send.

    Returns:
        Dict with status info:
          - success (bool)
          - sid (str, Twilio message SID if successful)
          - error (str, if failed)
    """
    if not is_configured():
        missing = [k for k, v in get_config_status().items() if not v]
        return {
            "success": False,
            "error": f"WhatsApp not configured. Missing: {', '.join(missing)}",
        }

    try:
        from twilio.rest import Client

        client = Client(
            os.getenv("TWILIO_ACCOUNT_SID"),
            os.getenv("TWILIO_AUTH_TOKEN"),
        )

        msg = client.messages.create(
            from_=os.getenv("TWILIO_WHATSAPP_FROM"),
            to=os.getenv("ALERT_PHONE_NUMBER"),
            body=message,
        )

        return {"success": True, "sid": msg.sid}

    except ImportError:
        return {
            "success": False,
            "error": "twilio package not installed. Run: pip install twilio",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def send_buy_zone_alert(scan_result) -> dict:
    """
    Send a formatted zone alert (buy or sell) for a ScanResult.

    Args:
        scan_result: A ScanResult object from scanner.py

    Returns:
        Dict with send status.
    """
    r = scan_result

    # Pick header based on zone type
    if r.below_support:
        header = f"🔴 {r.ticker} BELOW SUPPORT"
    elif r.in_buy_zone:
        header = f"🟢 {r.ticker} IN BUY ZONE"
    elif r.above_resistance:
        header = f"🟣 {r.ticker} ABOVE RESISTANCE"
    elif r.in_sell_zone:
        header = f"🟡 {r.ticker} IN SELL ZONE"
    else:
        header = f"📊 {r.ticker} ZONE ALERT"

    lines = [
        header,
        "━━━━━━━━━━━━━━━━━",
        f"Price: ${r.current_price:.2f}",
    ]

    if r.primary_support:
        lines.append(f"Buy Zone (Support): ${r.primary_support:.2f} ({r.distance_to_support_pct:+.1f}%)")
    if r.primary_resistance:
        lines.append(f"Sell Zone (Resistance): ${r.primary_resistance:.2f} ({r.distance_to_resistance_pct:+.1f}%)")

    lines += [
        "━━━━━━━━━━━━━━━━━",
        f"Verdict: {r.verdict}",
        f"Score:   {r.composite_score:.1f}/10",
        f"Target:  {r.price_target or 'N/A'}",
        "━━━━━━━━━━━━━━━━━",
        f"Analysis: {r.analysis_date}",
        "📈 Beat the ASPP Scanner",
    ]

    return send_alert("\n".join(lines))
