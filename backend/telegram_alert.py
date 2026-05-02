"""Telegram alert sender for RoadSense."""
import io, requests

TELEGRAM_BOT_TOKEN = "8411770719:AAG_8gLd4378mJDwtv4_k7nGS2xGIJLgVKU"
TELEGRAM_CHAT_ID = "5559724831"
ALERTS_ENABLED = True
ALERT_VERDICTS = {"SEVERE", "DAMAGED", "ABANDONED", "MIXED"}

_API_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


def _format_caption(title, lat, lon, verdict, confidence=None, extra_info=None):
    lines = []
    lines.append(f"\U0001F6A8  {title}")
    lines.append(f"\U0001F4CD  Location: {lat:.6f}, {lon:.6f}")
    lines.append(f"\U0001F4CA  Verdict: {verdict}")
    if confidence is not None:
        lines.append(f"\U0001F3AF  Confidence: {confidence:.2f}")
    if extra_info:
        for k, v in extra_info.items():
            if isinstance(v, float):
                lines.append(f"    - {k}: {v:.3f}")
            else:
                lines.append(f"    - {k}: {v}")
    maps_url = f"https://maps.google.com/?q={lat:.6f},{lon:.6f}"
    lines.append(f"\U0001F5FA  Map: {maps_url}")
    lines.append("")
    lines.append("Sent by RoadSense - VIT Vellore capstone")
    return "\n".join(lines)


def send_alert(title, lat, lon, verdict, confidence=None,
               image_bytes=None, extra_info=None, timeout=10.0):
    if not ALERTS_ENABLED:
        return False
    if verdict not in ALERT_VERDICTS:
        return False

    caption = _format_caption(title, lat, lon, verdict, confidence, extra_info)

    try:
        if image_bytes:
            url = f"{_API_BASE}/sendPhoto"
            files = {"photo": ("tile.png", io.BytesIO(image_bytes), "image/png")}
            data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
            r = requests.post(url, data=data, files=files, timeout=timeout)
        else:
            url = f"{_API_BASE}/sendMessage"
            data = {"chat_id": TELEGRAM_CHAT_ID, "text": caption,
                    "disable_web_page_preview": False}
            r = requests.post(url, data=data, timeout=timeout)

        if r.status_code == 200 and r.json().get("ok"):
            return True
        print(f"[telegram] Alert failed: HTTP {r.status_code} -- {r.text[:200]}")
        return False
    except Exception as e:
        print(f"[telegram] Alert exception: {e}")
        return False


if __name__ == "__main__":
    print("[telegram] Sending test alert...")
    ok = send_alert(
        title="TEST: RoadSense Alert System",
        lat=12.9692,
        lon=79.1559,
        verdict="SEVERE",
        confidence=0.78,
        extra_info={"damage_score": 0.78, "linearity": 0.45},
    )
    print("[telegram] OK" if ok else "[telegram] FAILED")