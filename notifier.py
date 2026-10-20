"""Xavfli korxonalarga ogohlantirish yuborish: xabar shakllantirish, SMS (Eskiz), email (SMTP), jurnal.

Sukut bo'yicha hech narsa yuborilmaydi (SIMULYATSIYA). Haqiqiy yuborish uchun kalitlar kerak:
ESKIZ_EMAIL, ESKIZ_PASSWORD (SMS) yoki SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD (email).
Kalitlarni KODGA yozmang: Streamlit Cloud'da Settings -> Secrets, kompyuterda muhit o'zgaruvchisi sifatida bering.
"""
import datetime as dt
import json
import os
import smtplib
import urllib.parse
import urllib.request
from email.message import EmailMessage

import pandas as pd

from scoring import LABELS, RECOMMENDATIONS, WEIGHTS

CONFIG_KEYS = ["ESKIZ_EMAIL", "ESKIZ_PASSWORD", "ESKIZ_SENDER", "SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD"]


def load_config(streamlit_secrets=None) -> dict:
    cfg = {}
    for k in CONFIG_KEYS:
        v = os.environ.get(k)
        if v is None and streamlit_secrets is not None:
            try:
                v = streamlit_secrets.get(k)
            except Exception:
                v = None
        if v:
            cfg[k] = str(v)
    return cfg


def channel_ready(channel: str, cfg: dict) -> bool:
    if channel == "SMS":
        return "ESKIZ_EMAIL" in cfg and "ESKIZ_PASSWORD" in cfg
    return all(k in cfg for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD"))


def weak_reasons(r: pd.Series, top=2):
    """Eng ko'p ball yo'qotilgan ko'rsatkichlar (sabablar)."""
    keys = [k for k in WEIGHTS if r[f"w_{k}"] > 0]
    lost = sorted(keys, key=lambda k: r[f"w_{k}"] * (100 - r[f"sub_{k}"]), reverse=True)
    return [k for k in lost[:top] if r[f"sub_{k}"] < 70]


def build_sms(r: pd.Series) -> str:
    reasons = ", ".join(LABELS[k].lower() for k in weak_reasons(r)) or "umumiy ko'rsatkichlar"
    debt = f"{r['tax_debt']:,.1f}".replace(",", " ")
    if r["risk_class"] == "Yuqori xavf":
        return (f"Hurmatli rahbar! {r['name']} (STIR {r['stir']}) soliq intizomi reytingida YUQORI XAVF guruhida "
                f"(ball {r['final_score']:.1f}/100). Soliq qarzi: {debt} mln so'm. Asosiy sabab: {reasons}. "
                f"Qarzni bartaraf etish uchun soliq inspeksiyasiga murojaat qiling.")
    return (f"Hurmatli rahbar! {r['name']} (STIR {r['stir']}) soliq intizomi reytingi {r['final_score']:.1f}/100 "
            f"(o'rtacha xavf). E'tibor bering: {reasons}. Soliq to'lovlarini o'z vaqtida amalga oshiring.")


def build_email(r: pd.Series):
    subject = f"Soliq intizomi bo'yicha ogohlantirish: {r['name']} (STIR {r['stir']})"
    lines = [f"Hurmatli rahbar!", "",
             f"{r['name']} (STIR {r['stir']}) korxonasi soliq intizomi va moliyaviy ishonchlilik reytingi bo'yicha "
             f"quyidagi natijaga ega:", "",
             f"  Yakuniy ball: {r['final_score']:.1f} / 100", f"  Xavf sinfi: {r['risk_class']}",
             f"  Soliq qarzi: {r['tax_debt']:,.1f} mln so'm",
             f"  Oxirgi 12 oyda kechikkan to'lovlar: {int(r['late_payments_count'])}", "",
             "Reytingga eng ko'p ta'sir qilgan omillar va tavsiyalar:"]
    for k in weak_reasons(r, 3):
        lines.append(f"  - {LABELS[k]}: {RECOMMENDATIONS[k]}")
    lines += ["", "Savollar bo'lsa, hududingizdagi soliq inspeksiyasiga murojaat qiling.", "",
              "Hurmat bilan,", "Tax Scoring axborot tizimi (o'quv prototipi)"]
    return subject, "\n".join(lines)


def sms_segments(text: str) -> int:
    n = len(text)
    return 1 if n <= 160 else -(-n // 153)


# ------------------------- yuborish funksiyalari -------------------------
def _eskiz_token(cfg):
    data = urllib.parse.urlencode({"email": cfg["ESKIZ_EMAIL"], "password": cfg["ESKIZ_PASSWORD"]}).encode()
    resp = json.load(urllib.request.urlopen(urllib.request.Request("https://notify.eskiz.uz/api/auth/login", data=data), timeout=15))
    return resp["data"]["token"]


def send_sms(phone: str, text: str, cfg: dict, token=None) -> str:
    data = urllib.parse.urlencode({"mobile_phone": phone.lstrip("+"), "message": text,
                                   "from": cfg.get("ESKIZ_SENDER", "4546")}).encode()
    req = urllib.request.Request("https://notify.eskiz.uz/api/message/sms/send", data=data,
                                 headers={"Authorization": f"Bearer {token}"})
    urllib.request.urlopen(req, timeout=15).read()
    return "Yuborildi"


def send_email(to: str, subject: str, body: str, cfg: dict) -> str:
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, cfg["SMTP_USER"], to
    msg.set_content(body)
    with smtplib.SMTP_SSL(cfg["SMTP_HOST"], int(cfg.get("SMTP_PORT", 465)), timeout=20) as s:
        s.login(cfg["SMTP_USER"], cfg["SMTP_PASSWORD"])
        s.send_message(msg)
    return "Yuborildi"


def dispatch(rows: pd.DataFrame, channel: str, cfg: dict, live: bool, override_to: str = "") -> list:
    """Har bir korxona uchun xabar yuboradi (yoki simulyatsiya qiladi) va jurnal yozuvlarini qaytaradi."""
    log, token = [], None
    for _, r in rows.iterrows():
        to = override_to.strip() or (r["phone"] if channel == "SMS" else r["email"])
        text = build_sms(r) if channel == "SMS" else build_email(r)[1]
        if not live:
            status = "Simulyatsiya (yuborilmadi)"
        elif not channel_ready(channel, cfg):
            status = "Xato: kanal sozlanmagan"
        else:
            try:
                if channel == "SMS":
                    token = token or _eskiz_token(cfg)
                    status = send_sms(to, text, cfg, token)
                else:
                    status = send_email(to, build_email(r)[0], text, cfg)
            except Exception as e:  # tarmoq, avtorizatsiya va h.k.
                status = f"Xato: {str(e)[:80]}"
        log.append({"Vaqt": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "STIR": r["stir"], "Korxona": r["name"],
                    "Kanal": channel, "Qabul qiluvchi": to, "Xavf sinfi": r["risk_class"],
                    "Ball": round(float(r["final_score"]), 1), "Holat": status, "Xabar": text})
    return log
