"""Tax Scoring Telegram boti (faqat standart kutubxona: urllib, long polling).

Ishga tushirish:
    set TELEGRAM_BOT_TOKEN=123456:ABC...        (Linux/Mac: export ...)
    set ADMIN_IDS=11111111,22222222             (ixtiyoriy; o'z ID ingizni botdan /id bilan bilib oling)
    python bot.py
"""
import html
import json
import os
import random
import re
import sys
import time
import urllib.request
import uuid
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

import auth

from ml_model import explain_local, load_model
from report import passport_pdf
from scoring import LABELS, WEIGHTS, recommendations

RAW_COLS = ["stir", "name", "region", "sector", "legal_form", "years_active", "employees", "revenue",
            "net_profit", "total_assets", "equity", "current_assets", "current_liabilities", "tax_accrued",
            "tax_debt", "late_payments_count", "avg_delay_days", "late_reports_count", "audit_violations", "vat_payer",
            "has_inventory", "inv_opening_value", "inv_purchases", "inv_closing_value"]
EMOJI = {"Past xavf": "🟢", "O'rtacha xavf": "🟡", "Yuqori xavf": "🔴"}
SESSION_TTL = 12 * 3600  # botdagi kirish 12 soatdan keyin tugaydi
PENDING_TTL = 300
RATE_LIMIT = (15, 60)  # 60 soniyada 15 tadan ortiq so'rov emas

COMMANDS = [("start", "Boshlash"), ("kirish", "Tizimga kirish (STIR va parol)"), ("mening", "Mening korxonam"),
            ("chiqish", "Tizimdan chiqish"), ("yordam", "Yordam"), ("id", "Mening Telegram ID'im")]

def esc(s):
    return html.escape(str(s), quote=False)  # faqat & < > ekranlanadi (Telegram HTML uchun yetarli)


def card(r: pd.Series) -> str:
    L = [f"{EMOJI[r['risk_class']]} <b>{esc(r['name'])}</b>",
         f"STIR: <code>{r['stir']}</code> · {esc(r['region'])} · {esc(r['sector'])}", "",
         f"Yakuniy ball: <b>{r['final_score']:.1f} / 100</b>",
         f"Xavf sinfi: <b>{r['risk_class']}</b>",
         f"Qarzdorlik ehtimoli: {r['pd']:.1%}",
         f"Soliq qarzi: {r['tax_debt']:,.1f} mln so'm",
         f"Kechikkan to'lovlar (12 oy): {int(r['late_payments_count'])}"]
    if r["has_inventory"] == 1:
        L.append(f"Aylanma mosligi (tovar balansi bo'yicha): {r['turnover_gap']:+.1%}")
    keys = [k for k in WEIGHTS if r[f"w_{k}"] > 0]
    weak = [k for k in sorted(keys, key=lambda k: r[f"w_{k}"] * (100 - r[f"sub_{k}"]), reverse=True)[:3]
            if r[f"sub_{k}"] < 70]
    if weak:
        L += ["", "<b>Eng zaif tomonlar:</b>"] + [f"• {LABELS[k]}: {r[f'sub_{k}']:.0f}/100" for k in weak]
    recs = recommendations(r)[:2]
    if recs:
        L += [""] + [f"💡 <b>{esc(k)}:</b> {esc(t)}" for k, t in recs]
    return "\n".join(L)


class Bot:
    def __init__(self, token, df, bundle, admins=(), state_path="data/bot_state.json"):
        self.token, self.df, self.bundle = token, df.reset_index(drop=True), bundle
        self.admins = {str(a) for a in admins}
        self.by_stir = {s: i for i, s in enumerate(self.df["stir"])}
        self.state_path = Path(state_path)
        self.state = {"sessions": {}, "alerted": []}
        if self.state_path.exists():
            try:
                self.state = json.loads(self.state_path.read_text())
            except Exception:
                pass
        self.state.setdefault("sessions", {})
        self.state.setdefault("alerted", [])
        self.pending = {}
        self.hits = defaultdict(deque)

    # ---------------- Telegram transporti ----------------
    def api(self, method, **params):
        req = urllib.request.Request(f"https://api.telegram.org/bot{self.token}/{method}",
                                     data=json.dumps(params).encode(), headers={"Content-Type": "application/json"})
        resp = json.load(urllib.request.urlopen(req, timeout=45))
        if not resp.get("ok"):
            raise RuntimeError(resp)
        return resp["result"]

    def send(self, chat, text, buttons=None):
        p = dict(chat_id=chat, text=text, parse_mode="HTML", disable_web_page_preview=True)
        if buttons:
            p["reply_markup"] = {"inline_keyboard": buttons}
        return self.api("sendMessage", **p)

    def send_document(self, chat, filename, content: bytes, caption=""):
        b = uuid.uuid4().hex
        parts = []
        for k, v in (("chat_id", str(chat)), ("caption", caption)):
            parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
        parts.append((f'--{b}\r\nContent-Disposition: form-data; name="document"; filename="{filename}"\r\n'
                      f'Content-Type: application/pdf\r\n\r\n').encode() + content + b"\r\n")
        parts.append(f"--{b}--\r\n".encode())
        req = urllib.request.Request(f"https://api.telegram.org/bot{self.token}/sendDocument", data=b"".join(parts),
                                     headers={"Content-Type": f"multipart/form-data; boundary={b}"})
        return json.load(urllib.request.urlopen(req, timeout=60))

    # ---------------- holat ----------------
    def save(self):
        try:
            self.state_path.parent.mkdir(exist_ok=True)
            self.state_path.write_text(json.dumps(self.state, ensure_ascii=False))
        except Exception as e:
            print("Holatni saqlab bo'lmadi:", e)

    def flooded(self, chat) -> bool:
        q, now = self.hits[chat], time.time()
        while q and now - q[0] > RATE_LIMIT[1]:
            q.popleft()
        q.append(now)
        return len(q) > RATE_LIMIT[0]

    # ---------------- yordamchilar ----------------
    def company_buttons(self, stir):
        return [[{"text": "📄 PDF pasport", "callback_data": f"pdf:{stir}"},
                 {"text": "🔍 Nima uchun?", "callback_data": f"why:{stir}"}],
                ]

    def show_company(self, chat, stir):
        r = self.df.iloc[self.by_stir[stir]]
        self.send(chat, card(r), self.company_buttons(stir))

    def explain_text(self, stir):
        i = self.by_stir[stir]
        e = explain_local(self.bundle, self.df.loc[[i], RAW_COLS], top=5)
        L = ["<b>Ballga eng ko'p ta'sir qilgan omillar:</b>", ""]
        for _, x in e.iterrows():
            d = x["PD ga ta'siri"] * 100
            if abs(d) < 0.5:
                continue
            L.append(f"{'🔺' if d > 0 else '🔻'} {esc(x['Belgi'])}: "
                     f"{'xavfni oshirmoqda' if d > 0 else 'xavfni kamaytirmoqda'} ({d:+.1f} foiz punkti)")
        return "\n".join(L) if len(L) > 2 else "Aniq bir omil ajralib turmaydi."

    # ---------------- kirish va ruxsatlar ----------------
    def is_admin(self, chat) -> bool:
        return str(chat) in self.admins

    def session_stir(self, chat):
        """Botda tizimga kirgan korxonaning STIR'i (muddati o'tgan bo'lsa None)."""
        sess = self.state["sessions"].get(str(chat))
        if not sess:
            return None
        if time.time() - sess["ts"] > SESSION_TTL:
            self.state["sessions"].pop(str(chat), None); self.save()
            return None
        return sess["stir"]

    def allowed(self, chat, stir) -> bool:
        return self.is_admin(chat) or self.session_stir(chat) == stir

    def delete_msg(self, chat, mid):
        if mid:
            try:
                self.api("deleteMessage", chat_id=chat, message_id=mid)
            except Exception:
                pass

    def handle_login_step(self, chat, text, mid):
        p = self.pending.get(chat)
        if p["step"] == "stir":
            if not re.fullmatch(r"\d{9}", text):
                return self.send(chat, "STIR 9 xonali raqam bo'lishi kerak. Qayta kiriting yoki /bekor.")
            self.pending[chat] = {"step": "pw", "stir": text, "ts": time.time()}
            return self.send(chat, "Endi <b>parolni</b> yuboring. Xavfsizlik uchun parol yozilgan xabar darhol o'chiriladi.")
        self.delete_msg(chat, mid)  # parol chatda qolmasligi uchun
        self.pending.pop(chat, None)
        user, msg = auth.verify(p["stir"], text)
        if not user or user["role"] != "company" or user["stir"] not in self.by_stir:
            return self.send(chat, msg or "Bu hisob bot uchun mos emas (bot yuridik shaxslar uchun).")
        self.state["sessions"][str(chat)] = {"stir": user["stir"], "ts": time.time()}
        self.save()
        self.send(chat, "✅ Tizimga kirdingiz.")
        self.show_company(chat, user["stir"])

    # ---------------- xabarlar ----------------
    def handle_update(self, u):
        if "callback_query" in u:
            return self.handle_callback(u["callback_query"])
        m = u.get("message")
        if m and "text" in m:
            self.handle_message(m["chat"]["id"], m["text"].strip(), m.get("message_id"))

    def handle_callback(self, cq):
        chat = cq["message"]["chat"]["id"]
        try:
            self.api("answerCallbackQuery", callback_query_id=cq["id"])
        except Exception:
            pass
        if self.flooded(chat):
            return self.send(chat, "⏳ So'rovlar juda tez. Bir daqiqadan keyin urinib ko'ring.")
        action, _, stir = cq.get("data", "").partition(":")
        if stir not in self.by_stir or not self.allowed(chat, stir):
            return self.send(chat, "⛔ Bu ma'lumotni ko'rishga ruxsatingiz yo'q. Avval /kirish orqali kiring.")
        if action == "pdf":
            r = self.df.iloc[self.by_stir[stir]]
            self.send_document(chat, f"soliq_pasporti_{stir}.pdf", passport_pdf(r), "Soliq pasporti (sintetik ma'lumotlar)")
        elif action == "why":
            self.send(chat, self.explain_text(stir))

    def handle_message(self, chat, text, mid=None):
        if self.flooded(chat):
            self.delete_msg(chat, mid) if chat in self.pending else None
            return self.send(chat, "⏳ So'rovlar juda tez. Bir daqiqadan keyin urinib ko'ring.")
        p = self.pending.get(chat)
        if p and time.time() - p["ts"] > PENDING_TTL:
            self.pending.pop(chat, None); p = None
        cmd, _, arg = text.partition(" ")
        cmd, arg = cmd.lower().split("@")[0], arg.strip()
        if p and not text.startswith("/"):
            return self.handle_login_step(chat, text, mid)
        if cmd in ("/start", "/yordam", "/help"):
            return self.send(chat, (
                "Assalomu alaykum! Men <b>Tax Scoring</b> botiman. Korxonangizning soliq intizomi va moliyaviy "
                "ishonchlilik reytingini ko'rsataman.\n\n"
                "• /kirish: STIR va parol bilan kirish (hisob veb-saytda ro'yxatdan o'tganda yaratiladi)\n"
                "• /mening: korxonamning natijasi\n• /chiqish: tizimdan chiqish\n\n"
                "Faqat <b>o'z</b> korxonangiz ma'lumotini ko'ra olasiz. Kirgan bo'lsangiz, xavf yuqori bo'lganda sizga ogohlantirish yuboriladi.\n"
                "⚠️ Barcha ma'lumotlar <b>sintetik</b> (o'quv prototipi)."))
        if cmd == "/id":
            return self.send(chat, f"Sizning Telegram ID: <code>{chat}</code>")
        if cmd == "/kirish":
            self.pending[chat] = {"step": "stir", "ts": time.time()}
            return self.send(chat, "Korxonangizning <b>STIR</b>ini (9 xonali, bu sizning loginingiz) yuboring. Bekor qilish: /bekor")
        if cmd == "/bekor":
            self.pending.pop(chat, None)
            return self.send(chat, "Bekor qilindi.")
        if cmd == "/chiqish":
            self.pending.pop(chat, None)
            if self.state["sessions"].pop(str(chat), None):
                self.save()
                return self.send(chat, "Tizimdan chiqdingiz.")
            return self.send(chat, "Siz tizimga kirmagansiz.")
        if cmd == "/mening":
            stir = self.session_stir(chat)
            return self.show_company(chat, stir) if stir else self.send(chat, "Avval /kirish orqali tizimga kiring.")
        if cmd == "/tekshir" and self.is_admin(chat):
            return self.lookup(chat, arg) if arg else self.send(chat, "STIR yoki nom kiriting: /tekshir 243785752")
        if cmd in ("/top10", "/statistika", "/yubor"):
            return self.admin(chat, cmd)
        if cmd.startswith("/"):
            return self.send(chat, "Noma'lum buyruq. /yordam ni bosing.")
        stir = self.session_stir(chat)
        if stir:
            return self.show_company(chat, stir)
        self.send(chat, "Ma'lumotni ko'rish uchun avval /kirish orqali tizimga kiring.")

    def lookup(self, chat, q):
        """Faqat Telegram administratorlari uchun: istalgan korxonani qidirish."""
        q = q.strip()
        m = re.search(r"\b\d{9}\b", q)
        if m:
            if m.group() in self.by_stir:
                return self.show_company(chat, m.group())
            return self.send(chat, "Bunday STIR topilmadi.")
        if len(q) < 3:
            return self.send(chat, "STIR (9 xonali raqam) yoki kamida 3 harfli nom kiriting.")
        hit = self.df[self.df["name"].str.contains(q, case=False, regex=False)]
        if hit.empty:
            return self.send(chat, "Hech narsa topilmadi.")
        if len(hit) == 1:
            return self.show_company(chat, hit.iloc[0]["stir"])
        self.send(chat, f"{len(hit)} ta natija. Aniqroq nom yoki STIR kiriting:\n" + "\n".join(
            f"{EMOJI[r['risk_class']]} <code>{r['stir']}</code> {esc(r['name'])}" for _, r in hit.head(6).iterrows()))

    # ---------------- admin ----------------
    def admin(self, chat, cmd):
        if str(chat) not in self.admins:
            return self.send(chat, "⛔ Bu buyruq faqat administratorlar uchun. ID ingizni /id orqali bilib, "
                                   "ADMIN_IDS muhit o'zgaruvchisiga yozing.")
        d = self.df
        if cmd == "/top10":
            t = d.nsmallest(10, "final_score")
            return self.send(chat, "<b>Eng xavfli 10 ta korxona:</b>\n\n" + "\n".join(
                f"{i}. {esc(r['name'])} (<code>{r['stir']}</code>): {r['final_score']:.1f} ball, qarz {r['tax_debt']:,.0f} mln"
                for i, (_, r) in enumerate(t.iterrows(), 1)))
        if cmd == "/statistika":
            vc = d["risk_class"].value_counts()
            return self.send(chat, "<b>Umumiy statistika</b>\n" + f"Korxonalar: {len(d):,}\nO'rtacha ball: {d['final_score'].mean():.1f}\n"
                             + "\n".join(f"{EMOJI[k]} {k}: {vc.get(k, 0)} ({vc.get(k, 0) / len(d):.0%})" for k in EMOJI)
                             + f"\nJami soliq qarzi: {d['tax_debt'].sum() / 1000:,.1f} mlrd so'm\n"
                             f"Botda kirganlar: {len(self.state['sessions'])}")
        sent = 0  # /yubor: botda tizimga kirgan va yuqori xavfli korxonalarga ogohlantirish
        for ch, sess in list(self.state["sessions"].items()):
            if time.time() - sess["ts"] > SESSION_TTL:
                continue
            key = f"{ch}:{sess['stir']}"
            r = self.df.iloc[self.by_stir[sess["stir"]]]
            if r["risk_class"] == "Yuqori xavf" and key not in self.state["alerted"]:
                self.send(ch, "⚠️ <b>Ogohlantirish!</b> Korxonangiz <b>yuqori xavf</b> guruhida:\n\n" + card(r), self.company_buttons(sess["stir"]))
                self.state["alerted"].append(key); sent += 1
        self.save()
        self.send(chat, f"Yuborilgan ogohlantirishlar: {sent} ta.")

    # ---------------- asosiy sikl ----------------
    def run(self):
        self.api("deleteWebhook")
        self.api("setMyCommands", commands=[{"command": c, "description": d} for c, d in COMMANDS])
        me = self.api("getMe")
        print(f"Bot ishga tushdi: @{me['username']}  (to'xtatish: Ctrl+C)")
        offset = None
        while True:
            try:
                for u in self.api("getUpdates", offset=offset, timeout=30, allowed_updates=["message", "callback_query"]):
                    offset = u["update_id"] + 1
                    try:
                        self.handle_update(u)
                    except Exception as e:
                        print("Xabarni qayta ishlashda xato:", repr(e))
            except KeyboardInterrupt:
                print("To'xtatildi."); return
            except Exception as e:
                print("Tarmoq xatosi:", repr(e)); time.sleep(5)


def load_bot():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        sys.exit("TELEGRAM_BOT_TOKEN o'rnatilmagan. @BotFather dan token oling va uni muhit o'zgaruvchisi sifatida bering.")
    stale = True
    if Path("data/scored_companies.csv").exists() and Path("models/best_model.joblib").exists():
        stale = pd.read_csv("data/scored_companies.csv", usecols=["final_score"])["final_score"].max() > 100 or \
                "has_inventory" not in pd.read_csv("data/scored_companies.csv", nrows=0).columns  # eski ma'lumot
    if stale:
        print("Ma'lumotlar yo'q, yaratilmoqda...")
        Path("data").mkdir(exist_ok=True); Path("models").mkdir(exist_ok=True)
        import main
        main.run()
    df = pd.read_csv("data/scored_companies.csv", dtype={"stir": str, "phone": str})
    admins = [a for a in os.environ.get("ADMIN_IDS", "").split(",") if a.strip()]
    return Bot(token, df, load_model(), admins)


if __name__ == "__main__":
    load_bot().run()
