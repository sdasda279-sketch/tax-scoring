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

from ml_model import explain_local, load_model
from report import passport_pdf
from scoring import LABELS, WEIGHTS, recommendations

RAW_COLS = ["stir", "name", "region", "sector", "legal_form", "years_active", "employees", "revenue",
            "net_profit", "total_assets", "equity", "current_assets", "current_liabilities", "tax_accrued",
            "tax_debt", "late_payments_count", "avg_delay_days", "late_reports_count", "audit_violations", "vat_payer",
            "has_inventory", "inv_opening_value", "inv_purchases", "inv_closing_value"]
EMOJI = {"Past xavf": "🟢", "O'rtacha xavf": "🟡", "Yuqori xavf": "🔴"}
MAX_SUBS = 5
RATE_LIMIT = (15, 60)  # 60 soniyada 15 tadan ortiq so'rov emas

COMMANDS = [("start", "Boshlash"), ("tekshir", "STIR bo'yicha tekshirish"), ("demo", "Namuna STIR'lar"),
            ("obuna", "Korxonaga obuna bo'lish"), ("mening", "Mening obunalarim"),
            ("bekor", "Obunani bekor qilish"), ("yordam", "Yordam"), ("id", "Mening Telegram ID'im")]

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
        self.state = {"subs": {}, "alerted": []}
        if self.state_path.exists():
            try:
                self.state = json.loads(self.state_path.read_text())
            except Exception:
                pass
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
                [{"text": "🔔 Obuna bo'lish", "callback_data": f"sub:{stir}"}]]

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

    def subscribe(self, chat, stir):
        subs = self.state["subs"].setdefault(str(chat), [])
        if stir in subs:
            return self.send(chat, "Siz bu korxonaga allaqachon obuna bo'lgansiz.")
        if len(subs) >= MAX_SUBS:
            return self.send(chat, f"Bir foydalanuvchi eng ko'pi bilan {MAX_SUBS} ta korxonaga obuna bo'la oladi.")
        subs.append(stir); self.save()
        self.send(chat, "🔔 Obuna qilindi. Korxona xavf sinfi <b>yuqori</b> bo'lsa, sizga ogohlantirish yuboriladi.")

    # ---------------- xabarlar ----------------
    def handle_update(self, u):
        if "callback_query" in u:
            return self.handle_callback(u["callback_query"])
        m = u.get("message")
        if m and "text" in m:
            self.handle_message(m["chat"]["id"], m["text"].strip())

    def handle_callback(self, cq):
        chat = cq["message"]["chat"]["id"]
        try:
            self.api("answerCallbackQuery", callback_query_id=cq["id"])
        except Exception:
            pass
        if self.flooded(chat):
            return self.send(chat, "⏳ So'rovlar juda tez. Bir daqiqadan keyin urinib ko'ring.")
        action, _, stir = cq.get("data", "").partition(":")
        if stir not in self.by_stir:
            return self.send(chat, "Korxona topilmadi.")
        if action == "pdf":
            r = self.df.iloc[self.by_stir[stir]]
            self.send_document(chat, f"soliq_pasporti_{stir}.pdf", passport_pdf(r), "Soliq pasporti (sintetik ma'lumotlar)")
        elif action == "why":
            self.send(chat, self.explain_text(stir))
        elif action == "sub":
            self.subscribe(chat, stir)
        elif action == "check":
            self.show_company(chat, stir)

    def handle_message(self, chat, text):
        if self.flooded(chat):
            return self.send(chat, "⏳ So'rovlar juda tez. Bir daqiqadan keyin urinib ko'ring.")
        cmd, _, arg = text.partition(" ")
        cmd, arg = cmd.lower().split("@")[0], arg.strip()
        if cmd in ("/start", "/yordam", "/help"):
            return self.send(chat, (
                "Assalomu alaykum! Men <b>Tax Scoring</b> botiman. Korxonaning soliq intizomi va moliyaviy "
                "ishonchlilik reytingini ko'rsataman.\n\n"
                "• 9 xonali <b>STIR</b> yoki korxona nomini yuboring\n• /demo: namuna STIR'lar\n"
                "• /obuna STIR: yuqori xavf haqida ogohlantirish olish\n• /mening: obunalarim\n• /bekor STIR: obunani bekor qilish\n\n"
                "⚠️ Barcha ma'lumotlar <b>sintetik</b> (o'quv prototipi)."))
        if cmd == "/id":
            return self.send(chat, f"Sizning Telegram ID: <code>{chat}</code>")
        if cmd == "/demo":
            L = ["Namuna korxonalar (STIR'ni nusxalab yuboring):", ""]
            for cls in ("Yuqori xavf", "O'rtacha xavf", "Past xavf"):
                sub = self.df[self.df["risk_class"] == cls]
                if len(sub):
                    r = sub.iloc[random.randrange(len(sub))]
                    L.append(f"{EMOJI[cls]} <code>{r['stir']}</code>: {esc(r['name'])}")
            return self.send(chat, "\n".join(L))
        if cmd == "/tekshir":
            return self.lookup(chat, arg) if arg else self.send(chat, "STIR yoki nom kiriting: /tekshir 243785752")
        if cmd == "/obuna":
            if arg in self.by_stir:
                return self.subscribe(chat, arg)
            return self.send(chat, "Format: /obuna 243785752 (9 xonali STIR)")
        if cmd == "/mening":
            subs = self.state["subs"].get(str(chat), [])
            if not subs:
                return self.send(chat, "Sizda obunalar yo'q.")
            return self.send(chat, "\n".join(f"{EMOJI[self.df.iloc[self.by_stir[s]]['risk_class']]} <code>{s}</code> "
                                             f"{esc(self.df.iloc[self.by_stir[s]]['name'])}" for s in subs))
        if cmd == "/bekor":
            subs = self.state["subs"].get(str(chat), [])
            if arg in subs:
                subs.remove(arg); self.save()
                return self.send(chat, "Obuna bekor qilindi.")
            return self.send(chat, "Format: /bekor 243785752 (obuna bo'lgan STIR)")
        if cmd in ("/top10", "/statistika", "/yubor"):
            return self.admin(chat, cmd)
        if cmd.startswith("/"):
            return self.send(chat, "Noma'lum buyruq. /yordam ni bosing.")
        self.lookup(chat, text)

    def lookup(self, chat, q):
        q = q.strip()
        m = re.search(r"\b\d{9}\b", q)
        if m:
            if m.group() in self.by_stir:
                return self.show_company(chat, m.group())
            return self.send(chat, "Bunday STIR topilmadi. Namuna uchun /demo ni bosing.")
        if len(q) < 3:
            return self.send(chat, "STIR (9 xonali raqam) yoki kamida 3 harfli nom kiriting.")
        hit = self.df[self.df["name"].str.contains(q, case=False, regex=False)]
        if hit.empty:
            return self.send(chat, "Hech narsa topilmadi.")
        if len(hit) == 1:
            return self.show_company(chat, hit.iloc[0]["stir"])
        btn = [[{"text": f"{EMOJI[r['risk_class']]} {r['name']} ({r['stir']})", "callback_data": f"check:{r['stir']}"}]
               for _, r in hit.head(6).iterrows()]
        self.send(chat, f"{len(hit)} ta natija topildi (dastlabki {len(btn)} tasi):", btn)

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
                             f"Obunachilar: {len(self.state['subs'])}")
        sent = 0  # /yubor: obuna bo'lgan va yuqori xavfli korxonalar haqida ogohlantirish
        for ch, stirs in self.state["subs"].items():
            for s in stirs:
                key = f"{ch}:{s}"
                r = self.df.iloc[self.by_stir[s]]
                if r["risk_class"] == "Yuqori xavf" and key not in self.state["alerted"]:
                    self.send(ch, "⚠️ <b>Ogohlantirish!</b> Siz kuzatayotgan korxona <b>yuqori xavf</b> guruhida:\n\n" + card(r),
                              self.company_buttons(s))
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
    df = pd.read_csv("data/scored_companies.csv", dtype={"stir": str})
    admins = [a for a in os.environ.get("ADMIN_IDS", "").split(",") if a.strip()]
    return Bot(token, df, load_model(), admins)


if __name__ == "__main__":
    load_bot().run()
