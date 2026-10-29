"""Foydalanuvchilar, parollar va rollar (SQLite, faqat standart kutubxona).

Rollar: admin (hamma narsa + foydalanuvchilarni boshqarish), inspector (tizimning barcha tahlil bo'limlari),
company (yuridik shaxs: faqat O'Z korxonasi; login = STIR).
Xavfsizlik (o'quv prototipi darajasi): parol PBKDF2-HMAC-SHA256 (tuz bilan) ko'rinishida saqlanadi; ketma-ket 5 ta xato
paroldan keyin hisob 5 daqiqaga bloklanadi; hodisalar jurnalga yoziladi. Real tizimda rasmiy identifikatsiya kerak.
"""
import datetime as dt
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
from contextlib import closing
from pathlib import Path

ITERATIONS = 200_000
MAX_FAILS = 5
LOCK_SECONDS = 300
CODE_TTL = 300
CODE_MAX_TRIES = 5
ROLES = {"admin": "Administrator", "inspector": "Inspektor", "company": "Yuridik shaxs"}
_SETUP = {"code": None}


def _path() -> str:
    return os.environ.get("AUTH_DB_PATH", "data/users.db")


def _conn():
    p = Path(_path())
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(p, timeout=10)
    c.row_factory = sqlite3.Row
    c.execute("""CREATE TABLE IF NOT EXISTS users(
        username TEXT PRIMARY KEY, role TEXT NOT NULL, stir TEXT, salt BLOB NOT NULL, pw_hash BLOB NOT NULL,
        iters INTEGER NOT NULL, active INTEGER NOT NULL DEFAULT 1, fails INTEGER NOT NULL DEFAULT 0,
        locked_until REAL NOT NULL DEFAULT 0, created TEXT, last_login TEXT)""")
    c.execute("CREATE TABLE IF NOT EXISTS audit(ts TEXT, username TEXT, event TEXT)")
    return c


def _run(sql, args=(), fetch=None):
    with closing(_conn()) as c:
        with c:
            cur = c.execute(sql, args)
            if fetch == "one":
                return cur.fetchone()
            if fetch == "all":
                return cur.fetchall()
    return None


def _now_s():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(event: str, username: str = ""):
    _run("INSERT INTO audit VALUES (?,?,?)", (_now_s(), (username or "")[:40], event))


def recent_events(n=50):
    return [dict(r) for r in _run("SELECT ts, username, event FROM audit ORDER BY rowid DESC LIMIT ?", (n,), "all")]


# ---------------------------------- parol ----------------------------------
def _hash(password: str, salt: bytes, iters: int) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iters)


def valid_password(pw: str):
    if len(pw or "") < 8:
        return False, "Parol kamida 8 belgidan iborat bo'lishi kerak."
    if not re.search(r"[A-Za-z]", pw) or not re.search(r"\d", pw):
        return False, "Parolda kamida bitta harf va bitta raqam bo'lishi kerak."
    return True, ""


def valid_username(u: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9_.-]{3,32}", u or ""))


def get_user(username: str):
    return _run("SELECT * FROM users WHERE username=?", ((username or "").strip().lower(),), "one")


def count_users() -> int:
    return _run("SELECT COUNT(*) AS n FROM users", fetch="one")["n"]


def has_admin() -> bool:
    return _run("SELECT 1 FROM users WHERE role='admin' AND active=1", fetch="one") is not None


def create_user(username: str, password: str, role: str, stir: str = None):
    username = (username or "").strip().lower()
    if role not in ROLES:
        return False, "Noto'g'ri rol."
    if role == "company":
        if not re.fullmatch(r"\d{9}", stir or "") or username != stir:
            return False, "Yuridik shaxs uchun login 9 xonali STIR bo'lishi kerak."
    elif not valid_username(username):
        return False, "Login 3-32 belgi: kichik harf, raqam, nuqta, pastki chiziq yoki chiziqcha."
    ok, msg = valid_password(password)
    if not ok:
        return False, msg
    if get_user(username):
        return False, "Bu login band."
    salt = secrets.token_bytes(16)
    _run("INSERT INTO users(username, role, stir, salt, pw_hash, iters, created) VALUES (?,?,?,?,?,?,?)",
         (username, role, stir if role == "company" else None, salt, _hash(password, salt, ITERATIONS), ITERATIONS, _now_s()))
    log(f"hisob yaratildi ({role})", username)
    return True, "Hisob yaratildi."


def verify(username: str, password: str):
    """(foydalanuvchi_dict | None, xabar). Ketma-ket xatolarda hisobni bloklaydi."""
    username = (username or "").strip().lower()
    row = get_user(username)
    if row is None:
        _hash(password or "", b"\0" * 16, ITERATIONS)  # vaqtni tenglashtirish (login bor-yo'qligini bildirmaslik uchun)
        log("kirish: xato (noma'lum login)", username)
        return None, "Login yoki parol noto'g'ri."
    now = time.time()
    if row["locked_until"] > now:
        return None, f"Hisob vaqtincha bloklangan. Taxminan {int((row['locked_until'] - now) // 60) + 1} daqiqadan keyin urinib ko'ring."
    if not row["active"]:
        return None, "Hisob faol emas. Administratorga murojaat qiling."
    if hmac.compare_digest(_hash(password or "", row["salt"], row["iters"]), row["pw_hash"]):
        _run("UPDATE users SET fails=0, locked_until=0, last_login=? WHERE username=?", (_now_s(), username))
        log("kirish: muvaffaqiyatli", username)
        return {"username": username, "role": row["role"], "stir": row["stir"]}, ""
    fails = row["fails"] + 1
    if fails >= MAX_FAILS:
        _run("UPDATE users SET fails=0, locked_until=? WHERE username=?", (now + LOCK_SECONDS, username))
        log("hisob bloklandi (ko'p xato)", username)
        return None, "Ko'p marta xato kiritildi: hisob 5 daqiqaga bloklandi."
    _run("UPDATE users SET fails=? WHERE username=?", (fails, username))
    log("kirish: xato parol", username)
    return None, "Login yoki parol noto'g'ri."


def set_password(username: str, new_password: str):
    ok, msg = valid_password(new_password)
    if not ok:
        return False, msg
    if not get_user(username):
        return False, "Foydalanuvchi topilmadi."
    salt = secrets.token_bytes(16)
    _run("UPDATE users SET salt=?, pw_hash=?, iters=?, fails=0, locked_until=0 WHERE username=?",
         (salt, _hash(new_password, salt, ITERATIONS), ITERATIONS, username.strip().lower()))
    log("parol o'zgartirildi", username)
    return True, "Parol o'zgartirildi."


def change_own_password(username: str, old: str, new: str):
    row = get_user(username)
    if not row or not hmac.compare_digest(_hash(old or "", row["salt"], row["iters"]), row["pw_hash"]):
        return False, "Joriy parol noto'g'ri."
    return set_password(username, new)


def _admins_left(excluding: str) -> int:
    return _run("SELECT COUNT(*) AS n FROM users WHERE role='admin' AND active=1 AND username<>?",
                (excluding.strip().lower(),), "one")["n"]


def set_active(username: str, active: bool):
    row = get_user(username)
    if not row:
        return False, "Foydalanuvchi topilmadi."
    if not active and row["role"] == "admin" and _admins_left(username) == 0:
        return False, "Oxirgi administratorni o'chirib bo'lmaydi."
    _run("UPDATE users SET active=? WHERE username=?", (1 if active else 0, username.strip().lower()))
    log("faollashtirildi" if active else "faolsizlantirildi", username)
    return True, "Bajarildi."


def unlock(username: str):
    _run("UPDATE users SET fails=0, locked_until=0 WHERE username=?", (username.strip().lower(),))
    log("blokdan chiqarildi", username)


def delete_user(username: str):
    row = get_user(username)
    if not row:
        return False, "Foydalanuvchi topilmadi."
    if row["role"] == "admin" and _admins_left(username) == 0:
        return False, "Oxirgi administratorni o'chirib bo'lmaydi."
    _run("DELETE FROM users WHERE username=?", (username.strip().lower(),))
    log("o'chirildi", username)
    return True, "O'chirildi."


def list_users():
    now = time.time()
    return [{"login": r["username"], "rol": ROLES[r["role"]], "stir": r["stir"] or "", "faol": bool(r["active"]),
             "bloklangan": r["locked_until"] > now, "yaratilgan": r["created"], "oxirgi_kirish": r["last_login"] or ""}
            for r in _run("SELECT * FROM users ORDER BY role, username", fetch="all")]


# ---------------------------------- sozlash va tasdiqlash kodi ----------------------------------
def bootstrap_admin(username: str, password: str):
    """ADMIN_USERNAME/ADMIN_PASSWORD (muhit yoki Secrets) berilsa, admin hisobi shu parol bilan ta'minlanadi."""
    username = (username or "").strip().lower()
    if not username or not password:
        return
    row = get_user(username)
    if row is None:
        ok, msg = create_user(username, password, "admin")
        if not ok:
            print(f"[AUTH] Admin yaratilmadi: {msg}", flush=True)
    elif row["role"] == "admin" and not hmac.compare_digest(_hash(password, row["salt"], row["iters"]), row["pw_hash"]):
        set_password(username, password)


def setup_code() -> str:
    """Birinchi admin yaratishda kerak bo'ladigan kod; faqat server konsoli/logida ko'rinadi."""
    if _SETUP["code"] is None:
        _SETUP["code"] = secrets.token_hex(4)
        print(f"[AUTH] Birinchi sozlash kodi (admin yaratish uchun): {_SETUP['code']}", flush=True)
    return _SETUP["code"]


def check_setup_code(entered: str) -> bool:
    """Sozlash kodini tekshiradi; 5 ta xatodan keyin 60 soniya kutish talab qilinadi (tanlab topishdan himoya)."""
    now = time.time()
    if _SETUP.get("until", 0) > now:
        return False
    if hmac.compare_digest((entered or "").strip(), setup_code()):
        return True
    _SETUP["fails"] = _SETUP.get("fails", 0) + 1
    if _SETUP["fails"] >= 5:
        _SETUP["fails"], _SETUP["until"] = 0, now + 60
    return False


def issue_code(stir: str) -> dict:
    return {"stir": stir, "code": f"{secrets.randbelow(10 ** 6):06d}", "exp": time.time() + CODE_TTL, "tries": 0}


def check_code(state: dict, entered: str):
    if time.time() > state["exp"]:
        return False, "Kod muddati tugadi. Yangi kod so'rang."
    state["tries"] += 1
    if state["tries"] > CODE_MAX_TRIES:
        return False, "Urinishlar soni tugadi. Yangi kod so'rang."
    if hmac.compare_digest((entered or "").strip(), state["code"]):
        return True, ""
    return False, "Kod noto'g'ri."


def mask_phone(phone: str) -> str:
    return phone[:6] + "*" * max(len(phone) - 8, 0) + phone[-2:]
