"""Sintetik (sun'iy) korxonalar ma'lumotlarini yaratish.

Haqiqiy soliq ma'lumotlari maxfiy bo'lgani uchun korxonalar O'zbekiston iqtisodiyotiga xos
tarmoq/hudud taqsimotlari bilan modellashtiriladi. Har bir korxonaga yashirin "xavf omili" (z)
beriladi; moliyaviy ko'rsatkichlar va soliq intizomi shu omil bilan bog'liq, lekin shovqin bilan.
"""
import numpy as np
import pandas as pd

# tarmoq: (mediana aylanma mln so'm, sigma, o'rtacha rentabellik, bazaviy xavf)
SECTORS = {
    "Savdo": (900, 1.1, 0.04, 0.15),
    "Ishlab chiqarish": (2500, 1.1, 0.08, 0.00),
    "Qurilish": (2000, 1.1, 0.06, 0.30),
    "Qishloq xo'jaligi": (1200, 1.0, 0.09, 0.10),
    "Transport va logistika": (1100, 1.0, 0.06, 0.10),
    "Xizmatlar": (500, 1.0, 0.12, -0.05),
    "Axborot texnologiyalari": (700, 1.1, 0.15, -0.25),
    "Umumiy ovqatlanish": (450, 0.9, 0.07, 0.20),
    "Turizm va mehmonxona": (600, 1.0, 0.07, 0.25),
    "Ta'lim va sog'liqni saqlash": (550, 0.9, 0.10, -0.20),
}
SECTOR_P = np.array([0.24, 0.14, 0.13, 0.08, 0.08, 0.13, 0.05, 0.05, 0.04, 0.06])

REGIONS = {
    "Toshkent shahri": 0.22, "Toshkent viloyati": 0.08, "Samarqand": 0.09, "Farg'ona": 0.09,
    "Andijon": 0.08, "Namangan": 0.07, "Qashqadaryo": 0.06, "Buxoro": 0.06,
    "Surxondaryo": 0.04, "Xorazm": 0.04, "Navoiy": 0.03, "Jizzax": 0.03,
    "Sirdaryo": 0.02, "Qoraqalpog'iston": 0.05,
}
LEGAL = {"MChJ": 0.72, "XK": 0.12, "AJ": 0.05, "QK": 0.04, "Fermer xo'jaligi": 0.07}

PREFIX = ["Orient", "Silk Road", "Registon", "Zarafshon", "Turon", "Sharq", "Bunyod", "Navro'z",
          "Oltin Vodiy", "Amir", "Mehr", "Baraka", "Ulug'bek", "Kamolot", "Yangi Asr", "Chinor",
          "Nur", "Tong", "Ishonch", "Humo", "Zafar", "Sog'lom", "Marg'ilon", "Kelajak"]
SECTOR_WORD = {
    "Savdo": "Savdo", "Ishlab chiqarish": "Prom", "Qurilish": "Qurilish",
    "Qishloq xo'jaligi": "Agro", "Transport va logistika": "Logistik",
    "Xizmatlar": "Servis", "Axborot texnologiyalari": "Tech",
    "Umumiy ovqatlanish": "Food", "Turizm va mehmonxona": "Turizm",
    "Ta'lim va sog'liqni saqlash": "Med",
}


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate(n: int = 3000, seed: int = 42, target_rate: float = 0.14) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(SECTORS)
    sec = rng.choice(names, n, p=SECTOR_P / SECTOR_P.sum())
    med = np.array([SECTORS[s][0] for s in sec], float)
    sig = np.array([SECTORS[s][1] for s in sec], float)
    mm = np.array([SECTORS[s][2] for s in sec], float)
    base = np.array([SECTORS[s][3] for s in sec], float)

    z = rng.normal(0, 1, n) + base  # yashirin xavf omili

    years = np.clip(np.round(rng.gamma(2.0, 3.5, n) - 0.35 * z), 0, 35).astype(int)
    revenue = np.clip(np.exp(np.log(med) + sig * rng.normal(size=n)), 30, 250_000)
    margin = np.clip(mm - 0.035 * z + rng.normal(0, 0.06, n), -0.35, 0.35)
    net_profit = revenue * margin
    turnover = np.exp(np.log(1.5) + 0.4 * rng.normal(size=n))
    total_assets = revenue / turnover
    eq_ratio = np.clip(0.45 - 0.12 * z + rng.normal(0, 0.15, n), -0.25, 0.95)
    equity = total_assets * eq_ratio
    cur_ratio = np.clip(np.exp(np.log(1.4) - 0.18 * z + 0.35 * rng.normal(size=n)), 0.2, 4.5)
    current_assets = total_assets * rng.uniform(0.3, 0.8, n)
    current_liabilities = current_assets / cur_ratio
    employees = np.maximum(1, np.round(revenue / np.exp(np.log(70) + 0.6 * rng.normal(size=n)))).astype(int)

    tax_accrued = revenue * rng.uniform(0.03, 0.09, n)
    has_debt = rng.random(n) < _sigmoid(-1.0 + 1.2 * z)
    debt_share = np.clip(np.exp(np.log(0.25) + 0.5 * z + 0.9 * rng.normal(size=n)), 0, 2.5)
    tax_debt = np.where(has_debt, tax_accrued * debt_share, 0.0)

    lam = np.exp(-0.1 + 0.6 * z + 0.5 * has_debt)
    late_pay = np.clip(rng.poisson(lam), 0, 12)
    avg_delay = np.where(late_pay > 0, rng.gamma(2.0, 6.0 * np.exp(0.3 * z)), 0.0)
    avg_delay = np.clip(avg_delay, 0, 180)
    late_rep = rng.binomial(12, _sigmoid(-2.3 + 0.8 * z))
    audits = rng.poisson(np.exp(-1.2 + 0.6 * z))
    vat = (revenue > 1000) | (rng.random(n) < 0.15)

    # Keyingi yilda xavfli qarzdorlik (90+ kun muddati o'tgan qarz) belgisi
    dtt = np.minimum(tax_debt / tax_accrued, 2.0)
    logit = (0.7 * z + 1.5 * dtt + 0.10 * late_pay + 0.015 * avg_delay + 0.25 * audits
             - 0.6 * (eq_ratio - 0.4) - 0.5 * (cur_ratio - 1.4) + rng.normal(0, 0.7, n))
    lo, hi = -15.0, 5.0
    for _ in range(50):  # kesishmani (intercept) maqsadli ulushga moslash
        mid = (lo + hi) / 2
        if _sigmoid(logit + mid).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    y = (rng.random(n) < _sigmoid(logit + (lo + hi) / 2)).astype(int)

    # STIR (9 xonali, noyob), nomlar
    stir = rng.choice(np.arange(200_000_000, 310_000_000), n, replace=False)
    legal = rng.choice(list(LEGAL), n, p=np.array(list(LEGAL.values())) / sum(LEGAL.values()))
    region = rng.choice(list(REGIONS), n, p=np.array(list(REGIONS.values())) / sum(REGIONS.values()))
    pre = rng.choice(PREFIX, n)
    name = [f'"{p} {SECTOR_WORD[s]}" {l}' for p, s, l in zip(pre, sec, legal)]

    rng2 = np.random.default_rng(seed + 1000)  # aloqa ma'lumotlari uchun alohida oqim
    ops = np.array(["90", "91", "93", "94", "95", "97", "98", "99", "33", "88", "77"])
    phone = [f"+998{o}{rng2.integers(1_000_000, 9_999_999)}" for o in rng2.choice(ops, n)]
    email = [f"info{str(s_)[-5:]}@example.com" for s_ in stir]

    return pd.DataFrame({
        "stir": stir.astype(str), "name": name, "phone": phone, "email": email, "region": region, "sector": sec,
        "legal_form": legal, "years_active": years, "employees": employees,
        "revenue": revenue.round(1), "net_profit": net_profit.round(1),
        "total_assets": total_assets.round(1), "equity": equity.round(1),
        "current_assets": current_assets.round(1), "current_liabilities": current_liabilities.round(1),
        "tax_accrued": tax_accrued.round(1), "tax_debt": tax_debt.round(2),
        "late_payments_count": late_pay, "avg_delay_days": avg_delay.round(1),
        "late_reports_count": late_rep, "audit_violations": audits,
        "vat_payer": vat.astype(int), "default_next_year": y,
    })


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/companies.csv", index=False)
    print(df.shape, "xavfli ulush:", df.default_next_year.mean().round(3))
