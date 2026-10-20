"""Qoidaga asoslangan (ekspert) skoring modeli.

Har bir ko'rsatkich 0-100 oralig'idagi qismiy ballga o'tkaziladi (bo'lak-chiziqli interpolyatsiya),
so'ng vaznlar bo'yicha yig'iladi. Ball 0-100 shkalasida beriladi (100 = eng ishonchli).
"""
import numpy as np
import pandas as pd
from features import add_ratios

WEIGHTS = {"debt": 25, "late": 20, "liquidity": 15, "equity": 15, "profit": 15, "reports": 10}
LABELS = {
    "debt": "Soliq qarzi / aylanma",
    "late": "To'lov intizomi (kechikishlar)",
    "liquidity": "Joriy likvidlik",
    "equity": "Avtonomlik (o'z kapitali ulushi)",
    "profit": "Rentabellik",
    "reports": "Hisobotlarni o'z vaqtida topshirish",
}
# (ko'rsatkich qiymati, ball) tugun nuqtalari
CURVES = {
    "debt": [(0, 100), (0.003, 85), (0.01, 60), (0.03, 25), (0.06, 0)],
    "liquidity": [(0.5, 0), (0.8, 25), (1.0, 50), (1.5, 85), (2.0, 100)],
    "equity": [(-0.1, 0), (0, 15), (0.2, 50), (0.4, 80), (0.6, 100)],
    "profit": [(-0.1, 0), (0, 25), (0.03, 50), (0.08, 80), (0.15, 100)],
    "reports": [(0, 100), (1, 85), (3, 55), (6, 20), (12, 0)],
    "late_count": [(0, 100), (1, 80), (3, 50), (6, 20), (12, 0)],
    "late_delay": [(0, 100), (5, 80), (15, 50), (30, 20), (60, 0)],
}
AUDIT_PENALTY_PER_CASE, AUDIT_PENALTY_MAX = 4.0, 10.0
RISK_THRESHOLDS = (60.0, 35.0)  # 100 ballik shkalada: >=60 past, 35-60 o'rta, <35 yuqori
RISK_NAMES = ["Past xavf", "O'rtacha xavf", "Yuqori xavf"]


def _interp(x, curve):
    xs, ys = zip(*curve)
    return np.interp(x, xs, ys)


def to_scale(s100):
    """Ball shkalasi: 0-100 (o'zgarishsiz). Boshqa shkala kerak bo'lsa, shu yerda o'zgartiriladi."""
    return np.asarray(s100, float)


def classify(s100):
    s = np.asarray(s100, float)
    hi, mid = RISK_THRESHOLDS
    return np.where(s >= hi, RISK_NAMES[0], np.where(s >= mid, RISK_NAMES[1], RISK_NAMES[2]))


def rule_score(df: pd.DataFrame) -> pd.DataFrame:
    """Qismiy ballar, hissalar (vazn*ball/100), rule_100 va rule_score ustunlarini qaytaradi."""
    d = add_ratios(df)
    out = pd.DataFrame(index=d.index)
    out["sub_debt"] = _interp(d["debt_to_revenue"], CURVES["debt"])
    out["sub_late"] = (0.6 * _interp(d["late_payments_count"], CURVES["late_count"])
                       + 0.4 * _interp(d["avg_delay_days"], CURVES["late_delay"]))
    out["sub_liquidity"] = _interp(d["current_ratio"], CURVES["liquidity"])
    out["sub_equity"] = _interp(d["equity_ratio"], CURVES["equity"])
    out["sub_profit"] = _interp(d["profit_margin"], CURVES["profit"])
    out["sub_reports"] = _interp(d["late_reports_count"], CURVES["reports"])
    total = sum(WEIGHTS[k] * out[f"sub_{k}"] / 100 for k in WEIGHTS)
    penalty = np.minimum(AUDIT_PENALTY_MAX, AUDIT_PENALTY_PER_CASE * d["audit_violations"])
    out["audit_penalty"] = penalty
    out["rule_100"] = np.clip(total - penalty, 0, 100)
    out["rule_score"] = to_scale(out["rule_100"])
    out["rule_class"] = classify(out["rule_100"])
    return out


def contributions(row_scores: pd.Series) -> pd.DataFrame:
    """Bitta korxona uchun: har bir ko'rsatkichning maksimal va olingan balli."""
    rows = [{"Ko'rsatkich": LABELS[k], "Vazn": WEIGHTS[k],
             "Qismiy ball (0-100)": round(float(row_scores[f"sub_{k}"]), 1),
             "Olingan ball": round(WEIGHTS[k] * float(row_scores[f"sub_{k}"]) / 100, 1)}
            for k in WEIGHTS]
    return pd.DataFrame(rows)


RECOMMENDATIONS = {
    "debt": "Soliq qarzini bosqichma-bosqich to'lash grafigini kelishish (qarz/aylanma nisbatini kamaytirish).",
    "late": "To'lov muddatlarini nazorat qilish, avtomatik to'lov eslatmalarini yoqish.",
    "liquidity": "Joriy majburiyatlarni qoplash uchun likvid aktivlarni oshirish yoki qisqa muddatli qarzlarni qayta moliyalashtirish.",
    "equity": "O'z kapitali ulushini oshirish (foydani taqsimlamaslik, ustav kapitalini ko'paytirish).",
    "profit": "Xarajatlarni optimallashtirish va rentabellikni oshirish choralarini ko'rish.",
    "reports": "Soliq hisobotlarini o'z vaqtida elektron topshirish uchun mas'ul shaxs va kalendar belgilash.",
}


def recommendations(row_scores: pd.Series, top=3):
    weak = sorted(WEIGHTS, key=lambda k: WEIGHTS[k] * (100 - row_scores[f"sub_{k}"]), reverse=True)
    return [(LABELS[k], RECOMMENDATIONS[k]) for k in weak[:top] if row_scores[f"sub_{k}"] < 70]
