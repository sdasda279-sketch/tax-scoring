"""Xom ma'lumotlardan moliyaviy koeffitsientlar (feature) hisoblash."""
import numpy as np
import pandas as pd

# Tovar qoldig'i hisobi yuritiladigan tarmoqlar: (tarmoq o'rtacha ustamasi tannarxga nisbatan, me'yoriy qoldiq aylanish kunlari)
SECTOR_INV = {
    "Savdo": (0.25, 45), "Ishlab chiqarish": (0.35, 60), "Qurilish": (0.30, 50),
    "Qishloq xo'jaligi": (0.40, 90), "Umumiy ovqatlanish": (0.80, 8),
}
DEFAULT_INV = (0.30, 45)

# Xom (kiritiladigan) ustunlar. Pul birligi: mln so'm
RAW_COLUMNS = [
    "years_active", "employees", "revenue", "net_profit", "total_assets", "equity",
    "current_assets", "current_liabilities", "tax_accrued", "tax_debt",
    "late_payments_count", "avg_delay_days", "late_reports_count",
    "audit_violations", "vat_payer", "has_inventory", "inv_opening_value", "inv_purchases",
    "inv_closing_value", "sector", "legal_form",
]

MODEL_NUM = [
    "years_active", "log_revenue", "employees", "debt_to_revenue", "debt_to_tax",
    "current_ratio", "equity_ratio", "profit_margin", "roa", "tax_burden",
    "late_payments_count", "avg_delay_days", "late_reports_count",
    "audit_violations", "vat_payer", "has_inventory", "turnover_gap", "stock_ratio", "stock_share",
]

# Foydalanuvchiga ko'rsatiladigan nomlar
FEATURE_LABELS = {
    "years_active": "Faoliyat yoshi (yil)",
    "log_revenue": "Aylanma hajmi (log)",
    "employees": "Xodimlar soni",
    "debt_to_revenue": "Soliq qarzi / aylanma",
    "debt_to_tax": "Soliq qarzi / hisoblangan soliq",
    "current_ratio": "Joriy likvidlik koeffitsienti",
    "equity_ratio": "Avtonomlik koeffitsienti",
    "profit_margin": "Sof rentabellik",
    "roa": "Aktivlar rentabelligi (ROA)",
    "tax_burden": "Soliq yuki",
    "late_payments_count": "Kechikkan to'lovlar soni",
    "avg_delay_days": "O'rtacha kechikish (kun)",
    "late_reports_count": "Kech topshirilgan hisobotlar",
    "audit_violations": "Tekshiruvdagi qoidabuzarliklar",
    "vat_payer": "QQS to'lovchisi",
    "has_inventory": "Tovar qoldig'i hisobi yuritiladi",
    "turnover_gap": "Aylanma mosligi (deklaratsiya va kutilgan farqi)",
    "stock_ratio": "Qoldiq aylanishi / tarmoq me'yori",
    "stock_share": "Qoldiq summasi / aylanma",
}


def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    rev = d["revenue"].clip(lower=1)
    d["debt_to_revenue"] = d["tax_debt"] / rev
    d["debt_to_tax"] = d["tax_debt"] / d["tax_accrued"].clip(lower=0.01)
    d["current_ratio"] = d["current_assets"] / d["current_liabilities"].clip(lower=0.01)
    d["equity_ratio"] = d["equity"] / d["total_assets"].clip(lower=0.01)
    d["profit_margin"] = d["net_profit"] / rev
    d["roa"] = d["net_profit"] / d["total_assets"].clip(lower=0.01)
    d["tax_burden"] = d["tax_accrued"] / rev
    d["log_revenue"] = np.log1p(d["revenue"].clip(lower=0))
    d["vat_payer"] = d["vat_payer"].astype(int)

    # --- Tovar balansi: aylanma manbasini tekshirish ---
    inv = d["has_inventory"].astype(int).eq(1).to_numpy()
    markup = d["sector"].map(lambda s: SECTOR_INV.get(s, DEFAULT_INV)[0]).to_numpy(float)
    norm = d["sector"].map(lambda s: SECTOR_INV.get(s, DEFAULT_INV)[1]).to_numpy(float)
    opening, purchases = d["inv_opening_value"].to_numpy(float), d["inv_purchases"].to_numpy(float)
    closing = d["inv_closing_value"].to_numpy(float)
    cogs = np.maximum(opening + purchases - closing, 0.01)          # sotilgan tovar tannarxi
    expected = cogs * (1 + markup)                                   # kutilgan aylanma
    declared = d["revenue"].to_numpy(float)
    d["inv_cogs"] = np.where(inv, cogs, 0.0)
    d["expected_revenue"] = np.where(inv, expected, 0.0)
    d["turnover_gap"] = np.where(inv, (declared - expected) / np.maximum(expected, 0.01), 0.0)
    stock_days = (opening + closing) / 2 / cogs * 365
    d["stock_days"] = np.where(inv, stock_days, 0.0)
    d["stock_ratio"] = np.where(inv, stock_days / norm, 1.0)
    d["stock_share"] = np.where(inv, closing / np.maximum(declared, 1), 0.0)
    return d


def build_X(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """ML modeli uchun belgilar matritsasi. columns berilsa, shu ustunlar tartibiga keltiriladi."""
    d = add_ratios(df)
    X = d[MODEL_NUM].astype(float).copy()
    sec = pd.get_dummies(d["sector"], prefix="sector").astype(float)
    leg = pd.get_dummies(d["legal_form"], prefix="legal").astype(float)
    X = pd.concat([X, sec, leg], axis=1)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X
