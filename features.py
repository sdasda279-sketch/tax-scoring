"""Xom ma'lumotlardan moliyaviy koeffitsientlar (feature) hisoblash."""
import numpy as np
import pandas as pd

# Xom (kiritiladigan) ustunlar. Pul birligi: mln so'm
RAW_COLUMNS = [
    "years_active", "employees", "revenue", "net_profit", "total_assets", "equity",
    "current_assets", "current_liabilities", "tax_accrued", "tax_debt",
    "late_payments_count", "avg_delay_days", "late_reports_count",
    "audit_violations", "vat_payer", "sector", "legal_form",
]

MODEL_NUM = [
    "years_active", "log_revenue", "employees", "debt_to_revenue", "debt_to_tax",
    "current_ratio", "equity_ratio", "profit_margin", "roa", "tax_burden",
    "late_payments_count", "avg_delay_days", "late_reports_count",
    "audit_violations", "vat_payer",
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
