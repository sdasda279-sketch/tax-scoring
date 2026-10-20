"""Korxona uchun PDF 'soliq pasporti' va Excel eksport."""
from io import BytesIO
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from scoring import contributions, recommendations

RISK_COL = {"Past xavf": "#2e9e5b", "O'rtacha xavf": "#e0a526", "Yuqori xavf": "#d64545"}


def _tbl(data, widths=None, header=True):
    t = Table(data, colWidths=widths)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 9),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef6")),
               ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(st)); return t


def passport_pdf(r: pd.Series) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=15 * mm, bottomMargin=15 * mm)
    ss = getSampleStyleSheet(); S = []
    S.append(Paragraph("SOLIQ PASPORTI (Tax Scoring)", ss["Title"]))
    S.append(Paragraph(f"{r['name']} | STIR: {r['stir']}", ss["Heading3"]))
    S.append(Spacer(1, 4))
    col = RISK_COL[r["risk_class"]]
    S.append(Paragraph(
        f"<font size=26 color='{col}'><b>{r['final_score']:.1f}</b></font> "
        f"<font size=11>/ 100 &nbsp;&nbsp; <b>{r['risk_class']}</b></font>",
        ParagraphStyle("big", parent=ss["Normal"], leading=36, spaceAfter=4)))
    S.append(Paragraph(
        f"Qoidaga asoslangan ball: {r['rule_score']:.1f} | ML xavf ehtimoli (PD): {r['pd']:.1%} | "
        f"Gibrid ball: {r['final_score']:.1f}", ss["Normal"]))
    S.append(Spacer(1, 8))
    info = [["Hudud", r["region"], "Tarmoq", r["sector"]],
            ["Tashkiliy shakl", r["legal_form"], "Faoliyat yoshi", f"{r['years_active']} yil"],
            ["Aylanma (mln so'm)", f"{r['revenue']:,.1f}", "Sof foyda (mln so'm)", f"{r['net_profit']:,.1f}"],
            ["Hisoblangan soliq", f"{r['tax_accrued']:,.1f}", "Soliq qarzi", f"{r['tax_debt']:,.2f}"],
            ["Kechikkan to'lovlar", int(r["late_payments_count"]), "O'rtacha kechikish (kun)", f"{r['avg_delay_days']:.1f}"],
            ["Kech hisobotlar", int(r["late_reports_count"]), "Tekshiruv qoidabuzarliklari", int(r["audit_violations"])]]
    S.append(_tbl(info, [38 * mm, 45 * mm, 50 * mm, 40 * mm], header=False))
    S.append(Spacer(1, 10))
    S.append(Paragraph("Ko'rsatkichlar bo'yicha ballar", ss["Heading4"]))
    c = contributions(r)
    S.append(_tbl([list(c.columns)] + c.values.tolist(), [75 * mm, 20 * mm, 40 * mm, 35 * mm]))
    if r["audit_penalty"] > 0:
        S.append(Paragraph(f"Tekshiruv qoidabuzarliklari uchun jarima: -{r['audit_penalty']:.0f} ball", ss["Normal"]))
    S.append(Spacer(1, 10))
    S.append(Paragraph("Tavsiyalar", ss["Heading4"]))
    recs = recommendations(r)
    if not recs:
        S.append(Paragraph("Jiddiy muammolar aniqlanmadi. Joriy to'lov intizomini saqlash tavsiya etiladi.", ss["Normal"]))
    for k, t in recs:
        S.append(Paragraph(f"<b>{k}:</b> {t}", ss["Normal"]))
    S.append(Spacer(1, 14))
    S.append(Paragraph("<i>Hujjat sintetik ma'lumotlar asosida tayyorlangan o'quv prototipi natijasidir.</i>", ss["Normal"]))
    doc.build(S)
    return buf.getvalue()


def to_excel(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Korxonalar")
    return buf.getvalue()
