"""Yuridik shaxslarning soliq qarzdorligi va moliyaviy ishonchliligini baholash (Tax Scoring) tizimi.
Ishga tushirish:  streamlit run app.py"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import plots
from features import FEATURE_LABELS
from data_generator import REGIONS, SECTORS, LEGAL
from ml_model import explain_local, hybrid_100, load_model, predict_pd, score_dataset
from report import passport_pdf, to_excel
from scoring import classify, contributions, recommendations, rule_score, to_scale

st.set_page_config(page_title="Tax Scoring", page_icon="📊", layout="wide")

# ---------- ma'lumotlarni yuklash (yo'q bo'lsa avtomatik yaratiladi) ----------
if not Path("data/scored_companies.csv").exists() or not Path("models/metrics.json").exists():
    with st.spinner("Birinchi ishga tushirish: ma'lumotlar yaratilmoqda va modellar o'qitilmoqda..."):
        import main as pipeline
        Path("data").mkdir(exist_ok=True); Path("models").mkdir(exist_ok=True)
        pipeline.run()


@st.cache_data
def load_scored():
    return pd.read_csv("data/scored_companies.csv", dtype={"stir": str})


@st.cache_data
def load_meta():
    return json.loads(Path("models/metrics.json").read_text())


@st.cache_resource
def get_bundle():
    return load_model()


df, meta, bundle = load_scored(), load_meta(), get_bundle()
RAW_COLS = ["stir", "name", "region", "sector", "legal_form", "years_active", "employees", "revenue",
            "net_profit", "total_assets", "equity", "current_assets", "current_liabilities", "tax_accrued",
            "tax_debt", "late_payments_count", "avg_delay_days", "late_reports_count", "audit_violations", "vat_payer"]

st.sidebar.title("📊 Tax Scoring")
st.sidebar.caption("Yuridik shaxslarning soliq qarzdorligi va moliyaviy ishonchliligini baholash axborot tizimi")
page = st.sidebar.radio("Bo'lim", ["Dashboard", "Korxonani tekshirish (STIR)", "Yangi korxona (what-if)",
                                   "Korxonalar ro'yxati", "Erta ogohlantirish", "Modellar taqqoslash", "Tizim haqida"])
st.sidebar.info("Barcha ma'lumotlar sintetik (sun'iy) — o'quv maqsadida yaratilgan.")


def show_company(r: pd.Series, raw_row: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Yakuniy ball (gibrid)", f"{r['final_score']:.0f}", help="300–850 shkalasi")
    c2.metric("Xavf sinfi", r["risk_class"])
    c3.metric("Scorecard bali", f"{r['rule_score']:.0f}")
    c4.metric("ML xavf ehtimoli (PD)", f"{r['pd']:.1%}")
    left, right = st.columns(2)
    with left:
        st.subheader("Ko'rsatkichlar bo'yicha ballar")
        cb = contributions(r)
        st.pyplot(plots.contribution_bar(cb))
        st.dataframe(cb, hide_index=True, use_container_width=True)
        if r["audit_penalty"] > 0:
            st.caption(f"Tekshiruv qoidabuzarliklari uchun jarima: −{r['audit_penalty']:.0f} ball")
    with right:
        st.subheader("Nima uchun bunday ball? (ML tushuntirishi)")
        st.pyplot(plots.local_bar(explain_local(bundle, raw_row)))
        st.subheader("Tavsiyalar")
        recs = recommendations(r)
        if not recs:
            st.success("Jiddiy muammolar aniqlanmadi.")
        for k, t in recs:
            st.write(f"**{k}:** {t}")


# ---------------------------------- DASHBOARD ----------------------------------
if page == "Dashboard":
    st.title("Umumiy ko'rsatkichlar")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Korxonalar soni", f"{len(df):,}")
    c2.metric("O'rtacha ball", f"{df['final_score'].mean():.0f}")
    c3.metric("Yuqori xavfli ulush", f"{(df['risk_class'] == 'Yuqori xavf').mean():.1%}")
    c4.metric("Jami soliq qarzi (mlrd so'm)", f"{df['tax_debt'].sum() / 1000:,.1f}")
    a, b = st.columns(2)
    a.subheader("Ballar taqsimoti"); a.pyplot(plots.score_hist(df))
    b.subheader("Xavf sinflari"); b.pyplot(plots.risk_bar(df))
    a, b = st.columns(2)
    a.subheader("Tarmoqlar bo'yicha o'rtacha ball"); a.pyplot(plots.sector_bar(df))
    b.subheader("Hududlar: yuqori xavfli korxonalar ulushi"); b.pyplot(plots.region_bar(df))
    st.subheader("Eng yuqori soliq qarzi bo'lgan 10 ta korxona")
    st.dataframe(df.nlargest(10, "tax_debt")[["stir", "name", "region", "sector", "tax_debt", "final_score", "risk_class"]],
                 hide_index=True, use_container_width=True)

# ---------------------------------- STIR BO'YICHA ----------------------------------
elif page == "Korxonani tekshirish (STIR)":
    st.title("Korxonani STIR yoki nomi bo'yicha tekshirish")
    q = st.text_input("STIR (9 xonali) yoki nom", placeholder="masalan: 309988287")
    st.caption(f"Namuna uchun STIR: {', '.join(df['stir'].head(3))}")
    if q:
        m = df[df["stir"].str.contains(q.strip()) | df["name"].str.contains(q.strip(), case=False, regex=False)]
        if m.empty:
            st.warning("Korxona topilmadi.")
        else:
            idx = m.index[0] if len(m) == 1 else st.selectbox(
                f"{len(m)} ta natija", m.index[:50], format_func=lambda i: f"{df.loc[i, 'stir']} — {df.loc[i, 'name']}")
            r = df.loc[idx]
            st.header(f"{r['name']}  ·  STIR {r['stir']}")
            st.caption(f"{r['region']} · {r['sector']} · {r['legal_form']} · {r['years_active']} yil")
            show_company(r, df.loc[[idx], RAW_COLS])
            st.download_button("📄 Soliq pasportini yuklab olish (PDF)", passport_pdf(r),
                               file_name=f"soliq_pasporti_{r['stir']}.pdf", mime="application/pdf")

# ---------------------------------- WHAT-IF ----------------------------------
elif page == "Yangi korxona (what-if)":
    st.title("Yangi korxona ma'lumotlarini kiritish")
    st.caption("Ko'rsatkichlarni o'zgartirib, ball qanday o'zgarishini ko'ring. Pul birligi: mln so'm.")
    with st.form("whatif"):
        a, b, c = st.columns(3)
        sector = a.selectbox("Tarmoq", list(SECTORS)); legal = b.selectbox("Tashkiliy shakl", list(LEGAL))
        years = c.number_input("Faoliyat yoshi (yil)", 0, 50, 5)
        emp = a.number_input("Xodimlar soni", 1, 10000, 25)
        revenue = b.number_input("Yillik aylanma", 1.0, 1e6, 1500.0)
        profit = c.number_input("Sof foyda", -1e5, 1e5, 90.0)
        assets = a.number_input("Jami aktivlar", 1.0, 1e6, 1000.0)
        equity = b.number_input("O'z kapitali", -1e5, 1e6, 450.0)
        ca = c.number_input("Joriy aktivlar", 1.0, 1e6, 600.0)
        cl = a.number_input("Joriy majburiyatlar", 1.0, 1e6, 400.0)
        tax = b.number_input("Yillik hisoblangan soliq", 0.1, 1e5, 90.0)
        debt = c.number_input("Soliq qarzi", 0.0, 1e5, 5.0)
        late = a.slider("Kechikkan to'lovlar (12 oy)", 0, 12, 1)
        delay = b.slider("O'rtacha kechikish (kun)", 0, 120, 5)
        lrep = c.slider("Kech topshirilgan hisobotlar (12 oy)", 0, 12, 0)
        aud = a.slider("Tekshiruvdagi qoidabuzarliklar", 0, 10, 0)
        vat = b.checkbox("QQS to'lovchisi", value=revenue > 1000)
        go = st.form_submit_button("Hisoblash", type="primary")
    if go:
        raw = pd.DataFrame([{"stir": "—", "name": "Yangi korxona", "region": "—", "sector": sector, "legal_form": legal,
                             "years_active": years, "employees": emp, "revenue": revenue, "net_profit": profit,
                             "total_assets": assets, "equity": equity, "current_assets": ca, "current_liabilities": cl,
                             "tax_accrued": tax, "tax_debt": debt, "late_payments_count": late, "avg_delay_days": delay,
                             "late_reports_count": lrep, "audit_violations": aud, "vat_payer": int(vat)}])
        res = score_dataset(raw, predict_pd(bundle, raw))
        show_company(res.iloc[0], raw)

# ---------------------------------- RO'YXAT ----------------------------------
elif page == "Korxonalar ro'yxati":
    st.title("Korxonalar ro'yxati")
    a, b, c = st.columns(3)
    risk = a.multiselect("Xavf sinfi", list(plots.COLORS), default=list(plots.COLORS))
    reg = b.multiselect("Hudud", sorted(df["region"].unique()))
    sec = c.multiselect("Tarmoq", sorted(df["sector"].unique()))
    f = df[df["risk_class"].isin(risk)]
    if reg: f = f[f["region"].isin(reg)]
    if sec: f = f[f["sector"].isin(sec)]
    show = ["stir", "name", "region", "sector", "revenue", "tax_debt", "late_payments_count",
            "rule_score", "pd", "final_score", "risk_class"]
    st.write(f"Topildi: **{len(f):,}** ta korxona")
    st.dataframe(f[show].sort_values("final_score"), hide_index=True, use_container_width=True)
    st.download_button("⬇️ Excelga yuklash", to_excel(f[show]), file_name="korxonalar.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------------------------- ERTA OGOHLANTIRISH ----------------------------------
elif page == "Erta ogohlantirish":
    st.title("Erta ogohlantirish tizimi")
    st.write("Hozircha **yuqori xavf** sinfiga tushmagan, ammo ML modeli bo'yicha qarzdorlik ehtimoli yuqori bo'lgan korxonalar.")
    thr = st.slider("PD chegarasi", 0.10, 0.60, 0.25, 0.01)
    w = df[(df["pd"] >= thr) & (df["risk_class"] != "Yuqori xavf")].sort_values("pd", ascending=False)
    st.metric("Aniqlangan korxonalar", len(w))
    st.dataframe(w[["stir", "name", "region", "sector", "tax_debt", "late_payments_count", "final_score", "pd", "risk_class"]],
                 hide_index=True, use_container_width=True)
    if len(w):
        st.download_button("⬇️ Ro'yxatni Excelga yuklash", to_excel(w), file_name="erta_ogohlantirish.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------------------------- MODELLAR ----------------------------------
elif page == "Modellar taqqoslash":
    st.title("Modellar taqqoslash")
    st.caption(f"Test to'plami: {meta['n_test']} ta korxona. Xavfli qarzdorlik ulushi: {meta['default_rate']:.1%}. "
               f"ML uchun tanlangan model (CV AUC bo'yicha): **{meta['best']}**")
    m = pd.DataFrame(meta["metrics"]).T
    st.dataframe(m.round(3), use_container_width=True)
    a, b = st.columns(2)
    a.subheader("ROC egri chiziqlari"); a.pyplot(plots.roc_plot(meta))
    b.subheader("Belgilar muhimligi"); b.pyplot(plots.importance_bar(meta))
    st.subheader("Xavf sinflari bo'yicha haqiqiy qarzdorlik ulushi (kalibrovka tekshiruvi)")
    g = df.groupby("risk_class")["default_next_year"].agg(["count", "mean"]).reindex(list(plots.COLORS))
    g.columns = ["Korxonalar soni", "Xavfli qarzdorlik ulushi"]
    st.dataframe(g.style.format({"Xavfli qarzdorlik ulushi": "{:.1%}"}), use_container_width=True)

# ---------------------------------- HAQIDA ----------------------------------
else:
    st.title("Tizim haqida")
    st.markdown("""
**Maqsad:** yuridik shaxslarning soliq intizomi va moliyaviy holatiga ko'ra ball (300–850) va xavf sinfini aniqlash.

**Gibrid yondashuv**
1. *Scorecard (ekspert model)*: 6 ta ko'rsatkich vaznlar bilan (qarz 25%, to'lov intizomi 20%, likvidlik 15%,
   avtonomlik 15%, rentabellik 15%, hisobotlar 10%) + tekshiruv qoidabuzarliklari uchun jarima.
2. *ML modeli*: Logistic Regression, Random Forest, Gradient Boosting. Maqsad: keyingi yilda xavfli qarzdorlik (PD).
3. *Gibrid ball* = 0.5 × scorecard + 0.5 × ML ball.

**Xavf sinflari** (100 ballik shkalada): ≥60 — past; 35–60 — o'rtacha; <35 — yuqori.

**Ma'lumotlar:** haqiqiy soliq ma'lumotlari maxfiy, shuning uchun sintetik ma'lumotlar ishlatilgan.
Haqiqiy ma'lumot bilan ishlash uchun `data/companies.csv` ni shu ustunlar formatida almashtirib, `python main.py` ni qayta ishga tushirish kifoya.
""")
