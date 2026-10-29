"""Yuridik shaxslarning soliq qarzdorligi va moliyaviy ishonchliligini baholash (Tax Scoring) tizimi.
Ishga tushirish:  streamlit run app.py"""
import json
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st

import plots
from features import FEATURE_LABELS, SECTOR_INV
from data_generator import REGIONS, SECTORS, LEGAL
from ml_model import explain_local, hybrid_100, load_model, predict_pd, score_dataset
import advisor
import auth
import brand
import theme
from notifier import build_email, build_sms, channel_ready, dispatch, load_config, sms_segments
from report import passport_pdf, to_excel
from scoring import classify, contributions, recommendations, rule_score, to_scale, turnover_verdict

try:
    st.set_page_config(page_title="Soliq Skoring", page_icon=brand.favicon(), layout="wide")
except Exception:  # eski Streamlit versiyalarida rasm yo'lini qabul qilmasa
    st.set_page_config(page_title="Soliq Skoring", page_icon="🏛️", layout="wide")

# ---------- ma'lumotlarni yuklash (yo'q bo'lsa avtomatik yaratiladi) ----------
def _needs_build():
    p = Path("data/scored_companies.csv")
    if not p.exists() or not Path("models/metrics.json").exists():
        return True
    cols = pd.read_csv(p, nrows=0).columns
    if "phone" not in cols or "has_inventory" not in cols:
        return True
    return pd.read_csv(p, usecols=["final_score"])["final_score"].max() > 100  # eski 300-850 shkala qolgan bo'lsa


if _needs_build():
    with st.spinner("Birinchi ishga tushirish: ma'lumotlar yaratilmoqda va modellar o'qitilmoqda..."):
        import main as pipeline
        Path("data").mkdir(exist_ok=True); Path("models").mkdir(exist_ok=True)
        pipeline.run()


@st.cache_data
def load_scored(file_version):
    """file_version (fayl o'zgargan vaqti) o'zgarsa, kesh yangilanadi."""
    d = pd.read_csv("data/scored_companies.csv", dtype={"stir": str, "phone": str})
    if "phone" not in d.columns:  # eski ma'lumot: aloqa ustunlarini STIR'dan hosil qilamiz
        ops = ["90", "91", "93", "94", "95", "97", "98", "99", "33", "88", "77"]
        d["phone"] = ["+998" + ops[int(x[-2:]) % 11] + str(1_000_000 + int(x) % 9_000_000) for x in d["stir"]]
        d["email"] = ["info" + x[-5:] + "@example.com" for x in d["stir"]]
    return d


@st.cache_data
def load_goods(file_version):
    p = Path("data/goods.csv")
    if p.exists():
        return pd.read_csv(p, dtype={"stir": str, "phone": str})
    return pd.DataFrame(columns=["stir", "group", "unit", "quantity", "unit_price", "value"])


@st.cache_data
def load_meta():
    return json.loads(Path("models/metrics.json").read_text())


@st.cache_resource
def get_bundle():
    return load_model()


goods = load_goods(Path("data/goods.csv").stat().st_mtime if Path("data/goods.csv").exists() else 0)
df, meta, bundle = load_scored(Path("data/scored_companies.csv").stat().st_mtime), load_meta(), get_bundle()

# ====================================== KIRISH (LOGIN) ======================================
SESSION_TIMEOUT = 30 * 60  # 30 daqiqa faolsizlikdan keyin sessiya tugaydi
DEMO_SHOW_CODE = os.environ.get("DEMO_SHOW_CODE", "1") != "0"  # SMS provayder bo'lmaganda kodni ekranda ko'rsatish (simulyatsiya)


def _secret(name):
    v = os.environ.get(name)
    if v:
        return v
    try:
        return st.secrets.get(name)
    except Exception:
        return None


@st.cache_resource
def _bootstrap_admin():
    auth.bootstrap_admin(_secret("ADMIN_USERNAME") or "admin", _secret("ADMIN_PASSWORD"))
    return True


def _send_registration_sms(phone, code):
    """SMS provayder sozlangan bo'lsa, kodni haqiqiy yuboradi; aks holda (False, sabab)."""
    cfg = load_config(st.secrets)
    if not channel_ready("SMS", cfg):
        return False, "SMS provayder sozlanmagan."
    try:
        from notifier import _eskiz_token, send_sms
        send_sms(phone, f"Tax Scoring tasdiqlash kodi: {code}", cfg, _eskiz_token(cfg))
        return True, ""
    except Exception as e:
        return False, f"SMS yuborilmadi: {str(e)[:80]}"


def auth_gate():
    _bootstrap_admin()
    a = st.session_state.get("auth")
    now = time.time()
    if a and now - a["last"] > SESSION_TIMEOUT:
        st.session_state.clear()
        a = None
        st.warning("Xavfsizlik uchun sessiya tugadi. Qayta kiring.")
    if a:
        a["last"] = now
        return a

    theme.login_screen()  # tungi fon, harakatlanuvchi tarmoq chiziqlari, katta logotip
    if not auth.has_admin():
        st.warning("Tizimda hali administrator yo'q. Birinchi administratorni yarating.")
        st.caption("Sozlash kodi server konsolida (terminal oynasida yoki Streamlit Cloud: Manage app → logs) "
                   "«[AUTH] Birinchi sozlash kodi» satrida chiqadi. Ommaviy saytda buning o'rniga Secrets'ga "
                   "ADMIN_USERNAME va ADMIN_PASSWORD yozing.")
        auth.setup_code()
        with st.form("setup"):
            code = st.text_input("Sozlash kodi")
            u = st.text_input("Administrator logini", value="admin")
            p = st.text_input("Parol", type="password")
            p2 = st.text_input("Parolni takrorlang", type="password")
            go = st.form_submit_button("Administratorni yaratish")
        if go:
            if not auth.check_setup_code(code):
                st.error("Sozlash kodi noto'g'ri.")
            elif p != p2:
                st.error("Parollar mos kelmadi.")
            else:
                ok, msg = auth.create_user(u, p, "admin")
                (st.success if ok else st.error)(msg + (" Endi kirishingiz mumkin." if ok else ""))
                if ok:
                    st.rerun()
        st.stop()

    tab_in, tab_reg = st.tabs(["🔐 Kirish", "🏢 Yuridik shaxs: ro'yxatdan o'tish"])
    with tab_in:
        st.caption("Inspektor/administrator: o'z logini. Yuridik shaxs: login = STIR.")
        with st.form("login"):
            lu = st.text_input("Login")
            lp = st.text_input("Parol", type="password")
            go = st.form_submit_button("Kirish", type="primary")
        if go:
            user, msg = auth.verify(lu, lp)
            if user:
                st.session_state["auth"] = {**user, "last": time.time()}
                st.session_state["splash"] = True  # keyingi sahifada katta logotip chiqadi
                st.rerun()
            else:
                st.error(msg)
    with tab_reg:
        st.caption("Korxona rahbari STIR bo'yicha hisob yaratadi. Shaxs korxonaning ro'yxatdan o'tgan telefon raqamiga "
                   "yuborilgan bir martalik kod bilan tasdiqlanadi.")
        reg = st.session_state.get("reg")
        if not reg:
            with st.form("reg1"):
                stir_in = st.text_input("STIR (9 xonali)")
                go1 = st.form_submit_button("Tasdiqlash kodini yuborish")
            if go1:
                sv = (stir_in or "").strip()
                hit = df[df["stir"] == sv] if (sv.isdigit() and len(sv) == 9) else df.iloc[0:0]
                if hit.empty:
                    st.error("Bu STIR bazada topilmadi.")
                elif auth.get_user(sv):
                    st.warning("Bu korxona uchun hisob allaqachon mavjud: «Kirish» bo'limidan kiring.")
                else:
                    state = auth.issue_code(sv)
                    phone = hit.iloc[0]["phone"]
                    sent = False
                    if not DEMO_SHOW_CODE:
                        sent, why = _send_registration_sms(phone, state["code"])
                        if not sent:
                            st.error("Kodni yuborib bo'lmadi. " + why)
                    if DEMO_SHOW_CODE or sent:
                        st.session_state["reg"] = {"state": state, "phone": phone, "name": hit.iloc[0]["name"]}
                        st.rerun()
        else:
            st.info(f"**{reg['name']}**: kod {auth.mask_phone(reg['phone'])} raqamiga yuborildi (amal qilish muddati 5 daqiqa).")
            if DEMO_SHOW_CODE:
                st.warning(f"🧪 SMS simulyatsiyasi (provayder ulanmagan): tasdiqlash kodi **{reg['state']['code']}**")
            with st.form("reg2"):
                code_in = st.text_input("Tasdiqlash kodi")
                pw1 = st.text_input("Yangi parol (kamida 8 belgi, harf va raqam)", type="password")
                pw2 = st.text_input("Parolni takrorlang", type="password")
                go2 = st.form_submit_button("Hisobni yaratish", type="primary")
            if go2:
                ok, msg = auth.check_code(reg["state"], code_in)
                if not ok:
                    st.error(msg)
                elif pw1 != pw2:
                    st.error("Parollar mos kelmadi.")
                else:
                    ok, msg = auth.create_user(reg["state"]["stir"], pw1, "company", reg["state"]["stir"])
                    if ok:
                        auth.log("ro'yxatdan o'tdi (kod bilan tasdiqlandi)", reg["state"]["stir"])
                        st.session_state.pop("reg", None)
                        st.success("Hisob yaratildi. «Kirish» bo'limida login sifatida STIR'ingizni kiriting.")
                    else:
                        st.error(msg)
            if st.button("Bekor qilish", key="reg_cancel"):
                st.session_state.pop("reg", None)
                st.rerun()
    st.stop()


user = auth_gate()
role = user["role"]
IS_STAFF = role in ("admin", "inspector")
theme.apply_base()
if st.session_state.pop("splash", False):
    theme.splash(user["username"])

RAW_COLS = ["stir", "name", "region", "sector", "legal_form", "years_active", "employees", "revenue",
            "net_profit", "total_assets", "equity", "current_assets", "current_liabilities", "tax_accrued",
            "tax_debt", "late_payments_count", "avg_delay_days", "late_reports_count", "audit_violations", "vat_payer",
            "has_inventory", "inv_opening_value", "inv_purchases", "inv_closing_value"]

COL_UZ = {
    "stir": "STIR", "name": "Korxona nomi", "region": "Hudud", "sector": "Tarmoq",
    "legal_form": "Tashkiliy shakl", "years_active": "Faoliyat yoshi (yil)", "employees": "Xodimlar soni",
    "revenue": "Aylanma (mln so'm)", "net_profit": "Sof foyda (mln so'm)",
    "tax_accrued": "Hisoblangan soliq (mln so'm)", "tax_debt": "Soliq qarzi (mln so'm)",
    "late_payments_count": "Kechikkan to'lovlar soni", "avg_delay_days": "O'rtacha kechikish (kun)",
    "rule_score": "Scorecard bali", "pd": "Qarzdorlik xavfi, %", "final_score": "Yakuniy ball",
    "risk_class": "Xavf sinfi", "expected_revenue": "Kutilgan aylanma (mln so'm)",
    "turnover_gap": "Aylanma farqi, %", "inv_closing_value": "Qoldiq tovar (mln so'm)",
}


def view(d, cols):
    """Jadvalni foydalanuvchi uchun tayyorlash: o'zbekcha ustunlar, yaxlitlangan raqamlar."""
    v = d[cols].copy()
    for c in ("rule_score", "final_score"):
        if c in v:
            v[c] = v[c].round(1)
    if "pd" in v:
        v["pd"] = (v["pd"] * 100).round(1)
    if "turnover_gap" in v:
        v["turnover_gap"] = (v["turnover_gap"] * 100).round(1)
    for c in ("revenue", "net_profit", "tax_accrued", "tax_debt", "avg_delay_days", "expected_revenue", "inv_closing_value"):
        if c in v:
            v[c] = v[c].round(1)
    return v.rename(columns=COL_UZ)


theme.sidebar_header(user["username"], auth.ROLES[role])
STAFF_PAGES = ["Dashboard", "Korxonani tekshirish (STIR)", "Tovar qoldig'i va aylanma manbasi",
                                   "Yangi korxona (what-if)",
                                   "Korxonalar ro'yxati", "Erta ogohlantirish", "Ogohlantirish yuborish",
                                   "Modellar taqqoslash", "Raqamlar izohi", "Tizim haqida"]
PAGES = ["Mening korxonam", "Raqamlar izohi"] if role == "company" else STAFF_PAGES + (["Foydalanuvchilar"] if role == "admin" else [])
if st.sidebar.button("🚪 Chiqish"):
    auth.log("chiqish", user["username"])
    st.session_state.clear()
    st.rerun()
with st.sidebar.expander("🔑 Parolni o'zgartirish"):
    with st.form("chpw", clear_on_submit=True):
        old_pw = st.text_input("Joriy parol", type="password")
        new_pw = st.text_input("Yangi parol", type="password")
        go_pw = st.form_submit_button("O'zgartirish")
    if go_pw:
        ok_pw, msg_pw = auth.change_own_password(user["username"], old_pw, new_pw)
        (st.success if ok_pw else st.error)(msg_pw)
page = st.sidebar.radio("Bo'lim", PAGES, format_func=theme.nav_label, label_visibility="collapsed")
if page not in PAGES:  # himoya: ruxsatsiz bo'limga o'tishga yo'l qo'yilmaydi
    st.error("Bu bo'limga ruxsat yo'q.")
    st.stop()
st.sidebar.info("Barcha ma'lumotlar sintetik (sun'iy) — o'quv maqsadida yaratilgan.")
if IS_STAFF:
    with st.sidebar.expander("🤖 AI sozlamalari"):
        st.checkbox("AI yoqilgan", value=True, key="ai_on")
        st.text_input("Claude API kaliti (ixtiyoriy)", type="password", key="ai_key_input",
                      help="Faqat shu sessiyada saqlanadi. Ommaviy saytda kalitni kodga emas, Secrets orqali bering.")
        if advisor.load_ai_config(st.secrets, st.session_state.get("ai_key_input", "")).get("key"):
            st.success("Kalit topildi: Claude AI ishlaydi.")
        else:
            st.info("Kalit yo'q: qolipli maslahatchi ishlaydi.")


def show_company(r: pd.Series, raw_row: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Yakuniy ball (gibrid)", f"{r['final_score']:.1f}", help="0–100 shkalasi")
    c2.metric("Xavf sinfi", r["risk_class"])
    c3.metric("Scorecard bali", f"{r['rule_score']:.1f}")
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


AI_SESSION_LIMIT = 25  # bitta foydalanuvchi sessiyasida eng ko'pi bilan nechta AI so'rovi
ai_cfg = advisor.load_ai_config(st.secrets, st.session_state.get("ai_key_input", ""))


def show_ai_result(res):
    st.markdown(res["text"])
    for w in res["warnings"]:
        st.warning("⚠️ " + w)
    st.caption(f"Rejim: {res['mode']} · {advisor.DISCLAIMER}")


def advisor_section(r, raw_row, staff=True):
    """Korxona kartochkasi ostidagi AI maslahatchi: reja, savol-javob, inspektor yordamchisi."""
    stir = r["stir"]
    st.divider()
    st.subheader("🤖 AI maslahatchi")
    use_ai = advisor.ai_enabled(ai_cfg, st.session_state.get("ai_on", True))
    ai_ok = use_ai and st.session_state.get("ai_calls", 0) < AI_SESSION_LIMIT
    if use_ai and not ai_ok:
        st.info("Bu sessiyada AI so'rovlari limiti tugadi, qolipli maslahatchi ishlaydi.")
    st.caption("🟢 Claude AI ulangan: hisob-kitoblar Python'da bajariladi, AI faqat tayyor natijalarni tushuntiradi."
               if ai_ok else "⚪ Qolipli maslahatchi: matnlar tayyor qoliplar asosida shakllanadi, hisob-kitoblar bir xil.")
    cache = st.session_state.setdefault("facts_cache", {})
    if stir not in cache:
        with st.spinner("Korxona ma'lumotlari tahlil qilinmoqda..."):
            cache[stir] = advisor.build_facts(r, raw_row, bundle)
    facts = cache[stir]

    def run(kind, question=None, history=()):
        with st.spinner("Javob tayyorlanmoqda..."):
            res = advisor.respond(kind, facts, ai_cfg if ai_ok else {}, question, history)
        if res["mode"].startswith("Claude"):
            st.session_state["ai_calls"] = st.session_state.get("ai_calls", 0) + 1
        return res

    tabs_ = st.tabs(["📋 Maslahat va 30/60/90 reja", "💬 Savol-javob"] + (["🔎 Inspektor yordamchisi"] if staff else []))
    t1, t2 = tabs_[0], tabs_[1]
    with t1:
        sim = pd.DataFrame(facts["simulyatsiya"])
        if len(sim):
            st.markdown("**Quyidagilar bajarilsa, ball qanday o'zgaradi?** (hisob-kitob: scorecard + ML modeli)")
            st.dataframe(sim[["ssenariy", "yangi_ball", "ball_ozgarishi", "yangi_sinf"]].rename(columns={
                "ssenariy": "Ssenariy", "yangi_ball": "Taxminiy yangi ball", "ball_ozgarishi": "Ball o'zgarishi",
                "yangi_sinf": "Yangi xavf sinfi"}), hide_index=True, use_container_width=True)
            st.caption(facts["simulyatsiya_izohi"])
        else:
            st.success("Ballni oshiradigan aniq qadam topilmadi: ko'rsatkichlar yaxshi.")
        if st.button("📋 Maslahat va 30/60/90 kunlik reja tayyorlash", key=f"plan_btn_{stir}"):
            st.session_state[f"plan_{stir}"] = run("plan")
        if f"plan_{stir}" in st.session_state:
            show_ai_result(st.session_state[f"plan_{stir}"])
    with t2:
        hist = st.session_state.setdefault(f"chat_{stir}", [])
        st.caption("Masalan: «Nega ballim past?», «Qarzni qanday kamaytiraman?», «Aylanma haqida ayt», «Nima qilsam yaxshilanadi?»")
        area = st.container()
        with st.form(f"chat_form_{stir}", clear_on_submit=True):
            qtxt = st.text_input("Savolingiz")
            sent = st.form_submit_button("Yuborish")
        if sent and qtxt.strip():
            res = run("chat", qtxt.strip(), list(hist))
            hist.append({"role": "user", "content": qtxt.strip()})
            hist.append({"role": "assistant", "content": res["text"], "mode": res["mode"], "warnings": res["warnings"]})
        if st.button("Suhbatni tozalash", key=f"chat_clear_{stir}"):
            hist.clear()
        with area:
            for m in hist:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
                    for w in m.get("warnings", []):
                        st.warning("⚠️ " + w)
                    if m["role"] == "assistant":
                        st.caption(f"Rejim: {m['mode']}")
    if staff:
        with tabs_[2]:
            st.caption("Ichki xulosa inspektor uchun: bu shubha signali va tekshiruv savollari, ayblov emas.")
            if st.button("🔎 Inspektor xulosasini tayyorlash", key=f"insp_btn_{stir}"):
                st.session_state[f"insp_{stir}"] = run("inspector")
            if f"insp_{stir}" in st.session_state:
                show_ai_result(st.session_state[f"insp_{stir}"])
    with st.expander("Maslahatchi tayanadigan tekshirilgan ma'lumotlar"):
        st.json(facts)


def inventory_detail(r):
    """Tovar balansi, aylanma manbasi, tovar guruhlari va ballga ta'siri (bitta korxona uchun)."""
    if r["has_inventory"] != 1:
        st.info(f"«{r['sector']}» tarmog'ida tovar qoldig'i hisobi yuritilmaydi. Shu sababli aylanma manbasi "
                "ko'rsatkichlari hisoblanmaydi va ballga ta'sir qilmaydi, korxona bu uchun jazolanmaydi.")
    else:
        markup, norm = SECTOR_INV[r["sector"]]
        a, b, c, d_ = st.columns(4)
        a.metric("Boshlang'ich qoldiq", f"{r['inv_opening_value']:,.1f}", help="mln so'm")
        b.metric("Kirim (xarid)", f"{r['inv_purchases']:,.1f}", help="mln so'm")
        c.metric("Oxirgi qoldiq", f"{r['inv_closing_value']:,.1f}", help="mln so'm, yil oxiriga")
        d_.metric("Sotilgan tovar tannarxi", f"{r['inv_cogs']:,.1f}", help="boshlang'ich + kirim − oxirgi qoldiq")
        a, b, c = st.columns(3)
        a.metric("Kutilgan aylanma", f"{r['expected_revenue']:,.1f}", help=f"tannarx × (1 + {markup:.0%} tarmoq ustamasi)")
        b.metric("Deklaratsiya qilingan aylanma", f"{r['revenue']:,.1f}")
        c.metric("Aylanma farqi", f"{r['turnover_gap']:+.1%}", help="(deklaratsiya − kutilgan) / kutilgan")
        level, text = turnover_verdict(r["turnover_gap"])
        {"ok": st.success, "warn": st.warning, "bad": st.error}[level](text)

        left, right = st.columns([3, 2])
        left.subheader("Aylanma manbasi")
        left.pyplot(plots.turnover_source(r))
        gt = goods[goods["stir"] == r["stir"]].copy()
        right.subheader("Qoldiq tovar guruhlari")
        if len(gt):
            gt["Ulushi, %"] = (gt["value"] / gt["value"].sum() * 100).round(1)
            gt = gt.drop(columns="stir").rename(columns={
                "group": "Tovar guruhi", "unit": "O'lchov", "quantity": "Miqdor",
                "unit_price": "Birlik narxi (so'm)", "value": "Summa (mln so'm)"})
            right.dataframe(gt, hide_index=True, use_container_width=True)
            right.caption(f"Jami qoldiq: {r['inv_closing_value']:,.2f} mln so'm · {len(gt)} ta tovar guruhi")
        st.subheader("Ballga ta'siri")
        a, b, c = st.columns(3)
        a.metric("Aylanma manbasi: qismiy ball", f"{r['sub_source']:.0f} / 100", help=f"Balldagi vazni: {int(r['w_source'])}%")
        b.metric("Qoldiq aylanuvchanligi: qismiy ball", f"{r['sub_stock']:.0f} / 100", help=f"Balldagi vazni: {int(r['w_stock'])}%")
        c.metric("Yakuniy ball", f"{r['final_score']:.1f} / 100")
        st.caption(f"Tarmoq o'rtacha ustamasi {markup:.0%}. Qoldiq tovar o'rtacha {r['stock_days']:.0f} kunda sotiladi "
                   f"(tarmoq me'yori ≈ {norm} kun).")


# ---------------------------------- DASHBOARD ----------------------------------
if page == "Dashboard":
    st.title("Umumiy ko'rsatkichlar")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Korxonalar soni", f"{len(df):,}")
    c2.metric("O'rtacha ball", f"{df['final_score'].mean():.1f}")
    c3.metric("Yuqori xavfli ulush", f"{(df['risk_class'] == 'Yuqori xavf').mean():.1%}")
    c4.metric("Jami soliq qarzi (mlrd so'm)", f"{df['tax_debt'].sum() / 1000:,.1f}")
    a, b = st.columns(2)
    a.subheader("Ballar taqsimoti"); a.pyplot(plots.score_hist(df))
    b.subheader("Xavf sinflari"); b.pyplot(plots.risk_bar(df))
    a, b = st.columns(2)
    a.subheader("Tarmoqlar bo'yicha o'rtacha ball"); a.pyplot(plots.sector_bar(df))
    b.subheader("Hududlar: yuqori xavfli korxonalar ulushi"); b.pyplot(plots.region_bar(df))
    st.subheader("Eng yuqori soliq qarzi bo'lgan 10 ta korxona")
    st.dataframe(view(df.nlargest(10, "tax_debt"), ["stir", "name", "region", "sector", "tax_debt", "final_score", "risk_class"]),
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
            st.caption(f"{r['region']} · {r['sector']} · {r['legal_form']} · {r['years_active']} yil · {r['phone']} · {r['email']}")
            show_company(r, df.loc[[idx], RAW_COLS])
            st.download_button("📄 Soliq pasportini yuklab olish (PDF)", passport_pdf(r),
                               file_name=f"soliq_pasporti_{r['stir']}.pdf", mime="application/pdf")
            advisor_section(r, df.loc[[idx], RAW_COLS])

# ---------------------------------- TOVAR QOLDIG'I ----------------------------------
elif page == "Tovar qoldig'i va aylanma manbasi":
    st.title("Tovar qoldig'i zaxirasi va aylanma manbasi")
    st.caption("Tovar balansi: boshlang'ich qoldiq + kirim − oxirgi qoldiq = sotilgan tovar tannarxi. Tannarxga tarmoq ustamasi "
               "qo'shilsa, kutilgan aylanma chiqadi va u deklaratsiya qilingan aylanma bilan solishtiriladi.")
    inv_df = df[df["has_inventory"] == 1]
    tab1, tab2 = st.tabs(["Korxona bo'yicha", "Umumiy tahlil"])
    with tab1:
        q = st.text_input("STIR yoki nom", key="inv_q", placeholder="masalan: " + inv_df["stir"].iloc[0])
        st.caption("Aylanmasi eng shubhali korxonalar STIR'i: " + ", ".join(inv_df.nsmallest(3, "turnover_gap")["stir"]))
        if q:
            m = df[df["stir"].str.contains(q.strip()) | df["name"].str.contains(q.strip(), case=False, regex=False)]
            if m.empty:
                st.warning("Korxona topilmadi.")
            else:
                idx = m.index[0] if len(m) == 1 else st.selectbox(
                    f"{len(m)} ta natija", m.index[:50], key="inv_sel",
                    format_func=lambda i: f"{df.loc[i, 'stir']} — {df.loc[i, 'name']}")
                r = df.loc[idx]
                st.header(f"{r['name']}  ·  STIR {r['stir']}")
                st.caption(f"{r['region']} · {r['sector']}")
                inventory_detail(r)
    with tab2:
        secs = st.multiselect("Tarmoq", sorted(inv_df["sector"].unique()))
        sub = inv_df[inv_df["sector"].isin(secs)] if secs else inv_df
        susp = sub[sub["turnover_gap"] < -0.25]
        a, b, c = st.columns(3)
        a.metric("Tovar hisobi yuritadigan korxonalar", f"{len(sub):,}")
        b.metric("Aylanmasi shubhali (farq < −25%)", f"{len(susp):,} ({len(susp) / max(len(sub), 1):.1%})")
        c.metric("Taxminiy yashirilgan aylanma (mlrd so'm)", f"{(susp['expected_revenue'] - susp['revenue']).sum() / 1000:,.1f}")
        a, b = st.columns(2)
        a.subheader("Kutilgan va deklaratsiya qilingan aylanma"); a.pyplot(plots.inventory_scatter(sub))
        b.subheader("Aylanma farqi taqsimoti"); b.pyplot(plots.gap_hist(sub))
        st.subheader("Aylanma manbasi eng shubhali 15 ta korxona")
        scols = ["stir", "name", "sector", "revenue", "expected_revenue", "turnover_gap", "final_score", "risk_class"]
        top = view(sub.nsmallest(15, "turnover_gap"), scols)
        st.dataframe(top, hide_index=True, use_container_width=True)
        st.download_button("⬇️ Shubhali korxonalar ro'yxati (Excel)", to_excel(view(susp.sort_values("turnover_gap"), scols)),
                           file_name="shubhali_aylanma.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.caption("Eslatma: bu sintetik ma'lumotdagi model hisobi. Real tizimda shubhali natija avtomatik jazo emas, "
                   "inspektor tekshiruvi uchun signal hisoblanadi.")

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
        has_inv = c.checkbox("Tovar qoldig'i hisobi yuritiladi", value=False)
        inv_open = a.number_input("Boshlang'ich tovar qoldig'i", 0.0, 1e6, 100.0)
        inv_purch = b.number_input("Yillik tovar kirimi (xarid)", 0.0, 1e6, 1100.0)
        inv_close = c.number_input("Oxirgi tovar qoldig'i", 0.0, 1e6, 120.0)
        go = st.form_submit_button("Hisoblash", type="primary")
    if go:
        raw = pd.DataFrame([{"stir": "—", "name": "Yangi korxona", "region": "—", "sector": sector, "legal_form": legal,
                             "years_active": years, "employees": emp, "revenue": revenue, "net_profit": profit,
                             "total_assets": assets, "equity": equity, "current_assets": ca, "current_liabilities": cl,
                             "tax_accrued": tax, "tax_debt": debt, "late_payments_count": late, "avg_delay_days": delay,
                             "late_reports_count": lrep, "audit_violations": aud, "vat_payer": int(vat),
                             "has_inventory": int(has_inv), "inv_opening_value": inv_open if has_inv else 0.0,
                             "inv_purchases": inv_purch if has_inv else 0.0,
                             "inv_closing_value": inv_close if has_inv else 0.0}])
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
    st.caption("Eng past ballli (eng xavfli) korxonalar tepada. Ustunlar izohi: «Raqamlar izohi» bo'limida.")
    st.dataframe(view(f.sort_values("final_score"), show), hide_index=True, use_container_width=True)
    st.download_button("⬇️ Excelga yuklash", to_excel(view(f.sort_values("final_score"), show)), file_name="korxonalar.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------------------------- ERTA OGOHLANTIRISH ----------------------------------
elif page == "Erta ogohlantirish":
    st.title("Erta ogohlantirish tizimi")
    st.write("Hozircha **yuqori xavf** sinfiga tushmagan, ammo ML modeli bo'yicha qarzdorlik ehtimoli yuqori bo'lgan korxonalar.")
    thr = st.slider("PD chegarasi", 0.10, 0.60, 0.25, 0.01)
    w = df[(df["pd"] >= thr) & (df["risk_class"] != "Yuqori xavf")].sort_values("pd", ascending=False)
    st.metric("Aniqlangan korxonalar", len(w))
    wcols = ["stir", "name", "region", "sector", "tax_debt", "late_payments_count", "final_score", "pd", "risk_class"]
    st.dataframe(view(w, wcols), hide_index=True, use_container_width=True)
    if len(w):
        st.download_button("⬇️ Ro'yxatni Excelga yuklash", to_excel(view(w, wcols)), file_name="erta_ogohlantirish.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------------------------- OGOHLANTIRISH YUBORISH ----------------------------------
elif page == "Ogohlantirish yuborish":
    st.title("Korxonalarga ogohlantirish yuborish")
    cfg = load_config(st.secrets)
    if "notif_log" not in st.session_state:
        st.session_state["notif_log"] = []
    st.info("Hozirgi rejim: **SIMULYATSIYA**. Xabarlar shakllantiriladi va jurnalga yoziladi, lekin hech kimga yuborilmaydi. "
            "Haqiqiy yuborish uchun SMS-provayder (Eskiz) yoki email kalitlari sozlanishi kerak. Korxonalar sintetik, "
            "shuning uchun haqiqiy yuborishni faqat o'zingizning test raqamingizga sinang.")

    a, b, c = st.columns(3)
    risk = a.multiselect("Xavf sinfi", list(plots.COLORS), default=["Yuqori xavf"])
    min_debt = b.number_input("Soliq qarzi kamida (mln so'm)", 0.0, 100000.0, 0.0, step=10.0)
    reg = c.multiselect("Hudud", sorted(df["region"].unique()))
    tgt = df[df["risk_class"].isin(risk) & (df["tax_debt"] >= min_debt)]
    if reg:
        tgt = tgt[tgt["region"].isin(reg)]
    tgt = tgt.sort_values("final_score")
    st.metric("Ogohlantirish yuboriladigan korxonalar", f"{len(tgt):,}")

    a, b = st.columns(2)
    channel = a.radio("Kanal", ["SMS", "Email"], horizontal=True)
    skip_done = b.checkbox("Avval ogohlantirilganlarni o'tkazib yuborish", value=True)
    done = {(x["STIR"], x["Kanal"]) for x in st.session_state["notif_log"]}
    if skip_done:
        tgt = tgt[[(r, channel) not in done for r in tgt["stir"]]]

    if len(tgt):
        st.subheader("Xabarni oldindan ko'rish")
        pick = st.selectbox("Korxona", tgt.index[:100],
                            format_func=lambda i: f"{tgt.loc[i, 'stir']} — {tgt.loc[i, 'name']} (ball {tgt.loc[i, 'final_score']:.1f})")
        r = tgt.loc[pick]
        if channel == "SMS":
            txt = build_sms(r)
            st.text_area("SMS matni", txt, height=150, disabled=True)
            st.caption(f"{len(txt)} belgi = {sms_segments(txt)} ta SMS | Qabul qiluvchi: {r['phone']}")
        else:
            subj, body = build_email(r)
            st.text_input("Mavzu", subj, disabled=True)
            st.text_area("Xat matni", body, height=300, disabled=True)
            st.caption(f"Qabul qiluvchi: {r['email']}")

        st.subheader("Yuborish")
        limit = st.slider("Bir martada nechta xabar", 1, min(200, len(tgt)), min(20, len(tgt)))
        ready = channel_ready(channel, cfg)
        live = st.checkbox("Haqiqiy yuborish", value=False, disabled=not ready,
                           help="Faqat kalitlar sozlangan bo'lsa yoqiladi." if not ready else None)
        override = ""
        if live:
            override = st.text_input("Test qabul qiluvchi (barcha xabarlar shu manzilga ketadi)",
                                     placeholder="+998901234567" if channel == "SMS" else "ism@gmail.com")
            st.warning("Test qabul qiluvchi bo'sh qoldirilsa, xabarlar korxonalarning o'z manzillariga yuboriladi!")
        confirm = st.checkbox("Yuborishni tasdiqlayman") if live else True
        if st.button("📨 Ogohlantirishlarni yuborish", type="primary", disabled=not confirm or (live and not override.strip())):
            with st.spinner("Yuborilmoqda..."):
                res = dispatch(tgt.head(limit), channel, cfg, live, override)
            st.session_state["notif_log"].extend(res)
            ok = sum(1 for x in res if not x["Holat"].startswith("Xato"))
            st.success(f"Qayta ishlandi: {len(res)} ta, muvaffaqiyatli: {ok} ta.")
    else:
        st.warning("Tanlangan shartlar bo'yicha yuboriladigan korxona yo'q.")

    st.subheader("Yuborish jurnali")
    if st.session_state["notif_log"]:
        lg = pd.DataFrame(st.session_state["notif_log"])
        st.dataframe(lg.drop(columns=["Xabar"]).iloc[::-1], hide_index=True, use_container_width=True)
        st.download_button("⬇️ Jurnalni Excelga yuklash", to_excel(lg), file_name="ogohlantirish_jurnali.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.caption("Hali hech narsa yuborilmagan.")

# ---------------------------------- MODELLAR ----------------------------------
elif page == "Modellar taqqoslash":
    st.title("Modellar taqqoslash")
    st.caption(f"Test to'plami: {meta['n_test']} ta korxona. Xavfli qarzdorlik ulushi: {meta['default_rate']:.1%}. "
               f"ML uchun tanlangan model (CV AUC bo'yicha): **{meta['best']}**")
    m = pd.DataFrame(meta["metrics"]).T.rename(columns={
        "Accuracy": "To'g'ri javoblar ulushi", "Precision": "Topilganlar aniqligi (Precision)",
        "Recall": "Qamrab olish (Recall)", "Brier": "Brier xatosi (kam = yaxshi)",
        "CV_AUC_mean": "AUC (5 marta o'zaro tekshiruv)"})
    st.dataframe(m.round(3), use_container_width=True)
    st.caption("AUC, Gini, KS qancha yuqori bo'lsa model xavfli va xavfsiz korxonalarni shuncha yaxshi ajratadi. "
               "Batafsil: «Raqamlar izohi» bo'limi.")
    a, b = st.columns(2)
    a.subheader("ROC egri chiziqlari"); a.pyplot(plots.roc_plot(meta))
    b.subheader("Belgilar muhimligi"); b.pyplot(plots.importance_bar(meta))
    st.subheader("Xavf sinflari bo'yicha haqiqiy qarzdorlik ulushi (kalibrovka tekshiruvi)")
    g = df.groupby("risk_class")["default_next_year"].agg(["count", "mean"]).reindex(list(plots.COLORS))
    g.columns = ["Korxonalar soni", "Xavfli qarzdorlik ulushi"]
    st.dataframe(g.style.format({"Xavfli qarzdorlik ulushi": "{:.1%}"}), use_container_width=True)

# ---------------------------------- MENING KORXONAM (yuridik shaxs) ----------------------------------
elif page == "Mening korxonam":
    mine = df[df["stir"] == user["stir"]]  # faqat o'z korxonasi: boshqa STIR so'rab bo'lmaydi
    if mine.empty:
        st.error("Sizning STIR bo'yicha korxona ma'lumotlar bazasida topilmadi. Administratorga murojaat qiling.")
        st.stop()
    idx = mine.index[0]
    r = df.loc[idx]
    st.title(r["name"])
    st.caption(f"STIR {r['stir']} · {r['region']} · {r['sector']} · {r['legal_form']} · {r['years_active']} yil")
    show_company(r, df.loc[[idx], RAW_COLS])
    st.download_button("📄 Soliq pasportini yuklab olish (PDF)", passport_pdf(r),
                       file_name=f"soliq_pasporti_{r['stir']}.pdf", mime="application/pdf")
    with st.expander("📦 Tovar qoldig'i va aylanma manbasi"):
        inventory_detail(r)
    advisor_section(r, df.loc[[idx], RAW_COLS], staff=False)

# ---------------------------------- FOYDALANUVCHILAR (admin) ----------------------------------
elif page == "Foydalanuvchilar":
    if role != "admin":
        st.error("Bu bo'lim faqat administrator uchun.")
        st.stop()
    st.title("Foydalanuvchilarni boshqarish")
    tl, ti, tc, tm, tj = st.tabs(["Ro'yxat", "Inspektor yaratish", "Korxona hisobi yaratish", "Parol va holat", "Jurnal"])
    with tl:
        ulist = auth.list_users()
        st.dataframe(pd.DataFrame(ulist).rename(columns={
            "login": "Login", "rol": "Rol", "stir": "STIR", "faol": "Faol", "bloklangan": "Bloklangan",
            "yaratilgan": "Yaratilgan", "oxirgi_kirish": "Oxirgi kirish"}), hide_index=True, use_container_width=True)
    with ti:
        with st.form("new_insp", clear_on_submit=True):
            nu = st.text_input("Inspektor logini", help="3-32 belgi: kichik harf, raqam, nuqta, pastki chiziq, chiziqcha")
            npw = st.text_input("Boshlang'ich parol", type="password")
            nrole = st.selectbox("Rol", ["inspector", "admin"], format_func=lambda x: auth.ROLES[x])
            go = st.form_submit_button("Yaratish")
        if go:
            ok, msg = auth.create_user(nu, npw, nrole)
            (st.success if ok else st.error)(msg)
    with tc:
        st.caption("Korxona o'zi ro'yxatdan o'tmagan bo'lsa (masalan, namoyish uchun), hisobni shu yerda yaratish mumkin. Login = STIR.")
        with st.form("new_comp", clear_on_submit=True):
            cs = st.text_input("Korxona STIR'i (9 xonali)")
            cpw = st.text_input("Boshlang'ich parol", type="password")
            go = st.form_submit_button("Yaratish")
        if go:
            if cs.strip() not in set(df["stir"]):
                st.error("Bu STIR bazada topilmadi.")
            else:
                ok, msg = auth.create_user(cs.strip(), cpw, "company", cs.strip())
                (st.success if ok else st.error)(msg)
    with tm:
        names = [u["login"] for u in auth.list_users()]
        sel = st.selectbox("Foydalanuvchi", names, key="adm_sel")
        npw2 = st.text_input("Yangi parol", type="password", key="adm_pw")
        c1, c2, c3 = st.columns(3)
        if c1.button("Parolni o'zgartirish"):
            ok, msg = auth.set_password(sel, npw2)
            (st.success if ok else st.error)(msg)
        if c2.button("Blokdan chiqarish"):
            auth.unlock(sel)
            st.success("Blokdan chiqarildi.")
        row = auth.get_user(sel)
        if row and c3.button("Faolsizlantirish" if row["active"] else "Faollashtirish"):
            if sel == user["username"]:
                st.error("O'z hisobingizni faolsizlantirib bo'lmaydi.")
            else:
                ok, msg = auth.set_active(sel, not row["active"])
                (st.success if ok else st.error)(msg)
        st.divider()
        sure = st.checkbox("Tanlangan foydalanuvchini o'chirishni tasdiqlayman", key="adm_del_ok")
        if st.button("🗑️ O'chirish", disabled=not sure):
            if sel == user["username"]:
                st.error("O'z hisobingizni o'chirib bo'lmaydi.")
            else:
                ok, msg = auth.delete_user(sel)
                (st.success if ok else st.error)(msg)
    with tj:
        st.caption("Oxirgi 50 ta hodisa (kirish, xato urinishlar, bloklash, hisob o'zgarishlari).")
        st.dataframe(pd.DataFrame(auth.recent_events(50)).rename(columns={"ts": "Vaqt", "username": "Login", "event": "Hodisa"}),
                     hide_index=True, use_container_width=True)

# ---------------------------------- RAQAMLAR IZOHI ----------------------------------
elif page == "Raqamlar izohi":
    st.title("Raqamlar nimani anglatadi?")
    st.markdown("""
### 1. Yakuniy ball (0–100)
Korxonaning **ishonchlilik bali**. Bank kreditlash skoringiga o'xshaydi: **ball qancha yuqori bo'lsa, korxona shuncha ishonchli**.
- **60 va undan yuqori**: past xavf, soliq intizomi yaxshi.
- **35–60**: o'rtacha xavf, e'tibor kerak.
- **35 dan past**: yuqori xavf, qarzdorlik ehtimoli katta.

### 2. Xavf sinfi
Ballga qarab korxona uch guruhdan biriga kiradi:
| Sinf | Ma'nosi | Tizimdagi haqiqiy qarzdorlik ulushi |
|---|---|---|
| 🟢 Past xavf | Soliqni o'z vaqtida to'laydi, moliyaviy holati yaxshi | taxminan 4% |
| 🟡 O'rtacha xavf | Ba'zi muammolar bor, kuzatish kerak | taxminan 10% |
| 🔴 Yuqori xavf | Qarzdorlik va kechikishlar ko'p, moliyaviy holati zaif | taxminan 36% |

### 3. Scorecard bali
**Ekspert qoidalari** bo'yicha hisoblangan ball. Tovar hisobi yuritiladigan korxonalarda sakkizta ko'rsatkich baholanadi:
soliq qarzi/aylanma (20%), to'lov intizomi (16%), joriy likvidlik (12%), avtonomlik (12%), rentabellik (12%),
hisobotlar (8%), **aylanma manbasi (14%)**, **qoldiq aylanuvchanligi (6%)**. Tovar hisobi yuritilmaydigan tarmoqlarda
(IT, xizmat, ta'lim, turizm, transport) oxirgi ikkitasi hisoblanmaydi va asosiy oltitasi 25/20/15/15/15/10 vaznda baholanadi,
shuning uchun korxona nohaq jazolanmaydi. Tekshiruvda qoidabuzarlik topilsa, ball kamayadi.
Qoidalarni inson tushunadi va tekshira oladi, shuning uchun bu model **shaffof**.

### 4. Qarzdorlik xavfi, % (PD, Probability of Default)
**Mashinaviy o'qitish modeli** hisoblagan ehtimollik: korxona **keyingi yilda xavfli qarzdorga aylanish** ehtimoli.
Masalan, 95% degani: 100 ta shunday korxonadan taxminan 95 tasi muammoli bo'ladi. 2% degani: deyarli xavfsiz.

### 5. Ular qanday birlashadi
**Yakuniy ball = 50% scorecard + 50% ML modeli.** Ikki yondashuv bir-birini to'ldiradi:
scorecard tushunarli, ML esa murakkab bog'liqliklarni ushlaydi.

### 6. Jadvaldagi ustunlar
- **STIR**: soliq to'lovchining identifikatsiya raqami (9 xonali).
- **Aylanma**: yillik tushum, mln so'mda.
- **Soliq qarzi**: to'lanmagan soliq summasi, mln so'mda.
- **Kechikkan to'lovlar soni**: oxirgi 12 oyda soliqni kechiktirib to'lash holatlari (0 dan 12 gacha).
- **Hisoblangan soliq**: yil davomida to'lanishi kerak bo'lgan soliq.

### 7. Moliyaviy koeffitsientlar
- **Joriy likvidlik** = joriy aktivlar / joriy majburiyatlar. 1.5 dan yuqori bo'lsa yaxshi: korxona qisqa muddatli qarzlarini yopa oladi.
- **Avtonomlik** = o'z kapitali / jami aktivlar. 0.4–0.5 dan yuqori bo'lsa yaxshi: korxona qarzga kam tayanadi.
- **Rentabellik** = sof foyda / aylanma. Foizlarda: aylanmaning qancha qismi foyda bo'lib qolishi.
- **Soliq qarzi / aylanma**: qarz yillik tushumning qancha ulushini tashkil etishi. Kichik bo'lgani yaxshi.

### 8. Misol: jadvaldagi birinchi qator
*"Registon Logistik" MChJ*: 12 oyning har 12 tasida to'lovni kechiktirgan (**12**), soliq qarzi **345 mln so'm**,
scorecard bali **3.7**, qarzdorlik xavfi **95%**, yakuniy ball **1.8**.
Xulosa: **yuqori xavf**. Bunday korxona soliq organi e'tiborida bo'lishi kerak.

### 9. Modellar sifati ko'rsatkichlari (Modellar taqqoslash bo'limi)
- **AUC** (0.5 dan 1 gacha): model xavfli va xavfsiz korxonalarni qanchalik yaxshi ajratishi. 0.5 = tasodifiy taxmin, 1 = mukammal. **0.85 dan yuqori juda yaxshi hisoblanadi.**
- **Gini** = 2×AUC − 1. Bank skoringida keng ishlatiladi. 0.7 va undan yuqori kuchli model.
- **KS** (0 dan 1 gacha): xavfli va xavfsiz guruhlar orasidagi eng katta farq. 0.4 dan yuqori yaxshi.
- **Qamrab olish (Recall)**: haqiqiy qarzdorlarning qanchasi topilgani.
- **Topilganlar aniqligi (Precision)**: "xavfli" deb belgilanganlarning qanchasi haqiqatan xavfli bo'lgani.
- **F1**: Precision va Recall o'rtasidagi muvozanat.
- **Brier xatosi**: ehtimollik bahosining aniqligi, 0 ga yaqin bo'lgani yaxshi.

### 10. Tovar qoldig'i va aylanma manbasi
- **Tovar balansi**: boshlang'ich qoldiq + kirim (xarid) − oxirgi qoldiq = sotilgan tovar tannarxi.
- **Kutilgan aylanma** = tannarx × (1 + tarmoq o'rtacha ustamasi).
- **Aylanma farqi** = (deklaratsiya − kutilgan) / kutilgan. Manfiy va katta bo'lsa (−25% dan past) aylanma yashirilgan bo'lishi mumkin, bu **shubhali** holat. Juda yuqori bo'lsa, tovar kirimi hujjatlashtirilmagan bo'lishi mumkin.
- **Qoldiq aylanishi (kun)** = o'rtacha qoldiq / tannarx × 365. Tarmoq me'yoridan ancha ko'p bo'lsa, tovar sotilmay yotibdi.
- Bu ko'rsatkichlar faqat tovar hisobi yuritiladigan tarmoqlarda hisoblanadi: savdo, ishlab chiqarish, qurilish, qishloq xo'jaligi, umumiy ovqatlanish.

> ⚠️ Barcha korxonalar sintetik (sun'iy) ma'lumot. Raqamlar tizimning ishlash tamoyilini ko'rsatadi, haqiqiy korxonalarga tegishli emas.
""")

# ---------------------------------- HAQIDA ----------------------------------
else:
    st.title("Tizim haqida")
    st.markdown("""
**Maqsad:** yuridik shaxslarning soliq intizomi va moliyaviy holatiga ko'ra ball (0–100) va xavf sinfini aniqlash.

**Gibrid yondashuv**
1. *Scorecard (ekspert model)*: 6 ta asosiy ko'rsatkich (qarz, to'lov intizomi, likvidlik, avtonomlik, rentabellik,
   hisobotlar) + tovar hisobi yuritiladigan tarmoqlar uchun 2 ta qo'shimcha ko'rsatkich (aylanma manbasi 14%,
   qoldiq aylanuvchanligi 6%) + tekshiruv qoidabuzarliklari uchun jarima.
2. *ML modeli*: Logistic Regression, Random Forest, Gradient Boosting. Maqsad: keyingi yilda xavfli qarzdorlik (PD).
3. *Gibrid ball* = 0.5 × scorecard + 0.5 × ML ball.

**Xavf sinflari** (100 ballik shkalada): ≥60 — past; 35–60 — o'rtacha; <35 — yuqori.

**Ma'lumotlar:** haqiqiy soliq ma'lumotlari maxfiy, shuning uchun sintetik ma'lumotlar ishlatilgan.
Haqiqiy ma'lumot bilan ishlash uchun `data/companies.csv` ni shu ustunlar formatida almashtirib, `python main.py` ni qayta ishga tushirish kifoya.
""")
