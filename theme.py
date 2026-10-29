"""Interfeys mavzusi: ranglar, katta logotip fon, kirish animatsiyasi (harakatlanuvchi tarmoq chiziqlari),
kirgandan keyingi katta logotipli "splash" va yon panel dizayni.

Animatsiyalar brauzer tomonida (JavaScript) ishlaydi; agar brauzer ularni bloklasa, sahifa baribir to'liq ishlaydi
(faqat CSS fon qoladi). Ranglar brand.PALETTE da.
"""
import json
import html as _html

import brand

P = brand.PALETTE

PAGE_META = {  # bo'lim kaliti: (belgi, ko'rinadigan nom)
    "Dashboard": ("🏛️", "Boshqaruv paneli"),
    "Korxonani tekshirish (STIR)": ("🔎", "Korxonani tekshirish"),
    "Tovar qoldig'i va aylanma manbasi": ("📦", "Tovar qoldig'i va aylanma"),
    "Yangi korxona (what-if)": ("🧪", "Yangi korxona (what-if)"),
    "Korxonalar ro'yxati": ("🗂️", "Korxonalar ro'yxati"),
    "Erta ogohlantirish": ("🚨", "Erta ogohlantirish"),
    "Ogohlantirish yuborish": ("📨", "Ogohlantirish yuborish"),
    "Modellar taqqoslash": ("📈", "Modellar taqqoslash"),
    "Raqamlar izohi": ("📘", "Raqamlar izohi"),
    "Tizim haqida": ("ℹ️", "Tizim haqida"),
    "Mening korxonam": ("🏢", "Mening korxonam"),
    "Foydalanuvchilar": ("🛡️", "Foydalanuvchilar"),
}


def nav_label(page: str) -> str:
    icon, name = PAGE_META.get(page, ("•", page))
    return f"{icon}  {name}"


def _fill(s: str) -> str:
    s = s.replace("@LOGO@", brand.logo_uri())
    for k, v in P.items():
        s = s.replace(f"@{k}@", v)
    return s


# ====================================== CSS ======================================
HIDE_HELPERS_CSS = """
[data-testid="stElementContainer"]:has(style), [data-testid="element-container"]:has(style) { display:none !important; }
[data-testid="stElementContainer"]:has(iframe[srcdoc*="tsinject"]), [data-testid="element-container"]:has(iframe[srcdoc*="tsinject"]) {
  position:fixed !important; left:-20px; top:-20px; width:1px; height:1px; opacity:0; pointer-events:none; overflow:hidden; }
footer, [data-testid="stAppDeployButton"], #MainMenu { visibility:hidden !important; }
"""

BASE_CSS = HIDE_HELPERS_CSS + """
html, body, .stApp { font-family: "Segoe UI", Inter, system-ui, -apple-system, Roboto, "Helvetica Neue", Arial, sans-serif; }
.stApp { background: linear-gradient(160deg, #F6F9FE 0%, #E7EEF9 100%) !important; position:relative; isolation:isolate; }
[data-testid="stAppViewContainer"], [data-testid="stMain"], section.main, [data-testid="stBottom"], [data-testid="stHeader"] { background: transparent !important; }

/* Orqa fondagi katta logotip */
.stApp::before { content:""; position:fixed; z-index:-1; pointer-events:none; top:50%; left:calc(50% + 10.5rem);
  width:74vmin; height:74vmin; transform:translate(-50%,-50%); background:url("@LOGO@") center/contain no-repeat;
  opacity:.075; animation: ts-breathe 16s ease-in-out infinite; }
@keyframes ts-breathe { 0%,100% { transform:translate(-50%,-50%) scale(1) rotate(0deg);} 50% { transform:translate(-50%,-50%) scale(1.045) rotate(2.5deg);} }

/* Sahifa sarlavhasi: banner */
[data-testid="stMain"] h1, section.main h1 { background: linear-gradient(120deg, @navy@ 0%, @navy2@ 55%, @blue@ 100%);
  color:#fff !important; padding:1.05rem 1.5rem !important; border-radius:16px; position:relative; overflow:hidden;
  font-size:1.65rem !important; font-weight:700 !important; letter-spacing:.2px; box-shadow:0 10px 28px rgba(10,31,68,.22);
  border-left:6px solid @gold@; margin-bottom:.6rem; }
[data-testid="stMain"] h1::after, section.main h1::after { content:""; position:absolute; right:-40px; top:-60px; width:190px; height:190px;
  background: radial-gradient(circle, rgba(233,180,76,.55), rgba(233,180,76,0) 65%); pointer-events:none; }
[data-testid="stMain"] h1 *, section.main h1 * { color:#fff !important; }
[data-testid="stMain"] h2, [data-testid="stMain"] h3, section.main h2, section.main h3 { color:@navy@ !important; }
[data-testid="stMain"] h2 a, [data-testid="stMain"] h3 a { display:none; }

/* Ko'rsatkich kartochkalari */
[data-testid="stMetric"] { background:@card@; border:1px solid @line@; border-left:4px solid @gold@; border-radius:14px;
  padding:.75rem 1rem; box-shadow:0 3px 14px rgba(10,31,68,.07); transition: transform .18s, box-shadow .18s; }
[data-testid="stMetric"]:hover { transform:translateY(-2px); box-shadow:0 8px 22px rgba(30,79,163,.16); }
[data-testid="stMetricValue"] { color:@navy@; font-weight:700; }
[data-testid="stMetricLabel"] { color:@muted@; }

/* Tablar */
button[data-baseweb="tab"] { font-weight:600; color:@muted@; }
button[data-baseweb="tab"][aria-selected="true"] { color:@blue@; }
[data-baseweb="tab-highlight"] { background:@gold@ !important; height:3px !important; }

/* Tugmalar */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
  border-radius:12px; border:1px solid @blue@; font-weight:600; transition: all .18s; }
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
  border-color:@gold@; box-shadow:0 6px 18px rgba(30,79,163,.25); transform:translateY(-1px); }
button[kind="primary"], button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-primaryFormSubmit"] {
  background: linear-gradient(135deg, @blue@, @navy2@) !important; color:#fff !important; border:none !important; }

/* Boshqa elementlar */
[data-testid="stExpander"] { border-radius:12px; border:1px solid @line@; background:rgba(255,255,255,.75); }
[data-testid="stDataFrame"], [data-testid="stImage"] img { border-radius:12px; }
[data-testid="stAlert"] { border-radius:12px; }
[data-testid="stForm"] { border-radius:14px; border:1px solid @line@; background:rgba(255,255,255,.7); }

/* ============ Yon panel ============ */
[data-testid="stSidebar"] { background: linear-gradient(185deg, @navy@ 0%, @navy2@ 58%, #15458F 100%) !important;
  border-right:1px solid rgba(233,180,76,.28); }
[data-testid="stSidebar"] > div:first-child { background:transparent !important; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span, [data-testid="stSidebar"] li,
[data-testid="stSidebar"] summary, [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color:#DCE6F7; }
[data-testid="stSidebar"] [data-testid="stAlert"] * { color:@ink@ !important; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea { color:#fff !important; background:transparent !important; }
[data-testid="stSidebar"] [data-baseweb="input"], [data-testid="stSidebar"] [data-baseweb="base-input"] {
  background:rgba(255,255,255,.08) !important; border-color:rgba(255,255,255,.22) !important; }
[data-testid="stSidebar"] .stButton > button { background:rgba(255,255,255,.07); color:#fff; border:1px solid rgba(233,180,76,.6); }
[data-testid="stSidebar"] .stButton > button:hover { background:rgba(233,180,76,.18); }
[data-testid="stSidebar"] [data-testid="stExpander"], [data-testid="stSidebar"] [data-testid="stForm"] {
  background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.14); }
[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.14); }

/* Yon panel: brend va foydalanuvchi */
.ts-brand { display:flex; align-items:center; gap:.7rem; padding:.3rem .2rem .9rem; border-bottom:1px solid rgba(233,180,76,.25); margin-bottom:.8rem; }
.ts-brand img { width:58px; height:58px; animation: ts-float 6s ease-in-out infinite; filter: drop-shadow(0 0 12px rgba(233,180,76,.5)); }
.ts-brand-name { color:#fff; font-weight:800; letter-spacing:.16em; font-size:1.02rem; line-height:1.1; }
.ts-brand-sub { color:#9FB6DD; font-size:.72rem; line-height:1.25; margin-top:.2rem; }
@keyframes ts-float { 0%,100% { transform:translateY(0) rotate(0deg);} 50% { transform:translateY(-3px) rotate(4deg);} }
.ts-user { display:flex; align-items:center; gap:.65rem; padding:.55rem .65rem; border-radius:14px; margin-bottom:.9rem;
  background:linear-gradient(135deg, rgba(255,255,255,.10), rgba(255,255,255,.03)); border:1px solid rgba(255,255,255,.13); }
.ts-ava { width:38px; height:38px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:800;
  color:@navy@; background:linear-gradient(135deg, @gold2@, @gold@); box-shadow:0 0 0 3px rgba(233,180,76,.25); }
.ts-uname { color:#fff; font-weight:700; font-size:.95rem; line-height:1.1; }
.ts-role { display:inline-block; margin-top:.18rem; font-size:.68rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase;
  color:@navy@; background:@gold@; padding:.06rem .5rem; border-radius:20px; }
.ts-navcap { color:#8FA9D6; font-size:.68rem; font-weight:700; letter-spacing:.2em; margin:.2rem 0 .35rem .2rem; }

/* Yon panel: navigatsiya (radio -> kartochkalar) */
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] { gap:.38rem; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { width:100%; background:rgba(255,255,255,.045); border:1px solid rgba(255,255,255,.08);
  border-radius:13px; padding:.52rem .8rem; cursor:pointer; transition: all .18s ease; margin:0; box-sizing:border-box; max-width:100%; }
[data-testid="stSidebar"] [data-testid="stRadio"] label > *:first-child:not(:has([data-testid="stMarkdownContainer"])) { display:none !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label p { font-weight:600; font-size:.92rem; margin:0; color:#E2EBFA; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:rgba(91,163,245,.2); border-color:rgba(91,163,245,.55); transform:translateX(4px); }
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) { background:linear-gradient(90deg, rgba(233,180,76,.32), rgba(233,180,76,.06));
  border-color:rgba(233,180,76,.75); box-shadow: inset 4px 0 0 @gold@, 0 4px 14px rgba(0,0,0,.18); }
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p { color:#fff; }

/* Mobil: katta logotip markazda */
@media (max-width: 900px) { .stApp::before { left:50%; width:90vmin; height:90vmin; } }
"""

LOGIN_CSS = HIDE_HELPERS_CSS + """
html, body { min-height:100vh; background: radial-gradient(circle at 18% 8%, #153A7A 0%, @navy@ 52%, #050F26 100%) fixed !important; }
.stApp { background:transparent !important; position:relative; z-index:1; }
[data-testid="stAppViewContainer"], [data-testid="stMain"], section.main, [data-testid="stHeader"], [data-testid="stBottom"] { background:transparent !important; }
[data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"], [data-testid="stToolbar"] { display:none !important; }
.block-container { max-width:500px !important; margin:4vh auto 0 !important; padding:1.6rem 2rem 1.3rem !important;
  background:rgba(7,22,52,.56); backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px);
  border:1px solid rgba(233,180,76,.32); border-radius:24px; box-shadow:0 30px 90px rgba(0,0,0,.5), inset 0 1px 0 rgba(255,255,255,.08);
  animation: ts-rise .9s cubic-bezier(.2,.8,.2,1) both; }
@keyframes ts-rise { from { opacity:0; transform:translateY(24px) scale(.97);} to { opacity:1; transform:none;} }
.block-container, .block-container p, .block-container label, .block-container span, .block-container li,
.block-container [data-testid="stMarkdownContainer"] { color:#E4ECFA; }
.block-container [data-testid="stAlert"] * { color:@ink@ !important; }
.block-container h1, .block-container h2, .block-container h3 { color:#fff !important; }
button[data-baseweb="tab"] { color:#AFC2E6 !important; font-weight:600; }
button[data-baseweb="tab"][aria-selected="true"] { color:#fff !important; }
[data-baseweb="tab-highlight"] { background:@gold@ !important; height:3px !important; }
[data-baseweb="tab-border"] { background:rgba(255,255,255,.15) !important; }
[data-testid="stForm"] { border:1px solid rgba(255,255,255,.14); background:rgba(255,255,255,.04); border-radius:16px; }
.stButton > button, [data-testid="stFormSubmitButton"] > button { border-radius:12px; font-weight:700; border:1px solid rgba(233,180,76,.6); }
button[kind="primary"], button[kind="primaryFormSubmit"], button[data-testid="stBaseButton-primaryFormSubmit"], button[data-testid="stBaseButton-primary"] {
  background:linear-gradient(135deg, @gold@, #D99A2B) !important; color:@navy@ !important; border:none !important; }
[data-testid="stFormSubmitButton"] > button { background:rgba(255,255,255,.09); color:#fff; }
[data-testid="stFormSubmitButton"] > button:hover { border-color:@gold@; box-shadow:0 8px 22px rgba(233,180,76,.3); }
/* JS ishlamasa: CSS bilan zaxira fon (katta logotip + siljuvchi chiziqlar) */
body:not(.ts-js) .stApp::before { content:""; position:fixed; z-index:-1; pointer-events:none; inset:0;
  background:url("@LOGO@") center/min(80vmin,780px) no-repeat; opacity:.10; }
body:not(.ts-js) .stApp::after { content:""; position:fixed; z-index:-1; pointer-events:none; inset:-50%;
  background: repeating-linear-gradient(60deg, rgba(120,180,255,.10) 0 1px, transparent 1px 90px), repeating-linear-gradient(-60deg, rgba(233,180,76,.08) 0 1px, transparent 1px 90px);
  animation: ts-drift 40s linear infinite; }
@keyframes ts-drift { to { transform: translate(90px, 52px); } }
.ts-hero { text-align:center; margin:.2rem 0 1rem; }
.ts-hero img { width:116px; height:116px; animation: ts-float 6s ease-in-out infinite; filter: drop-shadow(0 0 26px rgba(233,180,76,.55)); }
.ts-hero-title { color:#fff; font-weight:800; letter-spacing:.34em; font-size:1.55rem; margin-top:.4rem; text-indent:.34em; }
.ts-hero-sub { color:#A9BEE4; font-size:.82rem; line-height:1.4; margin-top:.35rem; }
@keyframes ts-float { 0%,100% { transform:translateY(0);} 50% { transform:translateY(-6px);} }
"""


# ====================================== JavaScript (brauzer / ota oyna) ======================================
BG_JS = r"""
(function(){
  var D = document, W = window;
  if (D.getElementById('ts-bg')) return;
  D.body.classList.add('ts-js');
  var wrap = D.createElement('div'); wrap.id = 'ts-bg';
  wrap.style.cssText = 'position:fixed;left:0;top:0;right:0;bottom:0;z-index:0;pointer-events:none;overflow:hidden';
  var logo = D.createElement('img'); logo.src = __LOGO__;
  logo.style.cssText = 'position:absolute;left:50%;top:50%;width:min(108vmin,1040px);height:min(108vmin,1040px);transform:translate(-50%,-50%);opacity:.2;' +
                       'animation:tsbgspin 90s linear infinite';
  var st = D.createElement('style'); st.id = 'ts-bg-style';
  st.textContent = '@keyframes tsbgspin{from{transform:translate(-50%,-50%) rotate(0deg)}to{transform:translate(-50%,-50%) rotate(360deg)}}';
  D.head.appendChild(st);
  var cv = D.createElement('canvas'); cv.style.cssText = 'position:absolute;left:0;top:0;width:100%;height:100%';
  wrap.appendChild(logo); wrap.appendChild(cv); D.body.appendChild(wrap);
  var ctx = cv.getContext('2d'), w = 0, h = 0, dpr = Math.min(W.devicePixelRatio || 1, 2), nodes = [], mouse = {x: -9999, y: -9999};
  var LINK = 195;
  function mk() { var a = Math.random() * 6.2832, s = 0.22 + Math.random() * 0.5;
    return {x: Math.random() * w, y: Math.random() * h, vx: Math.cos(a) * s, vy: Math.sin(a) * s, r: 1.2 + Math.random() * 1.8, g: Math.random() < 0.14}; }
  function size() { w = W.innerWidth; h = W.innerHeight; cv.width = w * dpr; cv.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    var n = Math.max(70, Math.min(170, Math.round(w * h / 9000))); while (nodes.length < n) nodes.push(mk()); nodes.length = n; }
  function move(e) { mouse.x = e.clientX; mouse.y = e.clientY; }
  size(); W.addEventListener('resize', size); D.addEventListener('mousemove', move);
  function frame() {
    if (!D.getElementById('ts-bg')) { W.removeEventListener('resize', size); D.removeEventListener('mousemove', move); return; }
    ctx.clearRect(0, 0, w, h);
    var i, j, p, q, dx, dy, d, a;
    for (i = 0; i < nodes.length; i++) { p = nodes[i]; p.x += p.vx; p.y += p.vy;
      if (p.x < -30) p.x = w + 30; else if (p.x > w + 30) p.x = -30; if (p.y < -30) p.y = h + 30; else if (p.y > h + 30) p.y = -30; }
    for (i = 0; i < nodes.length; i++) { p = nodes[i];
      for (j = i + 1; j < nodes.length; j++) { q = nodes[j]; dx = p.x - q.x; dy = p.y - q.y; d = Math.sqrt(dx * dx + dy * dy);
        if (d < LINK) { a = 1 - d / LINK;
          ctx.strokeStyle = (p.g || q.g) ? 'rgba(233,180,76,' + (a * 0.85) + ')' : 'rgba(120,185,255,' + (a * 0.72) + ')';
          ctx.lineWidth = 0.6 + a * 1.3; ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke(); } }
      dx = p.x - mouse.x; dy = p.y - mouse.y; d = Math.sqrt(dx * dx + dy * dy);
      if (d < 210) { a = 1 - d / 210; ctx.strokeStyle = 'rgba(246,210,122,' + (a * 0.9) + ')'; ctx.lineWidth = 0.8 + a * 1.2;
        ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(mouse.x, mouse.y); ctx.stroke(); } }
    for (i = 0; i < nodes.length; i++) { p = nodes[i]; ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, 6.2832);
      if (p.g) { ctx.fillStyle = 'rgba(246,210,122,0.95)'; ctx.shadowColor = 'rgba(233,180,76,0.9)'; ctx.shadowBlur = 10; }
      else { ctx.fillStyle = 'rgba(190,222,255,0.9)'; ctx.shadowBlur = 0; }
      ctx.fill(); }
    ctx.shadowBlur = 0;
    W.requestAnimationFrame(frame);
  }
  W.requestAnimationFrame(frame);
})();
"""

CLEAR_JS = r"""
(function(){ var D = document;
  var b = D.getElementById('ts-bg'); if (b) b.remove();
  var s = D.getElementById('ts-bg-style'); if (s) s.remove();
  D.body.classList.remove('ts-js'); })();
"""

SPLASH_JS = r"""
(function(){
  var D = document, W = window;
  var b = D.getElementById('ts-bg'); if (b) b.remove();
  var s0 = D.getElementById('ts-bg-style'); if (s0) s0.remove();
  D.body.classList.remove('ts-js');
  if (D.getElementById('ts-splash')) return;
  var css = '#ts-splash{position:fixed;left:0;top:0;right:0;bottom:0;z-index:2147483000;display:flex;flex-direction:column;align-items:center;justify-content:center;' +
    'background:radial-gradient(circle at 50% 38%,#1C54A8 0%,#0A1F44 58%,#050F26 100%);animation:tssout .75s ease 2.9s forwards;overflow:hidden}' +
    '#ts-splash .wrap{position:relative;width:min(62vmin,560px);height:min(62vmin,560px);display:flex;align-items:center;justify-content:center}' +
    '#ts-splash .ring{position:absolute;left:-9%;top:-9%;right:-9%;bottom:-9%;border-radius:50%;' +
    'background:conic-gradient(from 0deg,rgba(233,180,76,0),rgba(233,180,76,.9),rgba(233,180,76,0) 60%);' +
    '-webkit-mask:radial-gradient(circle,transparent 62%,#000 63%,#000 66%,transparent 67%);mask:radial-gradient(circle,transparent 62%,#000 63%,#000 66%,transparent 67%);' +
    'animation:tsspin 2.2s linear infinite;opacity:0;animation-delay:0s;animation-fill-mode:both}' +
    '#ts-splash img{position:relative;width:100%;height:100%;animation:tspop 1.1s cubic-bezier(.2,.9,.3,1.15) both,tsglow 2.4s ease-in-out 1.1s infinite;' +
    'filter:drop-shadow(0 0 38px rgba(233,180,76,.55))}' +
    '#ts-splash .t1{font:800 clamp(20px,3.6vw,38px)/1.1 "Segoe UI",system-ui,sans-serif;letter-spacing:.38em;text-indent:.38em;color:#fff;margin-top:26px;opacity:0;animation:tsfade .8s ease .7s forwards}' +
    '#ts-splash .t2{font:600 clamp(14px,1.8vw,19px)/1.3 "Segoe UI",system-ui,sans-serif;color:#F0C766;margin-top:12px;opacity:0;animation:tsfade .8s ease 1.1s forwards}' +
    '#ts-splash .bar{width:min(320px,62vw);height:4px;background:rgba(255,255,255,.16);border-radius:4px;margin-top:30px;overflow:hidden;opacity:0;animation:tsfade .5s ease .9s forwards}' +
    '#ts-splash .bar i{display:block;height:100%;width:0;background:linear-gradient(90deg,#E9B44C,#F6D27A);animation:tsload 2.1s ease .9s forwards}' +
    '@keyframes tssout{to{opacity:0;visibility:hidden}}' +
    '@keyframes tspop{from{opacity:0;transform:scale(.5) rotate(-14deg)}to{opacity:1;transform:none}}' +
    '@keyframes tsglow{0%,100%{filter:drop-shadow(0 0 30px rgba(233,180,76,.45))}50%{filter:drop-shadow(0 0 58px rgba(233,180,76,.85))}}' +
    '@keyframes tsspin{0%{transform:rotate(0);opacity:0}25%{opacity:1}100%{transform:rotate(360deg);opacity:1}}' +
    '@keyframes tsfade{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}' +
    '@keyframes tsload{to{width:100%}}';
  var st = D.createElement('style'); st.id = 'ts-splash-style'; st.textContent = css; D.head.appendChild(st);
  var o = D.createElement('div'); o.id = 'ts-splash';
  o.innerHTML = '<div class="wrap"><div class="ring"></div><img alt=""></div><div class="t1"></div><div class="t2"></div><div class="bar"><i></i></div>';
  o.querySelector('img').src = __LOGO__;
  o.querySelector('.t1').textContent = __TITLE__;
  o.querySelector('.t2').textContent = __WELCOME__;
  D.body.appendChild(o);
  W.setTimeout(function(){ if (o.parentNode) o.parentNode.removeChild(o); if (st.parentNode) st.parentNode.removeChild(st); }, 4300);
})();
"""


def _js(template: str, **vals) -> str:
    out = template
    for k, v in vals.items():
        out = out.replace(f"__{k}__", json.dumps(v, ensure_ascii=False))
    return out


def bg_script() -> str:
    return _js(BG_JS, LOGO=brand.logo_uri())


def splash_script(welcome: str) -> str:
    return _js(SPLASH_JS, LOGO=brand.logo_uri(), TITLE=brand.APP_NAME, WELCOME=welcome)


def _inject_html(code: str) -> str:
    payload = json.dumps(code, ensure_ascii=False).replace("</", "<\\/")
    return ("<!--tsinject--><script>(function(){try{var P=window.parent;var s=P.document.createElement('script');"
            f"s.textContent={payload};P.document.head.appendChild(s);P.document.head.removeChild(s);"
            "}catch(e){console.log('ts inject:',e);}})();</script>")


def inject_js(code: str):
    """Kodni ota oynada (Streamlit sahifasi) ishga tushiradi: iframe o'chirilsa ham animatsiya davom etadi."""
    import streamlit.components.v1 as components
    components.html(_inject_html(code), height=1)


# ====================================== Streamlit yordamchilari ======================================
def hero_html() -> str:
    return (f'<div class="ts-hero"><img src="{brand.logo_uri()}" alt="logo"/>'
            f'<div class="ts-hero-title">{brand.APP_NAME}</div>'
            f'<div class="ts-hero-sub">{_html.escape(brand.APP_TAGLINE)}<br/>o\'quv prototipi</div></div>')


def login_screen():
    """Kirish oynasi: tungi fon, harakatlanuvchi tarmoq chiziqlari, katta logotip."""
    import streamlit as st
    st.markdown(f"<style>{_fill(LOGIN_CSS)}</style>" + hero_html(), unsafe_allow_html=True)
    inject_js(bg_script())


def apply_base():
    import streamlit as st
    st.markdown(f"<style>{_fill(BASE_CSS)}</style>", unsafe_allow_html=True)


def splash(name: str):
    """Kirgandan keyin: katta logotip, ism bilan salomlashish, so'ng dasturga o'tish."""
    inject_js(splash_script(f"Xush kelibsiz, {name}"))


def sidebar_header(username: str, role_label: str):
    import streamlit as st
    initial = _html.escape((username or "?")[:1].upper())
    st.sidebar.markdown(
        f'<div class="ts-brand"><img src="{brand.logo_uri()}" alt="logo"/><div><div class="ts-brand-name">{brand.APP_NAME}</div>'
        f'<div class="ts-brand-sub">{_html.escape(brand.APP_TAGLINE)}</div></div></div>'
        f'<div class="ts-user"><div class="ts-ava">{initial}</div><div><div class="ts-uname">{_html.escape(username)}</div>'
        f'<span class="ts-role">{_html.escape(role_label)}</span></div></div>'
        f'<div class="ts-navcap">BO\'LIMLAR</div>', unsafe_allow_html=True)
