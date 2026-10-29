"""Brend: soliq logotipi (SVG), ranglar palitrasi va logotipni almashtirish imkoniyati.

Logotip original ("soliq muhri": ichida % belgisi, atrofida lavr shoxlari va yulduz). Rasmiy logotipingiz bo'lsa,
uni `assets/logo_custom.png` (yoki .svg/.jpg/.webp) nomi bilan qo'ying: dastur avtomatik shuni ishlatadi.
Barcha ranglar PALETTE da: interfeys ranglarini shu yerdan o'zgartirish mumkin.
"""
import base64
import math
from pathlib import Path

PALETTE = {
    "navy": "#0A1F44", "navy2": "#10306B", "blue": "#1E4FA3", "sky": "#5BA3F5",
    "gold": "#E9B44C", "gold2": "#F6D27A", "ink": "#0B1B33", "bg": "#F3F6FB", "card": "#FFFFFF",
    "line": "#D6DFEE", "green": "#1FA971", "amber": "#E9B44C", "red": "#D64545", "muted": "#5E6E8C",
}
APP_NAME = "SOLIQ SKORING"
APP_TAGLINE = "Soliq intizomi va moliyaviy ishonchlilik tizimi"
ASSETS = Path(__file__).parent / "assets"


def _leaf(cx, cy, ang, rx, ry, fill, op=1.0):
    return (f'<ellipse cx="{cx:.2f}" cy="{cy:.2f}" rx="{rx}" ry="{ry}" fill="{fill}" opacity="{op}" '
            f'transform="rotate({ang:.1f} {cx:.2f} {cy:.2f})"/>')


def logo_svg() -> str:
    P = PALETTE
    parts = []
    N = 9
    for side in (-1, 1):  # chap (-1) va o'ng (+1) lavr shoxi: pastdan yuqoriga o'sadi
        for i in range(N):
            t = math.radians(14 + i * (136 / (N - 1)))
            for R, tilt, fill, op in ((76, 30, "url(#gold)", 1.0), (64, -30, P["gold2"], 0.9)):
                x, y = 100 + side * R * math.sin(t), 100 + R * math.cos(t)
                dx, dy = side * R * math.cos(t), -R * math.sin(t)          # o'sish yo'nalishi (urinma)
                a_t = math.degrees(math.atan2(dy, dx))
                a_n = math.degrees(math.atan2(y - 100, x - 100))             # markazdan tashqariga normal
                diff = (a_n - a_t + 180) % 360 - 180
                ang = a_t + math.copysign(abs(tilt), diff) * (1 if tilt > 0 else -1)
                k = 1 - 0.035 * i                                            # uchi tomon barglar biroz kichrayadi
                parts.append(_leaf(x, y, ang, 11.5 * k, 4.3 * k, fill, op))
    star_pts = []
    for k in range(16):
        r = 11 if k % 2 == 0 else 4.6
        a = math.radians(k * 22.5 - 90)
        star_pts.append(f"{100 + r * math.cos(a):.2f},{22 + r * math.sin(a):.2f}")
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="200" height="200">
<defs>
<radialGradient id="disc" cx="50%" cy="38%" r="70%"><stop offset="0" stop-color="{P['blue']}"/><stop offset="0.65" stop-color="{P['navy2']}"/><stop offset="1" stop-color="{P['navy']}"/></radialGradient>
<linearGradient id="gold" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{P['gold2']}"/><stop offset="1" stop-color="{P['gold']}"/></linearGradient>
</defs>
<circle cx="100" cy="100" r="97" fill="url(#gold)"/>
<circle cx="100" cy="100" r="93" fill="url(#disc)"/>
<circle cx="100" cy="100" r="86" fill="none" stroke="{P['gold']}" stroke-width="0.9" opacity="0.55"/>
{''.join(parts)}
<polygon points="{' '.join(star_pts)}" fill="url(#gold)"/>
<g fill="none" stroke="#FFFFFF" stroke-width="6.5">
<circle cx="84" cy="86" r="10"/><circle cx="116" cy="114" r="10"/>
</g>
<line x1="124" y1="76" x2="76" y2="124" stroke="url(#gold)" stroke-width="8" stroke-linecap="round"/>
<rect x="70" y="141" width="60" height="3" rx="1.5" fill="{P['gold']}" opacity="0.8"/>
</svg>'''


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def custom_logo():
    """assets/logo_custom.* (foydalanuvchi qo'ygan rasmiy logotip) bo'lsa, (bayt, mime) qaytaradi."""
    for ext, mime in (("png", "image/png"), ("svg", "image/svg+xml"), ("jpg", "image/jpeg"),
                      ("jpeg", "image/jpeg"), ("webp", "image/webp")):
        p = ASSETS / f"logo_custom.{ext}"
        if p.exists():
            return p.read_bytes(), mime
    return None


def logo_uri() -> str:
    c = custom_logo()
    if c:
        return f"data:{c[1]};base64,{_b64(c[0])}"
    return "data:image/svg+xml;base64," + _b64(logo_svg().encode())


def logo_png_path():
    """PDF va favicon uchun PNG yo'li (bo'lmasa None)."""
    for name in ("logo_custom.png", "logo.png"):
        if (ASSETS / name).exists():
            return str(ASSETS / name)
    return None


def favicon():
    for name in ("logo_custom.png", "favicon.png", "logo.png"):
        if (ASSETS / name).exists():
            return str(ASSETS / name)
    return "🏛️"
