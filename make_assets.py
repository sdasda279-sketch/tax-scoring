"""Logotipdan PNG nusxalarini yaratadi (faqat dasturchi uchun; playwright kerak): python make_assets.py"""
import brand
from playwright.sync_api import sync_playwright

svg = brand.logo_svg()
with sync_playwright() as p:
    b = p.chromium.launch()
    for name, size in (("logo.png", 600), ("favicon.png", 96)):
        pg = b.new_page(viewport={"width": size, "height": size}, device_scale_factor=1)
        pg.set_content(f"<body style='margin:0;background:transparent'>{svg.replace('width=\"200\" height=\"200\"', f'width=\"{size}\" height=\"{size}\"')}</body>")
        pg.screenshot(path=f"assets/{name}", omit_background=True)
    b.close()
print("assets/logo.png va assets/favicon.png yaratildi")
