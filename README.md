# Tax Scoring: yuridik shaxslarning soliq qarzdorligi va moliyaviy ishonchliligini baholash tizimi

## Ishga tushirish
```bash
pip install -r requirements.txt
python main.py            # (ixtiyoriy) ma'lumot yaratish + modellarni o'qitish, natijalar konsolga chiqadi
streamlit run app.py      # veb-interfeys (data/models yo'q bo'lsa o'zi yaratadi)
```

## Tuzilma
| Fayl | Vazifasi |
|---|---|
| `taxscoring/data_generator.py` | Sintetik korxonalar (STIR, tarmoq, hudud, moliyaviy va soliq ko'rsatkichlari) |
| `taxscoring/features.py` | Koeffitsientlar: likvidlik, avtonomlik, rentabellik, qarz/aylanma ... |
| `taxscoring/scoring.py` | Qoidaga asoslangan scorecard (vaznlar, bo'lak-chiziqli ballar, xavf sinflari) |
| `taxscoring/ml_model.py` | Logistic Regression / Random Forest / Gradient Boosting, gibrid ball, tushuntirish |
| `taxscoring/report.py` | PDF "soliq pasporti", Excel eksport |
| `taxscoring/plots.py` | Grafiklar |
| `app.py` | Streamlit veb-ilovasi |

## Haqiqiy ma'lumot bilan ishlash
`data/companies.csv` ustunlari `taxscoring/data_generator.py` dagi formatda bo'lsa (oxirgi ustun `default_next_year` — 0/1), `python main.py --use-existing` ni ishga tushirish kifoya.

## Tovar qoldig'i va aylanma manbasi
`data/goods.csv` (korxona tovar guruhlari: miqdor, birlik narxi, summa) va `companies.csv` dagi `has_inventory`, `inv_opening_value`,
`inv_purchases`, `inv_closing_value` ustunlari tovar balansini beradi. Kutilgan aylanma = (boshlang'ich qoldiq + kirim - oxirgi qoldiq) x (1 + tarmoq ustamasi);
deklaratsiya bilan farqi skoringga (vazn 14%) va ML modeliga kiradi. Tarmoq ustamalari `features.py` dagi `SECTOR_INV` da.

## Kompyuterda ishlatish (internetsiz)
1. Python 3.10+ o'rnating (python.org; o'rnatishda "Add python.exe to PATH" ni belgilang).
2. Windows: `run.bat` ni ikki marta bosing. Mac/Linux: `./run.sh`.
3. Birinchi ishga tushishda kutubxonalar o'rnatiladi (internet faqat shu paytda kerak), keyin hammasi internetsiz ishlaydi.
4. Brauzerda `http://localhost:8501` ochiladi. To'xtatish: Ctrl+C.
SMS, email va Telegram bot internet talab qiladi; qolgan barcha bo'limlar internetsiz ishlaydi.

## AI maslahatchi
"Korxonani tekshirish (STIR)" sahifasi pastida: moliyaviy maslahat va 30/60/90 kunlik reja, savol-javob, inspektor yordamchisi.
- Hisob-kitoblar (ball, "agar ... qilinsa ball qanchaga o'zgaradi" simulyatsiyasi) Python'da bajariladi; til modeli faqat tushuntiradi.
- Kalit bo'lmasa yoki AI xato bersa, qolipli maslahatchi ishlaydi (internetsiz ham).
- Claude AI uchun kalit: muhit o'zgaruvchisi `ANTHROPIC_API_KEY` (Windows: `set ANTHROPIC_API_KEY=...`), yoki Streamlit Secrets,
  yoki yon paneldagi "AI sozlamalari" (faqat shu sessiya uchun). Model: `ANTHROPIC_MODEL` (sukut bo'yicha claude-sonnet-5-5).
- Himoya: sessiyada 25 ta, kuniga server bo'yicha 150 ta (`AI_DAILY_LIMIT`) AI so'rovi. Ommaviy saytda kalitni FAQAT Secrets orqali bering
  va Anthropic konsolida xarajat limitini o'rnating.

## Kirish, rollar va xavfsizlik
Tizimga login va parol bilan kiriladi; rol avtomatik aniqlanadi:
- **admin**: hamma bo'limlar + "Foydalanuvchilar" (inspektor/korxona hisoblari, parol, bloklash, jurnal).
- **inspector**: barcha tahlil bo'limlari (dashboard, ro'yxatlar, shubhali aylanma, inspektor yordamchisi, ogohlantirish).
- **company** (yuridik shaxs): faqat "Mening korxonam" va "Raqamlar izohi": faqat O'Z korxonasi (login = STIR).

**Birinchi admin.** Ikki yo'l:
1. Muhit o'zgaruvchisi / Streamlit Secrets: `ADMIN_USERNAME` (ixtiyoriy, sukut bo'yicha admin) va `ADMIN_PASSWORD` (kamida 8 belgi, harf va raqam).
   Ommaviy saytda FAQAT shu yo'lni ishlating.
2. Parol berilmasa, birinchi ochilishda sozlash oynasi chiqadi; kod server konsolida (terminal/`run.bat` oynasida yoki
   Streamlit Cloud: Manage app -> logs) `[AUTH] Birinchi sozlash kodi` satrida ko'rinadi.

**Yuridik shaxs ro'yxatdan o'tishi.** "Kirish" oynasidagi ikkinchi varaq: STIR kiritiladi, korxonaning telefon raqamiga bir martalik
kod yuboriladi (5 daqiqa, 5 urinish), kod to'g'ri bo'lsa parol belgilanadi. SMS provayder (Eskiz) ulanmagan bo'lsa, kod ekranda
ko'rsatiladi (SIMULYATSIYA; `DEMO_SHOW_CODE=0` bilan o'chiriladi va haqiqiy SMS yuboriladi).

**Himoya choralari:** parollar PBKDF2-SHA256 (tuz bilan) ko'rinishida; 5 ta xato paroldan keyin hisob 5 daqiqaga bloklanadi;
30 daqiqa faolsizlikdan keyin sessiya tugaydi; barcha kirish/o'zgarishlar jurnalga yoziladi; oxirgi adminni o'chirib bo'lmaydi.
Bu o'quv prototipi darajasidagi himoya: real tizimda rasmiy identifikatsiya (yagona kirish yoki elektron imzo), audit va
shifrlangan ma'lumotlar bazasi talab qilinadi.

**Muhim:** foydalanuvchilar `data/users.db` (SQLite) faylida saqlanadi (`AUTH_DB_PATH` bilan o'zgartiriladi). Streamlit Cloud'da
fayl tizimi vaqtinchalik: sayt qayta ishga tushganda yaratilgan hisoblar yo'qoladi, faqat Secrets'dagi admin qayta yaratiladi.
Doimiy hisoblar kerak bo'lsa, dasturni o'z kompyuteringizda (run.bat) ishlating.

## Interfeys va logotip
- Logotip: original "soliq muhri" (`brand.py`, `assets/`). Rasmiy logotipingiz bo'lsa, `assets/logo_custom.png` (yoki .svg/.jpg/.webp)
  nomi bilan qo'ying: kirish oynasi, splash, yon panel, fon, PDF va favicon shundan foydalanadi.
- Ranglar `brand.PALETTE` da (to'q ko'k, oltin, havorang; xavf sinflari: yashil/sariq/qizil). CSS va animatsiyalar `theme.py` da.
- Kirish oynasi: harakatlanuvchi, bir-biriga ulangan chiziqlar tarmog'i (sichqoncha bilan o'zaro ta'sir) va orqada katta logotip.
  Kirgandan keyin katta logotipli kutib olish ekrani (~3 soniya), so'ng dastur.
- Animatsiyalar brauzer JavaScript'i orqali ishlaydi; bloklansa, CSS bilan zaxira fon ko'rinadi, dastur esa to'liq ishlaydi.
- `.streamlit/config.toml` ixtiyoriy (asosiy mavzu rangi va menyuni yengillashtirish); Streamlit Cloud'da "Add file -> Create new file" da
  `.streamlit/config.toml` deb yozib yaratiladi. Usiz ham dizayn ishlaydi.
