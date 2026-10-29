# Telegram bot

Bot yuridik shaxslar uchun: faqat O'Z korxonasini ko'rsatadi, kirish login-parol bilan.

1. Telegram'da @BotFather ga `/newbot` yozing, token oling.
2. Muhit o'zgaruvchilari:
   - Windows (cmd):  `set TELEGRAM_BOT_TOKEN=SIZNING_TOKEN`  va (ixtiyoriy) `set ADMIN_IDS=SIZNING_TELEGRAM_ID`
   - Linux/Mac:      `export TELEGRAM_BOT_TOKEN=...`
3. `python bot.py`
4. Foydalanuvchi: `/kirish` -> STIR -> parol (parol yozilgan xabar darhol o'chiriladi) -> korxona kartochkasi, PDF pasport, "Nima uchun?".
   Hisob veb-saytda ro'yxatdan o'tganda yaratiladi. `/mening`, `/chiqish`. Botdagi kirish 12 soat amal qiladi.
5. `ADMIN_IDS` dagi Telegram foydalanuvchilari: `/tekshir STIR`, `/top10`, `/statistika`, `/yubor` (kirgan va xavfi yuqori korxonalarga ogohlantirish).
   O'z ID ingizni botga `/id` yozib bilib oling.

DIQQAT: bot hisoblar bazasini (`data/users.db`) o'zi ishlayotgan kompyuterdan o'qiydi. Veb-sayt va bot bir xil hisoblardan foydalanishi
uchun ikkalasini ham bitta kompyuterda (bir papkadan) ishga tushiring yoki `AUTH_DB_PATH` ni bir xil faylga yo'naltiring.
Internetdagi Streamlit Cloud sayti va sizning kompyuteringizdagi bot turli bazalardan foydalanadi.
