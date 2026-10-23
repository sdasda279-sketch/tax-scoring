@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === Tax Scoring ===
where python >nul 2>nul
if errorlevel 1 (
  echo Python topilmadi. python.org dan o'rnating va "Add python.exe to PATH" katakchasini belgilang.
  pause
  exit /b 1
)
if not exist ".venv" (
  echo Birinchi marta: muhit yaratilmoqda...
  python -m venv .venv
)
call .venv\Scripts\activate
if not exist ".venv\tayyor.txt" (
  echo Kutubxonalar o'rnatilmoqda (internet kerak, faqat bir marta)...
  pip install -r requirements.txt
  if errorlevel 1 (
    echo Kutubxonalarni o'rnatib bo'lmadi. Internetni tekshiring.
    pause
    exit /b 1
  )
  echo ok > .venv\tayyor.txt
)
echo Dastur ishga tushmoqda. Brauzer o'zi ochiladi. To'xtatish: shu oynada Ctrl+C.
streamlit run app.py
pause
