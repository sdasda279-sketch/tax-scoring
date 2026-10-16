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
