"""To'liq pipeline: ma'lumot yaratish -> modellarni o'qitish -> skoringlangan jadvalni saqlash.
Ishga tushirish:  python main.py [--n 3000]"""
import argparse
import pandas as pd
from data_generator import generate
from ml_model import score_dataset, train_all


def run(n=3000, seed=42, use_existing=False):
    if use_existing:
        df = pd.read_csv("data/companies.csv", dtype={"stir": str})
    else:
        df = generate(n, seed)
        df.to_csv("data/companies.csv", index=False)
    meta, oof = train_all(df)
    scored = score_dataset(df, oof)
    scored.to_csv("data/scored_companies.csv", index=False)
    return df, meta, scored


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-existing", action="store_true", help="data/companies.csv ni qayta yaratmasdan ishlatish")
    a = ap.parse_args()
    _, meta, scored = run(a.n, a.seed, a.use_existing)
    print("Eng yaxshi ML model:", meta["best"])
    for k, v in meta["metrics"].items():
        print(f"{k:34s} AUC={v['AUC']:.3f} KS={v['KS']:.3f} F1={v['F1']:.3f} Recall={v['Recall']:.3f}")
    print(scored["risk_class"].value_counts(normalize=True).round(3).to_dict())
