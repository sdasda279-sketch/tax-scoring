"""Mashinaviy o'qitish modellari va gibrid skoring."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_LABELS, MODEL_NUM, build_X
from .scoring import classify, rule_score, to_scale

TARGET = "default_next_year"
PD_FULL_ZERO = 0.40  # PD shu qiymatga yetganda ML ball 0 bo'ladi
W_RULE = 0.5         # gibrid modelda qoidaga asoslangan qismning ulushi

MODELS = {
    "Logistic Regression": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=300, min_samples_leaf=10, class_weight="balanced_subsample", random_state=42, n_jobs=-1),
    "Gradient Boosting": lambda: GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42),
}


def pd_to_ml100(pdv):
    return 100 * np.clip(1 - np.asarray(pdv, float) / PD_FULL_ZERO, 0, 1)


def hybrid_100(rule_100, pdv):
    return W_RULE * np.asarray(rule_100, float) + (1 - W_RULE) * pd_to_ml100(pdv)


def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def _metrics(y, p, thr):
    yh = (p >= thr).astype(int)
    auc = roc_auc_score(y, p)
    return {"AUC": auc, "Gini": 2 * auc - 1, "KS": ks_stat(y, p), "Accuracy": accuracy_score(y, yh),
            "Precision": precision_score(y, yh, zero_division=0), "Recall": recall_score(y, yh),
            "F1": f1_score(y, yh), "Brier": brier_score_loss(y, p)}


def train_all(df: pd.DataFrame, out_dir="models", seed=42):
    out = Path(out_dir); out.mkdir(exist_ok=True)
    X, y = build_X(df), df[TARGET].values
    cols = list(X.columns)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)

    results, rocs, fitted = {}, {}, {}
    for name, mk in MODELS.items():
        m = mk().fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        thr = float(np.quantile(p, 1 - ytr.mean()))  # taxminan bir xil ulush bo'yicha chegara
        r = _metrics(yte, p, thr)
        r["CV_AUC_mean"] = float(cross_val_score(mk(), X, y, cv=cv, scoring="roc_auc").mean())
        results[name] = r
        fpr, tpr, _ = roc_curve(yte, p)
        rocs[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        fitted[name] = m

    # Qoidaga asoslangan model (test to'plamida) va gibrid
    rs = rule_score(df.loc[Xte.index])
    risk_rule = 1 - rs["rule_100"].values / 100
    thr = float(np.quantile(risk_rule, 1 - ytr.mean()))
    results["Qoidaga asoslangan (scorecard)"] = {**_metrics(yte, risk_rule, thr), "CV_AUC_mean": float("nan")}
    fpr, tpr, _ = roc_curve(yte, risk_rule); rocs["Qoidaga asoslangan (scorecard)"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    best = max(MODELS, key=lambda k: results[k]["CV_AUC_mean"])
    pbest = fitted[best].predict_proba(Xte)[:, 1]
    hyb_risk = 1 - hybrid_100(rs["rule_100"].values, pbest) / 100
    thr = float(np.quantile(hyb_risk, 1 - ytr.mean()))
    results["Gibrid (scorecard + ML)"] = {**_metrics(yte, hyb_risk, thr), "CV_AUC_mean": float("nan")}
    fpr, tpr, _ = roc_curve(yte, hyb_risk); rocs["Gibrid (scorecard + ML)"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    # Belgilar muhimligi (permutatsiya usuli, test to'plamida)
    pi = permutation_importance(fitted[best], Xte, yte, scoring="roc_auc", n_repeats=8, random_state=seed)
    imp = pd.Series(pi.importances_mean, index=cols).sort_values(ascending=False)
    imp = imp[[c for c in imp.index if c in MODEL_NUM]].head(12)

    # To'liq to'plam uchun "out-of-fold" PD (halol baho: model o'z o'qitish ma'lumotini ko'rmaydi)
    oof = cross_val_predict(MODELS[best](), X, y, cv=cv, method="predict_proba")[:, 1]
    final = MODELS[best]().fit(X, y)  # yangi korxonalar uchun yakuniy model

    joblib.dump({"model": final, "columns": cols, "best": best, "median": X[MODEL_NUM].median().to_dict()},
                out / "best_model.joblib")
    meta = {"best": best, "metrics": results, "roc": rocs, "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
            "default_rate": float(y.mean()),
            "importance": {FEATURE_LABELS.get(k, k): float(v) for k, v in imp.items()}}
    (out / "metrics.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    return meta, oof


def score_dataset(df: pd.DataFrame, pd_values) -> pd.DataFrame:
    rs = rule_score(df)
    res = df.copy()
    res = pd.concat([res, rs], axis=1)
    res["pd"] = np.asarray(pd_values, float)
    res["ml_100"] = pd_to_ml100(res["pd"])
    res["final_100"] = hybrid_100(res["rule_100"], res["pd"])
    res["final_score"] = to_scale(res["final_100"])
    res["risk_class"] = classify(res["final_100"])
    return res


def load_model(path="models/best_model.joblib"):
    return joblib.load(path)


def predict_pd(bundle, df: pd.DataFrame):
    return bundle["model"].predict_proba(build_X(df, bundle["columns"]))[:, 1]


def explain_local(bundle, row: pd.DataFrame, top=6) -> pd.DataFrame:
    """Okklyuziya usuli: har bir belgi mediana bilan almashtirilsa PD qanchaga o'zgaradi.
    Musbat qiymat = shu belgi xavfni oshirmoqda."""
    X = build_X(row, bundle["columns"]).iloc[[0]]
    base = bundle["model"].predict_proba(X)[0, 1]
    rows = []
    for f in MODEL_NUM:
        if f == "log_revenue":
            continue
        X2 = X.copy(); X2[f] = bundle["median"][f]
        delta = base - bundle["model"].predict_proba(X2)[0, 1]
        rows.append((FEATURE_LABELS.get(f, f), float(X[f].iloc[0]), float(delta)))
    df = pd.DataFrame(rows, columns=["Belgi", "Qiymat", "PD ga ta'siri"])
    return df.reindex(df["PD ga ta'siri"].abs().sort_values(ascending=False).index).head(top).reset_index(drop=True)
