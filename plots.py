"""Matplotlib grafiklari (Streamlit va hisobotlar uchun)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"Past xavf": "#2e9e5b", "O'rtacha xavf": "#e0a526", "Yuqori xavf": "#d64545"}


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def score_hist(df, col="final_score"):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(df[col], bins=30, color="#4c78a8", edgecolor="white")
    ax.set_xlabel("Skoring bali (0-100)"); ax.set_ylabel("Korxonalar soni"); _clean(ax)
    fig.tight_layout(); return fig


def risk_bar(df):
    vc = df["risk_class"].value_counts().reindex(list(COLORS)).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(vc.index, vc.values, color=[COLORS[i] for i in vc.index])
    for i, v in enumerate(vc.values):
        ax.text(i, v, f"{int(v)}\n({v/vc.sum():.0%})", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, vc.max() * 1.25); ax.set_ylabel("Korxonalar soni"); _clean(ax)
    fig.tight_layout(); return fig


def sector_bar(df):
    g = df.groupby("sector")["final_score"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh(g.index, g.values, color="#4c78a8")
    ax.set_xlabel("O'rtacha ball"); ax.set_xlim(0, 100); _clean(ax)
    fig.tight_layout(); return fig


def region_bar(df):
    g = (df["risk_class"].eq("Yuqori xavf").groupby(df["region"]).mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(g.index, g.values, color="#d64545")
    ax.set_xlabel("Yuqori xavfli korxonalar ulushi, %"); _clean(ax)
    fig.tight_layout(); return fig


def contribution_bar(contrib_df):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    y = np.arange(len(contrib_df))
    ax.barh(y, contrib_df["Vazn"], color="#d9d9d9", label="Maksimal ball")
    ax.barh(y, contrib_df["Olingan ball"], color="#4c78a8", label="Olingan ball")
    ax.set_yticks(y); ax.set_yticklabels(contrib_df["Ko'rsatkich"], fontsize=8); ax.invert_yaxis()
    ax.legend(fontsize=8, frameon=False); _clean(ax)
    fig.tight_layout(); return fig


def roc_plot(meta):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, r in meta["roc"].items():
        auc = meta["metrics"][name]["AUC"]
        ax.plot(r["fpr"], r["tpr"], label=f"{name} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=7, frameon=False, loc="lower right"); _clean(ax)
    fig.tight_layout(); return fig


def importance_bar(meta):
    s = meta["importance"]
    fig, ax = plt.subplots(figsize=(6, 4))
    names, vals = list(s)[::-1], list(s.values())[::-1]
    ax.barh(names, vals, color="#4c78a8")
    ax.set_xlabel("Muhimlik (AUC pasayishi)"); _clean(ax)
    fig.tight_layout(); return fig


def local_bar(expl):
    fig, ax = plt.subplots(figsize=(6, 3))
    vals = expl["PD ga ta'siri"] * 100
    ax.barh(expl["Belgi"], vals, color=["#d64545" if v > 0 else "#2e9e5b" for v in vals])
    ax.invert_yaxis(); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Xavf ehtimoliga ta'siri (foiz punkti): qizil = oshiradi, yashil = kamaytiradi")
    ax.xaxis.label.set_size(7); ax.tick_params(labelsize=8); _clean(ax)
    fig.tight_layout(); return fig
