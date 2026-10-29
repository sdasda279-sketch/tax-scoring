"""Matplotlib grafiklari (Streamlit va hisobotlar uchun)."""
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

from brand import PALETTE as _P
import numpy as np

def _subplots(figsize):
    """pyplot'siz Figure: Streamlit serverida xotira to'planib qolmasligi uchun."""
    fig = Figure(figsize=figsize)
    return fig, fig.subplots()


COLORS = {"Past xavf": _P["green"], "O'rtacha xavf": _P["amber"], "Yuqori xavf": _P["red"]}


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def score_hist(df, col="final_score"):
    fig, ax = _subplots(figsize=(6, 3.4))
    ax.hist(df[col], bins=30, color=_P["blue"], edgecolor="white")
    ax.set_xlabel("Skoring bali (0-100)"); ax.set_ylabel("Korxonalar soni"); _clean(ax)
    fig.tight_layout(); return fig


def risk_bar(df):
    vc = df["risk_class"].value_counts().reindex(list(COLORS)).fillna(0)
    fig, ax = _subplots(figsize=(6, 3.4))
    ax.bar(vc.index, vc.values, color=[COLORS[i] for i in vc.index])
    for i, v in enumerate(vc.values):
        ax.text(i, v, f"{int(v)}\n({v/vc.sum():.0%})", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, vc.max() * 1.25); ax.set_ylabel("Korxonalar soni"); _clean(ax)
    fig.tight_layout(); return fig


def sector_bar(df):
    g = df.groupby("sector")["final_score"].mean().sort_values()
    fig, ax = _subplots(figsize=(6, 3.8))
    ax.barh(g.index, g.values, color=_P["blue"])
    ax.set_xlabel("O'rtacha ball"); ax.set_xlim(0, 100); _clean(ax)
    fig.tight_layout(); return fig


def region_bar(df):
    g = (df["risk_class"].eq("Yuqori xavf").groupby(df["region"]).mean() * 100).sort_values()
    fig, ax = _subplots(figsize=(6, 4))
    ax.barh(g.index, g.values, color=_P["red"])
    ax.set_xlabel("Yuqori xavfli korxonalar ulushi, %"); _clean(ax)
    fig.tight_layout(); return fig


def contribution_bar(contrib_df):
    fig, ax = _subplots(figsize=(6, 0.42 * len(contrib_df) + 0.8))
    y = np.arange(len(contrib_df))
    ax.barh(y, contrib_df["Vazn"], color=_P["line"], label="Maksimal ball")
    ax.barh(y, contrib_df["Olingan ball"], color=_P["blue"], label="Olingan ball")
    ax.set_yticks(y); ax.set_yticklabels(contrib_df["Ko'rsatkich"], fontsize=8); ax.invert_yaxis()
    ax.legend(fontsize=8, frameon=False); _clean(ax)
    fig.tight_layout(); return fig


def roc_plot(meta):
    fig, ax = _subplots(figsize=(5.5, 4.5))
    for name, r in meta["roc"].items():
        auc = meta["metrics"][name]["AUC"]
        ax.plot(r["fpr"], r["tpr"], label=f"{name} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=7, frameon=False, loc="lower right"); _clean(ax)
    fig.tight_layout(); return fig


def importance_bar(meta):
    s = meta["importance"]
    fig, ax = _subplots(figsize=(6, 4))
    names, vals = list(s)[::-1], list(s.values())[::-1]
    ax.barh(names, vals, color=_P["blue"])
    ax.set_xlabel("Muhimlik (AUC pasayishi)"); _clean(ax)
    fig.tight_layout(); return fig


def local_bar(expl):
    fig, ax = _subplots(figsize=(6, 3))
    vals = expl["PD ga ta'siri"] * 100
    ax.barh(expl["Belgi"], vals, color=[_P["red"] if v > 0 else _P["green"] for v in vals])
    ax.invert_yaxis(); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Xavf ehtimoliga ta'siri (foiz punkti): qizil = oshiradi, yashil = kamaytiradi")
    ax.xaxis.label.set_size(7); ax.tick_params(labelsize=8); _clean(ax)
    fig.tight_layout(); return fig


def turnover_source(r):
    """Aylanma manbasi: tovar balansidan kutilgan aylanma va deklaratsiya qilingan aylanma."""
    labels = ["Boshlang'ich\nqoldiq", "+ Kirim\n(xarid)", "- Oxirgi\nqoldiq", "= Sotilgan tovar\ntannarxi",
              "Kutilgan\naylanma", "Deklaratsiya\nqilingan aylanma"]
    vals = [r["inv_opening_value"], r["inv_purchases"], r["inv_closing_value"], r["inv_cogs"],
            r["expected_revenue"], r["revenue"]]
    gap = r["turnover_gap"]
    last = _P["red"] if gap < -0.25 else _P["amber"] if gap < -0.10 else _P["green"]
    fig, ax = _subplots(figsize=(7, 3.8))
    bars = ax.bar(labels, vals, color=[_P["sky"], "#3C7BD0", _P["gold"], _P["blue"], _P["muted"], last])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("mln so'm"); ax.tick_params(axis="x", labelsize=7.5); ax.set_ylim(0, max(vals) * 1.15); _clean(ax)
    fig.tight_layout(); return fig


def inventory_scatter(d):
    fig, ax = _subplots(figsize=(5.5, 4.5))
    for cls, col in COLORS.items():
        s = d[d["risk_class"] == cls]
        ax.scatter(s["expected_revenue"], s["revenue"], s=8, alpha=0.6, color=col, label=cls)
    lim = [max(d["expected_revenue"].min(), 1), d["expected_revenue"].max()]
    ax.plot(lim, lim, "k--", lw=0.8, label="Deklaratsiya = kutilgan")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Kutilgan aylanma (tovar balansidan), mln so'm"); ax.set_ylabel("Deklaratsiya qilingan aylanma, mln so'm")
    ax.legend(fontsize=7, frameon=False); _clean(ax)
    fig.tight_layout(); return fig


def gap_hist(d):
    fig, ax = _subplots(figsize=(5.5, 4.5))
    ax.hist(d["turnover_gap"].clip(-0.8, 1.0) * 100, bins=40, color=_P["blue"], edgecolor="white")
    ax.axvline(-10, color=_P["amber"], ls="--", lw=1); ax.axvline(-25, color=_P["red"], ls="--", lw=1)
    ax.set_xlabel("Aylanma farqi, %  (deklaratsiya - kutilgan) / kutilgan"); ax.set_ylabel("Korxonalar soni"); _clean(ax)
    fig.tight_layout(); return fig
