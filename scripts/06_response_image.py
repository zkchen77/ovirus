# -*- coding: utf-8 -*-
"""
@author: zkchen
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import textwrap
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QC = os.path.join(ROOT, "data", "mr_qc")
RES = os.path.join(ROOT, "data", "mr_results")
INST = os.path.join(ROOT, "data", "mr_instruments")
FIG = os.path.join(ROOT, "data", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica", "Liberation Sans"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 1200, "savefig.dpi": 1200, "savefig.bbox": "tight",
})

SK = ['ukb-b-1557', 'ukb-b-14486', 'finn-b-M13_OSTEONECROSIS', 'ukb-b-12141',
      'finn-b-M13_SCOLIOSIS', 'ukb-d-M13_SPINSTENOSIS',
      'finn-b-M13_SPONDYLOLISTHESIS', 'finn-b-M13_SPONDYLOSIS']
NAME = {
    'ukb-b-1557': 'Low back pain', 'ukb-b-14486': 'Osteoarthritis',
    'finn-b-M13_OSTEONECROSIS': 'Osteonecrosis', 'ukb-b-12141': 'Osteoporosis',
    'finn-b-M13_SCOLIOSIS': 'Scoliosis', 'ukb-d-M13_SPINSTENOSIS': 'Spinal stenosis',
    'finn-b-M13_SPONDYLOLISTHESIS': 'Spondylolisthesis',
    'finn-b-M13_SPONDYLOSIS': 'Spondylosis',
}


def short(s):
    if not isinstance(s, str):
        return s
    return s.split("||")[0].strip() if "||" in s else s.strip()


# --- Fig R1.1
def fig_threshold_audit():
    rows = []
    for f in sorted(glob.glob(os.path.join(INST, "exposure_*.csv"))):
        gid = os.path.basename(f)[len("exposure_"):-4]
        p = pd.read_csv(f)["pval.exposure"]
        rows.append({"id": gid, "grp": "skeletal" if gid in SK else "infection",
                     "n5e6": int((p < 5e-6).sum()), "n1e6": int((p < 1e-6).sum()),
                     "n5e7": int((p < 5e-7).sum()), "n5e8": int((p < 5e-8).sum())})
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(QC, "QC0_threshold_audit.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8.2, 4.2))

    s = d[d.grp == "skeletal"].set_index("id").reindex(SK)
    if s.isna().any().any():
        missing = s.index[s.isna().any(axis=1)].tolist()
        raise RuntimeError(f"The ID in SK cannot be found in exposure_ *. csv: {missing}")

    s["name"] = [NAME[i] for i in s.index]
    x = np.arange(len(s))
    w = 0.2
    for k, (col, lab, c) in enumerate([
        ("n5e6", "p<5e-6 (used)",            "#2c7bb6"),
        ("n1e6", "p<1e-6",                   "#abd9e9"),
        ("n5e7", "p<5e-7",                   "#fdae61"),
        ("n5e8", "p<5e-8 (conventional)",    "#d7191c"),
    ]):
        bars = ax.bar(x + (k - 1.5) * w, s[col], w,
                      label=lab, color=c, edgecolor="white", lw=.5)
        ax.bar_label(bars, fmt="%d", fontsize=6, padding=1, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(s["name"], rotation=35, ha="right")
    ax.set_ylabel("Number of instruments")
    ax.set_title(f"Skeletal exposures (n = {len(s)})", loc="left")
    ax.legend(fontsize=7.5, frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.22))
    ax.axhline(0, color="k", lw=.6)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    p = os.path.join(FIG, "FigR1.1_instrument_threshold_audit.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(" ->", os.path.basename(p))
    return d

# --- Fig R1.2
def _wrap_label(s, width=38):
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False)) or s

def fig_sensitivity_forest():
    q = pd.read_csv(os.path.join(QC, "QC5_summary.csv"))
    f = q[(q.direction == "o2v") & (q.p_Inverse_variance_weighted < 0.05)].copy()
    if f.empty:
        print("  [skip] no IVW-significant results to plot")
        return f
    f = f.sort_values("p_Inverse_variance_weighted").reset_index(drop=True)

    f["lab"] = f.apply(
        lambda r: _wrap_label(f"{short(r.exposure)} \u2192 {short(r.outcome)}",
                              width=38),
        axis=1,
    )
    n = len(f)

    methods = [
        ("Inverse_variance_weighted", "IVW",           "#2c7bb6"),
        ("Weighted_median",           "Weighted med.", "#fdae61"),
        ("MR_Egger",                  "MR-Egger",      "#d7191c"),
        ("Weighted_mode",             "Weighted mode", "#7fbc41"),
        ("Simple_mode",               "Simple mode",   "#8c6bb1"),
    ]

    fig_w = 20.0
    fig_h = max(8.0, 0.55 * n + 2.5)
    fig, axes = plt.subplots(
        1, 2, figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [2, 1], "wspace": 0.06},
    )

    # A
    ax = axes[0]
    y   = np.arange(n)[::-1]
    off = np.linspace(0.22, -0.22, len(methods))

    # jitter
    rng   = np.random.default_rng(77)
    jit_y = rng.uniform(-0.035, 0.035, size=n)
    jit_x = rng.uniform(-0.08,  0.08,  size=(n, len(methods)))

    # dividing line
    for yv in np.arange(-0.5, n, 1.0):
        ax.axhline(yv, color="#ebebeb", lw=0.7, zorder=1)

    for k, ((m, lab, c), o) in enumerate(zip(methods, off)):
        b   = f[f"b_{m}"].to_numpy()
        p   = f[f"p_{m}"].to_numpy()
        sig = p < 0.05
        xx  = np.sign(b) + jit_x[:, k]
        yy  = y + o + jit_y

        ax.scatter(xx[sig],  yy[sig],
                   color=c, s=130, edgecolor="white", lw=1.0,
                   zorder=3, label=lab)
        ax.scatter(xx[~sig], yy[~sig],
                   facecolor="white", edgecolor=c, lw=1.8, s=120, zorder=2)

    ax.axvline(0, color="grey", lw=0.8, ls=":")
    for xv in (-1, 1):
        ax.axvline(xv, color="k", lw=0.8, alpha=0.30, ls="--")
    ax.set_xlim(-1.45, 1.45)
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(["\u22121\nprotective", "0\nnull", "+1\nrisk"], fontsize=12)
    ax.set_yticks(y)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_yticklabels(f.lab, fontsize=11.5, linespacing=1.25)
    for label in ax.get_yticklabels():
        label.set_horizontalalignment("right")
    ax.tick_params(axis="y", pad=6)
    ax.set_title(f"A  Direction across five estimators "
                 f"(n = {n}; IVW p < 0.05)", loc="left", fontsize=14)

    handles = [Line2D([], [], marker="o", ls="", markersize=10,
                      color=c, label=lab) for _, lab, c in methods]
    handles.append(Line2D([], [], marker="o", ls="", markersize=10,
                          markerfacecolor="white", markeredgecolor="#555",
                          markeredgewidth=1.8, label="hollow: p \u2265 0.05"))
    ax.legend(handles=handles, fontsize=11, frameon=False, ncol=6,
              loc="upper center", bbox_to_anchor=(0.5, -0.08),
              handletextpad=0.3, columnspacing=1.5)
    ax.spines[["top", "right"]].set_visible(False)

    # B
    ax = axes[1]
    metrics = [
        ("het_pval_IVW",         "Het.\n(IVW)"),
        ("het_pval_Egger",       "Het.\n(Egger)"),
        ("egger_intercept_pval", "Pleiotropy\n(Egger int.)"),
    ]
    M = np.column_stack([f[c].to_numpy() for c, _ in metrics])
    M = np.clip(M, 1e-6, 1.0)
    L = -np.log10(M)
    vmax = max(3.0, float(L.max()))
    im = ax.imshow(L, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[1] for m in metrics], fontsize=11.5)
    ax.set_yticks([])
    ax.set_ylim(-0.5, n - 0.5)

    for i in range(n):
        for j in range(len(metrics)):
            is_dark   = L[i, j] > 0.6 * vmax
            color     = "white" if is_dark else "black"
            stroke_fg = "#2b2b2b" if is_dark else "white"
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=12, color=color,
                    path_effects=[pe.withStroke(linewidth=2.0,
                                                foreground=stroke_fg)])

    for j in (0.5, 1.5):
        ax.axvline(j, color="white", lw=2.0)
    ax.set_title("B  Diagnostic p-values", loc="left", fontsize=14)

    cb = fig.colorbar(im, ax=ax, fraction=.04, pad=.03)
    cb.set_label(r"$-\log_{10}(p)$", fontsize=11)
    cb.ax.tick_params(labelsize=10)

    p = os.path.join(FIG, "FigR1.2_sensitivity_forest.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(" ->", os.path.basename(p))
    return f

# --- Fig R1.3
def fig_pattern_congruity():
    q = pd.read_csv(os.path.join(QC, "QC5_summary.csv"))
    f = q[(q.direction == "o2v") &
          (q.p_Inverse_variance_weighted < 0.05) &
          (q.n_methods_concordant >= 4) &
          (q.het_pval_IVW >= 0.05) &
          (q.egger_intercept_pval >= 0.05) &
          (q.steiger_correct_direction == True) &
          (q.loo_n_sign_flip == 0)].copy()
    if f.empty:
        print("  [skip] no IVW-significant results to plot")
        return f

    # sorted
    f["_sign"] = np.sign(f["b_Inverse_variance_weighted"]).astype(int)
    expo_signs = f.groupby("exposure")["_sign"].apply(
        lambda s: frozenset(s) - {0}
    )

    def _grp_key(signs):
        if len(signs) > 1:  return 0
        if signs == frozenset({1}):  return 1
        if signs == frozenset({-1}): return 2
        return 3

    f["_grp"] = f["exposure"].map(lambda e: _grp_key(expo_signs[e]))
    f = (f.sort_values(["_grp", "exposure", "p_Inverse_variance_weighted"])
           .drop(columns=["_sign"])
           .reset_index(drop=True))

    grps = f["_grp"].to_numpy()
    f = f.drop(columns=["_grp"])

    f["lab"] = f.apply(
        lambda r: _wrap_label(f"{short(r.exposure)} \u2192 {short(r.outcome)}",
                              width=38),
        axis=1,
    )
    n = len(f)

    methods = [
        ("Inverse_variance_weighted", "IVW",           "#2c7bb6"),
        ("Weighted_median",           "Weighted med.", "#fdae61"),
        ("MR_Egger",                  "MR-Egger",      "#d7191c"),
        ("Weighted_mode",             "Weighted mode", "#7fbc41"),
        ("Simple_mode",               "Simple mode",   "#8c6bb1"),
    ]

    fig_w = 14.5
    fig_h = max(6.0, 0.55 * n + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    y   = np.arange(n)[::-1]
    off = np.linspace(0.22, -0.22, len(methods))

    rng   = np.random.default_rng(77)
    jit_y = rng.uniform(-0.035, 0.035, size=n)
    jit_x = rng.uniform(-0.08,  0.08,  size=(n, len(methods)))

    # Interline dividing line: 
    # Within the group → gray line; Inter group → Black line
    for yv in np.arange(-0.5, n, 1.0):
        i_top = n - 1.5 - yv
        i_bot = n - 0.5 - yv
        if (0 <= i_top < n) and (0 <= i_bot < n) and \
           grps[int(i_top)] != grps[int(i_bot)]:
            ax.axhline(yv, color="black", lw=1.3, zorder=1.5)
        else:
            ax.axhline(yv, color="#ebebeb", lw=0.7, zorder=1)

    for k, ((m, lab, c), o) in enumerate(zip(methods, off)):
        b   = f[f"b_{m}"].to_numpy()
        p   = f[f"p_{m}"].to_numpy()
        sig = p < 0.05
        xx  = np.sign(b) + jit_x[:, k]
        yy  = y + o + jit_y

        ax.scatter(xx[sig],  yy[sig],
                   color=c, s=130, edgecolor="white", lw=1.0,
                   zorder=3, label=lab)
        ax.scatter(xx[~sig], yy[~sig],
                   facecolor="white", edgecolor=c, lw=1.8, s=120, zorder=2)

    ax.axvline(0, color="grey", lw=0.8, ls=":")
    for xv in (-1, 1):
        ax.axvline(xv, color="k", lw=0.8, alpha=0.30, ls="--")
    ax.set_xlim(-1.45, 1.45)
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(["\u22121\nprotective", "0\nnull", "+1\nrisk"], fontsize=12)
    ax.set_yticks(y)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_yticklabels(f.lab, fontsize=11.5, linespacing=1.25)
    for label in ax.get_yticklabels():
        label.set_horizontalalignment("right")
    ax.tick_params(axis="y", pad=6)
    ax.set_title(f"Direction across five estimators "
                 f"(n = {n})", loc="left", fontsize=14)

    handles = [Line2D([], [], marker="o", ls="", markersize=10,
                      color=c, label=lab) for _, lab, c in methods]
    handles.append(Line2D([], [], marker="o", ls="", markersize=10,
                          markerfacecolor="white", markeredgecolor="#555",
                          markeredgewidth=1.8, label="hollow: p \u2265 0.05"))
    ax.legend(handles=handles, fontsize=11, frameon=False, ncol=6,
              loc="upper center", bbox_to_anchor=(0.5, -0.08),
              handletextpad=0.3, columnspacing=1.5)
    ax.spines[["top", "right"]].set_visible(False)

    p = os.path.join(FIG, "FigR1.3_pattern_congruity.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(" ->", os.path.basename(p))
    return f

# --- Fig R3.1
def fig_instrument_strength():
    snp = pd.concat([pd.read_csv(os.path.join(RES, f"{t}_snp_total.csv"))
                     for t in ("o2v", "v2o")], ignore_index=True)
    k = snp[snp["mr_keep"] == True].copy()
    k["F"] = k["beta.exposure"] ** 2 / k["se.exposure"] ** 2
    k["grp"] = np.where(k["id.exposure"].isin(SK), "skeletal exposure",
                        "infection exposure")

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.3))
    ax = axes[0]
    bins = np.linspace(k.F.min(), min(k.F.max(), 90), 60)
    for g, c in [("skeletal exposure", "#2c7bb6"), ("infection exposure", "#fdae61")]:
        ax.hist(k.loc[k.grp == g, "F"], bins=bins, alpha=.75, label=g, color=c)
    ax.axvline(10, color="#d7191c", ls="--", lw=1.2)
    ax.text(10.6, ax.get_ylim()[1] * .85, "F = 10\n(weak-instrument rule)",
            fontsize=7, color="#d7191c")
    ax.set_xlabel("F statistic  =  (beta / se)$^2$")
    ax.set_ylabel("Instrument count")
    ax.set_title("A  All retained instruments", loc="left")
    ax.legend(fontsize=6.5, frameon=False)

    ax = axes[1]
    k["name"] = k["id.exposure"].map(NAME).fillna("infection traits")
    g = k.groupby("name").F.agg(["mean", "min", "max"])
    g = g.sort_values("mean")
    ax.barh(range(len(g)), g["mean"], color="#2c7bb6", height=.6)
    ax.errorbar(g["mean"], range(len(g)),
                xerr=[g["mean"] - g["min"], g["max"] - g["mean"]],
                fmt="none", ecolor="k", elinewidth=.8, capsize=2)
    ax.set_yticks(range(len(g))); ax.set_yticklabels(g.index, fontsize=7)
    ax.axvline(10, color="#d7191c", ls="--", lw=1.2)
    ax.set_xlabel("F statistic (mean, min-max across instruments)")
    ax.set_title("B  By exposure", loc="left")

    plt.tight_layout()
    p = os.path.join(FIG, "FigR3.1_instrument_strength.png")
    plt.savefig(p); plt.close()
    print(" ->", os.path.basename(p),
          f"| min F = {k.F.min():.1f}, mean F = {k.F.mean():.1f}, "
          f"{(k.F > 10).mean()*100:.0f}% > 10")
    return k




if __name__ == "__main__":
    print("[06] building rebuttal figures ...")
    fig_threshold_audit()
    fig_sensitivity_forest()
    fig_pattern_congruity()
    fig_instrument_strength()
    print("[06] done.")
