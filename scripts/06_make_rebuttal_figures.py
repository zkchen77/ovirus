# -*- coding: utf-8 -*-
"""
06_make_rebuttal_figures.py
@author: zkchen

===========================
Outputs (data/figures/)
  FigR1_instrument_threshold_audit.png   why p<5e-6 and not p<5e-8
  FigR2_instrument_strength.png          F-statistic distribution
  FigR3_pvalue_distribution.png          histogram + QQ of the 312 IVW tests
  FigR4_sensitivity_forest.png           5 estimators for the retained pairs
  FigR5_leave_one_out.png                LOO for the two strongest pairs

=====================

"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

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


# --- Fig R1
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

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))

    ax = axes[0]
    s = d[d.grp == "skeletal"].set_index("id").loc[SK]
    s["name"] = [NAME[i] for i in s.index]
    x = np.arange(len(s))
    w = 0.2
    for k, (col, lab, c) in enumerate([("n5e6", "p<5e-6 (used)", "#2c7bb6"),
                                       ("n1e6", "p<1e-6", "#abd9e9"),
                                       ("n5e7", "p<5e-7", "#fdae61"),
                                       ("n5e8", "p<5e-8 (conventional)", "#d7191c")]):
        ax.bar(x + (k - 1.5) * w, s[col], w, label=lab, color=c, edgecolor="white", lw=.5)
    ax.set_xticks(x); ax.set_xticklabels(s["name"], rotation=35, ha="right")
    ax.set_ylabel("Number of instruments")
    ax.set_title("A  Skeletal exposures (n = 8)", loc="left")
    ax.legend(fontsize=7, frameon=False)
    ax.axhline(0, color="k", lw=.6)

    ax = axes[1]
    inf = d[d.grp == "infection"].sort_values("n5e6", ascending=False)
    x = np.arange(len(inf))
    for k, (col, c) in enumerate([("n5e6", "#2c7bb6"), ("n5e8", "#d7191c")]):
        ax.bar(x + (k - .5) * .4, inf[col], .4,
               label="p<5e-6 (used)" if k == 0 else "p<5e-8 (conventional)",
               color=c, edgecolor="white", lw=.5)
    ax.set_xticks(x)
    ax.set_xticklabels([i.replace("finn-b-AB1_", "")[:12] for i in inf.id],
                       rotation=90, fontsize=5.5)
    ax.set_ylabel("Number of instruments")
    ax.set_title(f"B  Infectious-disease outcomes (n = {len(inf)})", loc="left")
    ax.legend(fontsize=7, frameon=False)
    zero8 = int((inf.n5e8 == 0).sum())
    ax.text(0.98, 0.94, f"{zero8}/{len(inf)} have 0 instruments\nat p<5e-8",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            bbox=dict(fc="#fff6f6", ec="#d7191c", lw=.7))
    plt.tight_layout()
    p = os.path.join(FIG, "FigR1_instrument_threshold_audit.png")
    plt.savefig(p); plt.close()
    print(" ->", os.path.basename(p))
    return d


# --- Fig R2
def fig_instrument_strength():
    snp = pd.concat([pd.read_csv(os.path.join(RES, f"{t}_snp_total.csv"))
                     for t in ("o2v", "v2o")], ignore_index=True)
    k = snp[snp["mr_keep"] == True].copy()
    k["F"] = k["beta.exposure"] ** 2 / k["se.exposure"] ** 2
    k["grp"] = np.where(k["id.exposure"].isin(SK), "skeletal exposure",
                        "infection exposure")

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.3))
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

    ax = axes[2]
    q1 = pd.read_csv(os.path.join(QC, "QC1_instrument_strength.csv"))
    ns = q1.drop_duplicates(["id.exposure", "id.outcome"]).nsnp_harmonised
    ax.hist(ns, bins=range(int(ns.min()), int(ns.max()) + 2), color="#7fbc41",
            edgecolor="white", lw=.5)
    ax.set_xlabel("Instruments per exposure-outcome pair")
    ax.set_ylabel("Pairs")
    ax.set_title(f"C  n = {len(ns)} pairs, median = {int(ns.median())}", loc="left")

    plt.tight_layout()
    p = os.path.join(FIG, "FigR2_instrument_strength.png")
    plt.savefig(p); plt.close()
    print(" ->", os.path.basename(p),
          f"| min F = {k.F.min():.1f}, mean F = {k.F.mean():.1f}, "
          f"{(k.F > 10).mean()*100:.0f}% > 10")
    return k


# --- Fig R3
def fig_pvalue_distribution():
    q = pd.read_csv(os.path.join(QC, "QC5_summary.csv"))
    f = q[q.direction == "o2v"]["p_Inverse_variance_weighted"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    ax = axes[0]
    ax.hist(f, bins=20, range=(0, 1), color="#b8d4ea", edgecolor="#2c7bb6", lw=.7)
    ax.axhline(len(f) / 20, color="#d7191c", ls="--", lw=1.2,
               label=f"uniform expectation ({len(f)/20:.0f} per bin)")
    ax.set_xlabel("IVW p-value"); ax.set_ylabel("Number of pairs")
    ax.set_title("A  Forward direction, 312 tests", loc="left")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    obs = -np.log10(np.sort(f))
    exp = -np.log10(np.arange(1, len(f) + 1) / (len(f) + 1))
    ax.scatter(exp, obs, s=9, c="#2c7bb6", alpha=.7)
    lim = max(exp.max(), obs.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="#d7191c", lw=1.2)
    ax.set_xlabel("Expected  $-log_{10}(p)$"); ax.set_ylabel("Observed  $-log_{10}(p)$")
    ax.set_title("B  Q-Q plot", loc="left")
    lam = np.median(stats.chi2.ppf(1 - f, 1)) / stats.chi2.ppf(0.5, 1)
    ax.text(0.05, 0.92, f"$\\lambda_{{GC}}$ = {lam:.2f}", transform=ax.transAxes,
            fontsize=8, bbox=dict(fc="white", ec="grey", lw=.6))

    plt.tight_layout()
    p = os.path.join(FIG, "FigR3_pvalue_distribution.png")
    plt.savefig(p); plt.close()
    print(" ->", os.path.basename(p), f"| lambda_GC = {lam:.2f}, "
          f"{(f<0.05).sum()}/312 = {(f<0.05).mean()*100:.1f}% at p<0.05")
    return f


# --- Fig R4
def fig_sensitivity_forest():
    q = pd.read_csv(os.path.join(QC, "QC5_summary.csv"))
    f = q[(q.direction == "o2v") & (q.p_Inverse_variance_weighted < 0.05)].copy()
    f = f.sort_values("p_Inverse_variance_weighted")
    f["lab"] = f.apply(lambda r: f"{short(r.exposure)} → {short(r.outcome)[:26]}", axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.4),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    y = np.arange(len(f))[::-1]
    cols = {"Inverse_variance_weighted": "#2c7bb6", "Weighted_median": "#fdae61",
            "MR_Egger": "#d7191c", "Weighted_mode": "#7fbc41",
            "Simple_mode": "#8c6bb1"}
    off = {"Inverse_variance_weighted": .18, "Weighted_median": .09,
           "MR_Egger": 0.0, "Weighted_mode": -.09, "Simple_mode": -.18}
    for m, c in cols.items():
        b = f[f"b_{m}"].values
        p = f[f"p_{m}"].values
        # relative effect so that traits on different scales are comparable
        lo = b - 1.96 * np.abs(b) / stats.norm.ppf(1 - p / 2) if np.all(p > 0) else b * 0
        ax.scatter(b / np.abs(b), y + off[m], color=c, s=16, zorder=3,
                   label=m.replace("_", " "))
    ax.axvline(1, color="k", lw=.8, ls="-")
    ax.axvline(-1, color="k", lw=.8, ls="-")
    ax.axvline(0, color="grey", lw=.6, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels(f.lab, fontsize=7)
    ax.set_xlabel("Sign of the causal estimate (IVW-scaled)\n+1 = increases risk, -1 = protective")
    ax.set_xlim(-1.55, 1.55)
    ax.set_xticks([-1.5, -1, 0, 1, 1.5])
    ax.set_title("A  Direction across five estimators", loc="left")
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="lower left")

    ax = axes[1]
    metrics = [("het_pval_IVW", "Heterogeneity\n(Cochran Q, IVW)"),
               ("het_pval_Egger", "Heterogeneity\n(Cochran Q, Egger)"),
               ("egger_intercept_pval", "Pleiotropy\n(Egger intercept)")]
    M = np.zeros((len(f), len(metrics)))
    for j, (c, _) in enumerate(metrics):
        M[:, j] = f[c].values
    M = np.clip(M, 1e-6, 1)
    im = ax.imshow(-np.log10(M), aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=3)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[1] for m in metrics], fontsize=6.5, rotation=20, ha="right")
    ax.set_yticks(range(len(f))); ax.set_yticklabels([])
    for i in range(len(f)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if -np.log10(M[i, j]) > 1.5 else "black")
    ax.set_title("B  Diagnostic p-values",
                 loc="left", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=.03, pad=.04, label="$-log_{10}(p)$")

    plt.tight_layout()
    p = os.path.join(FIG, "FigR4_sensitivity_forest.png")
    plt.savefig(p); plt.close()
    print(" ->", os.path.basename(p))
    return f


# --- Fig R5
def fig_leave_one_out():
    loo = pd.read_csv(os.path.join(QC, "QC3_leave_one_out.csv"))
    q = pd.read_csv(os.path.join(QC, "QC5_summary.csv"))
    top = (q[(q.direction == "o2v") & (q.p_Inverse_variance_weighted < 0.05)]
           .nsmallest(2, "p_Inverse_variance_weighted"))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for ax, (_, r) in zip(axes, top.iterrows()):
        sub = loo[(loo["id.exposure"] == r["id.exposure"]) &
                  (loo["id.outcome"] == r["id.outcome"])]
        sub = sub.sort_values("dropped_SNP")
        x = np.arange(len(sub))
        ax.errorbar(x, sub.b_loo, yerr=1.96 * sub.se_loo, fmt="o", ms=4,
                    color="#2c7bb6", ecolor="#abd9e9", elinewidth=1.1, capsize=2)
        ax.axhline(sub.b_full.iloc[0], color="#d7191c", lw=1.3, ls="-",
                   label=f"full IVW = {sub.b_full.iloc[0]:.3f}")
        ax.axhline(0, color="k", lw=.6, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels(sub.dropped_SNP, rotation=90, fontsize=5.5)
        ax.set_xlabel("Instrument dropped")
        ax.set_ylabel("IVW estimate (95% CI)")
        ax.set_title(f"{short(r.exposure)} → {short(r.outcome)}\n"
                     f"p = {r.p_Inverse_variance_weighted:.4g}, "
                     f"n = {int(r.nsnp)} SNPs", loc="left", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
    plt.tight_layout()
    p = os.path.join(FIG, "FigR5_leave_one_out.png")
    plt.savefig(p); plt.close()
    print(" ->", os.path.basename(p))


if __name__ == "__main__":
    print("[06] building rebuttal figures ...")
    fig_threshold_audit()
    fig_instrument_strength()
    fig_pvalue_distribution()
    fig_sensitivity_forest()
    fig_leave_one_out()
    print("[06] done.")
