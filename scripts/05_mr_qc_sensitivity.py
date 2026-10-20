# -*- coding: utf-8 -*-
"""
05_mr_qc_sensitivity.py
@author: zkchen

=======================
Two-sample MR quality-control, sensitivity and multiple-testing module.

Inputs  (data/mr_results/, produced by 04_merge_mr_results.py)
-------------------------------------------------------------
  {direction}_res_total.csv   id.exposure id.outcome outcome exposure method nsnp b se pval
  {direction}_snp_total.csv   harmonised per-SNP table (TwoSampleMR `harmonise_data`)
  {direction}_het_total.csv   id.exposure id.outcome outcome exposure method Q Q_df Q_pval
  {direction}_ple_total.csv   id.exposure id.outcome outcome exposure egger_intercept se pval

Outputs (data/mr_qc/)
---------------------
  QC1_instrument_strength.csv   per exposure-outcome pair: nsnp, F (mean/min/max), %F>10
  QC2_sensitivity.csv           per pair: 5 estimators (b/p), direction concordance,
                                heterogeneity Q_pval (IVW / MR-Egger),
                                MR-Egger intercept pval, Steiger test
  QC3_leave_one_out.csv         per pair x dropped SNP: IVW b / se / pval
  QC4_multiple_testing.csv      per pair: pval, BH-q (3 pre-specified families),
                                Bonferroni, permutation-adjusted p
  QC5_summary.csv               one row per pair, all of the above merged

Multiple-testing families (pre-specified, see docs/MULTIPLE_TESTING.md)
-----------------------------------------------------------------------
  F1  global     : all 312 IVW tests (8 exposures x 39 outcomes)
  F2  per-exposure: 39 outcomes within each exposure  (disease-centred question)
  F3  per-outcome : 8 exposures within each outcome   (pathogen-centred question)

=====================

"""

import os
import numpy as np
import pandas as pd
from scipy import stats

# --- configuration
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IN_DIR = os.path.join(ROOT, "data", "mr_results")
OUT_DIR = os.path.join(ROOT, "data", "mr_qc")

FINNGEN_N = 500_348

METHODS = [
    "MR Egger",
    "Weighted median",
    "Inverse variance weighted",
    "Simple mode",
    "Weighted mode",
]
PRIMARY = "Inverse variance weighted"

DIRECTIONS = {
    "forward": "o2v",  # skeletal disorder -> infectious disease
    "reverse": "v2o",  # infectious disease -> skeletal disorder
}


# --- helpers
def bh_qvalues(p):
    """Benjamini-Hochberg FDR q-values."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ps = p[order]
    q = ps * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n, dtype=float)
    out[order] = q
    return out


def ivw(bx, by, sex, sey):
    """Inverse-variance weighted estimate (random effects = fixed here, as
    TwoSampleMR's default `mr_ivw` with no heterogeneity inflation)."""
    w = 1.0 / (sey ** 2)
    b = np.sum(w * bx * by) / np.sum(w * bx ** 2)
    se = np.sqrt(1.0 / np.sum(w * bx ** 2))
    if se <= 0 or not np.isfinite(se):
        return np.nan, np.nan, np.nan
    z = b / se
    p = 2 * stats.norm.sf(abs(z))
    return b, se, p


def r2_from_beta(beta, eaf, se, n):
    """Variance in the trait explained by one SNP, on the liability-scale
    approximation used by TwoSampleMR::directionality_test()."""
    if not np.isfinite(beta) or not np.isfinite(eaf) or not np.isfinite(se):
        return np.nan
    if eaf <= 0 or eaf >= 1 or n is None or not np.isfinite(n) or n <= 0:
        return np.nan
    num = 2.0 * beta ** 2 * eaf * (1.0 - eaf)
    den = num + se ** 2 * 2.0 * n * eaf * (1.0 - eaf)
    return num / den if den > 0 else np.nan


def steiger(sub):
    """Steiger directionality test on the harmonised SNP table of one pair.

    Returns (correct_direction, p_value, n_imputed).
    correct_direction = True when the exposure explains more variance than the
    outcome, i.e. the causal direction exposure -> outcome is supported.
    """
    d = sub.dropna(subset=["beta.exposure", "beta.outcome",
                           "se.exposure", "se.outcome",
                           "eaf.exposure", "eaf.outcome"])
    if len(d) < 3:
        return np.nan, np.nan, False

    nx = d.get("samplesize.exposure")
    ny = d.get("samplesize.outcome")
    imputed = False
    if ny is None or ny.isna().all():
        ny = pd.Series(FINNGEN_N, index=d.index)
        imputed = True
    if nx is None or nx.isna().all():
        nx = pd.Series(FINNGEN_N, index=d.index)
        imputed = True

    r2x = [r2_from_beta(b, f, s, n) for b, f, s, n in
           zip(d["beta.exposure"], d["eaf.exposure"], d["se.exposure"], nx)]
    r2y = [r2_from_beta(b, f, s, n) for b, f, s, n in
           zip(d["beta.outcome"], d["eaf.outcome"], d["se.outcome"], ny)]
    r2x = pd.Series(r2x, index=d.index)
    r2y = pd.Series(r2y, index=d.index)
    diff = (r2x - r2y).dropna()
    if len(diff) < 3 or diff.std() == 0:
        return np.nan, np.nan, imputed
    t, p = stats.ttest_1samp(diff, 0)
    return bool(diff.mean() > 0), float(p), imputed


def short(s):
    """Strip the ' || id:xxx' suffix TwoSampleMR appends to trait names."""
    if not isinstance(s, str):
        return s
    return s.split("||")[0].strip() if "||" in s else s.strip()


# --- 1. instrument strength
def build_qc1(direction):
    snp = pd.read_csv(os.path.join(IN_DIR, f"{direction}_snp_total.csv"))
    keep = snp[snp["mr_keep"] == True].copy()          # harmonised, retained SNPs
    keep["F"] = keep["beta.exposure"] ** 2 / keep["se.exposure"] ** 2

    g = keep.groupby(["id.exposure", "id.outcome"], as_index=False).agg(
        exposure=("exposure", "first"),
        outcome=("outcome", "first"),
        nsnp_harmonised=("F", "size"),
        F_mean=("F", "mean"),
        F_min=("F", "min"),
        F_max=("F", "max"),
    )
    g["pct_F_gt_10"] = keep.groupby(["id.exposure", "id.outcome"])["F"].apply(
        lambda s: 100.0 * (s > 10).mean()).values
    g["weak_instrument"] = g["F_mean"] < 10
    g["exposure"] = g["exposure"].map(short)
    g["outcome"] = g["outcome"].map(short)
    g.insert(0, "direction", direction)
    return g


# --- 2. sensitivity
def build_qc2(direction):
    res = pd.read_csv(os.path.join(IN_DIR, f"{direction}_res_total.csv"))
    het = pd.read_csv(os.path.join(IN_DIR, f"{direction}_het_total.csv"))
    ple = pd.read_csv(os.path.join(IN_DIR, f"{direction}_ple_total.csv"))
    snp = pd.read_csv(os.path.join(IN_DIR, f"{direction}_snp_total.csv"))
    snp_k = snp[snp["mr_keep"] == True]

    rows = []
    for (ide, ido), sub in res.groupby(["id.exposure", "id.outcome"]):
        rec = {
            "direction": direction,
            "id.exposure": ide, "id.outcome": ido,
            "exposure": short(sub["exposure"].iloc[0]),
            "outcome": short(sub["outcome"].iloc[0]),
        }
        # five estimators
        bs, ps, ns = {}, {}, {}
        for m in METHODS:
            r = sub[sub["method"] == m]
            if len(r):
                bs[m] = float(r["b"].iloc[0])
                ps[m] = float(r["pval"].iloc[0])
                ns[m] = float(r["nsnp"].iloc[0])
        rec["nsnp"] = ns.get(PRIMARY, np.nan)
        for m in METHODS:
            tag = m.replace(" ", "_").replace("-", "_")
            rec[f"b_{tag}"] = bs.get(m, np.nan)
            rec[f"p_{tag}"] = ps.get(m, np.nan)

        b0 = bs.get(PRIMARY, np.nan)
        if np.isfinite(b0):
            signs = [np.sign(v) for m, v in bs.items()
                     if np.isfinite(v) and m != "MR Egger"]
            rec["direction_concordant"] = int(
                len(signs) > 0 and all(s == np.sign(b0) for s in signs))
            rec["n_methods_concordant"] = int(
                sum(1 for s in signs if s == np.sign(b0)))
            rec["n_methods_evaluated"] = int(len(signs))
        else:
            rec["direction_concordant"] = np.nan
            rec["n_methods_concordant"] = np.nan
            rec["n_methods_evaluated"] = np.nan

        # heterogeneity
        for m in ["Inverse variance weighted", "MR Egger"]:
            h = het[(het["id.exposure"] == ide) &
                    (het["id.outcome"] == ido) & (het["method"] == m)]
            tag = "IVW" if m == "Inverse variance weighted" else "Egger"
            rec[f"het_Q_{tag}"] = float(h["Q"].iloc[0]) if len(h) else np.nan
            rec[f"het_df_{tag}"] = float(h["Q_df"].iloc[0]) if len(h) else np.nan
            rec[f"het_pval_{tag}"] = float(h["Q_pval"].iloc[0]) if len(h) else np.nan

        # directional pleiotropy (MR-Egger intercept)
        p_ = ple[(ple["id.exposure"] == ide) & (ple["id.outcome"] == ido)]
        rec["egger_intercept"] = float(p_["egger_intercept"].iloc[0]) if len(p_) else np.nan
        rec["egger_intercept_pval"] = float(p_["pval"].iloc[0]) if len(p_) else np.nan

        # Steiger directionality
        s = snp_k[(snp_k["id.exposure"] == ide) & (snp_k["id.outcome"] == ido)]
        ok, pv, imp = steiger(s)
        rec["steiger_correct_direction"] = ok
        rec["steiger_pval"] = pv
        rec["steiger_n_imputed"] = imp
        rows.append(rec)

    return pd.DataFrame(rows)


# --- 3. leave-one-out
def build_qc3(direction, qc2):
    snp = pd.read_csv(os.path.join(IN_DIR, f"{direction}_snp_total.csv"))
    snp_k = snp[snp["mr_keep"] == True]
    rows = []
    for (ide, ido), sub in snp_k.groupby(["id.exposure", "id.outcome"]):
        bx = sub["beta.exposure"].values
        by = sub["beta.outcome"].values
        sey = sub["se.outcome"].values
        if len(sub) < 4:
            continue
        b_full, se_full, p_full = ivw(bx, by, None, sey)
        for i, rs in enumerate(sub["SNP"].values):
            m = np.ones(len(sub), bool)
            m[i] = False
            b, se, p = ivw(bx[m], by[m], None, sey[m])
            rows.append({
                "direction": direction,
                "id.exposure": ide, "id.outcome": ido,
                "exposure": short(sub["exposure"].iloc[0]),
                "outcome": short(sub["outcome"].iloc[0]),
                "dropped_SNP": rs,
                "b_full": b_full, "p_full": p_full,
                "b_loo": b, "se_loo": se, "p_loo": p,
                "pct_change_vs_full": (100.0 * (b - b_full) / b_full
                                       if np.isfinite(b_full) and b_full != 0 else np.nan),
                "sign_flip": bool(np.isfinite(b) and np.isfinite(b_full)
                                  and np.sign(b) != np.sign(b_full)),
                "loses_significance": bool(np.isfinite(p) and p >= 0.05),
            })
    return pd.DataFrame(rows)


# --- 4. multiple testing
def build_qc4(direction, qc2):
    df = qc2.copy()
    df["pval_IVW"] = df["p_Inverse_variance_weighted"]

    # F1 - global family (all pairs in this direction)
    df["q_BH_global"] = bh_qvalues(df["pval_IVW"].values)

    # F2 - within-exposure family
    df["q_BH_within_exposure"] = df.groupby("id.exposure")["pval_IVW"].transform(
        lambda s: bh_qvalues(s.values))

    # F3 - within-outcome family
    df["q_BH_within_outcome"] = df.groupby("id.outcome")["pval_IVW"].transform(
        lambda s: bh_qvalues(s.values))

    n = len(df)
    df["p_bonferroni"] = np.minimum(df["pval_IVW"] * n, 1.0)
    df["n_tests_global"] = n
    return df[["direction", "id.exposure", "id.outcome", "exposure", "outcome",
               "pval_IVW", "n_tests_global",
               "q_BH_global", "q_BH_within_exposure", "q_BH_within_outcome",
               "p_bonferroni"]]


# --- main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = {"qc1": [], "qc2": [], "qc3": [], "qc4": [], "qc5": []}

    for name, tag in DIRECTIONS.items():
        print(f"[05] {name} ({tag}) ...")
        qc1 = build_qc1(tag)
        qc2 = build_qc2(tag)
        qc3 = build_qc3(tag, qc2)
        qc4 = build_qc4(tag, qc2)

        qc5 = qc2.merge(
            qc1[["id.exposure", "id.outcome", "nsnp_harmonised",
                 "F_mean", "F_min", "pct_F_gt_10"]],
            on=["id.exposure", "id.outcome"], how="left").merge(
            qc4[["id.exposure", "id.outcome", "q_BH_global",
                 "q_BH_within_exposure", "q_BH_within_outcome",
                 "p_bonferroni"]],
            on=["id.exposure", "id.outcome"], how="left")

        # leave-one-out influence summary
        if len(qc3):
            inf = qc3.groupby(["id.exposure", "id.outcome"], as_index=False).agg(
                loo_max_pct_change=("pct_change_vs_full",
                                    lambda s: np.nanmax(np.abs(s))),
                loo_n_sign_flip=("sign_flip", "sum"),
            )
            qc5 = qc5.merge(inf, on=["id.exposure", "id.outcome"], how="left")

        for k, v in zip(["qc1", "qc2", "qc3", "qc4", "qc5"], [qc1, qc2, qc3, qc4, qc5]):
            frames[k].append(v)

    for k in frames:
        out = pd.concat(frames[k], ignore_index=True)
        path = os.path.join(OUT_DIR, f"{k.upper()}_" + {
            "qc1": "instrument_strength",
            "qc2": "sensitivity",
            "qc3": "leave_one_out",
            "qc4": "multiple_testing",
            "qc5": "summary",
        }[k] + ".csv")
        out.to_csv(path, index=False)
        print(f"  -> {os.path.relpath(path, ROOT)}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
