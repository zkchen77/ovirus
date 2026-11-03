# -*- coding: utf-8 -*-
"""
@author: zkchen
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "data", "ml_results")
FIG = os.path.join(ROOT, "data", "figures")
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)

SEED = 77
N_NULL_DRAWS = 200_000
NULL_CHUNK = 20_000


def load_clean():
    df = pd.read_csv(os.path.join(DATA, "original_drugs.csv"),
                     encoding="utf-8-sig")
    df["Label"] = df["Label"].str.strip()
    valid = {"Risk", "Dual-effect"}
    bad = set(df["Label"].unique()) - valid
    if bad:
        raise ValueError(f"Unexpected Label values: {bad}")

    n_raw = len(df)
    dup_mask = df.duplicated(subset=["SMILES"], keep=False)
    n_dup_rows = int(dup_mask.sum())

    if n_dup_rows:
        grp = df.loc[dup_mask].groupby("SMILES")["Label"].nunique()
        conflict_smiles = set(grp[grp > 1].index)
        n_conflict = len(conflict_smiles)
        print(f"[load] {n_raw} rows, {n_dup_rows} share a SMILES "
              f"({df.duplicated(subset=['SMILES']).sum()} redundant), "
              f"{n_conflict} SMILES with conflicting labels")
        if n_conflict:
            df = df[~df["SMILES"].isin(conflict_smiles)]
            print(f"[load] DROPPING {n_conflict} conflicting SMILES "
                  f"({n_raw - len(df)} rows)")

    df = df.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
    print(f"[load] after cleanup: n = {len(df)}  "
          f"(Risk={(df['Label']=='Risk').sum()}, "
          f"Dual={(df['Label']=='Dual-effect').sum()})")
    return df, df["SMILES"].tolist(), df["Label"].tolist()


def balance_indices(labels, seed=SEED):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    pos = np.where(labels == "Risk")[0]
    neg = np.where(labels == "Dual-effect")[0]
    target = max(len(pos), len(neg))
    if len(pos) < target:
        pos = rng.choice(pos, size=target, replace=True)
    else:
        neg = rng.choice(neg, size=target, replace=True)
    idx = np.concatenate([pos, neg])
    rng.shuffle(idx)
    return idx


def null_auc_distribution(n_pos_test, n_neg_test, n_draws=N_NULL_DRAWS,
                          chunk=NULL_CHUNK, seed=SEED):
    rng = np.random.default_rng(seed)
    n_total = n_pos_test + n_neg_test
    y = np.array([1] * n_pos_test + [0] * n_neg_test)
    n_pos = y.sum(); n_neg = len(y) - n_pos
    aucs = np.empty(n_draws, dtype=np.float64)
    done = 0
    while done < n_draws:
        b = min(chunk, n_draws - done)
        a = rng.random((b, n_total))
        order = a.argsort(axis=1)
        ranks = order.argsort(axis=1) + 1
        aucs[done:done + b] = (
            ranks[:, y == 1].sum(axis=1) - n_pos * (n_pos + 1) / 2
        ) / (n_pos * n_neg)
        done += b
    return aucs


def load_published_auc(path):
    if not os.path.exists(path):
        return None, 0
    x = pd.read_excel(path)
    # first column holds feature names
    first_col = x.columns[0]
    x = x.set_index(first_col)
    # drop summary row/col if present
    x = x.drop(index=[i for i in x.index if str(i).strip().upper() == "AVERAGE"],
               errors="ignore")
    x = x.drop(columns=[c for c in x.columns if str(c).strip().upper() == "AVERAGE"],
               errors="ignore")
    auc = x.values.astype(float).flatten()
    auc = auc[~np.isnan(auc)]
    return auc, len(auc)


def main():
    df, smiles, labels = load_clean()
    n_clean = len(df)
    raw_counts = df["Label"].value_counts().to_dict()

    idx = balance_indices(labels)
    y_bal = np.array([1 if labels[i] == "Risk" else 0 for i in idx])
    n_bal = len(y_bal)

    _, _, _, y_val = train_test_split(
        np.zeros((n_bal, 1)), y_bal, test_size=0.2,
        random_state=SEED, stratify=y_bal)
    n_pos_test, n_neg_test = int(y_val.sum()), int((y_val == 0).sum())
    n_auc_values = n_pos_test * n_neg_test + 1
    p_chance = 1.0 / (n_pos_test * n_neg_test)

    smiles_bal = np.array([smiles[i] for i in idx])
    tr, va = train_test_split(np.arange(n_bal), test_size=0.2,
                              random_state=SEED, stratify=y_bal)
    train_set = set(smiles_bal[tr].tolist())
    leak = float(np.mean(np.isin(smiles_bal[va], list(train_set))))

    leak_fixed = 0.0

    # --- null AUC
    auc_null = null_auc_distribution(n_pos_test, n_neg_test)
    null_p95 = float(np.percentile(auc_null, 95))

    # --- published CV-AUCs
    p_xlsx = os.path.join(ROOT, "data", "ml_analyse", "total_cv_auc.xlsx")
    auc_pub, n_models = load_published_auc(p_xlsx)
    print(f"[xlsx] {p_xlsx} -> {n_models} cells"
          if auc_pub is not None else f"[xlsx] {p_xlsx} not found")

    rows = [
        {"item": "compounds in original_drugs.csv (after cleanup)", "value": n_clean,
         "note": f"raw labels = {raw_counts}"},
        {"item": "balanced sample size (leaky design)", "value": n_bal,
         "note": "minority class oversampled to match the majority"},
        {"item": "validation compounds (leaky design)", "value": len(y_val),
         "note": f"{n_pos_test} risk + {n_neg_test} dual-benefit"},
        {"item": "possible AUC values", "value": n_auc_values,
         "note": "AUC = k / (n_pos_test * n_neg_test)"},
        {"item": "P(AUC = 1.0) under a random scorer",
         "value": round(p_chance, 6),
         "note": "1 / (n_pos_test * n_neg_test)"},
        {"item": "expected # of AUC=1.0 cells by chance",
         "value": round(n_models * p_chance, 6),
         "note": f"out of {n_models} cells"},
        {"item": "observed # of AUC=1.0 cells",
         "value": int((auc_pub == 1).sum()) if auc_pub is not None else None,
         "note": "from total_cv_auc.xlsx"},
        {"item": "observed max CV-AUC",
         "value": round(float(auc_pub.max()), 4) if auc_pub is not None else None,
         "note": "from total_cv_auc.xlsx"},
        {"item": "null 95th percentile AUC",
         "value": round(null_p95, 4),
         "note": "random scorer, same fold size"},
        {"item": "observed mean CV-AUC",
         "value": round(float(auc_pub.mean()), 4) if auc_pub is not None else None,
         "note": "from total_cv_auc.xlsx"},
        {"item": "fraction of validation compounds duplicated in train (leaky)",
         "value": round(leak, 4),
         "note": "pre-split oversampling"},
        {"item": "fraction of validation compounds duplicated in train (fixed)",
         "value": leak_fixed,
         "note": "fold-internal undersampling; no pre-split duplication"},
    ]
    res = pd.DataFrame(rows)
    res["value"] = res["value"].astype(object)

    # binomial test only meaningful when at least one perfect cell exists
    if auc_pub is not None and (auc_pub == 1).any():
        from scipy import stats
        k_ones = int((auc_pub == 1).sum())
        bt = stats.binomtest(k_ones, len(auc_pub), p_chance)
        res.loc[res.item == "observed # of AUC=1.0 cells", "note"] += \
            f"; binomial p = {bt.pvalue:.2e}"
    else:
        res.loc[res.item == "observed # of AUC=1.0 cells", "note"] += \
            "; no perfect cell, binomial test skipped"

    res.to_csv(os.path.join(OUT, "validation_audit.csv"), index=False)
    print(res.to_string(index=False))

    # --- figure
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))

    # Panel A: null density vs published AUC density
    ax = axes[0]
    ax.hist(auc_null, bins=50, density=True, color="#b8d4ea",
            edgecolor="#2c7bb6", alpha=.85, label="random scorer (null)")
    if auc_pub is not None:
        ax.hist(auc_pub, bins=20, density=True, color="#fdae61",
                edgecolor="#e08214", alpha=.75,
                label=f"ovirus_stacked ({n_models} models)")
        ax.axvline(auc_pub.mean(), color="#e08214", ls="--", lw=1,
                   label=f"mean = {auc_pub.mean():.4f}")
    ax.axvline(null_p95, color="#2c7bb6", ls=":", lw=1,
               label=f"null 95th pct = {null_p95:.4f}")
    ax.set_xlabel("AUC on the single 80/20 validation split")
    ax.set_ylabel("Density")
    ax.set_title(f"A  Null vs ovirus_stacked AUC distribution\n"
                 f"(validation fold = {n_pos_test} risk + {n_neg_test} dual-effect)",
                 loc="left", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    if auc_pub is not None:
        bars = [
            ("null 95th\npercentile", null_p95, "#2c7bb6"),
            ("ovirus_stacked\nmean",   float(auc_pub.mean()), "#fdae61"),
            ("ovirus_stacked\nmax",    float(auc_pub.max()),  "#e08214"),
        ]
    else:
        bars = [("null 95th\npercentile", null_p95, "#2c7bb6")]
    labels_b = [b[0] for b in bars]
    vals_b   = [b[1] for b in bars]
    cols_b   = [b[2] for b in bars]
    ax.bar(labels_b, vals_b, color=cols_b, width=.55, alpha=.9)
    for i, v in enumerate(vals_b):
        ax.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=9)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("B  Observed CV-AUC vs null baseline", loc="left", fontsize=9)

    plt.tight_layout()
    fp = os.path.join(FIG, "FigR2.1_validation_audit.png")
    plt.savefig(fp); plt.close()
    print("\n ->", os.path.basename(fp))


if __name__ == "__main__":
    main()