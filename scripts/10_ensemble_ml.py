# -*- coding: utf-8 -*-
"""
@author: zkchen
================================================================
Repeated stratified CV + permutation test + batch ROC rendering
for the drug risk-classification pipeline.

Dataset (reference run: 3678 raw rows -> 3102 after cleanup)
------------------------------------------------------------
  raw rows           : 3678
  duplicate SMILES   : 1055 rows share a SMILES (576 redundant)
  label conflicts    : 255 SMILES carry BOTH 'Risk' and 'Dual-effect'
  after dedup        : n = 2847  (Risk=1015, Dual=1832)  -> pos:neg ~ 1:1.8
  NOTE: Risk is the MINORITY class.

Data-quality policy
-------------------
* SMILES with conflicting labels arise because the same compound was annotated 
against different target genes.  With 3000+ compounds we can afford to DROP 
the conflicting SMILES.
﻿
* class balance: pos:neg ~ 1:1.6 after dedup.  Training-time balance is 
FOLD-INTERNAL UNDERSAMPLING of the majority class to the minority count 
(no synthetic samples, no duplicates, no pre-split sampling).
Undersampling is done inside each training fold, so every Dual-effect compound 
is still seen by some folds and no information is permanently discarded.

Design
-----------------------------------------------
* fold-internal scaling, fold-internal balancing (no leak)
* N_SPLITS=5, N_REPEATS=3  repeated stratified CV
* multi-metric per fold: AUC / PR-AUC / balanced acc / MCC
* permutation test on shuffled labels, parallel over cells

Cost-control / caching
----------------------
  * mode_cv writes cv_oof.npz + cv_oof_y.npy + cv_repeated_stratified.csv
    + cv_meta.json, and refits the best (feature, model) on the FULL data,
    saved as best_model.joblib.
  * mode_plot reads the OOF cache instead of refitting.
  * mode_perm uses n_perm=20 by default, runs cells in parallel with joblib,
    and *always* tests every AUC==1.0 cell while restricting the rest to the
    top-3 models by mean AUC.

Model selection note
--------------------
best_model.joblib is chosen by maximum CV mean AUC, NOT by permutation p-value.

NRDL prediction caveat
----------------------
  flagged_high_risk uses vote_fraction > 0.5 over 56 balanced models.
  Because training folds are class-balanced, the ensemble is mildly biased
  toward predicting 'Risk'.  Treat flags as a screening signal, not as a
  clinical conclusion; external validation is required.

CLI
---
  python 10_ensemble_ml.py                     # default: plot from cache
  python 10_ensemble_ml.py --mode cv           # train + cache (skip if cached)
  python 10_ensemble_ml.py --mode cv --force   # retrain
  python 10_ensemble_ml.py --mode all          # diagnose+cv+perm+plot+predict
  python 10_ensemble_ml.py --mode perm --n-perm 20   # exact p
  python 10_ensemble_ml.py --mode predict

Cached artefacts (under data/ml_results/)
-----------------------------------------
  cv_repeated_stratified.csv     per-cell metrics
  cv_oof.npz                     OOF probability matrices, key '<feat>__<model>'
  cv_oof_y.npy                   label vector aligned to OOF columns
  cv_meta.json                   fold config, n, class counts, timestamp
  best_model.joblib              best cell refit on all data
  cv_permutation_null.csv        permutation p-values
  nrdl_*                         NRDL predictions
  figures_roc/                   png + svg figures
"""

import os
import json
import time
import argparse
import warnings
import functools
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.svm")
warnings.filterwarnings("ignore", message=".*probability.*deprecated.*")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT  = os.path.join(ROOT, "data", "ml_results")
os.makedirs(OUT, exist_ok=True)

SEED      = 77
N_SPLITS  = 5
N_REPEATS = 3
EPS       = 1e-9

# cached file paths
P_CV_CSV   = os.path.join(OUT, "cv_repeated_stratified.csv")
P_OOF      = os.path.join(OUT, "cv_oof.npz")
P_OOF_Y    = os.path.join(OUT, "cv_oof_y.npy")
P_META     = os.path.join(OUT, "cv_meta.json")
P_BEST     = os.path.join(OUT, "best_model.joblib")
P_PERM     = os.path.join(OUT, "cv_permutation_null.csv")

FIGDIR = Path(OUT) / "figures_roc"
FIGDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# cache helpers
# ============================================================
def _cache_ok():
    return all(os.path.exists(p) for p in (P_CV_CSV, P_OOF, P_OOF_Y, P_META))


def _load_meta():
    if not os.path.exists(P_META):
        return None
    with open(P_META, encoding="utf-8") as f:
        return json.load(f)


def _save_meta(meta):
    with open(P_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _cache_summary():
    m = _load_meta()
    if m is None:
        print("[cache] no meta; cache empty")
        return
    print(f"[cache] cv from {m.get('timestamp','?')}  "
          f"n={m.get('n')}  Risk={m.get('n_pos')}  "
          f"folds={m.get('n_splits')}x{m.get('n_repeats')}  "
          f"cells={m.get('n_cells')}")


# ============================================================
# 0. diagnose
# ============================================================
def diagnose_folds(y, n_splits=N_SPLITS, n_repeats=N_REPEATS, seed=SEED):
    from sklearn.model_selection import RepeatedStratifiedKFold
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=seed)
    rows = []
    for i, (tr, te) in enumerate(rskf.split(np.zeros(len(y)), y)):
        rows.append({"fold": i,
                     "n_train": len(tr), "n_pos_train": int(y[tr].sum()),
                     "n_test":  len(te), "n_pos_test":  int(y[te].sum())})
    df = pd.DataFrame(rows)
    print("\n=== Fold 诊断 (summary) ===")
    print(df.describe().loc[["min", "mean", "max"]].round(1).to_string())
    print(f"\ntest 正类=0: {(df.n_pos_test == 0).sum()}/{len(df)}  |  "
          f"=1: {(df.n_pos_test == 1).sum()}/{len(df)}  |  "
          f">=2: {(df.n_pos_test >= 2).sum()}/{len(df)}")
    return df


# ============================================================
# 1. features
# ============================================================
@functools.lru_cache(maxsize=1)
def build_features():
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, Descriptors
    from rdkit.Chem import rdFingerprintGenerator as rfg

    morgan_g = rfg.GetMorganGenerator(radius=2, fpSize=256)
    topo_g   = rfg.GetTopologicalTorsionGenerator(fpSize=256)
    atom_g   = rfg.GetAtomPairGenerator(fpSize=256)
    rdk_g    = rfg.GetRDKitFPGenerator(fpSize=256)

    def _maccs(mol):   return np.array(MACCSkeys.GenMACCSKeys(mol), dtype=float)
    def _morgan(mol):  return np.array(morgan_g.GetFingerprint(mol), dtype=float)
    def _topo(mol):    return np.array(topo_g.GetFingerprint(mol),   dtype=float)
    def _atom(mol):    return np.array(atom_g.GetFingerprint(mol),   dtype=float)
    def _rdk(mol):     return np.array(rdk_g.GetFingerprint(mol),    dtype=float)
    def _pattern(mol): return np.array(Chem.PatternFingerprint(mol, fpSize=256), dtype=float)

    def _desc(mol):
        return np.array([Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
                         Descriptors.TPSA(mol), Descriptors.NumHDonors(mol),
                         Descriptors.NumHAcceptors(mol),
                         Descriptors.NumRotatableBonds(mol)], dtype=float)

    return {
        "maccs":       (_maccs,   167, True),
        "morgan":      (_morgan,  256, True),
        "topofp":      (_topo,    256, True),
        "atomfp":      (_atom,    256, True),
        "rdkfp":       (_rdk,     256, True),
        "pattern":     (_pattern, 256, True),
        "descriptors": (_desc,      6, False),
    }


def featurise(smiles_list, feature_name):
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    fn, dim, _ = build_features()[feature_name]
    X  = np.zeros((len(smiles_list), dim))
    ok = np.ones(len(smiles_list), dtype=bool)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            ok[i] = False
            continue
        v = fn(mol)
        X[i, :len(v)] = v
    return X, ok


# ============================================================
# 2. models  (hyperparameters trimmed for 3000-sample cost)
# ============================================================
def make_models():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=150, random_state=SEED, n_jobs=-1,
            class_weight="balanced"),
        "LogisticRegression": Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000,
                                       class_weight="balanced",
                                       random_state=SEED)),
        ]),
        # SVC without probability=True: AUC only needs a ranking score,
        # and mode_predict converts decision_function via a logistic.
        "SVM": Pipeline([
            ("sc", StandardScaler()),
            ("clf", SVC(probability=False, class_weight="balanced",
                        random_state=SEED)),
        ]),
        "BernoulliNB": BernoulliNB(alpha=1.0),
        "MLP": Pipeline([
            ("sc", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32),
                                  max_iter=1000, early_stopping=True,
                                  validation_fraction=0.1,
                                  random_state=SEED)),
        ]),
        "KNN": Pipeline([
            ("sc", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5,
                                         weights="distance")),
        ]),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            tree_method="hist",
            eval_metric="logloss", random_state=SEED, n_jobs=-1)
    except ImportError:
        print("  ! xgboost unavailable")
    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = CatBoostClassifier(
            iterations=150, learning_rate=0.05, depth=4,
            random_seed=SEED, verbose=0, thread_count=-1)
    except ImportError:
        print("  ! catboost unavailable")
    return models


NEEDS_SCALING = {"LogisticRegression", "SVM", "MLP", "KNN"}


# ============================================================
# 3. fold-internal balancing  (undersample majority to minority)
# ============================================================
def balance_inside_fold(Xtr, ytr, rng):
    """Undersample the majority class to the minority count, inside the fold.

    No synthetic samples, no duplicates.  Done *after* the split so no
    validation compound is ever resampled into training.
    """
    cls, cnt = np.unique(ytr, return_counts=True)
    if len(cls) < 2:
        return Xtr, ytr
    target = cnt.min()
    idx = []
    for c, n in zip(cls, cnt):
        pool = np.where(ytr == c)[0]
        if n > target:
            idx.append(rng.choice(pool, size=target, replace=False))
        else:
            idx.append(pool)
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return Xtr[idx], ytr[idx]


# ============================================================
# 4. metrics
# ============================================================
def _safe_auc(y, p):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)


def _safe_prauc(y, p):
    from sklearn.metrics import average_precision_score
    if len(np.unique(y)) < 2:
        return np.nan
    return average_precision_score(y, p)


def _safe_mcc(y, yhat):
    from sklearn.metrics import matthews_corrcoef
    if len(np.unique(y)) < 2:
        return np.nan
    return matthews_corrcoef(y, yhat)


def _predict_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def repeated_stratified_cv(X, y, model_fn, model_name,
                           n_splits=N_SPLITS, n_repeats=N_REPEATS,
                           seed=SEED, balance=True,
                           scale_sparse=False,
                           return_oof=False):
    """Return dict: auc, prauc, bacc, mcc (lists) and, if return_oof,
    'oof' = (n_repeats, n_samples) matrix of out-of-fold probabilities."""
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=seed)
    rng = np.random.default_rng(seed)
    use_scaler = (model_name in NEEDS_SCALING) and (scale_sparse or X.shape[1] <= 20)

    res = {"auc": [], "prauc": [], "bacc": [], "mcc": []}
    oof_all = [] if return_oof else None

    for tr, te in rskf.split(X, y):
        Xtr, ytr = X[tr], y[tr]
        Xte, yte = X[te], y[te]

        if use_scaler:
            sc = StandardScaler().fit(Xtr)
            Xtr = sc.transform(Xtr)
            Xte = sc.transform(Xte)

        if balance:
            Xtr, ytr = balance_inside_fold(Xtr, ytr, rng)

        if len(np.unique(ytr)) < 2:
            continue

        m = clone(model_fn)
        try:
            m.fit(Xtr, ytr)
            p = _predict_scores(m, Xte)
        except Exception:
            continue

        yhat = (np.asarray(p) >= 0.5).astype(int) if hasattr(m, "predict_proba") \
               else (np.asarray(p) >= 0).astype(int)

        res["auc"].append(_safe_auc(yte, p))
        res["prauc"].append(_safe_prauc(yte, p))
        res["bacc"].append(balanced_accuracy_score(yte, yhat)
                           if len(np.unique(yte)) > 1 else np.nan)
        res["mcc"].append(_safe_mcc(yte, yhat))

        if return_oof:
            oof = np.full(len(y), np.nan)
            oof[te] = p
            oof_all.append(oof)

    for k in res:
        res[k] = [v for v in res[k] if not (isinstance(v, float) and np.isnan(v))]
    if return_oof:
        res["oof"] = np.vstack(oof_all) if oof_all else None
    return res


# ============================================================
# 5. data loading  (drop conflicting SMILES)
# ============================================================
def load_training(drop_conflicts=True):
    """Load, deduplicate, and (by default) drop SMILES with conflicting labels.

    Conflicting labels arise because the same compound was annotated against
    different target genes.  With 3000+ compounds we drop them rather than
    keep an arbitrary first occurrence.
    """
    df = pd.read_csv(os.path.join(DATA, "original_drugs.csv"),
                     encoding="utf-8-sig")
    df["Label"] = df["Label"].str.strip()
    valid = {"Risk", "Dual-effect"}
    bad = set(df["Label"].unique()) - valid
    if bad:
        raise ValueError(f"Unexpected Label values: {bad}. Expected subset of {valid}.")

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
        if n_conflict and drop_conflicts:
            n_rows_dropped = int(df["SMILES"].isin(conflict_smiles).sum())
            print(f"[load] DROPPING {n_conflict} conflicting SMILES "
                  f"({n_rows_dropped} rows) -- same compound, different target genes")
            df = df[~df["SMILES"].isin(conflict_smiles)]

    df = df.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
    y = (df["Label"] == "Risk").astype(int).values
    ratio = int((1 - y).sum()) / max(int(y.sum()), 1)
    print(f"[load] after dedup: n = {len(df)}  "
          f"(Risk={int(y.sum())}, Dual={int((1 - y).sum())}, "
          f"ratio 1:{ratio:.2f})")
    return df["SMILES"].tolist(), y, df["Drug Name"].tolist()


# ============================================================
# 6. diagnose
# ============================================================
def mode_diagnose():
    _, y, _ = load_training()
    diagnose_folds(y)


# ============================================================
# 7. CV (writes cache; reads cache unless force)
# ============================================================
def mode_cv(force=False):
    if not force and _cache_ok():
        _cache_summary()
        print("[cv] cache hit -> skip. Use --force to retrain.")
        return pd.read_csv(P_CV_CSV)

    print("[cv] cache miss or --force -> training ...")
    t0 = time.time()
    smiles, y, _ = load_training()
    feats = build_features()
    rows, oof_store = [], {}

    for fname, (_, _, is_sparse) in feats.items():
        X, ok = featurise(smiles, fname)
        X, yy = X[ok], y[ok]
        for mname, mfn in make_models().items():
            r = repeated_stratified_cv(X, yy, mfn, mname,
                                       scale_sparse=not is_sparse,
                                       balance=True,
                                       return_oof=True)
            if not r["auc"] and not r["mcc"]:
                continue
            rows.append({
                "feature": fname, "model": mname,
                "n_folds": len(r["auc"]),
                "auc_mean":  float(np.nanmean(r["auc"]))  if r["auc"]  else np.nan,
                "auc_median":float(np.nanmedian(r["auc"]))if r["auc"]  else np.nan,
                "prauc_mean":float(np.nanmean(r["prauc"]))if r["prauc"]else np.nan,
                "bacc_mean": float(np.nanmean(r["bacc"])) if r["bacc"] else np.nan,
                "mcc_mean":  float(np.nanmean(r["mcc"]))  if r["mcc"]  else np.nan,
                "mcc_sd":    float(np.nanstd(r["mcc"]))   if r["mcc"]  else np.nan,
            })
            if r.get("oof") is not None:
                oof_store[f"{fname}__{mname}"] = r["oof"]
            print(f"  [cv] {fname:12s} {mname:18s} "
                  f"AUC={rows[-1]['auc_mean']:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(P_CV_CSV, index=False)
    np.savez_compressed(P_OOF, **oof_store)
    np.save(P_OOF_Y, y)
    _save_meta({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n": int(len(y)), "n_pos": int(y.sum()),
        "n_splits": N_SPLITS, "n_repeats": N_REPEATS,
        "n_cells": int(len(res)), "seed": SEED,
        "balance": "fold-internal undersample",
        "elapsed_sec": round(time.time() - t0, 1),
    })

    # refit the best cell on all data, dump it
    best = res.sort_values("auc_mean", ascending=False).iloc[0]
    print(f"\n[cv] best cell: {best.feature} | {best.model}  AUC={best.auc_mean:.3f}")
    from joblib import dump
    Xb, okb = featurise(smiles, best.feature)
    Xb, yb = Xb[okb], y[okb]
    from sklearn.preprocessing import StandardScaler
    if best.model in NEEDS_SCALING and Xb.shape[1] > 20:
        scaler = StandardScaler().fit(Xb)
        Xb = scaler.transform(Xb)
    else:
        scaler = None
    rng = np.random.default_rng(SEED)
    Xb_o, yb_o = balance_inside_fold(Xb, yb, rng)
    m_best = clone(make_models()[best.model])
    m_best.fit(Xb_o, yb_o)
    dump({"model": m_best, "scaler": scaler, "feature": best.feature,
          "model_name": best.model, "auc_mean": float(best.auc_mean)},
         P_BEST)
    print(f"[cv] best model -> {os.path.basename(P_BEST)}")

    print("\n=== 按特征汇总 ===")
    print(res.groupby("feature")[["auc_mean", "prauc_mean", "mcc_mean"]]
             .mean().round(3).to_string())
    print("\n=== 按模型汇总 ===")
    print(res.groupby("model")[["auc_mean", "prauc_mean", "mcc_mean"]]
             .mean().round(3).to_string())
    print(f"\n[cv] done in {time.time()-t0:.1f}s  "
          f"cells={len(res)}/{len(feats)*len(make_models())}")
    return res


# ============================================================
# 8. permutation (parallel, AUC=1.0 always tested; others top-3 models)
# ============================================================
def _perm_one_cell(fname, mname, X, yy, mfn, is_sparse,
                   n_perm, seed, n_splits_null=2):
    rng = np.random.default_rng(seed)
    obs = repeated_stratified_cv(X, yy, mfn, mname,
                                 scale_sparse=not is_sparse, balance=True)
    if not obs["auc"]:
        return None
    obs_auc = float(np.mean(obs["auc"]))
    obs_mcc = float(np.mean(obs["mcc"])) if obs["mcc"] else np.nan

    null_auc, null_mcc = [], []
    for _ in range(n_perm):
        yp = rng.permutation(yy)
        r = repeated_stratified_cv(X, yp, mfn, mname,
                                   n_splits=n_splits_null, n_repeats=1,
                                   scale_sparse=not is_sparse, balance=True)
        null_auc += r["auc"]; null_mcc += r["mcc"]
    if not null_auc:
        return None
    p_auc = (1 + sum(1 for v in null_auc if v >= obs_auc)) / (1 + len(null_auc))
    p_mcc = ((1 + sum(1 for v in null_mcc if v >= obs_mcc))
             / (1 + len(null_mcc))
             if null_mcc and not np.isnan(obs_mcc) else np.nan)
    return {
        "feature": fname, "model": mname,
        "auc_obs": obs_auc,
        "auc_null_mean": float(np.mean(null_auc)),
        "auc_null_p95":  float(np.percentile(null_auc, 95)),
        "perm_p_auc": p_auc,
        "mcc_obs": obs_mcc,
        "mcc_null_mean": float(np.mean(null_mcc)) if null_mcc else np.nan,
        "perm_p_mcc": p_mcc,
    }


def mode_perm(n_perm=20, force=False):
    """Permutation test.

    Selection rule:
      * every cell with AUC >= 1-EPS is ALWAYS tested;
      * remaining cells: only the top-3 models by mean AUC.
    Parallelised over cells with joblib; each cell's permutations are serial.
    """
    if not _cache_ok():
        print("[perm] no CV cache -> run mode_cv first"); return None
    if os.path.exists(P_PERM) and not force:
        print(f"[perm] cache hit -> {os.path.basename(P_PERM)}. "
              f"Use --force to rerun.")
        return pd.read_csv(P_PERM)

    from joblib import Parallel, delayed

    cv = pd.read_csv(P_CV_CSV)
    perfect = cv[cv["auc_mean"] >= 1.0 - EPS]
    top_models = (cv.groupby("model")["auc_mean"].mean()
                    .sort_values(ascending=False).head(3).index.tolist())
    rest = cv[(cv["auc_mean"] < 1.0 - EPS) & (cv["model"].isin(top_models))]
    cells = pd.concat([perfect, rest]).drop_duplicates(["feature", "model"])
    print(f"[perm] testing {len(cells)} cells: "
          f"{len(perfect)} perfect + {len(rest)} top-3-model "
          f"({top_models})  n_perm={n_perm}")

    smiles, y, _ = load_training()
    feats = build_features()
    all_models = make_models()

    feat_data = {}
    for fname in cells.feature.unique():
        X, ok = featurise(smiles, fname)
        feat_data[fname] = (X[ok], y[ok], feats[fname][2])

    jobs = []
    for i, row in enumerate(cells.itertuples(index=False)):
        X, yy, is_sparse = feat_data[row.feature]
        jobs.append(delayed(_perm_one_cell)(
            row.feature, row.model, X, yy, all_models[row.model],
            is_sparse, n_perm, SEED + i))

    t0 = time.time()
    results = Parallel(n_jobs=-1, verbose=5, backend="loky")(jobs)
    results = [r for r in results if r]
    res = pd.DataFrame(results)
    res.to_csv(P_PERM, index=False)
    print(f"\n[perm] done in {time.time()-t0:.1f}s, "
          f"{len(res)} cells -> {os.path.basename(P_PERM)}")
    print(res.round(3).to_string(index=False))
    return res


# ============================================================
# 9. holdout (sanity only)
# ============================================================
def mode_holdout():
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    smiles, y, _ = load_training()
    feats = build_features()
    rows = []
    rng_global = np.random.default_rng(SEED)
    for fname, (_, _, is_sparse) in feats.items():
        X, ok = featurise(smiles, fname)
        X, yy = X[ok], y[ok]
        Xtr, Xte, ytr, yte = train_test_split(
            X, yy, test_size=0.25, random_state=SEED, stratify=yy)
        Xtr, ytr = balance_inside_fold(Xtr, ytr, rng_global)
        for mname, mfn in make_models().items():
            try:
                m = clone(mfn); m.fit(Xtr, ytr)
                p = _predict_scores(m, Xte)
                rows.append({"feature": fname, "model": mname,
                             "auc_holdout": roc_auc_score(yte, p)})
            except Exception:
                pass
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "cv_holdout.csv"), index=False)
    print(f"holdout mean AUC = {res.auc_holdout.mean():.3f} (n={len(res)})")
    return res


# ============================================================
# 10. predict NRDL
# ============================================================
def _to_prob(scores):
    """Logistic squash for decision_function outputs (SVM)."""
    s = np.asarray(scores, dtype=float)
    if s.min() >= 0.0 and s.max() <= 1.0:
        return s
    return 1.0 / (1.0 + np.exp(-s))


def mode_predict():
    from sklearn.preprocessing import StandardScaler
    smiles, y, _ = load_training()
    nrdl = pd.read_csv(os.path.join(DATA, "NRDL_drugs.csv"),
                       encoding="utf-8-sig")
    if nrdl["DrugName"].duplicated().any():
        print("[predict] WARNING: duplicate DrugName in NRDL")
    feats = build_features()
    probs, votes = {}, {}
    nrdl_ok = None
    for fname, (_, _, is_sparse) in feats.items():
        X, ok = featurise(smiles, fname)
        X, yy = X[ok], y[ok]
        Xn, okn = featurise(nrdl["SMILES"].tolist(), fname)
        Xn = Xn[okn]
        nrdl_ok = nrdl.loc[okn].reset_index(drop=True)
        for mname, mfn in make_models().items():
            Xtr, Xte = X.copy(), Xn.copy()
            if mname in NEEDS_SCALING and not is_sparse:
                sc = StandardScaler().fit(Xtr)
                Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            rng = np.random.default_rng(SEED)
            Xb, yb = balance_inside_fold(Xtr, yy, rng)
            try:
                m = clone(mfn); m.fit(Xb, yb)
                p = _to_prob(_predict_scores(m, Xte))
            except Exception:
                continue
            key = f"{fname}|{mname}"
            probs[key] = p
            votes[key] = (p >= 0.6).astype(int)

    if not probs:
        print("[predict] no model produced predictions"); return None
    P = pd.DataFrame(probs, index=nrdl_ok["DrugName"])
    V = pd.DataFrame(votes, index=nrdl_ok["DrugName"])
    P.to_csv(os.path.join(OUT, "nrdl_risk_probability.csv"))
    summary = pd.DataFrame({
        "DrugName": nrdl_ok["DrugName"],
        "n_models": V.shape[1],
        "votes_risk": V.sum(axis=1).values,
        "vote_fraction": V.mean(axis=1).values,
        "mean_probability": P.mean(axis=1).values,
    }).sort_values("vote_fraction", ascending=False)
    summary["flagged_high_risk"] = summary["vote_fraction"] > 0.5
    summary.to_csv(os.path.join(OUT, "nrdl_consensus.csv"), index=False)
    print(summary.head(15).to_string(index=False))
    print(f"\nflagged: {int(summary.flagged_high_risk.sum())} / {len(summary)}")
    return summary


# ============================================================
# 11. plot (reads OOF cache; 0 fits)
# ============================================================
def _save_fig(fig, stem):
    path = FIGDIR / stem
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{path}.svg", bbox_inches="tight")
    plt.close(fig)


def _trapz(y, x):
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


_model_color_cache = {}
def _model_colors(model_names, cmap_name="tab10"):
    """Return {model_name: rgba} with a stable assignment across subplots."""
    names = list(model_names)
    if all(n in _model_color_cache for n in names):
        return {n: _model_color_cache[n] for n in names}
    if len(names) <= 10:
        base = plt.get_cmap("tab10")
        cols = [base(i) for i in range(len(names))]
    elif len(names) <= 20:
        base = plt.get_cmap("tab20")
        cols = [base(i) for i in range(len(names))]
    else:
        base = plt.get_cmap("hsv")
        cols = [base(i / max(len(names) - 1, 1)) for i in range(len(names))]
    for n, c in zip(names, cols):
        _model_color_cache[n] = c
    return {n: _model_color_cache[n] for n in names}

def mode_plot(features=None, models=None, force=False):
    """Render ROC figures from cached OOF.  No fits are performed."""
    if not _cache_ok():
        print("[plot] OOF cache missing -> running mode_cv first")
        mode_cv(force=force)
    else:
        _cache_summary()

    from sklearn.metrics import roc_curve

    y = np.load(P_OOF_Y)
    z = np.load(P_OOF)
    cached = {k: z[k] for k in z.files}
    cv = pd.read_csv(P_CV_CSV)

    feats = list(cv["feature"].unique())
    all_models = list(cv["model"].unique())
    if features is None:
        features = feats
    if models is None:
        models = all_models

    overlay = {f: [] for f in features}
    auc_grid = pd.DataFrame(index=features, columns=models, dtype=float)

    for fname in features:
        for mname in models:
            key = f"{fname}__{mname}"
            if key not in cached:
                continue
            oof = cached[key]
            fpr_grid = np.linspace(0, 1, 200)
            tprs = []
            for rep in range(oof.shape[0]):
                p = oof[rep]
                mask = ~np.isnan(p)
                if mask.sum() < 2 or len(np.unique(y[mask])) < 2:
                    continue
                f, t, _ = roc_curve(y[mask], p[mask])
                tprs.append(np.interp(fpr_grid, f, t))
            if not tprs:
                continue
            tpr_mean = np.mean(tprs, axis=0)
            tpr_lo   = np.percentile(tprs, 5,  axis=0)
            tpr_hi   = np.percentile(tprs, 95, axis=0)
            auc_mean = _trapz(tpr_mean, fpr_grid)
            auc_grid.loc[fname, mname] = auc_mean

            fig, ax = plt.subplots(figsize=(4.2, 3.6))
            ax.fill_between(fpr_grid, tpr_lo, tpr_hi, alpha=.2,
                            color="#2c7bb6", label="5-95% across repeats")
            ax.plot(fpr_grid, tpr_mean, color="#2c7bb6", lw=2,
                    label=f"AUC = {auc_mean:.3f}")
            ax.plot([0, 1], [0, 1], "--", color="navy", lw=1)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"{fname}  |  {mname}\n"
                         f"repeated stratified CV ({N_SPLITS}x{N_REPEATS}), "
                         f"n={len(y)}", fontsize=9, loc="left")
            ax.legend(loc="lower right", fontsize=7, frameon=False)
            ax.grid(True, ls="--", alpha=.5)
            _save_fig(fig, f"roc/{fname}__{mname}")
            overlay[fname].append((mname, fpr_grid, tpr_mean, auc_mean))
            print(f"  [plot] {fname:12s} {mname:18s} AUC={auc_mean:.3f}")
    
    # Fixed order
    _all_model_names = sorted({m for rows in overlay.values() for (m, *_rest) in rows})
    _model_color = _model_colors(_all_model_names)

    for fname, rows in overlay.items():
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(5.6, 4.4))
        for mname, fpr_grid, tpr_mean, auc_mean in sorted(rows, key=lambda x: -x[3]):
            ax.plot(fpr_grid, tpr_mean, lw=1.6,
                    color=_model_color[mname],
                    label=f"{mname} ({auc_mean:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="navy", lw=1)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{fname}: all models, out-of-fold ROC\n"
                     f"{N_SPLITS}x{N_REPEATS} repeated stratified CV, n={len(y)}",
                     fontsize=9, loc="left")
        ax.legend(loc="lower right", fontsize=7, frameon=False, ncol=2)
        ax.grid(True, ls="--", alpha=.5)
        _save_fig(fig, f"roc_overlay/{fname}_all_models")


    if auc_grid.notna().any().any():
        fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(features) + 1.5))
        data = auc_grid.astype(float).values
        from matplotlib.colors import TwoSlopeNorm
        vmin, vmax = 0.4, 1.0
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.5, vmax=vmax)
        im = ax.imshow(data, aspect="auto", cmap="RdBu_r", norm=norm)

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if np.isnan(v):
                    continue
                rgba = im.cmap(norm(v))
                lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                txt_color = "white" if lum < 0.45 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color=txt_color)

        ax.set_title(f"AUC heatmap (red = high, blue = low), out-of-fold, "
                     f"{N_SPLITS}x{N_REPEATS} CV\n"
                     f"n={len(y)} (Risk={int(y.sum())}), white = 0.5 baseline",
                     fontsize=9, loc="left")
        fig.colorbar(im, ax=ax, fraction=.04, pad=.02, label="mean AUC")
        _save_fig(fig, "roc_overlay/auc_heatmap")

    auc_grid.to_csv(FIGDIR / "auc_heatmap.csv", float_format="%.4f")
    print(f"\n[plot] figures -> {FIGDIR}")
    return auc_grid


# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["diagnose", "cv", "perm", "holdout",
                             "predict", "plot", "all"])
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--force", action="store_true",
                    help="ignore caches and retrain")
    a = ap.parse_args()

    if a.mode == "all":
        print("\n" + "=" * 70 + "\n### RUNNING: diagnose\n" + "=" * 70)
        mode_diagnose()
        print("\n" + "=" * 70 + "\n### RUNNING: cv\n" + "=" * 70)
        mode_cv(force=a.force)
        print("\n" + "=" * 70 + "\n### RUNNING: perm\n" + "=" * 70)
        mode_perm(n_perm=a.n_perm, force=a.force)
        print("\n" + "=" * 70 + "\n### RUNNING: plot\n" + "=" * 70)
        mode_plot(force=a.force)
        print("\n" + "=" * 70 + "\n### RUNNING: predict\n" + "=" * 70)
        mode_predict()
    elif a.mode == "cv":
        mode_cv(force=a.force)
    elif a.mode == "perm":
        mode_perm(n_perm=a.n_perm, force=a.force)
    elif a.mode == "plot":
        mode_plot(force=a.force)
    elif a.mode == "diagnose":
        mode_diagnose()
    elif a.mode == "holdout":
        mode_holdout()
    elif a.mode == "predict":
        mode_predict()