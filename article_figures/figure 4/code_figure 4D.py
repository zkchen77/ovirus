# -*- coding: utf-8 -*-
"""
@author: zkchen
"""

import os
import argparse
import warnings
import functools
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.svm")
warnings.filterwarnings("ignore", message=".*probability.*deprecated.*")

import matplotlib
matplotlib.use("Agg")

from sklearn.base import clone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "figure 4")
OUT  = os.path.join(ROOT, "figure 4", "ml_results")
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


# features
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


# models
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


# fold-internal balancing 
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


def _predict_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


# data loading
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



# predict NRDL
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


# main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="ignore caches and retrain")
    a = ap.parse_args()
    mode_predict()
    