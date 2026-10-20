# -*- coding: utf-8 -*-
"""
04_merge_mr_results.py
@author: zkchen

======================
Concatenate the per-pair CSVs written by 02/03 into four flat tables per
direction.  Also writes the instrument-selection audit (QC0) if the R scripts
produced it.

  data/pairwise_forward/<outcome>/outcome_{res,snp,heterogeneity,pleiotropy}_<exposure>.csv
      ->  data/mr_results/o2v_{res,snp,het,ple}_total.csv
  data/pairwise_reverse/...
      ->  data/mr_results/v2o_{res,snp,het,ple}_total.csv
=====================

"""
import os
import glob
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIR = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "data", "mr_results")
os.makedirs(OUT, exist_ok=True)

PREFIX = {
    "res": "outcome_res_",
    "snp": "outcome_snp_",
    "het": "outcome_heterogeneity_",
    "ple": "outcome_pleiotropy_",
}

for tag, folder in (("o2v", "pairwise_forward"), ("v2o", "pairwise_reverse")):
    base = os.path.join(PAIR, folder)
    if not os.path.isdir(base):
        print(f"[04] skip {folder} (not present)")
        continue
    for kind, pref in PREFIX.items():
        files = glob.glob(os.path.join(base, "**", pref + "*.csv"), recursive=True)
        if not files:
            print(f"[04] no {pref}* in {folder}")
            continue
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        dst = os.path.join(OUT, f"{tag}_{kind}_total.csv")
        df.to_csv(dst, index=False)
        print(f"[04] {os.path.relpath(dst, ROOT)}  ({len(df)} rows from {len(files)} files)")

print("[04] done.")
