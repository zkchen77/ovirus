# OVirus

[![Python](https://img.shields.io/badge/Python-3.14%2B-blue.svg)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-%E2%89%A5%204.3-276DC3.svg)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**OVirus** is an integrated bioinformatics and machine-learning code package built around two mutually supporting computational tracks:

1. **Bidirectional Mendelian randomisation (MR)** — using 8 skeletal disorders as exposures and 39 infectious diseases as outcomes, this track performs bidirectional MR causal inference (312 exposure–outcome pairs per direction) with a full quality-control and sensitivity suite: instrument-selection auditing, five MR estimators, heterogeneity, directional pleiotropy, Steiger directionality, leave-one-out, and multiple-testing correction. Literature-derived gene sets are then mined from PubTator3, grouped by clinical pattern (Pattern I / II / III), and taken through STRING protein–protein interaction (PPI) network construction and GO-BP / KEGG / Reactome enrichment analysis.
2. **Drug-risk machine learning** — using molecular fingerprints (MACCS, Morgan, Topological Torsion, Atom Pair, RDKit, Pattern) and molecular descriptors, this track classifies "risk drugs" versus "dual-effect drugs" with repeated stratified cross-validation and permutation testing, and produces risk-probability predictions for NRDL drugs together with a validation audit.

The repository also ships `article_figures/` for one-command reproduction of every main figure / panel of the article.

---

## Repository layout

```
ovirus/
├── README.md                     # This file
├── LICENSE                       # MIT License
├── requirements.txt              # Python dependencies (pinned)
├── requirements-R.txt            # R dependencies (packages + usage notes)
│
├── config/                       # Initial analysis configuration
│   ├── all_traits.txt            # All 47 GWAS trait IDs (8 skeletal + 39 infectious)
│   ├── skeletal_exposures.txt    # 8 exposure traits (skeletal disorders)
│   ├── infection_outcomes.txt    # 39 outcome traits (infectious diseases)
│   └── openGWAS_dataset_manifest.xlsx  # OpenGWAS dataset manifest (IDs / names / sample sizes)
│
├── scripts/                      # Core pipeline code
│   ├── .Rproj                    # RStudio project file (working directory = scripts/)
│   ├── .Renviron                 # OpenGWAS API token location (see Notes below)
│   ├── 01_mr_extract_instruments.R   # Instrument extraction (LD clumping + threshold audit)
│   ├── 02_mr_forward.R           # Forward MR: skeletal disorders -> infectious diseases
│   ├── 03_mr_reverse.R           # Reverse MR: infectious diseases -> skeletal disorders
│   ├── 04_merge_mr_results.py    # Merge per-pair CSVs into four flat tables
│   ├── 05_mr_qc_sensitivity.py   # QC / sensitivity / multiple testing (QC1-QC5)
│   ├── 06_make_rebuttal_figures.py  # Generate rebuttal figures FigR1-FigR5
│   ├── 07_pubtator3_genes.py     # PubTator3 literature gene sets + Pattern co-occurrence sets
│   ├── 08_string_ppi.R           # STRING PPI networks and topology
│   ├── 09_enrichment.R           # GO-BP / KEGG / Reactome enrichment
│   ├── 10_ensemble_ml.py         # Drug-risk classification: repeated stratified CV + permutation + NRDL prediction
│   └── 11_validation_audit.py    # Validation audit (leakage re-analysis + null-model AUC distribution)
│
├── article_figures/              # Fast reproduction of article figures / panels
│   ├── figure 1/                 # Fig 1A-1F, Fig S1 (code + data)
│   ├── figure 2/                 # Fig 2A-2F (incl. ImmPort pathway gene lists)
│   ├── figure 3/                 # Fig 3A / 3B (TTD target-drug mapping)
│   └── figure 4/                 # Fig 4B (incl. original_drugs.csv / NRDL_drugs.csv and caches)
│
└── data/                         # Full-pipeline data (downloaded / intermediate / results)
    ├── original_drugs.csv        # Training set (Drug Name / SMILES / Label)
    ├── NRDL_drugs.csv            # Drugs to predict (DrugName / SMILES)
    ├── mr_instruments/           # Per-trait instruments (exposure_*.csv)
    ├── pairwise_forward/         # Forward per-pair harmonised / res / het / ple / steiger
    ├── pairwise_reverse/         # Reverse per-pair results
    ├── mr_results/               # Merged flat tables (o2v_/v2o_*_total.csv)
    ├── mr_qc/                    # QC0-QC5 tables
    ├── pubtator3/                # Raw PubTator3 relation tables (<DISEASE>+gene.csv)
    ├── gene_sets/                # patternI/II/III co-occurrence gene sets
    ├── string/                   # STRING web-export TSVs + network plots
    ├── ppi/                      # PPI topology / edge lists / hub tables
    ├── enrichment/               # GO-BP / KEGG / Reactome results
    ├── differentially expressed genes/  # Raw DEGs per infection (GEO datasets)
    ├── ml_analyse/               # Historical CV-AUC summary (total_cv_auc.xlsx)
    ├── ml_results/               # CV caches / permutation nulls / best_model.joblib / NRDL predictions
    └── figures/                  # FigR1-FigR6 outputs
```

---

## Requirements and installation

### 1. Python environment

- Python >= 3.13 (the full pipeline was verified under **Python 3.14.6 + Anaconda**)
- Install dependencies (versions pinned to the verified environment):

```bash
pip install -r requirements.txt
```

> If the RDKit pip install fails, use conda instead: `conda install -c conda-forge rdkit`

### 2. R environment

- R >= 4.3 + RStudio
- Package list with per-package usage notes: **[requirements-R.txt](requirements-R.txt)** (includes a one-shot install command):

```r
install.packages(c("TwoSampleMR", "ieugwasr", "igraph"))
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install(c("STRINGdb", "clusterProfiler", "org.Hs.eg.db",
                       "ReactomePA", "DOSE"))
```

### 3. OpenGWAS API access

The MR track (scripts 01-03) accesses the IEU OpenGWAS API through [ieugwasr](https://mrcieu.github.io/ieugwasr/) and requires an API token:

1. Sign in at <https://api.opengwas.io/profile/> and generate a JWT;
2. Store it in an R environment file (`~/.Renviron`, or `scripts/.Renviron` in this repo):

```r
OPENGWAS_JWT=xxxxxxxxxxxxxxxx
```

Scripts detect this variable at startup; without it, API calls in 01-03 will fail.

---

## Full pipeline (recommended execution order)

### Stage A: MR instruments and bidirectional MR (R, run in RStudio)

Open `scripts/.Rproj` in RStudio (the working directory is set to `scripts/` automatically), then run in order:

| Script | Function | Outputs |
|--------|----------|---------|
| `01_mr_extract_instruments.R` | Extract instruments for all 47 traits in `config/all_traits.txt`: p1 = 5e-6, LD clumping (r2 = 0.001, 10 Mb window, 1000 Genomes EUR reference panel) | `data/mr_instruments/exposure_*.csv`, `data/mr_qc/QC0_threshold_audit.csv` |
| `02_mr_forward.R` | Forward MR (8 exposures x 39 outcomes = 312 pairs): harmonisation, 5 estimators, heterogeneity, Egger intercept, Steiger | `data/pairwise_forward/` |
| `03_mr_reverse.R` | Reverse MR (39 x 8 = 312 pairs), same outputs | `data/pairwise_reverse/` |

> The reverse direction is not a formality: it is the test that defines the **asymmetry criterion** (retain only pairs significant in one direction but not the other).

### Stage B: MR result integration and QC (Python, run from the repository root)

```bash
python scripts/04_merge_mr_results.py
python scripts/05_mr_qc_sensitivity.py
python scripts/06_make_rebuttal_figures.py
```

- `04`: merges per-pair CSVs into `data/mr_results/{o2v,v2o}_{res,snp,het,ple}_total.csv`;
- `05`: writes QC1 (instrument strength, F statistics), QC2 (sensitivity summary), QC3 (leave-one-out), QC4 (multiple testing: global / within-exposure / within-outcome BH families + Bonferroni), QC5 (master table);
- `06`: renders `data/figures/FigR1-FigR5` (threshold audit, F distribution, p-value distribution + QQ, five-estimator forest, leave-one-out).

### Stage C: literature gene sets and functional annotation

```bash
python scripts/07_pubtator3_genes.py   # PubTator3 -> data/pubtator3/ + data/gene_sets/
```

Then, back in RStudio (`scripts/.Rproj`):

| Script | Function | Outputs |
|--------|----------|---------|
| `08_string_ppi.R` | Build PPIs from STRING web-export TSVs (combined_score >= 700, text-mining evidence sources excluded); compute topology and hubs | `data/ppi/*_topology.csv / *_edges.csv / *_hubs.csv` |
| `09_enrichment.R` | GO-BP / KEGG / Reactome enrichment for Pattern I / II / III co-occurrence gene sets (p = 0.05, q = 0.2, BH) | `data/enrichment/0{1,2,3}_*.csv` |

### Stage D: drug-risk machine learning (Python)

```bash
python scripts/10_ensemble_ml.py --mode all      # diagnose + cv + perm + plot + predict
python scripts/11_validation_audit.py            # leakage audit + null-model AUC distribution
```

- `10`: 7 molecular feature types x 8 classifiers, 5x3 repeated stratified cross-validation (**fold-internal undersampling**, no leakage), permutation testing (every AUC = 1 cell is always tested), and NRDL risk probabilities with consensus voting;
- `11`: reproduces and quantifies the validation leakage of the legacy pre-split oversampling design and reports the AUC null distribution of a random scorer as baseline.

> **Caveat**: in `10_ensemble_ml.py`, `flagged_high_risk` is based on a vote fraction > 0.5 across 56 balanced models. Because training folds are class-balanced, the ensemble is mildly biased toward predicting 'Risk' — treat the flags as a **screening signal**, not a clinical conclusion; external validation is required.

---

## Reproducing article figures (article_figures)

Each panel folder contains its `code_figure *.py` (or HTML) plus the required data files. Run from within the corresponding directory:

```bash
# Figure 1 (1A-1F and S1)
cd "article_figures/figure 1"
python "code_figure 1A.py"   # likewise 1B / 1C / 1D / 1E / 1F / S1

# Figure 2
cd "article_figures/figure 2"
python "code_figure 2C.py"   # 2B is a D3 interactive page: open code_figure 2B.html in a browser

# Figure 3
cd "article_figures/figure 3"
python "code_figure 3B.py"   # needs P1-01/P1-02 TTD downloads and the P1-07 mapping table

# Figure 4
cd "article_figures/figure 4"
python "code_figure 4B.py"   # plots from the local ml_results/ cache, no retraining needed
```

Data source for each figure:

| Figure | Content | Data files |
|--------|---------|------------|
| 1A / S1 | Bidirectional MR significance heatmap (Forward / Reverse / Bidirectional / None) | `fig 1A.csv` / `fig S1.csv` |
| 1B / 1C | Forest plots (effect size beta with 95% CI) | `fig 1B.csv` / `fig 1C.csv` |
| 1D-1F | UpSet intersection plots (pattern co-occurrence gene sets) | `fig 1D-1F.xlsx` |
| 2A / 2D-2F | STRING PPI networks (web-export) | `figure 2*.svg` |
| 2B | Enrichment bubble plot (interactive) | `fig 2B.csv` |
| 2C | ImmPort pathway gene-coverage heatmap | `fig 2C.xlsx` + `immport/*.txt` |
| 3A / 3B | Disease gene sets and TTD target-drug mapping | `figure 3*.xlsx`, `P1-*.txt/xlsx` |
| 4B | Out-of-fold ROC from repeated stratified CV | `ml_results/` caches |

---

## Configuration (config/)

| File | Contents |
|------|----------|
| `skeletal_exposures.txt` | 8 exposures: low back pain (ukb-b-1557), osteoarthritis (ukb-b-14486), osteonecrosis (finn-b-M13_OSTEONECROSIS), osteoporosis (ukb-b-12141), scoliosis (finn-b-M13_SCOLIOSIS), spinal stenosis (ukb-d-M13_SPINSTENOSIS), spondylolisthesis (finn-b-M13_SPONDYLOLISTHESIS), spondylosis (finn-b-M13_SPONDYLOSIS) |
| `infection_outcomes.txt` | 39 infectious-disease outcomes (FinnGen R13 `finn-b-AB1_*`, incl. respiratory tuberculosis, HIV, sepsis, varicella, etc.) |
| `all_traits.txt` | Union of the 47 traits above, consumed by `01` for batch instrument extraction |
| `openGWAS_dataset_manifest.xlsx` | Metadata manifest of the 47 GWAS datasets |

---

## Reproducibility and key parameters

- **Random seed**: fixed at `SEED = 77` throughout;
- **Instruments**: p1 = 5e-6 (relaxed from 5e-8; rationale documented in the QC0 threshold audit), clump r2 = 0.001 / 10 Mb, EUR reference panel;
- **MR estimators**: IVW (primary), MR-Egger, weighted median, simple mode, weighted mode;
- **Multiple testing**: three BH families (global 312 tests / 39 within exposure / 8 within outcome) + Bonferroni;
- **STRING networks**: web-export parameters combined_score >= 700 with text-mining evidence excluded; STRINGdb layout threshold 400;
- **Enrichment**: GO-BP (org.Hs.eg.db, ENTREZID), KEGG (hsa), Reactome (human), p = 0.05, q = 0.2, BH;
- **ML**: 5-fold x 3-repeat repeated stratified CV, fold-internal undersampling (no oversampling, no leakage), metrics AUC / PR-AUC / balanced accuracy / MCC.

## Notes

1. **API token security**: `scripts/.Renviron` stores `OPENGWAS_JWT`. 
2. **Network dependencies**: scripts 01-03 (OpenGWAS), 07 (PubTator3) and 08-09 (STRING / KEGG / Reactome) require internet access; the first STRINGdb run downloads the v12 database.
3. `data/` ships the complete intermediate results, so figures can be regenerated without recomputation (e.g. `article_figures/figure 4`).

## License

Released under the [MIT License](LICENSE).

## Citation

```bibtex
@software{ovirus,
  author = {Zikun Chen},
  title = {OVirus},
  year = {2026},
  url = {https://github.com/zkchen77/ovirus}
}
```

## Acknowledgments

- [TwoSampleMR](https://mrcieu.github.io/TwoSampleMR/) / [ieugwasr](https://mrcieu.github.io/ieugwasr/) / [OpenGWAS](https://opengwas.io/)
- [PubTator3](https://www.ncbi.nlm.nih.gov/research/pubtator3/) / [STRING](https://cn.string-db.org/) / [ImmPort](https://immport.org/shared/home) / [TTD](https://ttd.idrblab.cn/)
- [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.ai/), [CatBoost](https://catboost.ai/), [RDKit](https://www.rdkit.org/), [clusterProfiler](https://bioconductor.org/packages/clusterProfiler/)
