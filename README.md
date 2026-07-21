# Drug Risk Prediction Pipeline

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![RDKit](https://img.shields.io/badge/RDKit-2024.3.6-brightgreen.svg)](https://www.rdkit.org/)

A comprehensive machine learning pipeline for predicting drug risk using molecular fingerprints and descriptors. This tool extracts 7 types of molecular features from SMILES strings, trains 8 state-of-the-art classifiers, and generates risk probability predictions for new drug candidates.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Feature Types](#feature-types)
- [Machine Learning Models](#machine-learning-models)
- [Output Files](#output-files)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Citation](#citation)

## Features

- **7 Molecular Feature Types**: MACCS, Morgan, Topological Torsion, Atom Pair, RDKit, Pattern, and Molecular Descriptors
- **8 Machine Learning Models**: Random Forest, Logistic Regression, SVM, XGBoost, BernoulliNB, MLP, KNN, and CatBoost
- **Automated Data Balancing**: Intelligent oversampling for imbalanced datasets
- **ROC Curve Visualization**: Comprehensive model performance comparison
- **Risk Probability Predictions**: CSV output with ensemble predictions for NRDL drugs
- **Flexible Input**: Supports CSV file input for easy integration

## Installation

### Prerequisites

- Python 3.8 or higher (recommended: Python 3.12)
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/zkchen77/ovirus.git
cd ovirus
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python environment_check.py
```

## Data Preparation

### Input CSV Files

#### 1. `original_drugs.csv` (Training Data)

Contains known drugs with their risk labels:

| DrugName | SMILES | Label |
|----------|--------|-------|
| Dual_Moxonid... | CC1=NC(=C(C(=N1)Cl)NC2=NCCN2)OC | Dual benefit drugs |
| Risk_Saralas... | C[C@@H](C(=O)O)NC(=O)[C@@H]1CCCN1... | Risk drugs |

**Required Columns:**
- `DrugName`: Unique identifier for the drug
- `SMILES`: SMILES string representation
- `Label`: Risk category (must have exactly 2 distinct classes)

#### 2. `NRDL_drugs.csv` (Test/Prediction Data)

New drugs to predict risk scores:

| DrugName | SMILES |
|----------|--------|
| NRDL_Dexketo... | C[C@@H](C1=CC(=CC=C1)C(=O)C2=CC=CC=C2)C(=O)O |

**Required Columns:**
- `DrugName`: Unique identifier for the drug
- `SMILES`: SMILES string representation

## Usage

### Basic Usage

Run the main pipeline:

```bash
python ovirus_main.py
```

### Quick Test

Test your environment setup:

```bash
python environment_check.py
```

### Execution Flow

1. **Data Loading**: Reads CSV files from the current directory
2. **Feature Generation**: Extracts molecular fingerprints for each feature type
3. **Data Balancing**: Applies oversampling to handle imbalanced classes
4. **Model Training**: Trains all 8 models on the training data
5. **Evaluation**: Generates ROC curves and AUC scores
6. **Prediction**: Predicts risk probabilities for NRDL drugs
7. **Output**: Saves results to CSV files

## Feature Types

| Feature Type | Description | Dimensions |
|--------------|-------------|------------|
| **MACCS** | Molecular ACCess System keys | 166 bits |
| **Morgan** | Circular fingerprints (ECFP-like) | 256 bits |
| **Topological Torsion** | Topological torsion fingerprints | 256 bits |
| **Atom Pair** | Atom-pair fingerprints | 256 bits |
| **RDKit** | RDKit hashed fingerprints | 256 bits |
| **Pattern** | Substructure pattern fingerprints | 256 bits |
| **Descriptors** | 6 key molecular properties | 6 features |

### Molecular Descriptors

1. **MolWt**: Molecular weight
2. **MolLogP**: Lipophilicity (logP)
3. **TPSA**: Topological polar surface area
4. **NumHDonors**: Number of hydrogen bond donors
5. **NumHAcceptors**: Number of hydrogen bond acceptors
6. **NumRotatableBonds**: Number of rotatable bonds

## Machine Learning Models

| Model | Description | Best For |
|-------|-------------|----------|
| **Random Forest** | Ensemble of decision trees | High-dimensional data |
| **Logistic Regression** | Linear classifier | Interpretability |
| **SVM** | Support Vector Machine | Complex decision boundaries |
| **XGBoost** | Gradient boosting | High performance |
| **BernoulliNB** | Naive Bayes for binary features | MACCS fingerprints |
| **MLP** | Neural network | Non-linear patterns |
| **KNN** | K-nearest neighbors | Similarity-based predictions |
| **CatBoost** | Gradient boosting with categorical features | Robust performance |

## Output Files

The pipeline generates separate outputs for each feature type:

```
roc_curves_maccs.svg          # ROC curves for all models (MACCS)
risk_predictions_maccs.csv    # Risk predictions for NRDL drugs (MACCS)
roc_curves_morgan.svg         # ROC curves for all models (Morgan)
risk_predictions_morgan.csv   # Risk predictions for NRDL drugs (Morgan)
... (for each feature type)
```

### CSV Output Format

| Drug Name | RandomForest | LogisticRegression | SVM | XGBoost | ... | Mean_Risk |
|-----------|--------------|-------------------|-----|---------|-----|-----------|
| NRDL_Dexketo... | 0.8234 | 0.7891 | 0.8456 | 0.9123 | ... | 0.8426 |
| NRDL_Ketopro... | 0.3456 | 0.4123 | 0.3789 | 0.2987 | ... | 0.3589 |

- Values represent risk probability (0 = low risk, 1 = high risk)
- Drugs are sorted by mean risk score (highest to lowest)

## Project Structure

```
drug-risk-prediction/
│
├── ovirus_main.py          # Main pipeline script
├── environment_check.py      # Environment verification script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── LICENSE                   # MIT License
│
├── original_drugs.csv        # Training data (required)
├── NRDL_drugs.csv           # Test/prediction data (required)
│
├── roc_curves_*.svg         # ROC curve visualizations (generated)
├── risk_predictions_*.csv   # Prediction results (generated)
│
└── .gitignore               # Git ignore file
```

## Troubleshooting

### Common Issues

**1. "One class is missing in the dataset!"**

- Ensure your `original_drugs.csv` has exactly 2 unique labels
- Check for empty spaces or special characters in labels

**2. "No module named 'rdkit'"**

```bash
pip install rdkit-pypi
```

Or with conda:

```bash
conda install -c conda-forge rdkit
```

**3. SMILES parsing errors**

- Verify SMILES strings are valid
- Remove any extra spaces or special characters
- Check for brackets and parentheses balance

**4. Memory issues**

- Reduce fingerprint size (default: 256 bits)
- Use a smaller test dataset
- Consider using `nBits=128` for initial testing

### Performance Tips

- **For large datasets**: Use `nBits=128` to reduce memory usage
- **For better accuracy**: Use multiple feature types and ensemble predictions
- **For faster execution**: Reduce the number of models in the `models` dictionary

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{ovirus,
  author = {Zikun Chen},
  title = {Drug Risk Prediction Pipeline},
  year = {2026},
  url = {https://github.com/zkchen77/ovirus}
}
```

## Acknowledgments

- [RDKit](https://www.rdkit.org/) - Cheminformatics toolkit
- [scikit-learn](https://scikit-learn.org/) - Machine learning library
- [XGBoost](https://xgboost.ai/) - Gradient boosting framework
- [CatBoost](https://catboost.ai/) - Gradient boosting with categorical features

---

**Star this repository if you find it useful!**
