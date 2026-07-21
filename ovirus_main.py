# -*- coding: utf-8 -*-
"""
@author: zk
"""

"""
Molecular Feature Generation and Risk Prediction for Drug Compounds
This script extracts various molecular fingerprints and descriptors from SMILES strings,
trains multiple machine learning models to predict drug risk, and evaluates performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import roc_curve, auc

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 1. Feature Generation Functions
# =============================================================================

def smiles_to_maccs(smiles):
    """
    Generate MACCS (Molecular ACCess System) keys fingerprint.
    
    Parameters:
        smiles (str): SMILES string of the molecule
    
    Returns:
        np.ndarray: MACCS fingerprint as binary array (166 bits)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp)


def smiles_to_morgan(smiles, radius=2, nBits=256):
    """
    Generate Morgan fingerprint (circular fingerprint).
    
    Parameters:
        smiles (str): SMILES string of the molecule
        radius (int): Morgan algorithm radius (default: 2)
        nBits (int): Fingerprint length in bits (default: 256)
    
    Returns:
        np.ndarray: Morgan fingerprint as binary array
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits, dtype=np.int8)
    
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=nBits
    )
    
    fp = morgan_gen.GetFingerprint(mol)
    return np.array(fp)


def smiles_to_topological_torsion(smiles, nBits=256):
    """
    Generate topological torsion fingerprint.
    
    Parameters:
        smiles (str): SMILES string of the molecule
        nBits (int): Fingerprint length in bits (default: 256)
    
    Returns:
        np.ndarray: Topological torsion fingerprint as binary array
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits, dtype=np.int8)
    
    fp_gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=nBits)
    fp = fp_gen.GetFingerprint(mol)
    return np.array(fp)


def smiles_to_atom_pair(smiles, nBits=256):
    """
    Generate atom-pair fingerprint.
    
    Parameters:
        smiles (str): SMILES string of the molecule
        nBits (int): Fingerprint length in bits (default: 256)
    
    Returns:
        np.ndarray: Atom-pair fingerprint as binary array
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits, dtype=np.int8)
    
    fp_gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=nBits)
    fp = fp_gen.GetFingerprint(mol)
    return np.array(fp, dtype=np.int8)


def smiles_to_rdkfingerprint(smiles, fpSize=256, minPath=1, maxPath=7):
    """
    Generate RDKit fingerprint (hashed fingerprint based on subgraphs).
    
    Parameters:
        smiles (str): SMILES string of the molecule
        fpSize (int): Fingerprint length in bits (default: 256)
        minPath (int): Minimum path length (default: 1)
        maxPath (int): Maximum path length (default: 7)
    
    Returns:
        np.ndarray: RDKit fingerprint as binary array
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(fpSize, dtype=np.int8)
    
    fp = Chem.RDKFingerprint(
        mol,
        minPath=minPath,
        maxPath=maxPath,
        fpSize=fpSize,
        nBitsPerHash=2
    )
    arr = np.zeros((1,))
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.int8)


def smiles_to_pattern_fingerprint(smiles, fpSize=256):
    """
    Generate pattern fingerprint (substructure-based fingerprint).
    
    Parameters:
        smiles (str): SMILES string of the molecule
        fpSize (int): Fingerprint length in bits (default: 256)
    
    Returns:
        np.ndarray: Pattern fingerprint as binary array
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(fpSize, dtype=np.int8)
    
    fp = Chem.PatternFingerprint(mol, fpSize=fpSize)
    arr = np.zeros((1,))
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.int8)


def smiles_to_descriptors(smiles):
    """
    Extract 6 key molecular descriptors:
    0. MolWt              - Molecular weight
    1. MolLogP            - Lipophilicity (logP)
    2. TPSA               - Topological polar surface area
    3. NumHDonors         - Number of hydrogen bond donors
    4. NumHAcceptors      - Number of hydrogen bond acceptors
    5. NumRotatableBonds  - Number of rotatable bonds
    
    Parameters:
        smiles (str): SMILES string of the molecule
    
    Returns:
        np.ndarray: Array of 6 descriptor values (float32)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(6, dtype=np.float32)
    
    try:
        desc_list = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol)
        ]
        return np.array(desc_list, dtype=np.float32)
    except Exception:
        return np.zeros(6, dtype=np.float32)


def generate_features(smiles_list, ftype_name, **kwargs):
    """
    Generate molecular feature matrix based on specified feature type.
    
    Parameters:
        smiles_list (list): List of SMILES strings
        ftype_name (str): Feature type, supports:
            'maccs', 'morgan', 'topofp', 'atomfp', 'rdkfp', 'pattern', 'descriptors'
        **kwargs: Additional parameters for specific feature functions
    
    Returns:
        np.ndarray: Feature matrix (n_samples, n_features)
    """
    func_map = {
        "maccs": lambda s: smiles_to_maccs(s),
        "morgan": lambda s: smiles_to_morgan(s, **kwargs),
        "topofp": lambda s: smiles_to_topological_torsion(s, **kwargs),
        "atomfp": lambda s: smiles_to_atom_pair(s, **kwargs),
        "rdkfp": lambda s: smiles_to_rdkfingerprint(s, **kwargs),
        "pattern": lambda s: smiles_to_pattern_fingerprint(s, **kwargs),
        "descriptors": lambda s: smiles_to_descriptors(s),
    }
    
    if ftype_name not in func_map:
        raise ValueError(f"Unsupported feature type: {ftype_name}. "
                         f"Available: {list(func_map.keys())}")
    
    func = func_map[ftype_name]
    
    features = []
    for smile in smiles_list:
        feat = func(smile)
        features.append(feat)
    
    return np.vstack(features)


# =============================================================================
# 2. Data Balancing and Preprocessing
# =============================================================================

def balance_data(names, smiles, labels, random_state=77):
    """
    Balance the dataset by oversampling the minority class.
    
    Parameters:
        names, smiles, labels: Original data lists
        random_state (int): Random seed for reproducibility (default: 77)
    
    Returns:
        tuple: Balanced names, smiles, and labels as lists
    """
    rng = np.random.default_rng(seed=random_state)
    
    names, smiles, labels = np.array(names), np.array(smiles), np.array(labels)
    
    pos_mask = (labels == 'Dual benefit drugs')
    neg_mask = (labels == 'Risk drugs')
    
    pos_idx = np.where(pos_mask)[0]
    neg_idx = np.where(neg_mask)[0]
    
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    
    if n_pos == 0 or n_neg == 0:
        raise ValueError("One class is missing in the dataset!")
    
    target_size = max(n_pos, n_neg)
    
    if n_pos < target_size:
        sampled_pos_idx = rng.choice(pos_idx, size=target_size, replace=True)
        sampled_neg_idx = neg_idx
    else:
        sampled_neg_idx = rng.choice(neg_idx, size=target_size, replace=True)
        sampled_pos_idx = pos_idx
    
    sampled_idx = np.concatenate([sampled_pos_idx, sampled_neg_idx])
    rng.shuffle(sampled_idx)
    
    return names[sampled_idx].tolist(), smiles[sampled_idx].tolist(), labels[sampled_idx].tolist()


# =============================================================================
# 3. Model Training and Evaluation
# =============================================================================

def evaluate_models(X_train, y_train, X_test, models):
    """
    Train multiple models and return prediction probabilities.
    
    Parameters:
        X_train, y_train: Training data
        X_test: Test data
        models (dict): Dictionary of model_name: model_instance
    
    Returns:
        dict: Dictionary of model_name: prediction probabilities
    """
    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        probas = model.predict_proba(X_test)
        risk_prob = probas[:, 1]  # Label encoding: 0=non-risk, 1=risk
        results[model_name] = risk_prob
    return results


def plot_roc_curves(models, X_train, y_train, X_val, y_val, feature_name):
    """
    Plot ROC curves for all models.
    
    Parameters:
        models (dict): Dictionary of model_name: model_instance
        X_train, y_train: Training data
        X_val, y_val: Validation data
        feature_name (str): Name of feature type for plot title
    
    Returns:
        dict: ROC curve data (fpr, tpr, auc) for each model
    """
    roc_curves = {}
    
    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)
            
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_val)[:, 1]
            elif hasattr(model, "decision_function"):
                y_pred_proba = model.decision_function(X_val)
            else:
                print(f"⚠️ Model {model_name} does not support probability or decision function.")
                continue
            
            fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            roc_curves[model_name] = (fpr, tpr, roc_auc)
            
        except Exception as e:
            print(f"❌ Error computing ROC for model {model_name}: {e}")
            continue
    
    # Plot ROC curves
    plt.figure(figsize=(6, 4))
    for model_name, (fpr, tpr, roc_auc) in roc_curves.items():
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random (AUC = 0.5)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'ROC Curves for All Models ({feature_name} features)\n'
              f'Train Size: {len(y_train)}, Val Size: {len(y_val)}')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    roc_plot_path = f'roc_curves_{feature_name}.svg'
    plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n✅ ROC curves saved to: {roc_plot_path}")
    
    return roc_curves


def save_predictions(NRDL_names, predictions, feature_name):
    """
    Save prediction results to CSV file.
    
    Parameters:
        NRDL_names (list): List of drug names
        predictions (dict): Dictionary of model_name: prediction probabilities
        feature_name (str): Feature type for filename
    """
    full_results = {name: {} for name in NRDL_names}
    for model_name, probs in predictions.items():
        for i, drug_name in enumerate(NRDL_names):
            full_results[drug_name][model_name] = probs[i]
    
    df = pd.DataFrame.from_dict(full_results, orient='index')
    df.index.name = 'Drug Name'
    
    df['Mean_Risk'] = df.mean(axis=1)
    df = df.sort_values('Mean_Risk', ascending=False).drop(columns=['Mean_Risk'])
    
    output_csv = f'risk_predictions_{feature_name}.csv'
    df.to_csv(output_csv, float_format='%.4f')
    print(f"\n✅ Prediction results saved to: {output_csv}")


# =============================================================================
# 4. Main Program
# =============================================================================

if __name__ == "__main__":
    # Define feature types to iterate over
    feature_types = ["maccs", "morgan", "topofp", "atomfp", "rdkfp", "pattern", "descriptors"]
    
    # Read data from CSV files
    print("=" * 60)
    print("Loading data from CSV files...")
    print("=" * 60)
    
    # Load original drugs (training data)
    df_original = pd.read_csv('original_drugs.csv')
    original_drugs = list(zip(df_original['DrugName'], df_original['SMILES'], df_original['Label']))
    print(f"Loaded {len(original_drugs)} original drug entries")
    
    # Load NRDL drugs (test/prediction data)
    df_nrdl = pd.read_csv('NRDL_drugs.csv')
    NRDL_drugs = list(zip(df_nrdl['DrugName'], df_nrdl['SMILES']))
    print(f"Loaded {len(NRDL_drugs)} NRDL drug entries")
    print("=" * 60)
    
    # Extract names and SMILES from NRDL drugs
    NRDL_names = [name for name, _ in NRDL_drugs]
    NRDL_smiles = [smile for _, smile in NRDL_drugs]
    
    # Process original drug data
    orig_names, orig_smiles, orig_labels = zip(*original_drugs)
    
    # Iterate through all feature types
    for feature_name in feature_types:
        print(f"\n{'='*60}")
        print(f"Processing feature type: {feature_name.upper()}")
        print(f"{'='*60}")
        
        # Balance the training dataset
        names_balanced, smiles_balanced, labels_balanced = balance_data(
            orig_names, orig_smiles, orig_labels
        )
        
        # Generate features for training data
        X_features = generate_features(smiles_balanced, feature_name)
        
        # Encode labels: Risk drugs → 1, Dual benefit drugs → 0
        y_labels = np.array([1 if label == 'Risk drugs' else 0 for label in labels_balanced])
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_features)
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled,
            y_labels,
            test_size=0.2,
            random_state=77,
            stratify=y_labels
        )
        
        print("\n=== Data Diagnostics ===")
        print(f"Balanced labels: {np.bincount(y_labels)}")
        print(f"Training set labels: {np.bincount(y_train)}")
        print(f"Validation set labels: {np.bincount(y_val)}")
        print(f"X_train shape: {X_train.shape}")
        print(f"Any all-zero features? {np.any(X_train.sum(axis=0) == 0)}")
        
        # Calculate class weight for imbalanced data handling
        num_pos = (y_labels == 1).sum()
        num_neg = (y_labels == 0).sum()
        scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
        
        # Define machine learning models
        models = {
            'RandomForest': RandomForestClassifier(random_state=77),
            
            'LogisticRegression': LogisticRegression(
                max_iter=1000, random_state=77, class_weight='balanced'
            ),
            
            'SVM': SVC(
                probability=True, random_state=77, class_weight='balanced'
            ),
            
            'XGBoost': XGBClassifier(
                use_label_encoder=False, eval_metric='logloss', 
                random_state=77, scale_pos_weight=scale_pos_weight
            ),
            
            'BernoulliNB': BernoulliNB(),
            
            'MLP': MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation='relu', solver='adam', max_iter=1000,
                random_state=77, early_stopping=True,
                validation_fraction=0.1
            ),
            
            'KNN': KNeighborsClassifier(
                n_neighbors=5, weights='distance'
            ),
            
            'CatBoost': CatBoostClassifier(
                random_seed=77, auto_class_weights='Balanced',
                verbose=False, iterations=200, learning_rate=0.1, depth=4
            )
        }
        
        # Generate features for NRDL drugs (new drugs)
        NRDL_features = generate_features(NRDL_smiles, feature_name)
        scaled_NRDL_features = scaler.transform(NRDL_features)
        
        # Plot ROC curves
        print("\n--- Evaluating models on validation set ---")
        roc_curves = plot_roc_curves(
            models, X_train, y_train, X_val, y_val, feature_name
        )
        
        # Predict on NRDL drugs
        print("\n--- Predicting risk probabilities for NRDL drugs ---")
        predictions = evaluate_models(X_train, y_train, scaled_NRDL_features, models)
        
        # Save predictions to CSV
        save_predictions(NRDL_names, predictions, feature_name)
        
        print(f"\n✅ Completed analysis for {feature_name.upper()} feature type")
        print("=" * 60)

print("\n" + "="*60)
print("All feature types processed successfully!")
print("="*60)