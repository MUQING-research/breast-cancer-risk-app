# Module guide:
# - Role: Preserve a legacy breast-cancer workflow.
# - Workflow: Retain historical preprocessing, exploratory, or modelling behavior for comparison.
# - Design note: Review archived assumptions before using a routine in a new pipeline.
"""Shared constants, data loading, preprocessing, and metrics for breast cancer scripts."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (roc_auc_score, accuracy_score, roc_curve,
                              confusion_matrix, f1_score)

# ── Constants ──────────────────────────────────────────────────────────────────
SEED = 42
CELL_COLORS = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
    "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85",
]

VISUAL_LINEAR = [
    'mean texture', 'mean concavity', 'concave points error',
    'worst texture', 'worst concavity',
]
TRANSFORM_INV = ['compactness error', 'fractal dimension error']
TERTILE_COLS  = [
    'concavity error', 'mean fractal dimension',
    'texture error',   'symmetry error',
]


# ── Matplotlib style ───────────────────────────────────────────────────────────
# Function guide: setup_matplotlib is responsible for perform matplotlib.
# Inputs: **overrides. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def setup_matplotlib(**overrides):
    """Apply Cell-style matplotlib defaults. Keyword args override individual settings."""
    params = {
        'font.family'       : 'Arial',
        'font.sans-serif'   : ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans', 'sans-serif'],
        'font.size'         : 9,
        'axes.labelsize'    : 9,
        'axes.titlesize'    : 10,
        'axes.titleweight'  : 'bold',
        'axes.linewidth'    : 0.8,
        'xtick.labelsize'   : 8,
        'ytick.labelsize'   : 8,
        'legend.fontsize'   : 8,
        'xtick.direction'   : 'out',
        'ytick.direction'   : 'out',
        'xtick.major.width' : 0.8,
        'ytick.major.width' : 0.8,
        'xtick.major.size'  : 3.0,
        'ytick.major.size'  : 3.0,
        'axes.spines.top'   : True,
        'axes.spines.right' : True,
        'axes.grid'         : False,
        'figure.facecolor'  : 'white',
        'axes.facecolor'    : 'white',
        'figure.dpi'        : 300,
        'savefig.dpi'       : 300,
        'savefig.bbox'      : 'tight',
        'pdf.fonttype'      : 42,
        'ps.fonttype'       : 42,
        'lines.linewidth'   : 1.0,
        'lines.markersize'  : 4,
        'errorbar.capsize'  : 3,
    }
    params.update(overrides)
    matplotlib.rcParams.update(params)


# ── Data loading ───────────────────────────────────────────────────────────────
# Function guide: load_data is responsible for load data.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def load_data():
    """Return (X, y, X_train, X_test, y_train, y_test) — 80/20 stratified split."""
    np.random.seed(SEED)
    data = load_breast_cancer()
    X    = pd.DataFrame(data.data, columns=data.feature_names)
    y    = pd.Series(data.target, name='target')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    return X, y, X_train, X_test, y_train, y_test


# ── Preprocessing ──────────────────────────────────────────────────────────────
# Function guide: get_tertile_cuts is responsible for perform tertile cuts.
# Inputs: X_train. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def get_tertile_cuts(X_train):
    """Compute tertile (33rd / 67th percentile) cut points from training set."""
    return {col: tuple(np.percentile(X_train[col].values, [33.33, 66.67]))
            for col in TERTILE_COLS}


# Function guide: build_p1 is responsible for build p1.
# Inputs: X_df. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def build_p1(X_df):
    """Pipeline 1: raw features, no transformation."""
    return X_df.copy().astype(float)


# Function guide: build_p2 is responsible for build p2.
# Inputs: X_df. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def build_p2(X_df):
    """Pipeline 2: 1/x for TRANSFORM_INV vars, rest unchanged."""
    df = X_df.copy().astype(float)
    for col in TRANSFORM_INV:
        x = df[col].values
        df[col] = 1.0 / np.where(np.abs(x) < 1e-12, 1e-12, x)
    return df.rename(columns={c: f'inv_{c}' for c in TRANSFORM_INV})


# Function guide: build_p3 is responsible for build p3.
# Inputs: X_df, tertile_cuts. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def build_p3(X_df, tertile_cuts):
    """Pipeline 3: 1/x for TRANSFORM_INV + tertile dummies for TERTILE_COLS."""
    df = build_p2(X_df)
    for col in TERTILE_COLS:
        q1, q2 = tertile_cuts[col]
        cats    = pd.cut(df[col], bins=[-np.inf, q1, q2, np.inf],
                         labels=['T1', 'T2', 'T3'])
        dummies = pd.get_dummies(cats, prefix=col, drop_first=True).astype(float)
        df      = pd.concat([df.drop(columns=[col]), dummies], axis=1)
    return df


# Function guide: scale_fit_transform is responsible for perform fit transform.
# Inputs: X_tr, X_te, dummy_suffixes. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def scale_fit_transform(X_tr, X_te, dummy_suffixes=None):
    """RobustScaler fit on X_tr, applied to both. Skips dummy columns."""
    cont = ([c for c in X_tr.columns if not any(c.endswith(s) for s in dummy_suffixes)]
            if dummy_suffixes else list(X_tr.columns))
    scaler = RobustScaler()
    Xtr_s, Xte_s = X_tr.copy(), X_te.copy()
    Xtr_s[cont]  = scaler.fit_transform(X_tr[cont])
    Xte_s[cont]  = scaler.transform(X_te[cont])
    return Xtr_s, Xte_s


# ── Evaluation ─────────────────────────────────────────────────────────────────
# Function guide: evaluate is responsible for evaluate the requested operation.
# Inputs: model, X_te, y_te, label. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def evaluate(model, X_te, y_te, label):
    """Return dict of standard classification metrics + ROC curve arrays."""
    prob = model.predict_proba(np.asarray(X_te))[:, 1]
    pred = model.predict(np.asarray(X_te))
    auc  = roc_auc_score(y_te, prob)
    acc  = accuracy_score(y_te, pred)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1   = f1_score(y_te, pred)
    fpr, tpr, _ = roc_curve(y_te, prob)
    return dict(label=label, auc=auc, acc=acc, sens=sens,
                spec=spec, ppv=ppv, f1=f1, fpr=fpr, tpr=tpr)
