"""
breast_cancer_app.py — Breast Cancer Classification
Two-stage pipeline: (1) LASSO (lambda-1SE, 5-fold CV); (2) L2-regularized linear logistic regression
Wisconsin Breast Cancer Dataset (sklearn) · N=569 · 30 features
Cell Press visual style · Research & educational use only
"""
# Module guide:
# - Role: Define the deployed breast-cancer Shiny application, prediction workflow, and diagnostics.
# - Workflow: Load a precomputed bundle, validate inputs, make predictions, and render evidence-backed figures.
# - Design note: The deployed process must not depend on raw training feature matrices.
from __future__ import annotations

import html
import os
import pickle
import sys
import tempfile
import threading
import time
import warnings
from importlib.metadata import version

import requests
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from shiny import App, reactive, render, ui
import json
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             brier_score_loss)
from sklearn.calibration import calibration_curve as _sklearn_cal_curve

warnings.filterwarnings("ignore")

_CACHE_ROOT = Path(os.environ.get(
    "BC_CACHE_DIR",
    str(Path(__file__).parent / ".cache"),
))

if os.name == "nt":
    _TMP_ROOT = _CACHE_ROOT / "tmp"
    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(_TMP_ROOT)

# ── Palette ───────────────────────────────────────────────────────────────────
# Cell journal figure palette.
CELL_COLORS = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
    "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85",
]
CLR_MAL   = CELL_COLORS[0]
CLR_BEN   = CELL_COLORS[1]
CLR_TRAIN = CELL_COLORS[3]
CLR_1SE   = CELL_COLORS[7]
_MUTED    = "#64748B"
_NAVY     = CELL_COLORS[3]
_EDGE     = CELL_COLORS[5]
_FAINT    = CELL_COLORS[6]
_INK_BC   = "#111111"   # reference rules / misc ink
_AXIS_BC  = "#334155"   # axis spines
_TICK_BC  = "#475569"   # tick marks / tick labels
_LABEL_BC = "#1E293B"   # axis labels
_PANEL_BC = CELL_COLORS[0]  # red panel letters (A, B, ...)

plt.rcParams.update({
    "font.family":         "Arial",
    "font.sans-serif":     ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans", "sans-serif"],
    "font.size":           8.0,
    "axes.titlesize":      9.0,
    "axes.titleweight":    "bold",
    "axes.linewidth":      0.8,
    "axes.spines.top":     True,
    "axes.spines.right":   True,
    "axes.grid":           False,
    "xtick.direction":     "out",
    "ytick.direction":     "out",
    "xtick.major.size":    2.6,
    "ytick.major.size":    2.6,
    "xtick.major.width":   0.7,
    "ytick.major.width":   0.7,
    "xtick.labelsize":     7.5,
    "ytick.labelsize":     7.5,
    "axes.labelsize":      8.5,
    "axes.labelpad":       3,
    "lines.linewidth":     1.0,
    "lines.markersize":    3.2,
    "legend.fontsize":     7.5,
    "legend.frameon":      False,
    "legend.framealpha":   0.9,
    "legend.edgecolor":    _INK_BC,
    "legend.borderpad":    0.35,
    "legend.labelspacing": 0.22,
    "figure.facecolor":    "white",
    "axes.facecolor":      "white",
    "figure.dpi":          300,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "pdf.fonttype":        42,
    "ps.fonttype":         42,
})

# ── 1. Model training ────────────────────────────────────────────────────────
_BUNDLE_PATH = Path(__file__).parent / "breast_cancer_model_bundle.pkl"
BUNDLE_SCHEMA_VERSION = 6
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _source_dir in (
    _REPO_ROOT / "01_data_cleaning",
    _REPO_ROOT / "02_data_preprocessing",
):
    if _source_dir.is_dir():
        sys.path.insert(0, str(_source_dir))


# Function guide: _threshold_metrics computes threshold-based classification metrics.
# Inputs: y_true, p_benign, threshold. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _threshold_metrics(y_true: np.ndarray, p_benign: np.ndarray,
                       threshold: float = 0.5) -> dict[str, float]:
    """Evaluate a decision rule with malignancy as the positive class."""
    y_malignant = (np.asarray(y_true) == 0).astype(int)
    p_malignant = 1.0 - np.asarray(p_benign, dtype=float)
    pred_malignant = (p_malignant >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_malignant, pred_malignant, labels=[0, 1]).ravel()
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "acc": float((tp + tn) / len(y_malignant)),
        "sens": float(tp / (tp + fn)) if tp + fn else 0.0,
        "spec": float(tn / (tn + fp)) if tn + fp else 0.0,
        "ppv": float(tp / (tp + fp)) if tp + fp else 0.0,
        "npv": float(tn / (tn + fn)) if tn + fn else 0.0,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
    }


# Function guide: _youden_threshold is responsible for select a training-only decision threshold.
# Inputs: encoded target labels and predicted benign probabilities.
# Outputs: the threshold that maximizes sensitivity plus specificity minus one.
def _youden_threshold(y_true: np.ndarray, p_benign: np.ndarray) -> float:
    """Select the Youden-index threshold with malignancy as the positive class."""
    y_malignant = (np.asarray(y_true) == 0).astype(int)
    p_malignant = 1.0 - np.asarray(p_benign, dtype=float)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y_malignant, p_malignant)
    finite = np.isfinite(thresholds)
    if not finite.any():
        raise ValueError("The ROC curve did not produce a finite decision threshold.")
    youden = true_positive_rate[finite] - false_positive_rate[finite]
    best_index = int(np.argmax(youden))
    return float(thresholds[finite][best_index])


# Function guide: _lasso_cv_search is responsible for select the LASSO parameter by cross-validation.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _lasso_cv_search(X, y, candidates, cv):
    """Refit preprocessing inside every CV training fold."""
    search = GridSearchCV(
        Pipeline([
            ("scaler", RobustScaler()),
            ("lasso", LogisticRegression(
                penalty="l1", solver="liblinear", max_iter=5000,
                random_state=42)),
        ]),
        {"lasso__C": candidates}, cv=cv, scoring="roc_auc",
        n_jobs=1, error_score="raise", refit=True,
    )
    return search.fit(X, y)


# Function guide: _final_model_cv_search is responsible for select the final logistic parameter by cross-validation.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _final_model_cv_search(X, y, selected_features, decisions, candidates, cv):
    """Tune the final regularized model with fold-local design preprocessing."""
    from eda_and_preprocessing import build_design_matrix

    search = GridSearchCV(
        Pipeline([
            ("design", build_design_matrix(selected_features, decisions)),
            ("scaler", RobustScaler()),
            ("lr", LogisticRegression(
                penalty="l2", solver="lbfgs", max_iter=5000,
                tol=1e-8, random_state=42,
            )),
        ]),
        {"lr__C": candidates}, cv=cv, scoring="roc_auc",
        n_jobs=1, error_score="raise", refit=True,
    )
    return search.fit(X, y)


# Function guide: _hosmer_lemeshow is responsible for compute the Hosmer-Lemeshow calibration statistic.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _hosmer_lemeshow(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    """Hosmer-Lemeshow goodness-of-fit test (decile-of-risk)."""
    q = np.unique(np.percentile(y_prob, np.linspace(0, 100, n_bins + 1)))
    q[-1] += 1e-8
    bins = np.digitize(y_prob, q[1:-1])
    chi2 = 0.0
    n_groups = 0
    for b in np.unique(bins):
        mask = bins == b
        if mask.sum() == 0:
            continue
        n_groups += 1
        n_b  = mask.sum()
        obs  = float(y_true[mask].sum())
        exp  = float(y_prob[mask].sum())
        nobs = n_b - obs
        nexp = n_b - exp
        if exp > 1e-10:
            chi2 += (obs - exp) ** 2 / exp
        if nexp > 1e-10:
            chi2 += (nobs - nexp) ** 2 / nexp
    df = n_groups - 2
    p_value = float(stats.chi2.sf(chi2, df)) if df > 0 else float("nan")
    return float(chi2), p_value, df


# Function guide: _compute_vif is responsible for compute variance inflation factors.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _compute_vif(X_sc: np.ndarray, names: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for i, nm in enumerate(names):
        y_ = X_sc[:, i]
        X_ = np.delete(X_sc, i, axis=1)
        r2 = LinearRegression().fit(X_, y_).score(X_, y_) if X_.shape[1] else 0.0
        result[nm] = 1.0 / max(1e-10, 1.0 - r2)
    return result


# Function guide: _train_model_bundle is responsible for train the model and assemble the deployment bundle.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _train_model_bundle(cleaned_table: pd.DataFrame | None = None) -> dict:
    """Train full pipeline; return bundle dict — no raw DataFrames."""
    _seed = 42
    np.random.seed(_seed)

    _data = load_breast_cancer()
    if cleaned_table is None:
        X_all = pd.DataFrame(_data.data, columns=_data.feature_names)
        y_all = pd.Series(_data.target, name="target")
    else:
        X_all = cleaned_table[list(_data.feature_names)].copy()
        y_all = cleaned_table["target"].copy()
    if not np.isfinite(X_all.to_numpy(dtype=float)).all() or (X_all < 0).any().any():
        raise ValueError("Training features must be finite, non-negative measurements.")
    if set(y_all.unique()) != {0, 1}:
        raise ValueError("Training target must use 0=malignant and 1=benign.")
    _n_malignant = int((y_all == 0).sum())
    _n_benign    = int((y_all == 1).sum())

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=_seed, stratify=y_all)
    _y_tr = y_train.values.astype(int)
    _y_te = y_test.values.astype(int)

    _N_TOTAL = len(y_all)
    _N_TRAIN = len(_y_tr)
    _N_TEST  = len(_y_te)

    from eda_and_preprocessing import screen_variables, build_design_matrix, export_diagnostics
    _decisions, _all_plot_data, _summary = screen_variables(
        X_train, _y_tr, linear_only=True)
    _audit_dir = Path(os.environ.get(
        "BC_TRAINING_OUTPUT_DIR",
        str(_BUNDLE_PATH.parent / ".cache" / "training"),
    ))
    export_diagnostics(X_train, _y_tr, _decisions, _all_plot_data, _summary, _audit_dir)

    # Stage 1 — LASSO with 5-fold CV
    _CS  = np.logspace(-4, 2, 60)
    _cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=_seed)
    _search = _lasso_cv_search(X_train.values, _y_tr, _CS, _cv)
    _cv_scores = np.asarray([
        _search.cv_results_[f"split{i}_test_score"] for i in range(_cv.n_splits)
    ])
    _mean_auc  = _cv_scores.mean(axis=0)
    _se_auc    = _cv_scores.std(axis=0, ddof=1) / np.sqrt(_cv_scores.shape[0])
    _idx_min   = int(np.argmax(_mean_auc))
    _c_min     = _CS[_idx_min]
    _thr_1se   = _mean_auc[_idx_min] - _se_auc[_idx_min]
    _idx_1se   = int(np.where(_mean_auc >= _thr_1se)[0][0])
    _c_1se     = _CS[_idx_1se]
    _nz_min = int((_search.best_estimator_.named_steps["lasso"].coef_[0] != 0).sum())

    # Refit at λ1se
    _pipe_lasso = Pipeline([
        ("scaler", RobustScaler()),
        ("lasso",  LogisticRegression(
            penalty="l1", solver="liblinear", C=_c_1se,
            max_iter=5000, random_state=_seed,
        )),
    ])
    _pipe_lasso.fit(X_train.values, _y_tr)

    _lc        = _pipe_lasso.named_steps["lasso"].coef_[0]
    _sel_mask  = _lc != 0
    _sel_cols  = X_train.columns[_sel_mask].tolist()
    _n_sel     = len(_sel_cols)
    if not _n_sel:
        raise ValueError("The one-standard-error rule selected no predictors.")
    _feat_names = list(X_train.columns)
    _sel_idx   = [_feat_names.index(f) for f in _sel_cols]
    _lasso_coef = {f: float(_lc[i]) for f, i in zip(_sel_cols, np.where(_sel_mask)[0])}

    # Stage 2 - regularized linear logistic regression on processed features
    X_tr_sel = X_train[_sel_cols].values
    X_te_sel = X_test[_sel_cols].values
    _FINAL_CS = np.logspace(-4, 2, 25)
    _final_search = _final_model_cv_search(
        X_tr_sel, _y_tr, _sel_cols, _decisions, _FINAL_CS, _cv)
    _final_cv_scores = np.asarray([
        _final_search.cv_results_[f"split{i}_test_score"]
        for i in range(_cv.n_splits)
    ])
    _final_mean_auc = _final_cv_scores.mean(axis=0)
    _final_se_auc = _final_cv_scores.std(axis=0, ddof=1) / np.sqrt(_final_cv_scores.shape[0])
    _final_c = float(_final_search.best_params_["lr__C"])
    _pipe_lr = _final_search.best_estimator_

    _lr_coef = _pipe_lr.named_steps["lr"].coef_[0]
    _lr_int  = float(_pipe_lr.named_steps["lr"].intercept_[0])
    _design_cols = list(_pipe_lr.named_steps["design"].get_feature_names_out(_sel_cols))
    _design_cols = [name.split("__", 1)[1] for name in _design_cols]
    _lr_coef_map = {f: float(_lr_coef[k]) for k, f in enumerate(_design_cols)}

    _prob_train = _pipe_lr.predict_proba(X_tr_sel)[:, 1]
    _prob_test  = _pipe_lr.predict_proba(X_te_sel)[:, 1]
    _auc_train  = roc_auc_score(_y_tr, _prob_train)
    _auc_test   = roc_auc_score(_y_te, _prob_test)
    _fpr_tr, _tpr_tr, _ = roc_curve(1 - _y_tr, 1 - _prob_train)
    _fpr_te, _tpr_te, _ = roc_curve(1 - _y_te, 1 - _prob_test)
    _train_metrics = _threshold_metrics(_y_tr, _prob_train, 0.5)
    _test_metrics = _threshold_metrics(_y_te, _prob_test, 0.5)
    _youden_thr = _youden_threshold(_y_tr, _prob_train)
    _train_metrics_youden = _threshold_metrics(
        _y_tr, _prob_train, _youden_thr)
    _test_metrics_youden = _threshold_metrics(
        _y_te, _prob_test, _youden_thr)

    _brier_train = float(brier_score_loss(_y_tr, _prob_train))
    _brier_test  = float(brier_score_loss(_y_te, _prob_test))
    _null_brier  = float(np.mean((_y_te - float(_y_tr.mean())) ** 2))

    _cal_frac, _cal_mean = _sklearn_cal_curve(
        1 - _y_te, 1 - _prob_test, n_bins=10, strategy="quantile")
    _hl_chi2, _hl_p, _hl_df = _hosmer_lemeshow(1 - _y_te, 1 - _prob_test)

    _X_tr_design = _pipe_lr.named_steps["design"].transform(X_tr_sel)
    _X_tr_sc_sel = _pipe_lr.named_steps["scaler"].transform(_X_tr_design)
    _vif = _compute_vif(_X_tr_sc_sel, _design_cols)
    _raw_vif = _compute_vif(RobustScaler().fit_transform(X_tr_sel), _sel_cols)
    _influence_summary = _review_influence(
        _X_tr_sc_sel, _y_tr, X_train, _audit_dir, _final_c)

    # LASSO regularization path
    _X_scaled   = _pipe_lasso.named_steps["scaler"].transform(X_train.values)
    _C_PATH     = np.logspace(-4, 2, 120)
    _LOG_C      = np.log10(_C_PATH)
    _PATH_COEFS = np.zeros((len(_C_PATH), X_train.shape[1]))
    for _i, _c in enumerate(_C_PATH):
        _m = LogisticRegression(penalty="l1", solver="liblinear", C=_c,
                                max_iter=5000, random_state=_seed)
        _m.fit(_X_scaled, _y_tr)
        _PATH_COEFS[_i] = _m.coef_[0]

    # Feature ranges (for sliders)
    _feat_ranges: dict[str, tuple[float, float, float]] = {}
    _train_medians: dict[str, float] = {}
    for f in _sel_cols:
        v = X_train[f].values
        _train_medians[f] = float(np.median(X_train[f].values))
        _feat_ranges[f] = (float(v.min()), float(v.max()), _train_medians[f])

    _lrt = {name: _decisions[name]["lrt"] for name in _sel_cols}
    _lrt_plot = {name: _all_plot_data[name] for name in _sel_cols}
    for name, decision in _decisions.items():
        decision["selected"] = name in _sel_cols
        if name in _sel_cols:
            index = _sel_cols.index(name)
            fitted = _pipe_lr.named_steps["design"].named_transformers_[f"feature{index}"]
            if "center" in fitted.named_steps:
                decision["centering_constant"] = float(fitted.named_steps["center"].mean_[0])

    return dict(
        BUNDLE_SCHEMA_VERSION=BUNDLE_SCHEMA_VERSION,
        TRAINING_METADATA={
            "seed": _seed, "test_size": 0.2, "stratified": True,
            "cv_folds": 5, "cv_preprocessing": "fitted within each training fold",
            "positive_class": "malignant (target=0)",
            "probability_arrays": "P(benign), converted for malignant-positive evaluation",
            "train_performance": "apparent; includes feature selection and refitting",
            "final_model": "L2-regularized linear logistic regression after training-set preprocessing",
            "final_model_cv": "5-fold stratified training-only CV with fold-local design preprocessing",
            "final_model_C": _final_c,
            "cv_performance": "LASSO selection and final-model regularization were tuned on training data only",
            "threshold_selection": "Youden index selected on training predictions and locked before test evaluation",
            "default_threshold": _youden_thr,
            "limitations": ["GAM smooth p-values are approximate association tests, not tests of linearity; LRT is reported separately.",
                            "Feature selection and form screening are conditional on the training sample; apparent metrics are optimistic.",
                            "Single internal holdout; no external validation."],
            "versions": {name: version(name) for name in
                         ("numpy", "pandas", "scipy", "scikit-learn")},
        },
        EDA_DECISIONS=_decisions, TRAIN_METRICS_05=_train_metrics,
        TRAIN_METRICS_YOUDEN=_train_metrics_youden,
        DESIGN_COLS=_design_cols, RAW_VIF=_raw_vif,
        INFLUENCE_SUMMARY=_influence_summary,
        TEST_METRICS_05=_test_metrics,
        TEST_METRICS_YOUDEN=_test_metrics_youden,
        YOUDEN_THRESHOLD=_youden_thr,
        pipe_lasso=_pipe_lasso, pipe_lr=_pipe_lr,
        CS=_CS, MEAN_AUC=_mean_auc, SE_AUC=_se_auc,
        FINAL_CS=_FINAL_CS, FINAL_MEAN_AUC=_final_mean_auc,
        FINAL_SE_AUC=_final_se_auc, FINAL_C=_final_c,
        C_MIN=_c_min, C_1SE=_c_1se, THR_1SE=_thr_1se,
        NZ_MIN=_nz_min, N_SEL=_n_sel,
        SEL_MASK=_sel_mask, SEL_COLS=_sel_cols,
        FEAT_NAMES=_feat_names, SEL_IDX=_sel_idx, LASSO_COEF=_lasso_coef,
        LR_COEF=_lr_coef, LR_INT=_lr_int, LR_COEF_MAP=_lr_coef_map,
        PROB_TRAIN=_prob_train, PROB_TEST=_prob_test,
        y_tr=_y_tr, y_te=_y_te,
        AUC_TRAIN=_auc_train, AUC_TEST=_auc_test,
        FPR_TR=_fpr_tr, TPR_TR=_tpr_tr, FPR_TE=_fpr_te, TPR_TE=_tpr_te,
        ACC05=_test_metrics["acc"], SENS05=_test_metrics["sens"],
        SPEC05=_test_metrics["spec"], PPV05=_test_metrics["ppv"],
        NPV05=_test_metrics["npv"], F1_05=_test_metrics["f1"],
        BRIER_TRAIN=_brier_train, BRIER_TEST=_brier_test, NULL_BRIER=_null_brier,
        CAL_FRAC=_cal_frac, CAL_MEAN=_cal_mean,
        HL_CHI2=_hl_chi2, HL_P=_hl_p, HL_DF=_hl_df,
        VIF=_vif, LRT=_lrt, LRT_PLOT_DATA=_lrt_plot,
        LOG_C=_LOG_C, PATH_COEFS=_PATH_COEFS, C_PATH=_C_PATH,
        FEAT_RANGES=_feat_ranges, TRAIN_MEDIANS=_train_medians,
        N_TOTAL=_N_TOTAL, N_TRAIN=_N_TRAIN, N_TEST=_N_TEST,
        n_malignant=_n_malignant, n_benign=_n_benign,
    )


# Function guide: _review_influence is responsible for perform the review influence step.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _review_influence(design, target, raw_train, output_dir, model_c) -> dict:
    """Inspect extremes and approximate ridge-logistic influence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    q1, q3 = raw_train.quantile(0.25), raw_train.quantile(0.75)
    extreme = ((raw_train < q1 - 3 * (q3 - q1))
               | (raw_train > q3 + 3 * (q3 - q1)))
    summary = {
        "extreme_rows": int(extreme.any(axis=1).sum()),
        "extremes_by_variable": extreme.sum().to_dict(),
        "exclusions": 0, "winsorisation": "none",
        "decision": "All values are finite and non-negative; no verified data errors. Retain extreme observations and flag influence for review.",
    }
    try:
        design = np.asarray(design, dtype=float)
        fitted = LogisticRegression(
            penalty="l2", solver="lbfgs", C=float(model_c),
            max_iter=5000, tol=1e-8, random_state=42,
        ).fit(design, np.asarray(target, dtype=int))
        probability = np.clip(
            fitted.predict_proba(design)[:, 1], 1e-8, 1.0 - 1e-8)
        weights = np.clip(probability * (1.0 - probability), 1e-8, None)
        design_with_intercept = np.column_stack([
            np.ones(len(design)), design])
        weighted_design = design_with_intercept * np.sqrt(weights)[:, None]
        penalty = np.eye(design_with_intercept.shape[1]) / max(float(model_c), 1e-8)
        penalty[0, 0] = 0.0
        information = weighted_design.T @ weighted_design + penalty
        covariance = np.linalg.pinv(information, rcond=1e-12)
        leverage = np.einsum(
            "ij,jk,ik->i", weighted_design, covariance, weighted_design)
        leverage = np.clip(leverage, 0.0, 1.0 - 1e-8)
        pearson = (np.asarray(target, dtype=float) - probability) / np.sqrt(weights)
        term_count = design_with_intercept.shape[1]
        cooks = (pearson ** 2 * leverage) / (
            term_count * np.maximum(1.0 - leverage, 1e-8) ** 2)
        finite = np.isfinite(leverage) & np.isfinite(cooks)
        summary.update({
            "influence_available": bool(finite.all()),
            "influence_method": "approximate ridge-logistic leverage and Cook distance",
            "regularization_C": float(model_c),
            "high_leverage_rows": int((leverage > 2 * (design.shape[1] + 1) / len(target)).sum()),
            "high_cooks_rows": int((cooks > 4 / len(target)).sum()),
            "maximum_cooks": float(np.max(cooks[finite])) if finite.any() else None,
        })
        pd.DataFrame({"training_row": raw_train.index, "leverage": leverage,
                      "cooks_distance": cooks}).to_csv(
            output_dir / "influence_diagnostics.csv", index=False)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        summary.update({"influence_available": False, "reason": str(error)})
    (output_dir / "outlier_influence_review.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


# Function guide: _startup_from_scratch is responsible for build deployment state when no bundle exists.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _startup_from_scratch() -> dict:
    """Clean reference data before training; keep row-level exports local."""
    output_dir = Path(os.environ.get(
        "BC_TRAINING_OUTPUT_DIR",
        str(_BUNDLE_PATH.parent / ".cache" / "training"),
    ))
    from clean_dataset import clean_breast_cancer_dataset

    cleaned = clean_breast_cancer_dataset(output_dir)
    cleaned_path = output_dir / "breast_cancer_cleaned.csv"
    bundle = _train_model_bundle(pd.read_csv(cleaned_path))
    decisions = {
        "metadata": bundle["TRAINING_METADATA"],
        "variables": bundle["EDA_DECISIONS"],
    }
    (_BUNDLE_PATH.parent / "preprocessing_decisions.json").write_text(
        json.dumps(decisions, indent=2, allow_nan=False), encoding="utf-8")
    return bundle


# Function guide: _save_bundle is responsible for persist the model bundle.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _save_bundle(bundle: dict) -> None:
    _BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _BUNDLE_PATH.with_suffix(".pkl.tmp")
    with temporary_path.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.replace(_BUNDLE_PATH)


print("=" * 56, flush=True)
print("Breast Cancer App — initialising", flush=True)
print("=" * 56, flush=True)

SEED = 42

if _BUNDLE_PATH.exists() and os.environ.get("BC_FORCE_REBUILD") != "1":
    print("  Loading bundle ...", flush=True)
    with open(_BUNDLE_PATH, "rb") as _f:
        _B = pickle.load(_f)
    if _B.get("BUNDLE_SCHEMA_VERSION") != BUNDLE_SCHEMA_VERSION:
        raise RuntimeError("Outdated model bundle. Run 03_model_training_and_evaluation/train_and_export_model_bundle.py before deployment.")
else:
    print("  Training from scratch ...", flush=True)
    _B = _startup_from_scratch()
    _save_bundle(_B)
    print(f"  Bundle saved: {_BUNDLE_PATH}", flush=True)

pipe_lasso   = _B["pipe_lasso"]
pipe_lr      = _B["pipe_lr"]
CS           = _B["CS"]
MEAN_AUC     = _B["MEAN_AUC"]
SE_AUC       = _B["SE_AUC"]
C_MIN        = _B["C_MIN"]
C_1SE        = _B["C_1SE"]
THR_1SE      = _B["THR_1SE"]
NZ_MIN       = _B["NZ_MIN"]
N_SEL        = _B["N_SEL"]
SEL_MASK     = _B["SEL_MASK"]
SEL_COLS     = _B["SEL_COLS"]
FEAT_NAMES   = _B["FEAT_NAMES"]
SEL_IDX      = _B["SEL_IDX"]
LASSO_COEF   = _B["LASSO_COEF"]
LR_COEF      = _B["LR_COEF"]
LR_INT       = _B["LR_INT"]
LR_COEF_MAP  = _B["LR_COEF_MAP"]
DESIGN_COLS  = _B["DESIGN_COLS"]
EDA_DECISIONS = _B["EDA_DECISIONS"]
PROB_TRAIN   = _B["PROB_TRAIN"]
PROB_TEST    = _B["PROB_TEST"]
y_tr         = _B["y_tr"]
y_te         = _B["y_te"]
AUC_TRAIN    = _B["AUC_TRAIN"]
AUC_TEST     = _B["AUC_TEST"]
FPR_TR       = _B["FPR_TR"]
TPR_TR       = _B["TPR_TR"]
FPR_TE       = _B["FPR_TE"]
TPR_TE       = _B["TPR_TE"]
ACC05        = _B["ACC05"]
SENS05       = _B["SENS05"]
SPEC05       = _B["SPEC05"]
PPV05        = _B["PPV05"]
NPV05        = _B["NPV05"]
F1_05        = _B["F1_05"]
YOUDEN_THRESHOLD = _B["YOUDEN_THRESHOLD"]
DEFAULT_THRESHOLD = YOUDEN_THRESHOLD
TRAIN_METRICS_YOUDEN = _B["TRAIN_METRICS_YOUDEN"]
TEST_METRICS_YOUDEN = _B["TEST_METRICS_YOUDEN"]
BRIER_TRAIN  = _B["BRIER_TRAIN"]
BRIER_TEST   = _B["BRIER_TEST"]
NULL_BRIER   = _B["NULL_BRIER"]
CAL_FRAC     = _B["CAL_FRAC"]
CAL_MEAN     = _B["CAL_MEAN"]
HL_CHI2      = _B["HL_CHI2"]
HL_P         = _B["HL_P"]
HL_DF        = _B["HL_DF"]
VIF          = _B["VIF"]
RAW_VIF      = _B["RAW_VIF"]
LRT          = _B["LRT"]
LRT_PLOT_DATA = _B["LRT_PLOT_DATA"]
LOG_C        = _B["LOG_C"]
PATH_COEFS   = _B["PATH_COEFS"]
C_PATH       = _B["C_PATH"]
FINAL_CS     = _B["FINAL_CS"]
FINAL_MEAN_AUC = _B["FINAL_MEAN_AUC"]
FINAL_SE_AUC = _B["FINAL_SE_AUC"]
FINAL_C      = _B["FINAL_C"]
FEAT_RANGES  = _B["FEAT_RANGES"]
_train_centers = getattr(pipe_lr.named_steps["scaler"], "center_", None)
if "TRAIN_MEDIANS" in _B:
    TRAIN_MEDIANS = {k: float(v) for k, v in _B["TRAIN_MEDIANS"].items()}
elif _train_centers is not None and len(_train_centers) == len(SEL_COLS):
    TRAIN_MEDIANS = {
        feat: float(center) for feat, center in zip(SEL_COLS, _train_centers)
    }
else:
    TRAIN_MEDIANS = {feat: float(FEAT_RANGES[feat][2]) for feat in SEL_COLS}
N_TOTAL      = _B["N_TOTAL"]
N_TRAIN      = _B["N_TRAIN"]
N_TEST       = _B["N_TEST"]
n_malignant  = _B["n_malignant"]
n_benign     = _B["n_benign"]

print(f"  λmin C={C_MIN:.5f} ({NZ_MIN} vars)  λ1se C={C_1SE:.5f} ({N_SEL} vars)",
      flush=True)
print(f"  AUC  train={AUC_TRAIN:.4f}  test={AUC_TEST:.4f}", flush=True)
print(f"  Final ridge C={FINAL_C:.5f}  CV AUC={FINAL_MEAN_AUC.max():.4f}", flush=True)
print(f"  Brier train={BRIER_TRAIN:.4f}  test={BRIER_TEST:.4f}  null={NULL_BRIER:.4f}",
      flush=True)
print(f"  HL test  chi2={HL_CHI2:.2f}  p={HL_P:.3f}  df={HL_DF}", flush=True)
print("  Ready.", flush=True)


TEST_METRICS_05 = _threshold_metrics(y_te, PROB_TEST, 0.5)
TRAIN_METRICS_05 = _threshold_metrics(y_tr, PROB_TRAIN, 0.5)


# Function guide: _validated_predictors is responsible for validate user predictors and convert them to model input.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _validated_predictors(frame: pd.DataFrame) -> np.ndarray:
    """Validate data at the server boundary before model prediction."""
    if frame.empty:
        raise ValueError("The CSV contains no data rows.")
    if len(frame) > 10000:
        raise ValueError("Upload at most 10,000 rows per batch.")
    missing = [name for name in SEL_COLS if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    try:
        selected = frame[SEL_COLS].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("All required feature columns must contain numeric values.") from exc
    values = selected.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Required features must contain finite values with no blanks or missing values.")
    if (values < 0).any():
        raise ValueError("Nuclear morphology measurements cannot be negative.")
    return values


# ── 2. Figure helpers ────────────────────────────────────────────────────────
# Function guide: _sid is responsible for perform the internal operation.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _sid(feat: str) -> str:
    """Slider input ID from feature name."""
    return "f_" + feat.replace(" ", "_")


# Function guide: _P is responsible for perform the internal operation.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _P(ax, letter, fs=10.5):
    """Journal-style panel label: bold Cell red letter above top-left."""
    ax.set_title(letter, fontsize=fs, fontweight="bold",
                 loc="left", pad=3, color=_PANEL_BC)


# Function guide: _style_axis is responsible for perform the style axis step.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _style_axis(ax):
    """Apply closed-box journal-style axes to one matplotlib axis."""
    ax.grid(False)
    ax.tick_params(direction="out", length=2.6, width=0.7, colors=_TICK_BC,
                   labelsize=7.5)
    ax.xaxis.label.set_color(_LABEL_BC)
    ax.yaxis.label.set_color(_LABEL_BC)
    ax.xaxis.label.set_size(8.5)
    ax.yaxis.label.set_size(8.5)
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_color(_AXIS_BC)
        ax.spines[sp].set_linewidth(0.8)


# Function guide: _make_linearity_fig is responsible for create an application figure or UI component.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _make_linearity_fig():
    """A–G: log-odds linearity check (LRT) for each selected feature"""
    n_cols = 4 if len(SEL_COLS) > 4 else max(1, len(SEL_COLS))
    n_rows = int(np.ceil(len(SEL_COLS) / n_cols))
    fig = plt.figure(figsize=(7.0, 3.5))
    gs  = fig.add_gridspec(n_rows, n_cols, wspace=0.42, hspace=0.58)

    for k, feat in enumerate(SEL_COLS):
        r, c = divmod(k, n_cols)
        ax   = fig.add_subplot(gs[r, c])
        _style_axis(ax)
        mids   = LRT_PLOT_DATA[feat]["mids"]
        logits = LRT_PLOT_DATA[feat]["logits"]
        ax.scatter(mids, logits, color=CLR_TRAIN, s=22, zorder=3,
                   alpha=0.82, edgecolors="none")
        ax.fill_between(LRT_PLOT_DATA[feat]["grid"], LRT_PLOT_DATA[feat]["gam_lower"],
                        LRT_PLOT_DATA[feat]["gam_upper"], color=CLR_BEN, alpha=0.2)
        ax.plot(LRT_PLOT_DATA[feat]["grid"], LRT_PLOT_DATA[feat]["gam_logit"],
                color=CLR_MAL, lw=1)
        ax.plot(LRT_PLOT_DATA[feat]["grid"],
                LRT_PLOT_DATA[feat]["linear_logit"],
                color=_EDGE, lw=1.0, ls="--")
        r_lrt  = LRT[feat]
        linear = r_lrt["linear"]
        t_col  = _NAVY if linear else CLR_1SE
        verdict = EDA_DECISIONS[feat]["functional_form"]
        ax.text(0.97, 0.97, f"LRT p={r_lrt['p']:.3f}\nedf={EDA_DECISIONS[feat]['gam_edf']:.2f}; {verdict}",
                transform=ax.transAxes, fontsize=6.8, va="top", ha="right",
                color=t_col, style="italic",
                bbox=dict(facecolor="white", edgecolor="none",
                           alpha=0.75, pad=0.5))
        ax.set_xlabel(feat.replace("worst ", "").title(), labelpad=2)
        ax.set_ylabel("Malignancy log-odds", labelpad=2)
        _P(ax, chr(ord("A") + k))
        ax.tick_params(labelsize=8.0)

    for k in range(len(SEL_COLS), n_rows * n_cols):
        r, c = divmod(k, n_cols)
        fig.add_subplot(gs[r, c]).set_visible(False)
    fig.tight_layout(pad=0.55)
    return fig


# Keep offline report renderers separate from responsive application views.
import visualizations as charts

_PERF_SRC = charts.png(charts.roc_figure(globals()))
_CALIBRATION_SRC = charts.png(charts.calibration_figure(globals()))
_CV_SRC = charts.png(charts.cv_figure(globals()))
_VIF_SRC = charts.png(charts.vif_figure(globals()))
print("  Static figures rendered.", flush=True)


# ── 3. Analytics: global visitor map ─────────────────────────────────────────
# Runtime integrations are optional. The app should still start when these
# variables are absent, with visit logging and geo-enrichment disabled.
_IPINFO_TOKEN = os.environ.get("IPINFO_TOKEN", "").strip()
_SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
_SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip()
_ANALYTICS_CONFIGURED = bool(
    _SUPABASE_URL and _SUPABASE_KEY
)
_ANALYTICS_STATE = {
    "error": None,
    "mode": "remote" if _ANALYTICS_CONFIGURED else "local",
    "retry_at": 0.0,
}
_APP_NAME_BC  = "breast-cancer-classifier"
_ANALYTICS_RETRY_SECONDS = 60.0
_LOCAL_VISITS_LIMIT_BC = 500
_LOCAL_VISITS_BC: list[dict] = []
_LOCAL_VISITS_LOCK_BC = threading.Lock()

_COUNTRY_NAMES = {
    "AF":"Afghanistan","AL":"Albania","DZ":"Algeria","AR":"Argentina",
    "AU":"Australia","AT":"Austria","BD":"Bangladesh","BE":"Belgium",
    "BR":"Brazil","BG":"Bulgaria","CA":"Canada","CL":"Chile",
    "CN":"China","CO":"Colombia","HR":"Croatia","CZ":"Czech Republic",
    "DK":"Denmark","EG":"Egypt","FI":"Finland","FR":"France",
    "DE":"Germany","GH":"Ghana","GR":"Greece","HK":"Hong Kong",
    "HU":"Hungary","IN":"India","ID":"Indonesia","IR":"Iran",
    "IQ":"Iraq","IE":"Ireland","IL":"Israel","IT":"Italy",
    "JP":"Japan","JO":"Jordan","KZ":"Kazakhstan","KE":"Kenya",
    "KR":"South Korea","KW":"Kuwait","LB":"Lebanon","MY":"Malaysia",
    "MX":"Mexico","MA":"Morocco","NL":"Netherlands","NZ":"New Zealand",
    "NG":"Nigeria","NO":"Norway","PK":"Pakistan","PE":"Peru",
    "PH":"Philippines","PL":"Poland","PT":"Portugal","QA":"Qatar",
    "RO":"Romania","RU":"Russia","SA":"Saudi Arabia","SG":"Singapore",
    "ZA":"South Africa","ES":"Spain","SE":"Sweden","CH":"Switzerland",
    "TW":"Taiwan","TH":"Thailand","TN":"Tunisia","TR":"Turkey",
    "UA":"Ukraine","AE":"United Arab Emirates","GB":"United Kingdom",
    "US":"United States","VN":"Vietnam","YE":"Yemen","ZW":"Zimbabwe",
}


# Function guide: _country_name is responsible for perform the country name step.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _country_name(code: str) -> str:
    return _COUNTRY_NAMES.get((code or "").upper(), code or "")


# Function guide: _lookup_ip_location is responsible for perform the lookup ip location step.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _lookup_ip_location(ip: str):
    if not ip or ip in ("127.0.0.1", "::1"):
        return None, None, None, None
    try:
        if _IPINFO_TOKEN:
            response = requests.get(
                f"https://ipinfo.io/{ip}/json",
                params={"token": _IPINFO_TOKEN},
                timeout=4,
            )
            if response.status_code != 200:
                return None, None, None, None
            data = response.json()
            loc = data.get("loc", "")
            lat, lon = map(float, loc.split(",")) if loc else (None, None)
            return data.get("country"), data.get("city"), lat, lon

        response = requests.get(f"https://ipwho.is/{ip}", timeout=4)
        data = response.json() if response.status_code == 200 else {}
        if not data.get("success"):
            return None, None, None, None
        return (
            data.get("country_code"),
            data.get("city"),
            data.get("latitude"),
            data.get("longitude"),
        )
    except Exception:
        return None, None, None, None


# Function guide: _sb_headers is responsible for perform the sb headers step.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _sb_headers():
    return {
        "apikey":        _SUPABASE_KEY,
        "Authorization": f"Bearer {_SUPABASE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=minimal",
    }


# One shared session per process: connection pooling + shorter TLS handshakes
# for the (otherwise) many small analytics requests.
_HTTP_SESSION = requests.Session()
_HTTP_SESSION.headers.update({"User-Agent": _APP_NAME_BC})


# Function guide: _map_coordinate_bc is responsible for perform the map coordinate bc step.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _map_coordinate_bc(value, lower, upper):
    """Return a finite coordinate inside the requested range."""
    try:
        coordinate = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(coordinate) or not lower <= coordinate <= upper:
        return None
    return coordinate


# Function guide: _normalise_visit_bc is responsible for perform the normalise visit bc step.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _normalise_visit_bc(visit):
    if not isinstance(visit, dict):
        return None
    lat = _map_coordinate_bc(visit.get("lat"), -90.0, 90.0)
    lon = _map_coordinate_bc(visit.get("lon"), -180.0, 180.0)
    if lat is None or lon is None:
        return None
    return {
        "country": str(visit.get("country") or "").strip(),
        "city": str(visit.get("city") or "").strip(),
        "lat": lat,
        "lon": lon,
    }


# Function guide: _normalise_visits_bc is responsible for perform the normalise visits bc step.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _normalise_visits_bc(visits):
    normalised = (_normalise_visit_bc(visit) for visit in (visits or []))
    return [visit for visit in normalised if visit is not None]


# Function guide: _record_local_visit_bc is responsible for perform the record local visit bc step.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _record_local_visit_bc(country, city, lat, lon):
    visit = _normalise_visit_bc({
        "country": country,
        "city": city,
        "lat": lat,
        "lon": lon,
    })
    if visit is None:
        return
    with _LOCAL_VISITS_LOCK_BC:
        _LOCAL_VISITS_BC.append(visit)
        del _LOCAL_VISITS_BC[:-_LOCAL_VISITS_LIMIT_BC]


# Function guide: _local_visits_bc is responsible for perform the local visits bc step.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _local_visits_bc():
    with _LOCAL_VISITS_LOCK_BC:
        return [dict(visit) for visit in _LOCAL_VISITS_BC]


# Function guide: _log_visit_bc is responsible for perform the log visit bc step.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _log_visit_bc(country, city, lat, lon):
    if not (_SUPABASE_URL and _SUPABASE_KEY):
        return
    if lat is None or lon is None:
        return
    try:
        _HTTP_SESSION.post(
            f"{_SUPABASE_URL}/rest/v1/visits",
            headers=_sb_headers(),
            json={"app_name": _APP_NAME_BC, "country": country,
                  "city": city, "lat": lat, "lon": lon},
            timeout=5,
        )
    except Exception:
        pass


# Function guide: _fetch_visits_bc is responsible for perform the fetch visits bc step.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _fetch_visits_bc():
    if not (_SUPABASE_URL and _SUPABASE_KEY):
        _ANALYTICS_STATE["error"] = "configuration"
        _ANALYTICS_STATE["mode"] = "local"
        return _local_visits_bc()
    if time.monotonic() < _ANALYTICS_STATE["retry_at"]:
        return _local_visits_bc()
    try:
        hdrs = {k: v for k, v in _sb_headers().items() if k != "Prefer"}
        r = _HTTP_SESSION.get(
            f"{_SUPABASE_URL}/rest/v1/visits",
            headers=hdrs,
            params={"app_name": f"eq.{_APP_NAME_BC}",
                    "select": "country,city,lat,lon",
                    "limit": "2000"},
            timeout=5,
        )
        if r.status_code == 200:
            data = r.json()
            _ANALYTICS_STATE["error"] = None
            _ANALYTICS_STATE["mode"] = "remote"
            _ANALYTICS_STATE["retry_at"] = 0.0
            return _normalise_visits_bc(data if isinstance(data, list) else [])
        _ANALYTICS_STATE["error"] = f"http-{r.status_code}"
        _ANALYTICS_STATE["mode"] = "local"
        _ANALYTICS_STATE["retry_at"] = (
            time.monotonic() + _ANALYTICS_RETRY_SECONDS
        )
        return _local_visits_bc()
    except Exception:
        _ANALYTICS_STATE["error"] = "connection"
        _ANALYTICS_STATE["mode"] = "local"
        _ANALYTICS_STATE["retry_at"] = (
            time.monotonic() + _ANALYTICS_RETRY_SECONDS
        )
        return _local_visits_bc()


_WORLD_GEO_PATH_BC = Path(__file__).parent / "world.geojson"
_WORLD_GEO_BC = None
_WORLD_PATCHES_BC = None


# Function guide: _load_world_geo_bc is responsible for perform the load world geo bc step.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _load_world_geo_bc():
    global _WORLD_GEO_BC
    if _WORLD_GEO_BC is None and _WORLD_GEO_PATH_BC.exists():
        with open(_WORLD_GEO_PATH_BC, encoding="utf-8") as _f:
            _WORLD_GEO_BC = json.load(_f)
    return _WORLD_GEO_BC


# Function guide: _world_patches_bc is responsible for perform the world patches bc step.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _world_patches_bc():
    """World-outline matplotlib Polygons, parsed once per process."""
    global _WORLD_PATCHES_BC
    if _WORLD_PATCHES_BC is not None:
        return _WORLD_PATCHES_BC
    from matplotlib.patches import Polygon

    geo = _load_world_geo_bc()
    patches = []
    if geo:
        for feat in geo.get("features", []):
            geom = feat.get("geometry") or {}
            gtype = geom.get("type", "")
            coords = geom.get("coordinates", [])
            try:
                if gtype == "Polygon":
                    pts = np.array(coords[0])[:, :2]
                    patches.append(Polygon(pts, closed=True))
                elif gtype == "MultiPolygon":
                    for poly in coords:
                        pts = np.array(poly[0])[:, :2]
                        patches.append(Polygon(pts, closed=True))
            except Exception:
                pass
    _WORLD_PATCHES_BC = patches
    return patches


# Function guide: _make_visit_map_bc is responsible for create an application figure or UI component.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _make_visit_map_bc(visits, user_lat=None, user_lon=None,
                       analytics_mode="remote"):
    from matplotlib.collections import PatchCollection

    valid = _normalise_visits_bc(visits)
    point_groups = {}
    for visit in valid:
        key = (
            visit["city"], visit["country"],
            round(visit["lat"], 2), round(visit["lon"], 2),
        )
        point_groups[key] = point_groups.get(key, 0) + 1
    points = sorted(
        ((*key, count) for key, count in point_groups.items()),
        key=lambda item: (-item[4], item[0], item[1]),
    )

    user_lat = _map_coordinate_bc(user_lat, -90.0, 90.0)
    user_lon = _map_coordinate_bc(user_lon, -180.0, 180.0)

    fig, ax = plt.subplots(figsize=(6.6, 3.15), facecolor="white")
    ax.set_facecolor("white")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-70, 85)
    _style_axis(ax)

    patches = _world_patches_bc()
    if patches:
        pc = PatchCollection(
            patches, facecolor=_FAINT, edgecolor=_EDGE,
            linewidth=0.4, alpha=0.35, zorder=1,
        )
        ax.add_collection(pc)

    if points:
        lons = [point[3] for point in points]
        lats = [point[2] for point in points]
        sizes = [min(80.0, 22.0 + 12.0 * np.sqrt(point[4]))
                 for point in points]
        ax.scatter(lons, lats, s=sizes, color=CLR_BEN, alpha=0.85,
                   marker="o", zorder=3, linewidths=0,
                   label=f"Visitors (n={len(valid)})")
        labelled_points = [point for point in points if point[0]][:4]
        for i, (name, _country, lat_c, lon_c, count) in enumerate(labelled_points):
            if lon_c < -100:
                dx = 5
            elif lon_c > 105:
                dx = -4
            else:
                dx = 5 if i % 2 == 0 else -4
            dy = 4 if i % 3 == 0 else (-7 if i % 3 == 1 else 9)
            label = f"{name} ({count})" if count > 1 else name
            ax.annotate(label, xy=(lon_c, lat_c),
                        xytext=(dx, dy), textcoords="offset points",
                        ha="left" if dx >= 0 else "right",
                        fontsize=7.0, color=_NAVY, zorder=5, clip_on=False)

    if user_lat is not None and user_lon is not None:
        ax.scatter([user_lon], [user_lat], s=32, color=CLR_MAL,
                   marker="o", zorder=4, linewidths=0, label="You")

    if not points and user_lat is None:
        empty_label = (
            "Waiting for a live visitor"
            if analytics_mode == "local" else "No mapped visits yet"
        )
        ax.text(
            0.5, 0.055, empty_label,
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7.5, color=_MUTED, zorder=5,
        )

    ax.set_xticks([])
    ax.set_yticks([])

    if points or user_lat is not None:
        ax.legend(fontsize=7.5, loc="lower left", frameon=False)

    fig.tight_layout(pad=0.6)
    return fig


# ── 4. CSS ───────────────────────────────────────────────────────────────────
# Keep the active interface theme in one deployable stylesheet.
_CSS = (Path(__file__).parent / "compact_theme.css").read_text(encoding="utf-8")

_TEXT_SCALE_JS = """
(() => {
  const storageKey = "medictio-text-scale-v2";
  const allowed = new Set(["1", "1.125", "1.25"]);
  const applyScale = (value) => {
    const scale = allowed.has(value) ? value : "1.125";
    document.documentElement.style.setProperty("--app-scale", scale);
    return scale;
  };
  document.addEventListener("DOMContentLoaded", () => {
    const control = document.getElementById("text-scale");
    if (!control) return;
    let saved = "1";
    try { saved = localStorage.getItem(storageKey) || saved; } catch (_) {}
    control.value = applyScale(saved);
    control.addEventListener("change", () => {
      const value = applyScale(control.value);
      try { localStorage.setItem(storageKey, value); } catch (_) {}
    });
  });
})();
"""


# Function guide: _slider_step is responsible for perform the slider step step.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _slider_step(lo: float, hi: float) -> float:
    rng = hi - lo
    if rng == 0:
        return 0.001
    mag = 10 ** (np.floor(np.log10(rng)) - 2)
    return round(float(mag), 10)


# Function guide: _make_inputs is responsible for create the application input controls.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _make_inputs():
    items = []
    for f in SEL_COLS:
        lo, hi, _ = FEAT_RANGES[f]
        med = TRAIN_MEDIANS[f]
        step     = _slider_step(lo, hi)
        decimals = max(0, -int(np.floor(np.log10(step)))) if step > 0 else 4
        items.append(
            ui.input_numeric(_sid(f), f.title(),
                             value=round(med, decimals),
                             min=round(lo, decimals),
                             max=round(hi, decimals),
                             step=step)
        )
    return items


# Function guide: _overview_tile is responsible for perform the overview tile step.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _overview_tile(label: str, value: str, detail: str, accent: str) -> ui.Tag:
    return ui.tags.div(
        ui.tags.div(label, class_="overview-label"),
        ui.tags.div(value, class_="overview-value"),
        ui.tags.div(detail, class_="overview-detail"),
        class_=f"overview-tile {accent}",
    )


# Function guide: _section_head is responsible for perform the section head step.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _section_head(_kicker: str, title: str, copy: str) -> ui.Tag:
    return ui.tags.div(
        ui.tags.h4(title, class_="section-title"),
        ui.tags.p(copy, class_="section-copy"),
        class_="section-head",
    )


# Function guide: _hero_metadata is responsible for perform the hero metadata step.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _hero_metadata() -> ui.Tag:
    return ui.tags.div(
        ui.tags.span(ui.tags.strong(f"{N_TOTAL}"), " cases", class_="hero-meta-item"),
        ui.tags.span(
            ui.tags.strong(f"{N_TRAIN} / {N_TEST}"),
            " train / test",
            class_="hero-meta-item",
        ),
        ui.tags.span(
            ui.tags.strong("80 / 20"),
            f" outcome-stratified, seed {SEED}",
            class_="hero-meta-item",
        ),
        ui.tags.span(
            ui.tags.strong(f"{n_malignant} / {n_benign}"),
            " malignant / benign",
            class_="hero-meta-item",
        ),
        ui.tags.span(
            ui.tags.strong(f"{N_SEL} of {len(FEAT_NAMES)}"),
            " model inputs",
            class_="hero-meta-item",
        ),
        ui.tags.span(
            "Training-only preprocessing; held-out test evaluation.",
            class_="hero-meta-note",
        ),
        class_="hero-meta",
        **{"aria-label": "Dataset and split overview"},
    )


# Function guide: _note_block is responsible for perform the note block step.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _note_block(title: str, copy: str) -> ui.Tag:
    return ui.tags.div(
        ui.tags.div(title, class_="note-title"),
        ui.tags.p(copy, class_="note-copy"),
        class_="note-block",
    )


# Function guide: _input_panel is responsible for perform the input panel step.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _input_panel():
    fields = _make_inputs()
    return ui.tags.aside(
        ui.tags.h4("Sample measurements", class_="input-title"),
        ui.tags.p("Defaults are based on training-set medians.",
                  class_="input-copy"),
        ui.tags.div("Size", class_="input-group-label"),
        ui.tags.div(fields[0], fields[2], class_="input-grid"),
        ui.tags.div("Texture & shape", class_="input-group-label"),
        ui.tags.div(fields[1], *fields[3:], class_="input-grid"),
        ui.input_action_button("submit", "Run Prediction", class_="btn btn-primary w-100"),
        ui.tags.p("Inputs are applied when you run the prediction.", class_="input-hint"),
        class_="input-panel", **{"aria-label": "Prediction inputs"},
    )


# Function guide: _view_notes is responsible for perform the view notes step.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _view_notes(*notes):
    return ui.tags.details(
        ui.tags.summary("Reading guide"),
        ui.tags.div(*notes, class_="note-grid"),
        class_="detail-panel view-notes",
        open=True,
    )


app_ui = ui.page_fluid(
    ui.tags.style(_CSS),
    ui.tags.script(_TEXT_SCALE_JS),
    ui.tags.div(
        ui.tags.div(
            ui.tags.h1("Breast Cancer Classification", class_="page-title"),
            ui.tags.p(
                "Wisconsin Diagnostic Breast Cancer | LASSO selection + L2 logistic regression",
                class_="page-subtitle",
            ),
            _hero_metadata(),
            class_="hero-copy",
        ),
        ui.tags.div(
            ui.tags.span("Research use only", class_="app-status"),
            ui.tags.label("Text size", class_="text-scale-label", **{"for": "text-scale"}),
            ui.tags.select(
                ui.tags.option("Standard", value="1", selected="selected"),
                ui.tags.option("Comfortable", value="1.125"),
                ui.tags.option("Large", value="1.25"),
                id="text-scale",
                class_="text-scale-select",
                **{"aria-label": "Interface text size"},
            ),
            class_="hero-actions",
        ),
        class_="hero-banner",
    ),

    ui.navset_tab(
        ui.nav_panel(
            "Prediction",
            _section_head(
                "Individual prediction",
                "Individual prediction",
                "Enter seven measurements to estimate malignancy probability. Compare inputs with training-set medians below.",
            ),
            ui.tags.div(
                _input_panel(),
                ui.tags.div(
                    ui.tags.div(
                        ui.output_ui("pred_chips"),
                        **{"aria-live": "polite", "aria-atomic": "true"},
                    ),
                    ui.card(
                        ui.card_header("Measurements and training reference"),
                        ui.output_ui("feat_table"),
                    ),
                    class_="prediction-results",
                ),
                class_="prediction-workspace",
            ),
            _view_notes(
                _note_block(
                    "Classification rule",
                    f"The default rule classifies a case as malignant when P(malignant) >= {DEFAULT_THRESHOLD:.3f}. The Decision Threshold tab shows how other cutoffs change performance.",
                ),
                _note_block(
                    "Feature context",
                    "Each entered value is compared with the corresponding training-set median to show the direction and magnitude of the difference.",
                ),
                _note_block(
                    "Intended use",
                    "This tool is intended for research and education. It illustrates model behavior and must not replace clinical assessment.",
                ),
            ),
            ui.tags.details(
                ui.tags.summary("Visitor activity"),
                ui.layout_columns(
                ui.card(
                    ui.card_header("Global Visitor Map"),
                    ui.tags.div(
                        ui.output_ui("visit_map"),
                        class_="plot-frame plot-map",
                    ),
                    class_="equal-card",
                ),
                ui.card(
                    ui.card_header("Visit Statistics"),
                    ui.tags.div(ui.output_ui("visit_stats"), class_="result-frame result-map"),
                    class_="equal-card",
                ),
                col_widths=[8, 4],
                ),
                class_="detail-panel",
            ),
            ui.tags.p(
                f"Predictions use {N_SEL} LASSO-selected features and RobustScaler parameters "
                "fitted on the training set. For research and educational use only.",
                class_="disclaimer",
            ),
        ),

        ui.nav_panel(
            "Batch Prediction",
            _section_head(
                "Batch prediction",
                "Score a cohort from a CSV file",
                "Upload a CSV file, verify the required columns, review the first 20 results, and download the complete scored dataset.",
            ),
            _view_notes(
                _note_block(
                    "CSV schema",
                    "The uploaded file must contain every retained feature column required by the deployed model.",
                ),
                _note_block(
                    "Inline review",
                    "The preview shows the first 20 rows so column and value issues can be identified before download.",
                ),
                _note_block(
                    "Downloaded output",
                    "The downloaded file retains the original columns and adds probabilities for the malignant and benign classes, plus the predicted class.",
                ),
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Upload CSV"),
                    ui.input_file("batch_file", "CSV file",
                                  accept=[".csv"],
                                  placeholder="Choose CSV file…"),
                    ui.tags.p("Required column names", class_="input-group-label"),
                    ui.tags.ul(
                        *[ui.tags.li(ui.tags.code(c)) for c in SEL_COLS],
                        class_="batch-schema",
                    ),
                    ui.tags.div(
                        ui.output_ui("batch_status"),
                        **{"aria-live": "polite", "aria-atomic": "true"},
                    ),
                    ui.download_button("batch_download", "Download Predictions",
                                       class_="btn btn-primary btn-sm batch-download"),
                ),
                ui.card(
                    ui.card_header("Prediction Preview (first 20 rows)"),
                    ui.output_ui("batch_table"),
                ),
                col_widths=[4, 8],
            ),
        ),

        ui.nav_panel(
            "Decision Threshold",
            _section_head(
                "Threshold analysis",
                "Explore classification trade-offs",
                "Use the training-set score distribution to see how alternative probability cutoffs change sensitivity, specificity, and classification errors.",
            ),
            _view_notes(
                _note_block(
                    "Why it matters",
                    "Changing the threshold trades sensitivity against specificity. The confusion matrix and performance metrics update together.",
                ),
                _note_block(
                    "Reference set",
                    "All calculations in this tab use training-set predictions to isolate the effect of the cutoff.",
                ),
                _note_block(
                    "Default baseline",
                    f"Metrics elsewhere in the app use the training-set Youden threshold ({DEFAULT_THRESHOLD:.3f}) unless stated otherwise.",
                ),
            ),
            ui.layout_columns(
                ui.card(
                    ui.tags.div(
                        ui.output_ui("hist_img"),
                        class_="chart-image chart-square",
                    ),
                    class_="equal-card",
                ),
                ui.card(
                    ui.card_header("Threshold Selection and Confusion Matrix"),
                    ui.input_slider("threshold", "Decision Threshold",
                    min=0.001, max=0.999, value=DEFAULT_THRESHOLD, step=0.001),
                    ui.tags.p(
                        "Classified as malignant when P(malignant) >= threshold; "
                        "benign otherwise.",
                        class_="threshold-caption",
                    ),
                    ui.output_ui("cm_display"),
                    class_="equal-card",
                ),
                col_widths=[7, 5],
            ),
            ui.card(
                ui.card_header("Performance at the Selected Threshold"),
                ui.output_ui("metrics_table"),
            ),
        ),

        ui.nav_panel(
            "Model Evaluation",
            _section_head(
                "Model evaluation",
                "Selection, performance, calibration, and diagnostics",
                "Review held-out performance, selected coefficients, and the diagnostic figures stored with the deployed model bundle.",
            ),
            ui.tags.div(
                _overview_tile(
                    "Cohort",
                    f"{N_TOTAL}",
                    f"{n_benign} benign / {n_malignant} malignant cases",
                    "accent-blue cohort-tile",
                ),
                _overview_tile(
                    "Selected features",
                    f"{N_SEL} of {len(FEAT_NAMES)}",
                    "selected using the λ₁ₛₑ rule",
                    "accent-teal",
                ),
                _overview_tile(
                    "Test discrimination",
                    f"{AUC_TEST:.3f}",
                    f"training AUC {AUC_TRAIN:.3f} / test AUC {AUC_TEST:.3f}",
                    "accent-navy",
                ),
                _overview_tile(
                    "Test Brier score",
                    f"{BRIER_TEST:.3f}",
                    f"HL p={HL_P:.3f} · null Brier {NULL_BRIER:.3f}",
                    "accent-salmon",
                ),
                class_="overview-grid evaluation-overview",
            ),
            ui.navset_pill(
                ui.nav_panel(
                    "Performance",
                    ui.tags.div(
                        ui.card(ui.tags.div(ui.output_ui("fig_perf"), class_="chart-image chart-square")),
                        ui.card(ui.tags.div(ui.output_ui("fig_calibration"), class_="chart-image chart-square")),
                        class_="chart-grid",
                    ),
                    ui.output_ui("perf_metrics_table"),
                ),
                ui.nav_panel(
                    "Feature selection",
                    ui.tags.div(
                        ui.input_select("path_feature", "Inspect a retained input",
                                        {f: f.title() for f in SEL_COLS}, selected=SEL_COLS[0]),
                        class_="chart-toolbar",
                    ),
                    ui.tags.div(
                        ui.card(
                            ui.tags.div(ui.output_ui("fig_feat_sel"), class_="chart-image chart-square"),
                        ),
                        ui.card(ui.tags.div(ui.output_ui("fig_cv"), class_="chart-image chart-square")),
                        class_="chart-grid",
                    ),
                ),
                ui.nav_panel(
                    "Diagnostics",
                    ui.tags.div(
                        ui.tags.div(
                            ui.input_select("linearity_feature", "Inspect a retained input",
                                            {f: f.title() for f in SEL_COLS}, selected=SEL_COLS[0]),
                            class_="chart-toolbar",
                        ),
                        ui.tags.div(
                            ui.card(
                                ui.tags.div(ui.output_ui("fig_linearity"), class_="chart-image chart-square"),
                            ),
                            ui.card(ui.tags.div(ui.output_ui("fig_vif"), class_="chart-image chart-square")),
                            class_="chart-grid",
                        ),
                        class_="diagnostics-panel",
                    ),
                ),
                id="evaluation_view",
            ),
            ui.tags.details(
                ui.tags.summary(f"Inspect all {len(DESIGN_COLS)} fitted coefficients"),
                ui.output_ui("coef_table"), class_="detail-panel",
            ),
        ),

        ui.nav_panel(
            "Methods",
            _section_head(
                "Reproducibility",
                "Bundle, diagnostics, and modeling assumptions",
                "All preprocessing, feature retention, scaling, and diagnostics shown here are loaded from the saved deployment bundle.",
            ),
            ui.output_ui("methods_panel"),
        ),

        id="main_tab",
    ),

    title="Breast Cancer Dashboard",
    class_="dashboard-shell",
)


# ── 5. Server ────────────────────────────────────────────────────────────────

# Function guide: server is responsible for register the reactive Shiny server logic.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def server(input, output, session):

    # ── Visit logging ─────────────────────────────────────────────────────────
    _user_loc = {
        "country": "", "city": "", "lat": None, "lon": None,
    }
    try:
        _hdrs = session.http_conn.headers
        _ip = (
            _hdrs.get("x-forwarded-for") or
            _hdrs.get("x-real-ip") or ""
        ).split(",")[0].strip()
    except Exception:
        _ip = ""

    # Function guide: _do_log is responsible for perform the do log step.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def _do_log(ip: str) -> None:
        country, city, lat, lon = _lookup_ip_location(ip)
        visit = _normalise_visit_bc({
            "country": country, "city": city, "lat": lat, "lon": lon,
        })
        if visit is not None:
            _user_loc.update(visit)
            _record_local_visit_bc(**visit)
        _log_visit_bc(country, city, lat, lon)

    threading.Thread(target=_do_log, args=(_ip,), daemon=True).start()

    # Auto-refresh the map/stats a few times after load so the visitor's own
    # just-logged visit (written asynchronously above) appears without a
    # manual page refresh. Bounded to a handful of ticks, then stops.
    _refresh_tick = reactive.value(0)
    _refresh_n    = {"c": 0}

    @reactive.effect
    # Function guide: _auto_refresh is responsible for perform the auto refresh step.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def _auto_refresh():
        _refresh_n["c"] += 1
        if _refresh_n["c"] <= 2:
            reactive.invalidate_later(3)
            _refresh_tick.set(_refresh_n["c"])

    @reactive.calc
    # Function guide: _visits is responsible for perform the internal operation.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def _visits():
        _refresh_tick.get()  # re-fetch when the tick advances
        return _fetch_visits_bc()

    @render.ui
    # Function guide: visit_map is responsible for perform the visit map step.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def visit_map():
        fig = _make_visit_map_bc(
            _visits(), _user_loc["lat"], _user_loc["lon"],
            _ANALYTICS_STATE["mode"],
        )
        fig.set_size_inches(7, 3.5)
        fig.axes[0].set_title("Global visitor activity", loc="left", fontsize=10, fontweight="bold")
        charts.finish(fig, "Source: app visit logs. Locations are approximate; unavailable locations are omitted.")
        return ui.tags.img(src=charts.png(fig), alt="Aggregate visitor locations on a world map")

    @render.ui
    # Function guide: visit_stats is responsible for perform the visit stats step.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def visit_stats():
        from collections import Counter
        visits = _normalise_visits_bc(_visits())
        total  = len(visits)
        counts = Counter(
            f"{v.get('city')}, {_country_name(v.get('country'))}" if v.get("country") else v.get("city")
            for v in visits if v.get("city")
        )
        top    = counts.most_common(10)
        rows   = "".join(
            f"<tr><td>{i+1}. {html.escape(str(c))}</td>"
            f"<td class='num'>{n}</td></tr>"
            for i, (c, n) in enumerate(top)
        )
        empty_message = (
            "Waiting for the first mappable visitor."
            if not total else "No city labels are available yet."
        )
        empty = (
            '<p class="visit-stats-empty">'
            f"{empty_message}</p>"
            if not top else ""
        )
        scope_note = ""
        if _ANALYTICS_STATE["mode"] == "local":
            scope_note = (
                '<p class="visit-scope-note">Live locations for this running app '
                'instance. Persistent history is currently unavailable.</p>'
            )
        return ui.HTML(f"""
<div class="visit-stats-shell">
  <div class="visit-total">
    <div class="visit-total-value">{total}</div>
    <div class="visit-total-label">Total Visits</div>
  </div>
  <table class="tbl">
    <thead><tr>
      <th scope="col">City</th>
      <th scope="col">Visits</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
  {empty}
  {scope_note}
</div>
""")

    # ── Individual prediction ─────────────────────────────────────────────────

    @reactive.calc
    # Function guide: _submitted_inputs is responsible for perform the submitted inputs step.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def _submitted_inputs() -> dict:
        input.submit()
        with reactive.isolate():
            try:
                values = {f: float(input[_sid(f)]()) for f in SEL_COLS}
                _validated_predictors(pd.DataFrame([values]))
                return values
            except (ValueError, TypeError):
                from shiny.types import SafeException
                raise SafeException("Enter a finite, non-negative number for every measurement.") from None

    @reactive.calc
    # Function guide: _patient_prob is responsible for perform the patient prob step.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def _patient_prob() -> float:
        vals = _submitted_inputs()
        X = _validated_predictors(pd.DataFrame([vals]))
        return float(pipe_lr.predict_proba(X)[0, 1])

    @render.ui
    # Function guide: pred_chips is responsible for perform the pred chips step.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def pred_chips():
        p_ben = _patient_prob()
        p_mal = 1.0 - p_ben
        cls  = "Malignant" if p_mal >= DEFAULT_THRESHOLD else "Benign"
        prediction_class = "malignant" if cls == "Malignant" else "benign"
        chips = (
            '<div class="chip chip-malignant">'
            f'<span class="chip-lbl">P(Malignant)</span>'
            f'<span class="chip-val">{p_mal*100:.1f}%</span></div>'
            '<div class="chip chip-benign">'
            f'<span class="chip-lbl">P(Benign)</span>'
            f'<span class="chip-val">{p_ben*100:.1f}%</span></div>'
            f'<div class="chip chip-{prediction_class}">'
            f'<span class="chip-lbl">Classification</span>'
            f'<span class="chip-val">{cls}</span>'
            f'<span class="chip-detail">Malignancy cutoff: {DEFAULT_THRESHOLD:.3f}</span></div>'
        )
        return ui.HTML(f'<div class="infobar">{chips}</div>')


    @render.ui
    # Function guide: feat_table is responsible for perform the feat table step.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def feat_table():
        vals = _submitted_inputs()
        rows = ""
        for f in SEL_COLS:
            val  = vals[f]
            med  = TRAIN_MEDIANS[f]
            diff = val - med
            sign = "↑" if diff > 0 else ("↓" if diff < 0 else "—")
            delta_class = "positive" if diff > 0 else ("negative" if diff < 0 else "neutral")
            rows += (
                f"<tr><td>{f.title()}</td>"
                f"<td class='num'>{val:.4g}</td>"
                f"<td class='num'>{med:.4g}</td>"
                f"<td class='num feature-delta feature-delta-{delta_class}'>"
                f"{sign} {abs(diff):.4g}</td></tr>"
            )
        return ui.HTML(f"""
<div class="feature-table-wrap">
  <table class="tbl">
    <thead><tr>
      <th scope="col">Feature</th><th scope="col">Value</th>
      <th scope="col">Training Median</th><th scope="col">Δ</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>
<p class="feature-note">
  The model uses the already processed measurements as linear terms. No spline,
  quadratic, or model-applied nonlinear transformation is used.
</p>
""")

    # ── Batch prediction ──────────────────────────────────────────────────────

    @reactive.calc
    # Function guide: _batch_df is responsible for perform the batch df step.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def _batch_df():
        fi = input.batch_file()
        if not fi:
            return None, "No file uploaded."
        try:
            df = pd.read_csv(fi[0]["datapath"])
        except Exception as e:
            return None, f"Read error: {e}"
        try:
            X_b = _validated_predictors(df)
            p_benign = pipe_lr.predict_proba(X_b)[:, 1]
        except (ValueError, TypeError, OverflowError) as exc:
            return None, str(exc)
        p_malignant = 1.0 - p_benign
        df["P_malignant"] = p_malignant.round(4)
        df["P_benign"]    = p_benign.round(4)
        df["Prediction"]  = np.where(
            p_malignant >= DEFAULT_THRESHOLD, "Malignant", "Benign")
        return df, f"Processed {len(df)} rows successfully."

    @render.ui
    # Function guide: batch_status is responsible for perform the batch status step.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def batch_status():
        _, msg = _batch_df()
        ok  = msg and msg.startswith("Processed")
        status_class = "ok" if ok else "error"
        return ui.HTML(
            f'<p class="batch-status-message batch-status-{status_class}">'
            f'{html.escape(msg or "")}</p>'
        )

    @render.ui
    # Function guide: batch_table is responsible for perform the batch table step.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def batch_table():
        df, _ = _batch_df()
        if df is None:
            return ui.tags.div(
                ui.tags.h4("Your prediction preview will appear here"),
                ui.tags.p("Upload a CSV with the seven required measurements. "
                          "Validation messages appear beside the upload control."),
                ui.tags.ol(
                    ui.tags.li("Match the required column names exactly."),
                    ui.tags.li("Review the first 20 scored rows."),
                    ui.tags.li("Download the complete prediction file."),
                ),
                class_="batch-empty",
            )
        show = df.head(20)
        cols = list(show.columns)
        thead = "".join(
            f"<th scope=\"col\">{html.escape(str(c))}</th>" for c in cols
        )
        tbody = ""
        for _, row in show.iterrows():
            cells = ""
            for c in cols:
                v = row[c]
                if c == "Prediction":
                    prediction_class = "benign" if v == "Benign" else "malignant"
                    cells += (
                        f'<td class="prediction-{prediction_class}">'
                        f'{html.escape(str(v))}</td>'
                    )
                elif c in ("P_benign", "P_malignant"):
                    cells += f'<td class="num">{v:.4f}</td>'
                else:
                    cells += f"<td>{html.escape(str(v))}</td>"
            tbody += f"<tr>{cells}</tr>"
        return ui.HTML(f"""
<div class="batch-table-wrap">
  <table class="tbl">
    <thead><tr>{thead}</tr></thead>
    <tbody>{tbody}</tbody>
  </table>
</div>
""")

    @render.download(filename="breast_cancer_predictions.csv")
    # Function guide: batch_download is responsible for perform the batch download step.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def batch_download():
        df, _ = _batch_df()
        if df is None:
            yield ""
        else:
            yield df.to_csv(index=False)

    # ── Decision threshold ────────────────────────────────────────────────────

    @render.ui
    # Function guide: hist_img is responsible for perform the hist img step.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def hist_img():
        return ui.tags.img(
            src=charts.png(charts.threshold_figure(globals(), float(input.threshold()))),
            alt="Training probabilities by true class with the selected malignancy cutoff",
        )

    @render.ui
    # Function guide: cm_display is responsible for perform the cm display step.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def cm_display():
        thr  = float(input.threshold())
        metrics = _threshold_metrics(y_tr, PROB_TRAIN, thr)
        tn = metrics["tn"]
        fp = metrics["fp"]
        fn = metrics["fn"]
        tp = metrics["tp"]
        return ui.HTML(f"""
<div class="threshold-context">
  <p class="threshold-caption">
    Threshold = {thr:.2f} · Positive class = Malignant · Training set (n={N_TRAIN})
  </p>
  <div class="cm-wrap" role="table" aria-label="Training confusion matrix">
    <div class="cm-corner"></div>
    <div class="cm-col-hdr">Predicted<br>Malignant</div>
    <div class="cm-col-hdr">Predicted<br>Benign</div>
    <div class="cm-row-hdr">Actual<br>Malignant</div>
    <div class="cm-cell cm-tp">
      <span class="cm-n">{tp}</span>
      <span class="cm-desc"><b>True positive</b><br>Malignancy identified</span>
    </div>
    <div class="cm-cell cm-fn">
      <span class="cm-n">{fn}</span>
      <span class="cm-desc"><b>False negative</b><br>Malignancy missed</span>
    </div>
    <div class="cm-row-hdr">Actual<br>Benign</div>
    <div class="cm-cell cm-fp">
      <span class="cm-n">{fp}</span>
      <span class="cm-desc"><b>False positive</b><br>Benign case flagged</span>
    </div>
    <div class="cm-cell cm-tn">
      <span class="cm-n">{tn}</span>
      <span class="cm-desc"><b>True negative</b><br>Benign identified</span>
    </div>
  </div>
</div>
""")

    @render.ui
    # Function guide: metrics_table is responsible for perform the metrics table step.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def metrics_table():
        thr  = float(input.threshold())
        metrics = _threshold_metrics(y_tr, PROB_TRAIN, thr)
        metric_tiles = "".join(
            f"<div class='threshold-metric'><span>{lbl}</span>"
            f"<strong>{val:.4f}</strong></div>"
            for lbl, val in [
                ("Accuracy",                           metrics["acc"]),
                ("Sensitivity (malignant cases)",      metrics["sens"]),
                ("Specificity (benign cases)",         metrics["spec"]),
                ("Positive predictive value",          metrics["ppv"]),
                ("Negative predictive value",          metrics["npv"]),
                ("F1 score (malignant class)",         metrics["f1"]),
            ]
        )
        return ui.HTML(f"""
<div class="threshold-metric-grid">{metric_tiles}</div>
<p class="metric-caption">
  Evaluated on the training set (n={N_TRAIN}), with malignancy treated as the positive class.
</p>
""")

    # ── Results figures (pre-rendered data URIs) ──────────────────────────────

    @render.ui
    # Function guide: fig_feat_sel is responsible for perform the fig feat sel step.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def fig_feat_sel():
        feature = input.path_feature() or SEL_COLS[0]
        return ui.tags.img(src=charts.png(charts.path_figure(globals(), feature)),
                           alt=f"LASSO coefficient path for {feature}")

    @render.ui
    # Function guide: fig_perf is responsible for perform the fig perf step.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def fig_perf():
        return ui.tags.img(src=_PERF_SRC, alt="Model Performance")

    @render.ui
    # Function guide: fig_linearity is responsible for perform the fig linearity step.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def fig_linearity():
        feature = input.linearity_feature() or SEL_COLS[0]
        return ui.tags.img(src=charts.png(charts.linearity_figure(globals(), feature)),
                           alt=f"GAM functional-form diagnostic for {feature}")

    @render.ui
    # Function guide: fig_calibration is responsible for perform the fig calibration step.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def fig_calibration():
        return ui.tags.img(src=_CALIBRATION_SRC, alt="Training and test probability calibration")

    @render.ui
    # Function guide: fig_cv is responsible for perform the fig cv step.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def fig_cv():
        return ui.tags.img(src=_CV_SRC, alt="Training five-fold cross-validation AUC and one-SE rule")

    @render.ui
    # Function guide: fig_vif is responsible for perform the fig vif step.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def fig_vif():
        return ui.tags.img(src=_VIF_SRC, alt="Collinearity VIF")

    # ── Coefficient table (kept for Methods panel reference) ──────────────────

    @render.ui
    # Function guide: coef_table is responsible for perform the coef table step.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def coef_table():
        rows = ""
        order = np.argsort(np.abs(LR_COEF))[::-1]
        for i in order:
            f   = DESIGN_COLS[i]
            rc  = LR_COEF[i]
            rows += (
                f"<tr><td>{f}</td>"
                f"<td class='num coefficient-value'>{rc:+.4f}</td></tr>"
            )
        return ui.HTML(f"""
<div class="card">
  <div class="card-header">Fitted Design-Matrix Coefficients</div>
  <div class="card-body">
    <div class="coefficient-table-wrap">
      <table class="tbl">
        <thead><tr>
          <th scope="col">Design term</th>
          <th scope="col" class="coefficient-header">L2-Regularized Coefficient</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    <p class="coefficient-note">
      These {len(DESIGN_COLS)} terms represent the {N_SEL} selected processed inputs.
      Coefficients are on the benign log-odds scale and are conditional on feature
      selection.
    </p>
  </div>
</div>
""")

    # ── Performance metrics table (Performance & Calibration tab) ─────────────

    @render.ui
    # Function guide: perf_metrics_table is responsible for perform the perf metrics table step.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def perf_metrics_table():
        hl_interpretation = (
            "no evidence of lack of fit"
            if HL_P >= 0.10 else "evidence of lack of fit"
        )
        training = TRAIN_METRICS_YOUDEN
        rows_html = "".join(
            f"<tr><td>{label}</td><td class='num'>{train:.4f}</td>"
            f"<td class='num'>{test:.4f}</td></tr>"
            for label, train, test in [
                ("AUC", AUC_TRAIN, AUC_TEST),
                ("Brier score", BRIER_TRAIN, BRIER_TEST),
                ("Accuracy", training["acc"], TEST_METRICS_YOUDEN["acc"]),
                ("Sensitivity", training["sens"], TEST_METRICS_YOUDEN["sens"]),
                ("Specificity", training["spec"], TEST_METRICS_YOUDEN["spec"]),
                ("Positive predictive value", training["ppv"], TEST_METRICS_YOUDEN["ppv"]),
                ("Negative predictive value", training["npv"], TEST_METRICS_YOUDEN["npv"]),
                ("F1 score", training["f1"], TEST_METRICS_YOUDEN["f1"]),
            ]
        )
        return ui.HTML(f"""
<div class="card performance-card">
  <div class="card-header">
    Model Performance | Train / Test (Youden threshold = {YOUDEN_THRESHOLD:.3f})
  </div>
  <div class="card-body">
    <table class="tbl">
      <thead><tr><th scope="col">Metric</th><th scope="col">Train (apparent)</th><th scope="col">Test (held-out)</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    <p class="metric-caption">
      The Brier score is the mean squared error of the probability predictions;
      lower values are better (null model = {NULL_BRIER:.4f}).
      Hosmer-Lemeshow χ²={HL_CHI2:.2f}, p={HL_P:.3f}, df={HL_DF}
      → {hl_interpretation}. A non-significant result does not prove good calibration.
    </p>
  </div>
</div>
""")

    # ── Methods panel ─────────────────────────────────────────────────────────

    @render.ui
    # Function guide: methods_panel is responsible for perform the methods panel step.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def methods_panel():
        nonlinear_features = []
        nonlinear_overview = "none; linear-only model specified"
        lin_rows = "".join(
            f"<tr><td>{f.title()}</td>"
            f"<td class='num'>{TRAIN_MEDIANS[f]:.4g}</td>"
            f"<td class='num'>{LRT[f]['chi2']:.2f}</td>"
            f"<td class='num'>{LRT[f]['p']:.3f}</td>"
            f"<td>{EDA_DECISIONS[f]['transformation']} / {EDA_DECISIONS[f]['functional_form']}</td></tr>"
            for f in SEL_COLS
        )
        vif_rows = "".join(
            f"<tr><td><code>{f}</code></td>"
            f"<td class='num'>{VIF[f]:.3f}</td>"
            f"<td>{'Acceptable (<5)' if VIF[f] < 5 else ('Moderate (5–10)' if VIF[f] < 10 else 'Severe (>10)')}</td></tr>"
            for f in DESIGN_COLS
        )
        hl_interpretation = (
            "no evidence of lack of fit"
            if HL_P >= 0.10 else "evidence of lack of fit"
        )
        return ui.HTML(f"""
<div class="methods">

  <div class="card method-card">
    <div class="card-header">Linearity Assessment (LRT, α = 0.10)</div>
    <div class="card-body method-card-body">
      <table class="tbl">
        <thead><tr>
          <th scope="col">Feature</th><th scope="col">Training Median</th>
          <th scope="col">LRT χ²</th><th scope="col">p-value</th>
          <th scope="col">Applied transform / form</th>
        </tr></thead>
        <tbody>{lin_rows}</tbody>
      </table>
      <p class="method-note">
        The LRT compares a linear logistic GLM with a cubic-spline alternative
        (sklearn SplineTransformer; n_knots=3, degree=3, df_extra=3).
        Evidence against linearity is assessed at α = 0.10.
        The prespecified GAM effective-degrees-of-freedom rule selects the applied
        form. GAM smooth p-values test association, not nonlinearity.
      </p>
    </div>
  </div>

  <details class="detail-panel">
     <summary>Collinearity Assessment (VIF) - all linear design terms</summary>
    <div class="card-body method-card-body">
      <table class="tbl">
        <thead><tr>
          <th scope="col">Design term</th><th scope="col">VIF</th><th scope="col">Verdict</th>
        </tr></thead>
        <tbody>{vif_rows}</tbody>
      </table>
      <p class="method-note">
        VIF computed on the scaled training design matrix.
        VIF = 1/(1−R²) from regressing each feature on the others.
        VIF &lt; 5: acceptable · 5–10: moderate · &gt;10: severe multicollinearity.
        VIF is a screening diagnostic; it is not the sole basis for removing a
        predictor from this prespecified linear model.
      </p>
    </div>
  </details>

  <div class="card method-card">
    <div class="card-header">Model Performance — Train / Test (Youden threshold = {YOUDEN_THRESHOLD:.3f})</div>
    <div class="card-body method-card-body">
      <table class="tbl">
        <thead><tr><th scope="col">Metric</th><th scope="col">Value</th></tr></thead>
        <tbody>
          <tr><td>AUC (Training / Test)</td>
              <td class='num'>{AUC_TRAIN:.4f} / {AUC_TEST:.4f}</td></tr>
          <tr><td>Brier Score (Training / Test)</td>
              <td class='num'>{BRIER_TRAIN:.4f} / {BRIER_TEST:.4f}</td></tr>
          <tr><td>Null Brier (prevalence model)</td>
              <td class='num'>{NULL_BRIER:.4f}</td></tr>
          <tr><td>Hosmer-Lemeshow χ² (df={HL_DF})</td>
              <td class='num'>{HL_CHI2:.2f}  p={HL_P:.3f}</td></tr>
          <tr><td>Accuracy (Train / Test)</td><td class='num'>{TRAIN_METRICS_YOUDEN["acc"]:.4f} / {TEST_METRICS_YOUDEN["acc"]:.4f}</td></tr>
          <tr><td>Sensitivity (malignant; Train / Test)</td><td class='num'>{TRAIN_METRICS_YOUDEN["sens"]:.4f} / {TEST_METRICS_YOUDEN["sens"]:.4f}</td></tr>
          <tr><td>Specificity (benign; Train / Test)</td><td class='num'>{TRAIN_METRICS_YOUDEN["spec"]:.4f} / {TEST_METRICS_YOUDEN["spec"]:.4f}</td></tr>
          <tr><td>Positive predictive value (Train / Test)</td><td class='num'>{TRAIN_METRICS_YOUDEN["ppv"]:.4f} / {TEST_METRICS_YOUDEN["ppv"]:.4f}</td></tr>
          <tr><td>Negative predictive value (Train / Test)</td><td class='num'>{TRAIN_METRICS_YOUDEN["npv"]:.4f} / {TEST_METRICS_YOUDEN["npv"]:.4f}</td></tr>
          <tr><td>F1 score (malignant; Train / Test)</td><td class='num'>{TRAIN_METRICS_YOUDEN["f1"]:.4f} / {TEST_METRICS_YOUDEN["f1"]:.4f}</td></tr>
        </tbody>
      </table>
      <p class="method-note">
        Training results are apparent performance after feature selection and refitting.
        The test set is the held-out internal evaluation.
        Hosmer-Lemeshow test: {hl_interpretation}
        (χ²={HL_CHI2:.2f}, p={HL_P:.3f}).
        Brier skill = 1 − Brier/NullBrier =
        {1 - BRIER_TEST/NULL_BRIER:.3f}.
      </p>
    </div>
  </div>

  <div class="card">
    <div class="card-header">Pipeline Description</div>
    <div class="card-body method-narrative method-card-body">
      <section class="method-section">
      <h4>Dataset</h4>
      <p>The Wisconsin Diagnostic Breast Cancer dataset (sklearn; n = {N_TOTAL})
      contains 30 nuclear-morphology measurements derived from digitized
      fine-needle aspirate images. The outcome is malignant
      (n = {n_malignant}) or benign (n = {n_benign}). Data were divided using an
      80/20 stratified training/test split with random seed 42.</p>
      </section>

      <section class="method-section">
      <h4>Scaling</h4>
      <p>Features were scaled with RobustScaler by subtracting the median and
      dividing by the interquartile range. All scaling parameters were estimated
      from the training set to prevent data leakage. RobustScaler was used because
      several nuclear-morphology measurements are right-skewed.</p>
      </section>

      <section class="method-section">
      <h4>Stage 1 — LASSO Feature Selection</h4>
      <p>L1-penalized logistic regression (liblinear solver) was evaluated over 60
      log-spaced values of C ∈ [10⁻⁴, 10²] using 5-fold stratified
      cross-validation with AUC as the selection criterion. RobustScaler is refit
      within every training fold. These CV scores describe LASSO tuning and do not
      validate the final refitted model. The λ₁ₛₑ rule chooses
      the most regularized model whose mean cross-validated AUC is within one
      standard error of the maximum. This selected {N_SEL} features at
      C = {C_1SE:.5f}; λ_min selected {NZ_MIN} features at C = {C_MIN:.5f}.</p>
      </section>

      <section class="method-section">
      <h4>Stage 2 — L2-Regularized Logistic Regression</h4>
      <p>An L2-regularized logistic regression model (lbfgs solver) was tuned with
      five-fold cross-validation on the training set and refit using
      {len(DESIGN_COLS)} linear design terms derived from the {N_SEL} LASSO-selected
      processed inputs. No spline, quadratic, or model-applied nonlinear
      transformation is used. The resulting design is robust-scaled before fitting.</p>
      </section>

      <section class="method-section">
      <h4>Model Diagnostics</h4>
      <p>Preprocessing transformations were selected using the training split.
      The final model is constrained to linear terms. Diagnostic linearity tests
      are not used to add nonlinear terms.
      At α = 0.10, {len(nonlinear_features)} features showed evidence of nonlinearity:
      {nonlinear_overview}. LRT and GAM results are retained as diagnostics only
      and do not change the fitted design. Preprocessing decisions are recorded
      in preprocessing_decisions.json. The VIF plot describes the processed inputs
      used by the linear model.</p>
      <p>The L2 penalty limits coefficient inflation and separation while preserving
      the linear model specification.
      Apparent performance after selection is optimistic. These results have not
      been externally validated, and the held-out test set was not used to tune
      the model.</p>
      </section>

      <section class="method-section">
      <h4>Calibration</h4>
      <p>The Hosmer-Lemeshow test (10 quantile-based risk groups, df = {HL_DF})
      yielded χ² = {HL_CHI2:.2f}, p = {HL_P:.3f}, indicating
      {hl_interpretation}. A non-significant result does not establish good calibration.
      The test-set Brier score is {BRIER_TEST:.4f}, compared with
      {NULL_BRIER:.4f} for the prevalence-only model, giving a Brier skill score of
      {1 - BRIER_TEST/NULL_BRIER:.3f}.</p>
      </section>
    </div>
  </div>

</div>
""")


# ── 6. App ────────────────────────────────────────────────────────────────────

app = App(app_ui, server)
