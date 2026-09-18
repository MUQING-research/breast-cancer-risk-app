# Module guide:
# - Role: Estimate deployment-model performance with repeated cross-validation.
# - Workflow: Load the saved pipeline, refit it inside repeated stratified folds,
#   summarize discrimination, calibration, classification, and feature stability.
# - Design note: This script is an offline audit only and never changes the
#   deployment bundle or exports raw feature matrices.
"""Run repeated stratified cross-validation for the deployed model."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold


DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / ".cache" / "training"
DEFAULT_BUNDLE_PATH = DEFAULT_PROJECT_DIR / "04_model_deployment" / "breast_cancer_model_bundle.pkl"


# Function guide: _calibration_metrics is responsible for fit calibration intercept and slope.
# Inputs: malignant outcome labels and predicted malignant probabilities.
# Outputs: unpenalized logistic calibration intercept and slope.
def _calibration_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Fit a logistic calibration line to logit-transformed predictions."""
    probability = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1.0 - 1e-8)
    outcome = np.asarray(y_true, dtype=float)
    logit_probability = np.log(probability / (1.0 - probability))

    def objective(parameters: np.ndarray) -> float:
        linear_predictor = parameters[0] + parameters[1] * logit_probability
        return float(np.sum(
            np.logaddexp(0.0, linear_predictor) - outcome * linear_predictor
        ))

    result = minimize(
        objective,
        x0=np.array([0.0, 1.0]),
        method="L-BFGS-B",
        bounds=[(-25.0, 25.0), (-25.0, 25.0)],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success or not np.isfinite(result.fun):
        return float("nan"), float("nan")
    return float(result.x[0]), float(result.x[1])


# Function guide: _classification_metrics is responsible for compute threshold metrics.
# Inputs: encoded target labels, predicted malignant probabilities, and a decision threshold.
# Outputs: classification metrics with malignancy treated as the positive class.
def _classification_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute threshold metrics with target=0 (malignant) as positive."""
    actual = (np.asarray(target, dtype=int) == 0).astype(int)
    predicted = (np.asarray(probabilities, dtype=float) >= threshold).astype(int)
    true_positive = int(((actual == 1) & (predicted == 1)).sum())
    true_negative = int(((actual == 0) & (predicted == 0)).sum())
    false_positive = int(((actual == 0) & (predicted == 1)).sum())
    false_negative = int(((actual == 1) & (predicted == 0)).sum())
    return {
        "accuracy": float((true_positive + true_negative) / len(actual)),
        "sensitivity": float(true_positive / (true_positive + false_negative)),
        "specificity": float(true_negative / (true_negative + false_positive)),
        "ppv": float(true_positive / (true_positive + false_positive))
        if true_positive + false_positive else 0.0,
        "npv": float(true_negative / (true_negative + false_negative))
        if true_negative + false_negative else 0.0,
        "f1": float(2.0 * true_positive / (
            2.0 * true_positive + false_positive + false_negative
        )) if 2 * true_positive + false_positive + false_negative else 0.0,
    }


# Function guide: _youden_threshold is responsible for select a fold-local threshold.
# Inputs: encoded target labels and predicted malignant probabilities from a training fold.
# Outputs: the finite threshold that maximizes the Youden index.
def _youden_threshold(target: np.ndarray, probabilities: np.ndarray) -> float:
    """Select a Youden threshold using training-fold predictions only."""
    malignant = (np.asarray(target, dtype=int) == 0).astype(int)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        malignant, np.asarray(probabilities, dtype=float))
    finite = np.isfinite(thresholds)
    if not finite.any():
        raise ValueError("The ROC curve did not produce a finite decision threshold.")
    index = int(np.argmax(
        true_positive_rate[finite] - false_positive_rate[finite]))
    return float(thresholds[finite][index])


# Function guide: _fold_metrics is responsible for compute fold-level metrics.
# Inputs: target labels and predicted malignant probabilities for one test fold.
# Outputs: discrimination, calibration, and threshold metrics for that fold.
def _fold_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute all prespecified metrics for one validation fold."""
    malignant = (np.asarray(target, dtype=int) == 0).astype(int)
    intercept, slope = _calibration_metrics(malignant, probabilities)
    metrics = {
        "roc_auc": float(roc_auc_score(malignant, probabilities)),
        "pr_auc": float(average_precision_score(malignant, probabilities)),
        "brier": float(brier_score_loss(malignant, probabilities)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }
    metrics.update(_classification_metrics(target, probabilities, threshold))
    return metrics


# Function guide: _bootstrap_summary is responsible for estimate patient-level confidence intervals.
# Inputs: one aggregated prediction per patient, encoded targets, bootstrap count, and random seed.
# Outputs: point estimates and percentile 95 percent confidence intervals.
def _bootstrap_summary(
    target: np.ndarray,
    probabilities: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, object]]:
    """Bootstrap patient-level confidence intervals for aggregated predictions."""
    target = np.asarray(target, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    point = _fold_metrics(target, probabilities)
    rng = np.random.default_rng(seed)
    values = {name: [] for name in point}
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(target), size=len(target))
        sample_target = target[indices]
        if np.unique(sample_target).size < 2:
            continue
        sample_metrics = _fold_metrics(sample_target, probabilities[indices])
        for name, value in sample_metrics.items():
            if np.isfinite(value):
                values[name].append(float(value))
    summary = {}
    for name, estimate in point.items():
        samples = np.asarray(values[name], dtype=float)
        summary[name] = {
            "estimate": float(estimate),
            "ci95": [
                float(np.percentile(samples, 2.5)),
                float(np.percentile(samples, 97.5)),
            ] if len(samples) else [None, None],
            "bootstrap_replicates": int(len(samples)),
        }
    return summary


# Function guide: _summarise_folds is responsible for aggregate fold metrics.
# Inputs: a list of metric dictionaries from repeated test folds.
# Outputs: mean, standard deviation, minimum, and maximum for each metric.
def _summarise_folds(fold_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Summarize variability across all repeated test folds."""
    names = fold_metrics[0].keys()
    summary = {}
    for name in names:
        values = np.asarray([row[name] for row in fold_metrics], dtype=float)
        finite = values[np.isfinite(values)]
        summary[name] = {
            "mean": float(np.mean(finite)),
            "sd": float(np.std(finite, ddof=1)),
            "minimum": float(np.min(finite)),
            "maximum": float(np.max(finite)),
        }
    return summary


# Function guide: _load_inputs is responsible for load the cleaned table and model bundle.
# Inputs: the deployment bundle path and optional cleaned-table path.
# Outputs: feature table, encoded target, and the unmodified deployment bundle.
def _load_inputs(
    bundle_path: Path,
    cleaned_path: Path | None,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Load the same data contract and pipeline used by deployment."""
    with bundle_path.open("rb") as handle:
        bundle = pickle.load(handle)
    if cleaned_path is not None and cleaned_path.exists():
        table = pd.read_csv(cleaned_path)
        features = [str(name) for name in load_breast_cancer().feature_names]
        return table[features], table["target"].astype(int), bundle
    source = load_breast_cancer(as_frame=True)
    return source.data.copy(), source.target.astype(int), bundle


# Function guide: run_repeated_validation is responsible for execute the offline validation workflow.
# Inputs: model bundle, input table, output path, repeated-fold settings, and bootstrap settings.
# Outputs: a JSON report plus auditable fold and patient-level prediction tables.
def run_repeated_validation(
    bundle_path: Path = DEFAULT_BUNDLE_PATH,
    cleaned_path: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    n_splits: int = 5,
    n_repeats: int = 10,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Evaluate the fixed deployment pipeline without changing its settings."""
    X, target, bundle = _load_inputs(bundle_path, cleaned_path)
    selected_features = list(bundle["SEL_COLS"])
    pipeline_template = bundle["pipe_lr"]
    lasso_template = bundle["pipe_lasso"]
    all_features = list(bundle["FEAT_NAMES"])
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=seed,
    )

    repeated_predictions = np.full((n_repeats, len(target)), np.nan, dtype=float)
    repeated_youden_predictions = np.full(
        (n_repeats, len(target)), np.nan, dtype=float)
    fold_rows = []
    selection_counts = dict.fromkeys(all_features, 0)
    for split_number, (train_index, test_index) in enumerate(
        splitter.split(X, target)
    ):
        repeat_index = split_number // n_splits
        X_train = X.iloc[train_index][selected_features].to_numpy()
        X_test = X.iloc[test_index][selected_features].to_numpy()
        y_train = target.iloc[train_index].to_numpy(dtype=int)
        y_test = target.iloc[test_index].to_numpy(dtype=int)

        pipeline = clone(pipeline_template)
        pipeline.fit(X_train, y_train)
        p_train = 1.0 - pipeline.predict_proba(X_train)[:, 1]
        p_benign = pipeline.predict_proba(X_test)[:, 1]
        p_malignant = 1.0 - p_benign
        repeated_predictions[repeat_index, test_index] = p_malignant
        threshold = _youden_threshold(y_train, p_train)
        repeated_youden_predictions[repeat_index, test_index] = (
            p_malignant >= threshold).astype(float)

        lasso = clone(lasso_template)
        lasso.fit(X.iloc[train_index][all_features].to_numpy(), y_train)
        selected_mask = lasso.named_steps["lasso"].coef_[0] != 0
        for feature, selected in zip(all_features, selected_mask):
            selection_counts[feature] += int(selected)

        row = {
            "repeat": repeat_index + 1,
            "fold": split_number % n_splits + 1,
            "train_n": len(train_index),
            "test_n": len(test_index),
        }
        row.update(_fold_metrics(y_test, p_malignant))
        row["youden_threshold"] = threshold
        row.update({
            f"youden_{name}": value
            for name, value in _fold_metrics(
                y_test, p_malignant, threshold).items()
        })
        fold_rows.append(row)

    if (not np.isfinite(repeated_predictions).all()
            or not np.isfinite(repeated_youden_predictions).all()):
        raise RuntimeError("Repeated cross-validation did not produce one prediction per patient and repeat.")
    patient_probabilities = repeated_predictions.mean(axis=0)
    patient_youden_predictions = (
        repeated_youden_predictions.mean(axis=0) >= 0.5).astype(float)
    fold_metrics = [
        {name: value for name, value in row.items()
         if name not in {"repeat", "fold", "train_n", "test_n"}}
        for row in fold_rows
    ]
    report = {
        "validation_name": "Repeated stratified cross-validation",
        "data": {
            "n_total": int(len(target)),
            "n_malignant": int((target == 0).sum()),
            "n_benign": int((target == 1).sum()),
            "positive_class": "malignant (target=0)",
        },
        "protocol": {
            "outer_folds": int(n_splits),
            "repeats": int(n_repeats),
            "total_test_folds": int(n_splits * n_repeats),
            "random_seed": int(seed),
            "preprocessing": "Deployment pipeline refit independently inside every fold",
            "feature_set": "Fixed to the seven predictors selected by the deployment bundle",
            "regularization": "Fixed to the deployment bundle; no retuning during this audit",
            "threshold": "Youden index selected independently inside each training fold and locked before its test fold",
            "interpretation": "Conditional validation of the deployed feature set and model settings; it does not replace nested feature-selection validation or external validation.",
        },
        "fold_metrics": _summarise_folds(fold_metrics),
        "pooled_patient_metrics": _bootstrap_summary(
            target.to_numpy(), patient_probabilities, n_bootstrap, seed
        ),
        "pooled_youden_classification": _classification_metrics(
            target.to_numpy(), patient_youden_predictions, threshold=0.5
        ),
        "feature_selection_stability": {
            feature: {
                "selected_folds": int(count),
                "total_folds": int(n_splits * n_repeats),
                "selection_frequency": float(count / (n_splits * n_repeats)),
            }
            for feature, count in selection_counts.items()
        },
        "source_bundle": str(bundle_path),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "repeated_cv_metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.DataFrame(fold_rows).to_csv(
        output_dir / "repeated_cv_fold_metrics.csv", index=False
    )
    pd.DataFrame({
        "row_index": np.arange(len(target)),
        "target": target.to_numpy(dtype=int),
        "p_malignant_repeated_mean": patient_probabilities,
    }).to_csv(output_dir / "repeated_cv_patient_predictions.csv", index=False)
    (output_dir / "feature_selection_stability.json").write_text(
        json.dumps(report["feature_selection_stability"], indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return report


# Function guide: main is responsible for parse arguments and run validation.
# Inputs: command-line options. Outputs: a human-readable completion message.
def main() -> None:
    """Parse command-line options and run repeated validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--cleaned", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = run_repeated_validation(
        bundle_path=args.bundle,
        cleaned_path=args.cleaned,
        output_dir=args.output_dir,
        n_splits=args.folds,
        n_repeats=args.repeats,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    pooled = report["pooled_patient_metrics"]
    print("Repeated validation completed.")
    for metric in ("roc_auc", "pr_auc", "brier", "calibration_slope"):
        values = pooled[metric]
        print(f"{metric}: {values['estimate']:.4f} [{values['ci95'][0]:.4f}, {values['ci95'][1]:.4f}]")
    fold_metrics = report["fold_metrics"]
    print(
        "youden threshold mean: "
        f"{fold_metrics['youden_threshold']['mean']:.4f}"
    )
    for metric in ("accuracy", "sensitivity", "specificity", "f1"):
        print(
            f"youden_{metric}: "
            f"{fold_metrics['youden_' + metric]['mean']:.4f}"
        )


if __name__ == "__main__":
    main()
