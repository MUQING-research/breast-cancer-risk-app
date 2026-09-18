# Module guide:
# - Role: Train, evaluate, or test the breast-cancer model workflow.
# - Workflow: Consume prepared artifacts, compare train and test performance, and validate bundles or figures.
# - Design note: Keep regression tests focused on reproducible artifacts and public contracts.
"""Train the deployable model and export row-level audit files locally."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[1]
SHINYAPP_DIR = REPO_ROOT / "04_model_deployment"
TRAINING_DIR = Path(__file__).resolve().parent
PREPROCESSING_DIR = REPO_ROOT / "02_data_preprocessing"
DATA_CLEANING_DIR = REPO_ROOT / "01_data_cleaning"
sys.path.insert(0, str(SHINYAPP_DIR))
sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(PREPROCESSING_DIR))
sys.path.insert(0, str(DATA_CLEANING_DIR))
_default_output_dir = Path(__file__).parent / ".cache" / "training"
os.environ.setdefault("BC_TRAINING_OUTPUT_DIR", str(_default_output_dir))
os.environ.setdefault("BC_CACHE_DIR", str(Path(__file__).parent / ".cache"))


# Function guide: main is responsible for run the module main workflow.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def main() -> None:
    os.environ["BC_FORCE_REBUILD"] = "1"
    import breast_cancer_app as model

    output_dir = Path(os.environ["BC_TRAINING_OUTPUT_DIR"])
    cleaned = pd.read_csv(output_dir / "breast_cancer_cleaned.csv")
    train, test = train_test_split(
        cleaned, test_size=0.2, stratify=cleaned["target"], random_state=42)
    rows = []
    for name in model.FEAT_NAMES:
        values = train[name].to_numpy()
        rows.append({
            "feature": name, "variable_type": "continuous", "n": len(values),
            "missing": int(train[name].isna().sum()),
            "mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)),
            "median": float(np.median(values)),
            "q1": float(np.quantile(values, 0.25)),
            "q3": float(np.quantile(values, 0.75)),
            "minimum": float(np.min(values)), "maximum": float(np.max(values)),
            "shapiro_p": float(stats.shapiro(values).pvalue),
            "skewness": float(stats.skew(values, bias=False)),
            "excess_kurtosis": float(stats.kurtosis(values, bias=False)),
            "mann_whitney_p": float(stats.mannwhitneyu(
                train.loc[train.target == 0, name],
                train.loc[train.target == 1, name]).pvalue),
        })
    pd.DataFrame(rows).to_csv(output_dir / "eda_summary_stats.csv", index=False)
    for label, frame in (("train", train), ("test", test)):
        raw_design = model.pipe_lr.named_steps["design"].transform(frame[model.SEL_COLS].to_numpy())
        design = model.pipe_lr.named_steps["scaler"].transform(raw_design)
        pd.DataFrame(design, columns=model.DESIGN_COLS).to_csv(
            output_dir / f"X_{label}_design.csv", index=False)
        frame[["target"]].to_csv(output_dir / f"y_{label}.csv", index=False)
    manifest = {
        "train_n": model.N_TRAIN, "test_n": model.N_TEST,
        "selected_features": model.SEL_COLS,
        "C_min": float(model.C_MIN), "C_1se": float(model.C_1SE),
        "final_model": "L2-regularized logistic regression",
        "final_C": float(model.FINAL_C),
        "final_cv_auc": float(max(model.FINAL_MEAN_AUC)),
        "train": {"auc": model.AUC_TRAIN, "brier": model.BRIER_TRAIN,
                  **model.TRAIN_METRICS_05},
        "test": {"auc": model.AUC_TEST, "brier": model.BRIER_TEST,
                 **model.TEST_METRICS_05},
        "youden_threshold": float(model.YOUDEN_THRESHOLD),
        "train_youden": model.TRAIN_METRICS_YOUDEN,
        "test_youden": model.TEST_METRICS_YOUDEN,
        "threshold_selection": "Youden index selected on training predictions and locked before test evaluation",
        "nonlinear_predictors": [],
        "model_form": "linear terms only after training-set preprocessing transformations",
        "row_level_exports": "Local .cache/training only; never deployed",
    }
    (output_dir / "training_metrics.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    from eda_and_preprocessing import export_form_diagnostics
    export_form_diagnostics(model.EDA_DECISIONS, output_dir)
    figure = model._make_linearity_fig()
    figure.savefig(output_dir / 'eda_linearity_continuous.png', dpi=300)
    model.plt.close(figure)


if __name__ == "__main__":
    main()
