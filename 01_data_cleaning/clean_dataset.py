# Module guide:
# - Role: Load and clean raw breast-cancer data.
# - Workflow: Create cleaned tables, dictionaries, and audit outputs before modelling.
# - Design note: Keep source cleaning separate from training-fitted preprocessing.
"""Create the cleaned breast-cancer table and its base data audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer


# Function guide: build_clean_dataset is responsible for build clean dataset.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
# Function guide: build_clean_dataset is responsible for build clean dataset.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def build_clean_dataset() -> tuple[pd.DataFrame, list[dict], dict]:
    """Load the source table and apply source-level cleaning only."""
    source = load_breast_cancer()
    cleaned = pd.DataFrame(source.data, columns=source.feature_names)
    cleaned = cleaned.astype(float)
    cleaned["target"] = source.target.astype(int)

    dictionary = [{
        "variable": str(name),
        "variable_type": "continuous",
        "definition": str(name),
        "unit": "Dataset-provided measurement scale",
        "missing_codes": [],
        "valid_range": "finite and non-negative",
    } for name in source.feature_names]
    dictionary.append({
        "variable": "target",
        "variable_type": "binary",
        "definition": "0=malignant; 1=benign",
        "unit": "none",
    })

    audit = {
        "sample_size": len(cleaned),
        "predictor_count": len(source.feature_names),
        "outcome": "target",
        "positive_class": "malignant (target=0)",
        "duplicate_rows": int(cleaned.duplicated().sum()),
        "missing_per_variable": cleaned.isna().sum().to_dict(),
        "rows_with_missing": int(cleaned.isna().any(axis=1).sum()),
        "invalid_numeric_values": int((
            ~np.isfinite(cleaned[source.feature_names].to_numpy(dtype=float))
            | (cleaned[source.feature_names].to_numpy(dtype=float) < 0)
        ).sum()),
        "cleaning": "Type normalization only; no outcome-aware transformations or exclusions.",
    }
    return cleaned, dictionary, audit


# Function guide: clean_breast_cancer_dataset is responsible for clean breast cancer dataset.
# Inputs: output_dir. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
# Function guide: clean_breast_cancer_dataset is responsible for clean breast cancer dataset.
# Inputs: output_dir: str | Path. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def clean_breast_cancer_dataset(output_dir: str | Path) -> pd.DataFrame:
    """Write base-cleaning outputs and return the cleaned table."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cleaned, dictionary, audit = build_clean_dataset()
    cleaned.to_csv(output_path / "breast_cancer_cleaned.csv", index=False)
    (output_path / "data_dictionary.json").write_text(
        json.dumps(dictionary, indent=2), encoding="utf-8")
    (output_path / "data_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")
    return cleaned


# Function guide: main is responsible for run the module main workflow.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
# Function guide: main is responsible for run the module main workflow.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "03_model_training_and_evaluation" / ".cache" / "training",
    )
    args = parser.parse_args()
    cleaned = clean_breast_cancer_dataset(args.output_dir)
    print(f"Saved cleaned breast-cancer table with {len(cleaned)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
