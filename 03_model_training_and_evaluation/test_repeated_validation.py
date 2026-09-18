# Module guide:
# - Role: Test the repeated-validation audit contract.
# - Workflow: Run a small deterministic audit and verify its files and metrics.
# - Design note: Keep the test independent from the full 50-fold production audit.
"""Regression tests for the repeated-validation audit."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repeated_validation as validation


# Class guide: RepeatedValidationTests is responsible for validate audit outputs.
# Inputs: the saved deployment bundle and a temporary output directory.
# Outputs: assertions about the audit schema and reproducible metric fields.
class RepeatedValidationTests(unittest.TestCase):
    # Function guide: test_small_audit_writes_expected_outputs is responsible for validate the audit contract.
    # Inputs: a two-fold, two-repeat validation configuration.
    # Outputs: no return value; raises an assertion when the audit contract changes.
    def test_small_audit_writes_expected_outputs(self):
        bundle_path = Path(__file__).resolve().parents[1] / "04_model_deployment" / "breast_cancer_model_bundle.pkl"
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            report = validation.run_repeated_validation(
                bundle_path=bundle_path,
                output_dir=output_dir,
                n_splits=2,
                n_repeats=2,
                n_bootstrap=20,
                seed=42,
            )
            self.assertEqual(report["protocol"]["total_test_folds"], 4)
            self.assertEqual(report["data"]["n_total"], 569)
            self.assertIn("roc_auc", report["pooled_patient_metrics"])
            self.assertIn("calibration_slope", report["pooled_patient_metrics"])
            self.assertIn("pooled_youden_classification", report)
            self.assertIn("youden_threshold", report["fold_metrics"])
            self.assertEqual(
                set(report["feature_selection_stability"]),
                set(report["data"].get("feature_names", validation.load_breast_cancer().feature_names)),
            )
            saved = json.loads((output_dir / "repeated_cv_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["protocol"], report["protocol"])
            self.assertTrue((output_dir / "repeated_cv_fold_metrics.csv").exists())
            self.assertTrue((output_dir / "repeated_cv_patient_predictions.csv").exists())
            self.assertTrue((output_dir / "feature_selection_stability.json").exists())


if __name__ == "__main__":
    unittest.main()
