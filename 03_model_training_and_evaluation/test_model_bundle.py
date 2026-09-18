# Module guide:
# - Role: Train, evaluate, or test the breast-cancer model workflow.
# - Workflow: Consume prepared artifacts, compare train and test performance, and validate bundles or figures.
# - Design note: Keep regression tests focused on reproducible artifacts and public contracts.
"""Focused regression checks for the deployment model bundle and evaluation behavior."""
from __future__ import annotations

import pickle
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.base import clone
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_model_deployment"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_data_preprocessing"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_data_cleaning"))
import breast_cancer_app as model


# Class guide: ModelBundleTests is responsible for perform the requested operation.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
class ModelBundleTests(unittest.TestCase):
    # Function guide: test_preprocessing_is_refit_in_each_cv_fold is responsible for test preprocessing is refit in each cv fold.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def test_preprocessing_is_refit_in_each_cv_fold(self):
        values = np.arange(120, dtype=float).reshape(40, 3)
        target = np.tile([0, 1], 20)
        fit_sizes = []
        original = RobustScaler.fit

        # Function guide: record_fit is responsible for record fit.
        # Inputs: scaler, X, y. Outputs and side effects follow the routine contract.
        # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
        def record_fit(scaler, X, y=None):
            fit_sizes.append(len(X))
            return original(scaler, X, y)

        with patch.object(RobustScaler, "fit", record_fit):
            model._lasso_cv_search(
                values, target, [0.1, 1.0],
                StratifiedKFold(4, shuffle=True, random_state=42))
        self.assertEqual(fit_sizes.count(30), 8)
        self.assertEqual(fit_sizes.count(40), 1)

    # Function guide: test_saved_predictions_match_untouched_holdout is responsible for test saved predictions match untouched holdout.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def test_saved_predictions_match_untouched_holdout(self):
        data = load_breast_cancer(as_frame=True)
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.2,
            stratify=data.target, random_state=42)
        self.assertEqual((len(y_train), len(y_test)), (455, 114))
        self.assertFalse(set(X_train.index) & set(X_test.index))
        np.testing.assert_array_equal(y_test, model.y_te)
        design = model.pipe_lr.named_steps["design"]
        raw_train = X_train[model.SEL_COLS].to_numpy()
        independently_fitted = clone(design).fit(raw_train)
        expected_design = independently_fitted.transform(raw_train)
        # Independently optimized power transforms differ slightly across platforms.
        np.testing.assert_allclose(
            design.transform(raw_train), expected_design, rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(
            model.pipe_lr.named_steps["scaler"].center_,
            np.median(expected_design, axis=0), rtol=1e-7, atol=1e-8)
        fitted_before = pickle.dumps(model.pipe_lr)
        for frame, saved in ((X_train, model.PROB_TRAIN),
                             (X_test, model.PROB_TEST)):
            predicted = model.pipe_lr.predict_proba(
                frame[model.SEL_COLS].to_numpy())[:, 1]
            np.testing.assert_allclose(predicted, saved, rtol=1e-9, atol=1e-9)
        self.assertEqual(fitted_before, pickle.dumps(model.pipe_lr))

    # Function guide: test_malignancy_positive_metrics_and_roc is responsible for test malignancy positive metrics and roc.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def test_malignancy_positive_metrics_and_roc(self):
        result = model._threshold_metrics(
            np.array([0, 0, 0, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
        self.assertEqual(result["sens"], 2 / 3)
        self.assertEqual(result["spec"], 1.0)
        self.assertEqual(result["tp"], 2)
        expected_fpr, expected_tpr, _ = roc_curve(
            1 - model.y_te, 1 - model.PROB_TEST)
        np.testing.assert_array_equal(expected_fpr, model.FPR_TE)
        np.testing.assert_array_equal(expected_tpr, model.TPR_TE)
        self.assertEqual(model._B["TEST_METRICS_05"], model.TEST_METRICS_05)
        self.assertEqual(model._B["TRAIN_METRICS_05"], model.TRAIN_METRICS_05)
        self.assertEqual(model.SENS05, model.TEST_METRICS_05["sens"])

    # Function guide: test_youden_threshold_is_training_locked is responsible for verify threshold selection.
    # Inputs: the saved training and test predictions from the deployment bundle.
    # Outputs: assertions that the threshold is selected from training predictions and both metric sets exist.
    def test_youden_threshold_is_training_locked(self):
        threshold = model._youden_threshold(model.y_tr, model.PROB_TRAIN)
        self.assertAlmostEqual(threshold, model.YOUDEN_THRESHOLD)
        self.assertEqual(
            model._B["TRAIN_METRICS_YOUDEN"], model.TRAIN_METRICS_YOUDEN)
        self.assertEqual(
            model._B["TEST_METRICS_YOUDEN"], model.TEST_METRICS_YOUDEN)
        expected = model._threshold_metrics(
            model.y_te, model.PROB_TEST, model.YOUDEN_THRESHOLD)
        self.assertEqual(expected, model.TEST_METRICS_YOUDEN)

    # Function guide: test_batch_predictor_validation is responsible for test batch predictor validation.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def test_batch_predictor_validation(self):
        good = pd.DataFrame([model.TRAIN_MEDIANS])
        self.assertEqual(model._validated_predictors(good).shape,
                         (1, len(model.SEL_COLS)))
        for invalid in (float("nan"), float("inf"), -1, "text"):
            bad = good.astype(object).copy()
            bad.iloc[0, 0] = invalid
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                model._validated_predictors(bad)
        with self.assertRaises(ValueError):
            model._validated_predictors(good.iloc[:0])
        with self.assertRaises(ValueError):
            model._validated_predictors(good.drop(columns=[model.SEL_COLS[0]]))

    # Function guide: test_bundle_contains_no_patient_feature_matrices is responsible for test bundle contains no patient feature matrices.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def test_bundle_contains_no_patient_feature_matrices(self):
        forbidden = {"X_train", "X_test", "X_all", "X", "exog", "endog",
                     "_training_data", "_fit_X", "dataframe", "training_data"}
        seen = set()

        # Function guide: inspect is responsible for perform the requested operation.
        # Inputs: value, path. Outputs and side effects follow the routine contract.
        # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
        def inspect(value, path):
            if id(value) in seen:
                return
            seen.add(id(value))
            self.assertNotIsInstance(value, pd.DataFrame, path)
            if isinstance(value, np.ndarray) and value.ndim == 2:
                self.assertNotIn(value.shape[0], (455, 114, 569), path)
            if isinstance(value, dict):
                for key, child in value.items():
                    self.assertNotIn(str(key), forbidden, path)
                    inspect(child, f"{path}.{key}")
            elif isinstance(value, (list, tuple)):
                for index, child in enumerate(value):
                    inspect(child, f"{path}[{index}]")
            elif hasattr(value, "__dict__") and not isinstance(value, type):
                inspect(vars(value), path)

        inspect(model._B, "bundle")
        self.assertEqual(model._B["BUNDLE_SCHEMA_VERSION"], model.BUNDLE_SCHEMA_VERSION)
        self.assertEqual(len(model._B["EDA_DECISIONS"]), 30)


if __name__ == "__main__":
    unittest.main()
