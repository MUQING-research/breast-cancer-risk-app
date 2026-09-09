"""Focused regression checks for deployment and model evaluation contracts."""
from __future__ import annotations

import pickle
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.base import clone
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler

import breast_cancer_app as model


class ModelContractTests(unittest.TestCase):
    def test_preprocessing_is_refit_in_each_cv_fold(self):
        values = np.arange(120, dtype=float).reshape(40, 3)
        target = np.tile([0, 1], 20)
        fit_sizes = []
        original = RobustScaler.fit

        def record_fit(scaler, X, y=None):
            fit_sizes.append(len(X))
            return original(scaler, X, y)

        with patch.object(RobustScaler, "fit", record_fit):
            model._lasso_cv_search(
                values, target, [0.1, 1.0],
                StratifiedKFold(4, shuffle=True, random_state=42))
        self.assertEqual(fit_sizes.count(30), 8)
        self.assertEqual(fit_sizes.count(40), 1)

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
        np.testing.assert_allclose(design.transform(raw_train), expected_design)
        np.testing.assert_allclose(
            model.pipe_lr.named_steps["scaler"].center_,
            np.median(expected_design, axis=0))
        fitted_before = pickle.dumps(model.pipe_lr)
        for frame, saved in ((X_train, model.PROB_TRAIN),
                             (X_test, model.PROB_TEST)):
            predicted = model.pipe_lr.predict_proba(
                frame[model.SEL_COLS].to_numpy())[:, 1]
            np.testing.assert_allclose(predicted, saved, rtol=1e-9, atol=1e-9)
        self.assertEqual(fitted_before, pickle.dumps(model.pipe_lr))

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

    def test_bundle_contains_no_patient_feature_matrices(self):
        forbidden = {"X_train", "X_test", "X_all", "X", "exog", "endog",
                     "_training_data", "_fit_X", "dataframe", "training_data"}
        seen = set()

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
