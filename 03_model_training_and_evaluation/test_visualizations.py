# Module guide:
# - Role: Train, evaluate, or test the breast-cancer model workflow.
# - Workflow: Consume prepared artifacts, compare train and test performance, and validate bundles or figures.
# - Design note: Keep regression tests focused on reproducible artifacts and public contracts.
"""Regression checks for complete, legible, fixed-ratio frontend figures."""

import base64
import struct
import sys
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_model_deployment"))
import breast_cancer_app as application
import visualizations as charts


# Class guide: VisualizationTests is responsible for perform the requested operation.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
class VisualizationTests(unittest.TestCase):
    # Function guide: figures is responsible for perform the requested operation.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def figures(self):
        data = vars(application)
        figures = [
            charts.roc_figure(data), charts.calibration_figure(data),
            charts.cv_figure(data), charts.vif_figure(data),
            charts.threshold_figure(data, 0.5),
        ]
        for feature in application.SEL_COLS:
            figures.extend((charts.path_figure(data, feature),
                            charts.linearity_figure(data, feature)))
        return figures

    # Function guide: test_calibration_uses_exactly_ten_equal_frequency_points is responsible for test calibration uses exactly ten equal frequency points.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def test_calibration_uses_exactly_ten_equal_frequency_points(self):
        for target, probability, expected_sizes in (
            (application.y_tr, application.PROB_TRAIN, {45, 46}),
            (application.y_te, application.PROB_TEST, {11, 12}),
        ):
            predicted, observed, group_sizes = charts._quantile_calibration_points(
                target, probability, n_points=10)
            self.assertEqual(len(predicted), 10)
            self.assertEqual(len(observed), 10)
            self.assertEqual(len(group_sizes), 10)
            self.assertEqual(set(group_sizes), expected_sizes)
            self.assertTrue(np.all(np.diff(predicted) >= 0))
            self.assertTrue(np.all((observed >= 0) & (observed <= 1)))

    # Function guide: test_calibration_uses_malignant_probability_and_canonical_bins is responsible for verify calibration semantics.
    # Inputs: saved target labels and probability predictions.
    # Outputs: assertions that the plotted points match sklearn's quantile calibration definition.
    def test_calibration_uses_malignant_probability_and_canonical_bins(self):
        for target, probability in (
            (application.y_tr, application.PROB_TRAIN),
            (application.y_te, application.PROB_TEST),
        ):
            expected_observed, expected_predicted = calibration_curve(
                (np.asarray(target) == 0).astype(int),
                1.0 - np.asarray(probability, dtype=float),
                n_bins=10,
                strategy="quantile",
            )
            predicted, observed, _ = charts._quantile_calibration_points(
                target, probability, n_points=10)
            np.testing.assert_allclose(predicted, expected_predicted)
            np.testing.assert_allclose(observed, expected_observed)

    # Function guide: test_figure_bounds_style_and_png_dimensions is responsible for test figure bounds style and png dimensions.
    # Inputs: the component state. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def test_figure_bounds_style_and_png_dimensions(self):
        for fig in self.figures():
            with self.subTest(title=fig.axes[0].get_title(loc="left")):
                try:
                    fig.canvas.draw()
                    renderer = fig.canvas.get_renderer()
                    bounds = fig.bbox
                    self.assertIsNone(fig._suptitle)
                    self.assertTrue(fig.texts, "A caption must be included in the image.")
                    for ax in fig.axes:
                        self.assertTrue(all(spine.get_visible() for spine in ax.spines.values()))
                        self.assertFalse(any(line.get_visible() for line in ax.get_xgridlines()))
                        self.assertFalse(any(line.get_visible() for line in ax.get_ygridlines()))
                        self.assertEqual(ax.title.get_fontsize(), 10)
                        labels = [ax.xaxis.label, ax.yaxis.label, ax._left_title]
                        # Matplotlib creates extra ticks beyond the view limits but
                        # does not draw them. Check only the ticks in the final image.
                        xlo, xhi = sorted(ax.get_xlim())
                        ylo, yhi = sorted(ax.get_ylim())
                        labels += [t for t in ax.get_xticklabels() if xlo <= t.get_position()[0] <= xhi]
                        labels += [t for t in ax.get_yticklabels() if ylo <= t.get_position()[1] <= yhi]
                        labels += list(ax.texts)
                        if ax.get_legend() is not None:
                            labels += list(ax.get_legend().get_texts())
                        for item in labels + list(fig.texts):
                            if item.get_visible() and item.get_text():
                                box = item.get_window_extent(renderer)
                                self.assertGreaterEqual(box.x0, bounds.x0 - 2)
                                self.assertGreaterEqual(box.y0, bounds.y0 - 2)
                                self.assertLessEqual(box.x1, bounds.x1 + 2)
                                self.assertLessEqual(box.y1, bounds.y1 + 2)
                    expected = tuple(int(round(size * 300)) for size in fig.get_size_inches())
                    payload = base64.b64decode(charts.png(fig).split(",", 1)[1])
                    self.assertEqual(payload[:8], b"\x89PNG\r\n\x1a\n")
                    self.assertEqual(struct.unpack(">II", payload[16:24]), expected)
                finally:
                    plt.close(fig)


if __name__ == "__main__":
    unittest.main()
