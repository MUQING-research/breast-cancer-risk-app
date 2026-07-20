"""Regenerate the README figures from the deployment bundle."""

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.calibration import calibration_curve

mpl.rcParams.update({
    'font.family'      : 'Arial',
    'font.size'        : 9,
    'axes.titlesize'   : 10,
    'axes.titleweight' : 'bold',
    'axes.labelsize'   : 9,
    'xtick.labelsize'  : 8,
    'ytick.labelsize'  : 8,
    'legend.fontsize'  : 8,
    'axes.spines.top'  : True,
    'axes.spines.right': True,
    'xtick.direction'  : 'out',
    'ytick.direction'  : 'out',
    'figure.dpi'       : 300,
    'savefig.dpi'      : 300,
    'savefig.bbox'     : 'tight',
    'pdf.fonttype'     : 42,   # embeds fonts as TrueType in PDF
    'ps.fonttype'      : 42,
})

CELL_COLORS = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
    "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85",
]

PRIMARY_COLOR = CELL_COLORS[0]
SECONDARY_COLOR = CELL_COLORS[1]
TERTIARY_COLOR = CELL_COLORS[2]
TEXT_COLOR = CELL_COLORS[3]
WARNING_COLOR = CELL_COLORS[4]
MUTED_COLOR = CELL_COLORS[5]
LIGHT_COLOR = CELL_COLORS[6]
SEVERE_COLOR = CELL_COLORS[7]
BROWN_COLOR = CELL_COLORS[8]
REFERENCE_COLOR = CELL_COLORS[9]

mpl.rcParams.update({
    "text.color": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.edgecolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "bc_bundle.pkl"
ASSET_DIR = ROOT / "assets"


def _load_bundle() -> dict:
    """Load the privacy-preserving deployment bundle."""
    with BUNDLE_PATH.open("rb") as handle:
        return pickle.load(handle)


def _style_axis(ax) -> None:
    """Apply the shared closed-box Cell figure style."""
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(TEXT_COLOR)
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(
        direction="out",
        length=3,
        width=0.8,
        colors=TEXT_COLOR,
    )
    ax.grid(False)


def _panel_title(ax, letter: str, title: str) -> None:
    ax.set_title(f"{letter}. {title}", loc="left", color=TEXT_COLOR)


def _save_figure(fig, filename: str, *, use_tight_layout: bool = True) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    if use_tight_layout:
        fig.tight_layout()
    fig.savefig(ASSET_DIR / filename, format="png", dpi=300)
    plt.close(fig)


def _display_name(feature: str) -> str:
    return feature.replace("worst ", "").capitalize()


def _model_performance_figure(bundle: dict) -> None:
    fig = plt.figure(figsize=(7.0, 5.25), dpi=300, layout="constrained")
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(1.0, 1.15),
    )

    ax = fig.add_subplot(grid[0, 0])
    _style_axis(ax)
    ax.plot(
        [0, 1],
        [0, 1],
        color=REFERENCE_COLOR,
        linewidth=0.8,
        linestyle="--",
        label="Chance",
    )
    ax.plot(
        bundle["FPR_TR"],
        bundle["TPR_TR"],
        color=SECONDARY_COLOR,
        linewidth=1.0,
        label=f"Train AUC = {bundle['AUC_TRAIN']:.3f}",
    )
    ax.plot(
        bundle["FPR_TE"],
        bundle["TPR_TE"],
        color=PRIMARY_COLOR,
        linewidth=1.0,
        marker="o",
        markersize=4,
        label=f"Test AUC = {bundle['AUC_TEST']:.3f}",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    _panel_title(ax, "A", "Discrimination")
    ax.legend(loc="lower right", frameon=False)

    train_fraction, train_mean = calibration_curve(
        np.asarray(bundle["y_tr"], dtype=int),
        np.asarray(bundle["PROB_TRAIN"], dtype=float),
        n_bins=10,
        strategy="quantile",
    )
    test_fraction, test_mean = calibration_curve(
        np.asarray(bundle["y_te"], dtype=int),
        np.asarray(bundle["PROB_TEST"], dtype=float),
        n_bins=10,
        strategy="quantile",
    )

    ax = fig.add_subplot(grid[0, 1])
    _style_axis(ax)
    ax.plot(
        [0, 1],
        [0, 1],
        color=REFERENCE_COLOR,
        linewidth=0.8,
        linestyle="--",
        label="Ideal",
    )
    ax.plot(
        train_mean,
        train_fraction,
        color=SECONDARY_COLOR,
        linewidth=1.0,
        marker="o",
        markersize=4,
        markerfacecolor="white",
        markeredgecolor=SECONDARY_COLOR,
        label=f"Train Brier = {bundle['BRIER_TRAIN']:.3f}",
    )
    ax.plot(
        test_mean,
        test_fraction,
        color=PRIMARY_COLOR,
        linewidth=1.0,
        marker="o",
        markersize=4,
        label=f"Test Brier = {bundle['BRIER_TEST']:.3f}",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Predicted probability of benign disease")
    ax.set_ylabel("Observed benign fraction")
    _panel_title(ax, "B", "Calibration")
    ax.legend(loc="upper left", frameon=False)

    ax = fig.add_subplot(grid[1, :])
    _style_axis(ax)
    coefficients = np.asarray(bundle["LR_COEF"], dtype=float)
    order = np.argsort(coefficients)
    ordered_values = coefficients[order]
    ordered_names = [bundle["SEL_COLS"][index].title() for index in order]
    colors = [
        SECONDARY_COLOR if value >= 0 else PRIMARY_COLOR
        for value in ordered_values
    ]
    y_positions = np.arange(len(ordered_names))
    ax.barh(
        y_positions,
        ordered_values,
        color=colors,
        edgecolor="none",
        height=0.58,
    )
    ax.axvline(0, color=REFERENCE_COLOR, linewidth=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ordered_names)
    ax.set_xlabel("Logistic regression coefficient for benign outcome")
    _panel_title(ax, "C", "Fitted coefficients")

    coefficient_span = max(float(np.ptp(ordered_values)), 1.0)
    lower_limit = min(float(ordered_values.min()) - 0.65, -0.65)
    upper_limit = max(float(ordered_values.max()) + 0.65, 0.65)
    ax.set_xlim(lower_limit, upper_limit)
    for y_pos, value in zip(y_positions, ordered_values):
        offset = coefficient_span * 0.025
        ax.text(
            value + (offset if value >= 0 else -offset),
            y_pos,
            f"{value:.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8,
            color=TEXT_COLOR,
        )
    _save_figure(fig, "model_performance.png", use_tight_layout=False)


def _feature_selection_figure(bundle: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), dpi=300)
    for ax in axes:
        _style_axis(ax)

    selected_indices = list(bundle["SEL_IDX"])
    selected_set = set(selected_indices)
    log_c = np.asarray(bundle["LOG_C"], dtype=float)
    path_coefficients = np.asarray(bundle["PATH_COEFS"], dtype=float)

    ax = axes[0]
    for feature_index in range(path_coefficients.shape[1]):
        if feature_index not in selected_set:
            ax.plot(
                log_c,
                path_coefficients[:, feature_index],
                color=MUTED_COLOR,
                linewidth=0.8,
                alpha=0.22,
            )
    for color_index, feature_index in enumerate(selected_indices):
        ax.plot(
            log_c,
            path_coefficients[:, feature_index],
            color=CELL_COLORS[color_index],
            linewidth=1.0,
            label=bundle["FEAT_NAMES"][feature_index],
        )
    ax.axhline(0, color=REFERENCE_COLOR, linewidth=0.8)
    ax.axvline(
        np.log10(bundle["C_MIN"]),
        color=BROWN_COLOR,
        linewidth=0.8,
        linestyle="--",
    )
    ax.axvline(
        np.log10(bundle["C_1SE"]),
        color=SEVERE_COLOR,
        linewidth=0.8,
        linestyle=":",
    )
    ax.set_xlim(log_c.min(), log_c.max())
    ax.set_xlabel(r"$\log_{10}(C)$")
    ax.set_ylabel("Coefficient")
    _panel_title(ax, "A", "Regularisation path")
    legend = ax.legend(
        loc="upper left",
        ncol=2,
        frameon=False,
        handlelength=1.2,
        columnspacing=0.8,
        title=f"n = {bundle['N_SEL']} selected",
    )
    legend.get_title().set_fontweight("bold")

    ax = axes[1]
    log_cs = np.log10(np.asarray(bundle["CS"], dtype=float))
    mean_auc = np.asarray(bundle["MEAN_AUC"], dtype=float)
    se_auc = np.asarray(bundle["SE_AUC"], dtype=float)
    ax.fill_between(
        log_cs,
        mean_auc - se_auc,
        mean_auc + se_auc,
        color=SECONDARY_COLOR,
        alpha=0.18,
    )
    ax.plot(log_cs, mean_auc, color=SECONDARY_COLOR, linewidth=1.0)
    ax.axvline(
        np.log10(bundle["C_MIN"]),
        color=BROWN_COLOR,
        linewidth=0.8,
        linestyle="--",
        label=fr"$\lambda_{{min}}$ ({bundle['NZ_MIN']})",
    )
    ax.axvline(
        np.log10(bundle["C_1SE"]),
        color=SEVERE_COLOR,
        linewidth=0.8,
        linestyle=":",
        label=fr"$\lambda_{{1SE}}$ ({bundle['N_SEL']})",
    )
    ax.axhline(
        bundle["THR_1SE"],
        color=SEVERE_COLOR,
        linewidth=0.8,
        linestyle="--",
        alpha=0.55,
    )
    ax.set_xlabel(r"$\log_{10}(C)$")
    ax.set_ylabel("5-fold CV AUC")
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.3f"))
    _panel_title(ax, "B", "Cross-validation AUC")
    ax.legend(loc="lower left", frameon=False)

    _save_figure(fig, "model_feature_selection.png")


def _linearity_figure(bundle: dict) -> None:
    selected_features = list(bundle["SEL_COLS"])
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 5.0), dpi=300)

    for index, feature in enumerate(selected_features):
        ax = axes.flat[index]
        _style_axis(ax)
        plot_data = bundle["LRT_PLOT_DATA"][feature]
        midpoints = np.asarray(plot_data["mids"], dtype=float)
        empirical_logits = np.asarray(plot_data["logits"], dtype=float)
        ax.scatter(
            midpoints,
            empirical_logits,
            color=SECONDARY_COLOR,
            marker="o",
            s=16,
            edgecolors="none",
            zorder=3,
        )
        if len(midpoints) > 2:
            slope, intercept = np.polyfit(midpoints, empirical_logits, 1)
            x_line = np.linspace(midpoints.min(), midpoints.max(), 100)
            ax.plot(
                x_line,
                slope * x_line + intercept,
                color=TEXT_COLOR,
                linewidth=1.0,
                linestyle="--",
            )

        result = bundle["LRT"][feature]
        is_linear = bool(result["linear"])
        verdict = "Linear" if is_linear else "Non-linear"
        verdict_color = TEXT_COLOR if is_linear else PRIMARY_COLOR
        ax.text(
            0.97,
            0.96,
            f"p={result['p']:.3f}\n{verdict}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color=verdict_color,
        )
        ax.set_xlabel("Feature value")
        if index % 4 == 0:
            ax.set_ylabel("Empirical log-odds")
        _panel_title(ax, chr(ord("A") + index), _display_name(feature))

    summary_ax = axes.flat[-1]
    _style_axis(summary_ax)
    summary_ax.set_xticks([])
    summary_ax.set_yticks([])
    _panel_title(summary_ax, "H", "Diagnostic summary")
    nonlinear = [
        feature
        for feature in selected_features
        if not bool(bundle["LRT"][feature]["linear"])
    ]
    summary_ax.text(
        0.08,
        0.82,
        "Non-linear at alpha=0.10",
        transform=summary_ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    for row_index, feature in enumerate(nonlinear):
        result = bundle["LRT"][feature]
        summary_ax.text(
            0.08,
            0.62 - row_index * 0.12,
            f"- {_display_name(feature)} (p={result['p']:.3f})",
            transform=summary_ax.transAxes,
            fontsize=8,
            color=PRIMARY_COLOR,
        )
    summary_ax.text(
        0.08,
        0.10,
        "Functional-form diagnostic;\nnot a feature-selection test.",
        transform=summary_ax.transAxes,
        fontsize=8,
        color=BROWN_COLOR,
    )

    _save_figure(fig, "model_linearity_diagnostics.png")


def _vif_figure(bundle: dict) -> None:
    ordered = sorted(bundle["VIF"].items(), key=lambda item: item[1], reverse=True)
    names = [_display_name(name) for name, _ in ordered]
    values = np.asarray([value for _, value in ordered], dtype=float)
    colors = [
        SEVERE_COLOR
        if value > 10
        else WARNING_COLOR
        if value > 5
        else SECONDARY_COLOR
        for value in values
    ]

    fig, ax = plt.subplots(figsize=(7.0, 3.5), dpi=300)
    _style_axis(ax)
    y_positions = np.arange(len(names))
    ax.barh(
        y_positions,
        values,
        color=colors,
        edgecolor="none",
        height=0.56,
    )
    ax.axvline(
        5,
        color=WARNING_COLOR,
        linewidth=0.8,
        linestyle="--",
        label="Moderate: VIF=5",
    )
    ax.axvline(
        10,
        color=SEVERE_COLOR,
        linewidth=0.8,
        linestyle=":",
        label="Severe: VIF=10",
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Variance inflation factor")
    _panel_title(ax, "A", "Collinearity among retained predictors")
    ax.set_xlim(0, max(10.3, float(values.max()) * 1.22))
    for y_pos, value in zip(y_positions, values):
        ax.text(
            value + 0.12,
            y_pos,
            f"{value:.2f}",
            va="center",
            fontsize=8,
            color=TEXT_COLOR,
        )
    ax.legend(loc="lower right", frameon=False)

    _save_figure(fig, "model_vif_diagnostics.png")


def main() -> None:
    bundle = _load_bundle()
    _model_performance_figure(bundle)
    _feature_selection_figure(bundle)
    _linearity_figure(bundle)
    _vif_figure(bundle)
    print("README figures regenerated with the shared Cell color system.")


if __name__ == "__main__":
    main()
