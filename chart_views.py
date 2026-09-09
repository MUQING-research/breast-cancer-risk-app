"""Readable, self-contained Cell-style figures for the application UI."""

from __future__ import annotations

import base64
import io
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": ["Arial", "sans-serif"],
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

CELL_COLORS = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
    "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85",
]


def canvas(title, wide=False):
    fig, ax = plt.subplots(figsize=(7.0 if wide else 3.5, 3.5))
    ax.set_title(title, loc="left", pad=10)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(True)
    ax.tick_params(direction="out")
    ax.grid(False)
    return fig, ax


def finish(fig, caption):
    """Reserve a real caption region inside the image, outside the data axes."""
    wide = fig.get_figwidth() > 5
    caption_lines = textwrap.fill(caption, width=112 if wide else 53)
    footer = 0.16 if wide else 0.20
    fig.tight_layout(pad=0.9, rect=(0, footer, 1, 1))
    fig.text(0.05, 0.035, caption_lines, ha="left", va="bottom",
             fontsize=8, linespacing=1.3, color=CELL_COLORS[3])
    return fig


def png(fig):
    buffer = io.BytesIO()
    with mpl.rc_context({"savefig.bbox": None}):
        fig.savefig(buffer, format="png", dpi=300, bbox_inches=None)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def legend(ax, **kwargs):
    return ax.legend(frameon=True, facecolor="white", framealpha=0.94,
                     edgecolor="none", fontsize=8, **kwargs)


def roc_figure(data):
    fig, ax = canvas("Discrimination | ROC")
    for suffix, label, color, style in (
        ("TR", "Train", CELL_COLORS[0], "-"),
        ("TE", "Test", CELL_COLORS[1], "--"),
    ):
        auc = data["AUC_TRAIN" if suffix == "TR" else "AUC_TEST"]
        ax.plot(data["FPR_" + suffix], data["TPR_" + suffix],
                color=color, linestyle=style, linewidth=1,
                label=f"{label} AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color=CELL_COLORS[2], linestyle=":", linewidth=1)
    ax.set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel="1 - Specificity",
           ylabel="Sensitivity")
    legend(ax, loc="lower right")
    return finish(fig, f"WDBC | Train n={data['N_TRAIN']}; test n={data['N_TEST']}. "
                  "The dotted diagonal is chance discrimination.")


def calibration_figure(data):
    from sklearn.calibration import calibration_curve

    fig, ax = canvas("Probability calibration")
    for label, target, probability, brier, color, style in (
        ("Train", data["y_tr"], data["PROB_TRAIN"], data["BRIER_TRAIN"], CELL_COLORS[0], "-"),
        ("Test", data["y_te"], data["PROB_TEST"], data["BRIER_TEST"], CELL_COLORS[1], "--"),
    ):
        observed, predicted = calibration_curve(
            1 - target, 1 - probability, n_bins=10, strategy="quantile")
        ax.plot(predicted, observed, color=color, linestyle=style,
                linewidth=1, marker="o", markersize=4,
                label=f"{label} Brier = {brier:.3f}")
    ax.plot([0, 1], [0, 1], color=CELL_COLORS[2], linestyle=":", linewidth=1)
    ax.set(xlim=(0, 1), ylim=(0, 1.04),
           xlabel="Predicted malignancy probability", ylabel="Observed malignant fraction")
    legend(ax, loc="upper left")
    return finish(fig, "WDBC | 10 quantile bins per split. The dotted diagonal "
                  "indicates perfect calibration; lower Brier is better.")


def path_figure(data, feature):
    fig, ax = canvas("LASSO coefficient path")
    names = list(data["FEAT_NAMES"])
    for index, name in enumerate(names):
        selected = name == feature
        ax.plot(data["LOG_C"], data["PATH_COEFS"][:, index],
                color=CELL_COLORS[0] if selected else CELL_COLORS[5],
                linewidth=1, alpha=1 if selected else 0.20,
                label=feature.title() if selected else None,
                zorder=3 if selected else 1)
    ax.axvline(np.log10(data["C_1SE"]), color=CELL_COLORS[1], linestyle="--",
               linewidth=1, label="Selected C (1-SE)")
    ax.set(xlabel="log10(C)", ylabel="Standardized coefficient")
    legend(ax, loc="upper left")
    return finish(fig, "WDBC | Training only. Highlight: selected input; faint "
                  "curves: other candidates. Change the input above to inspect its path.")


def cv_figure(data):
    fig, ax = canvas("Five-fold cross-validation")
    x = np.log10(data["CS"])
    mean, error = data["MEAN_AUC"], data["SE_AUC"]
    ax.plot(x, mean, color=CELL_COLORS[0], linewidth=1, label="Mean AUC")
    ax.fill_between(x, mean - error, mean + error, color=CELL_COLORS[0], alpha=0.16)
    ax.axvline(np.log10(data["C_1SE"]), color=CELL_COLORS[1], linewidth=1,
               linestyle="--", label="Selected C (1-SE)")
    ax.axvline(np.log10(data["C_MIN"]), color=CELL_COLORS[2], linewidth=1,
               linestyle=":", label="Best mean AUC")
    ax.set(xlabel="log10(C)", ylabel="Cross-validated AUC")
    legend(ax, loc="lower right")
    return finish(fig, f"WDBC | Training only; band = +/- 1 SE. "
                  f"The 1-SE rule retains {data['N_SEL']} of {len(data['FEAT_NAMES'])} raw inputs.")


def linearity_figure(data, feature):
    plot = data["LRT_PLOT_DATA"][feature]
    decision = data["EDA_DECISIONS"][feature]
    fig, ax = canvas(feature.title())
    ax.plot(plot["grid"], plot["gam_logit"], color=CELL_COLORS[0], linewidth=1,
            label="GAM")
    ax.fill_between(plot["grid"], plot["gam_lower"], plot["gam_upper"],
                    color=CELL_COLORS[0], alpha=0.15, label="95% CI")
    ax.scatter(plot["mids"], plot["logits"], c=CELL_COLORS[1], marker="o",
               s=16, label="Binned log-odds", zorder=3)
    ax.plot(plot["grid"], plot["linear_logit"], color=CELL_COLORS[2],
            linestyle=":", linewidth=1, label="Linear GLM")
    ax.set(xlabel="Input value", ylabel="Malignancy log-odds")
    legend(ax, loc="best")
    return finish(fig, f"Training diagnostic | edf={decision['gam_edf']:.2f}; "
                  f"LRT p={data['LRT'][feature]['p']:.3g}. "
                  f"Applied: {decision['transformation']} / {decision['functional_form']}.")


def vif_figure(data):
    names = data["SEL_COLS"]
    values = [data["RAW_VIF"][name] for name in names]
    fig, ax = canvas("Raw-input collinearity")
    ax.barh(np.arange(len(names)), values, color=CELL_COLORS[0], edgecolor="none", height=0.55)
    ax.set_yticks(np.arange(len(names)), [textwrap.fill(name.title(), 15) for name in names])
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.30)
    for index, value in enumerate(values):
        ax.text(value + max(values) * 0.025, index, f"{value:.1f}", va="center", fontsize=8)
    ax.axvline(5, color=CELL_COLORS[1], linestyle="--", linewidth=1, label="VIF = 5")
    ax.axvline(10, color=CELL_COLORS[2], linestyle=":", linewidth=1, label="VIF = 10")
    ax.set_xlabel("Variance inflation factor")
    legend(ax, loc="lower right")
    return finish(fig, "WDBC | Training inputs before spline expansion. "
                  "Full design-matrix VIF values are available in Methods.")


def threshold_figure(data, threshold):
    fig, ax = canvas("Training probability distribution")
    probabilities = 1 - data["PROB_TRAIN"]
    for value, label, color in ((0, "Malignant", CELL_COLORS[0]), (1, "Benign", CELL_COLORS[1])):
        ax.hist(probabilities[data["y_tr"] == value], bins=np.linspace(0, 1, 26),
                color=color, alpha=0.65, edgecolor="none", label=label)
    ax.axvline(threshold, color=CELL_COLORS[2], linestyle="--", linewidth=1,
               label=f"Cutoff = {threshold:.2f}")
    ax.set(xlim=(0, 1), xlabel="Predicted malignancy probability", ylabel="Cases")
    legend(ax, loc="upper center")
    return finish(fig, f"WDBC | Training n={data['N_TRAIN']}. "
                  "Probabilities at or above the cutoff are classified as malignant.")
