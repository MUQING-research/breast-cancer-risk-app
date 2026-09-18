# Module guide:
# - Role: Run leakage-safe breast-cancer EDA and preprocessing.
# - Workflow: Fit transformations on training data and export the design matrix and preprocessing decisions.
# - Design note: Preserve the training-only boundary for every data-dependent choice.
"""Offline, training-only diagnostics and fitted model design construction."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer, PolynomialFeatures, PowerTransformer,
    RobustScaler, SplineTransformer,
)

mpl.rcParams.update({
    'font.family': 'Arial', 'font.size': 9,
    'axes.titlesize': 10, 'axes.titleweight': 'bold',
    'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'axes.spines.top': True, 'axes.spines.right': True,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
CELL_COLORS = [
    '#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F',
    '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85',
]


# Function guide: transformation is responsible for perform the requested operation.
# Inputs: name. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def transformation(name):
    if name == 'log1p':
        return FunctionTransformer(np.log1p, feature_names_out='one-to-one')
    if name == 'sqrt':
        return FunctionTransformer(np.sqrt, feature_names_out='one-to-one')
    if name == 'yeo_johnson':
        return PowerTransformer(method='yeo-johnson', standardize=False)
    return FunctionTransformer(feature_names_out='one-to-one')


# Function guide: _linear_fit is responsible for perform fit.
# Inputs: values, target. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _linear_fit(values, target):
    fitted = Pipeline([
        ('scale', RobustScaler()),
        ('model', LogisticRegression(penalty=None, max_iter=5000, tol=1e-8)),
    ]).fit(values, target)
    loss = log_loss(target, fitted.predict_proba(values)[:, 1])
    return fitted, float(loss)


# Function guide: _lrt is responsible for perform the requested operation.
# Inputs: values, target. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _lrt(values, target):
    linear, loss = _linear_fit(values, target)
    spline = SplineTransformer(n_knots=3, degree=3, include_bias=False)
    basis = spline.fit_transform(values)
    _, spline_loss = _linear_fit(basis, target)
    statistic = max(0.0, 2 * len(target) * (loss - spline_loss))
    degrees = (np.linalg.matrix_rank(np.column_stack([np.ones(len(values)), basis]))
               - np.linalg.matrix_rank(np.column_stack([np.ones(len(values)), values])))
    return float(statistic), float(stats.chi2.sf(statistic, degrees)), linear


# Function guide: screen_variables is responsible for perform variables.
# Inputs: X_train, y_train, linear_only. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def screen_variables(X_train, y_train, linear_only=False):
    """Screen diagnostics without accessing test rows.

    When ``linear_only`` is true, preprocessing transformations are retained,
    but the final model is restricted to linear terms. Diagnostic GAM and LRT
    summaries remain available for the audit trail.
    """
    from pygam import LogisticGAM, s

    target = 1 - np.asarray(y_train, dtype=int)
    decisions, plot_data, summaries = {}, {}, []
    for name in X_train.columns:
        raw = X_train[[name]].to_numpy(dtype=float)
        raw_skew = float(stats.skew(raw[:, 0], bias=False))
        candidates = ['raw']
        if abs(raw_skew) >= 0.5:
            candidates += ['log1p', 'sqrt', 'yeo_johnson']
        reports, fitted = {}, {}
        for candidate in candidates:
            transformer = transformation(candidate)
            values = transformer.fit_transform(raw)
            _, loss = _linear_fit(values, target)
            reports[candidate] = {
                'skewness': float(stats.skew(values[:, 0], bias=False)),
                'excess_kurtosis': float(stats.kurtosis(values[:, 0], bias=False)),
                'linear_log_loss': loss,
            }
            fitted[candidate] = (transformer, values)
        choice = 'raw'
        acceptable = [
            candidate for candidate in candidates if candidate != 'raw'
            and abs(reports[candidate]['skewness']) < abs(raw_skew) * 0.9
            and 2 * len(target) * (reports['raw']['linear_log_loss']
                                   - reports[candidate]['linear_log_loss']) > 2
        ]
        if acceptable:
            choice = min(acceptable, key=lambda item: reports[item]['linear_log_loss'])
        transformer, values = fitted[choice]
        statistic, lrt_p, linear = _lrt(values, target)
        gam = LogisticGAM(s(0, n_splines=10)).gridsearch(
            values, target, progress=False)
        edf = float(gam.statistics_['edof'])
        smooth_p = float(gam.statistics_['p_values'][0])
        if linear_only:
            form = 'linear'
        elif edf <= 1.3 or smooth_p >= 0.10:
            form = 'linear'
        elif edf <= 2.0:
            form = 'quadratic'
        else:
            form = 'spline'
        grid_raw = np.linspace(raw.min(), raw.max(), 100)
        grid = transformer.transform(grid_raw.reshape(-1, 1))
        probability = np.clip(gam.predict_mu(grid), 1e-9, 1 - 1e-9)
        interval = np.clip(gam.confidence_intervals(grid), 1e-9, 1 - 1e-9)
        logit = lambda probability: np.log(probability / (1 - probability))
        cuts = np.unique(np.quantile(raw[:, 0], np.linspace(0, 1, 11)))
        bin_ids = np.digitize(raw[:, 0], cuts[1:-1])
        mids, logits = [], []
        for group in np.unique(bin_ids):
            mask = bin_ids == group
            probability_bin = (target[mask].sum() + 0.5) / (mask.sum() + 1)
            mids.append(float(raw[mask].mean()))
            logits.append(float(logit(probability_bin)))
        plot_data[name] = {
            'grid': grid_raw.tolist(), 'mids': mids, 'logits': logits,
            'linear_logit': linear.decision_function(grid).tolist(),
            'gam_logit': logit(probability).tolist(),
            'gam_lower': logit(interval[:, 0]).tolist(),
            'gam_upper': logit(interval[:, 1]).tolist(),
        }
        decisions[name] = {
            'variable_type': 'continuous', 'missing_rule': 'reject_missing',
            'missing_train_count': int(X_train[name].isna().sum()),
            'transformation': choice, 'functional_form': form, 'encoding': 'numeric',
            'candidate_screen': reports, 'gam_edf': edf, 'gam_smooth_p': smooth_p,
            'gam_p_interpretation': 'Approximate smooth-term association p-value, not a test of nonlinearity; smoothing is estimated.',
            'lrt': {'chi2': statistic, 'p': lrt_p, 'linear': lrt_p >= 0.10},
            'centering_constant': float(values.mean()) if form == 'quadratic' else None,
            'power_lambda': float(transformer.lambdas_[0]) if choice == 'yeo_johnson' else None,
            'n_knots': 3 if form == 'spline' else None,
            'decision_rationale': (
                'Preprocessing selects the variable transformation; the final model '
                'uses the selected transformed variable as a linear term. GAM and '
                'LRT results are retained for audit.'
                if linear_only else
                'Transformation requires improved skewness and linear likelihood; '
                'form follows the prespecified GAM edf/p rule, with LRT reported independently.'
            ),
        }
        summaries.append({
            'feature': name, 'n': len(raw), 'missing': int(np.isnan(raw).sum()),
            'mean': float(raw.mean()), 'sd': float(raw.std(ddof=1)),
            'median': float(np.median(raw)), 'minimum': float(raw.min()),
            'maximum': float(raw.max()), 'shapiro_p': float(stats.shapiro(raw[:, 0]).pvalue),
            'skewness': raw_skew, 'excess_kurtosis': float(stats.kurtosis(raw[:, 0], bias=False)),
            'mann_whitney_p': float(stats.mannwhitneyu(raw[target == 0, 0], raw[target == 1, 0]).pvalue),
        })
    return decisions, plot_data, summaries


# Function guide: build_design_matrix is responsible for build design matrix.
# Inputs: selected_features, decisions. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def build_design_matrix(selected_features, decisions):
    """Apply preprocessing transformations and return linear model terms only."""
    transformers = []
    for index, name in enumerate(selected_features):
        steps = [('transform', transformation(decisions[name]['transformation']))]
        transformers.append((f'feature{index}', Pipeline(steps), [index]))
    return ColumnTransformer(transformers, remainder='drop', verbose_feature_names_out=True)


# Function guide: export_diagnostics is responsible for export diagnostics.
# Inputs: X_train, target, decisions, plot_data, summaries, output_dir. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def export_diagnostics(X_train, target, decisions, plot_data, summaries, output_dir):
    """Export training-only diagnostics; all categorical routes are inapplicable."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(output_dir / 'eda_summary_stats.csv', index=False)
    n_columns, n_rows = 3, int(np.ceil(X_train.shape[1] / 3))
    for mode, filename in [('histogram', 'eda_continuous_histograms.png'),
                           ('qq', 'eda_qq_plots.png')]:
        fig, axes = plt.subplots(n_rows, n_columns, figsize=(10.5, 2.5 * n_rows))
        for index, name in enumerate(X_train.columns):
            ax = axes.flat[index]
            values = X_train[name].to_numpy()
            if mode == 'histogram':
                bins = np.histogram_bin_edges(values, bins=20)
                for outcome, label in ((0, 'Malignant'), (1, 'Benign')):
                    ax.hist(values[np.asarray(target) == outcome], bins=bins, density=True,
                            color=CELL_COLORS[outcome], alpha=0.5, edgecolor='none', label=label)
                ax.set_ylabel('Density')
            else:
                (theoretical, observed), (slope, intercept, _) = stats.probplot(values)
                ax.plot(theoretical, observed, 'o', markersize=4, color=CELL_COLORS[0])
                ax.plot(theoretical, slope * theoretical + intercept, color=CELL_COLORS[1], lw=1)
                row = summaries[index]
                ax.text(0.02, 0.98, f"SW p={row['shapiro_p']:.3g}\nSkew={row['skewness']:.2f}; kurt={row['excess_kurtosis']:.2f}",
                        transform=ax.transAxes, va='top', fontsize=8)
                ax.set_xlabel('Normal quantile')
            ax.set_title(name.title())
            ax.grid(False)
        plt.tight_layout()
        fig.savefig(output_dir / filename, dpi=300)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    counts = [(np.asarray(target) == code).sum() for code in (0, 1)]
    ax.bar(['Malignant', 'Benign'], counts, color=CELL_COLORS[:2], edgecolor='none')
    ax.set_ylabel('Training count')
    ax.set_title('Target class balance')
    plt.tight_layout()
    fig.savefig(output_dir / 'eda_target_categorical.png', dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.text(0.5, 0.5, 'Not applicable:\nall predictors are continuous.', ha='center', va='center')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(output_dir / 'eda_linearity_categorical.png', dpi=300)
    plt.close(fig)
    pearson, spearman = X_train.corr(), X_train.corr(method='spearman')
    fig, ax = plt.subplots(figsize=(18, 18))
    ax.imshow(pearson, cmap='RdBu_r', vmin=-1, vmax=1)
    for row in range(len(pearson)):
        for col in range(len(pearson)):
            ax.text(col, row, f'r {pearson.iloc[row, col]:.2f}\nrho {spearman.iloc[row, col]:.2f}',
                    ha='center', va='center', fontsize=6)
    ax.set_xticks(range(len(pearson)), pearson.columns, rotation=90)
    ax.set_yticks(range(len(pearson)), pearson.columns)
    plt.tight_layout()
    fig.savefig(output_dir / 'eda_correlation_matrix.png', dpi=300)
    plt.close(fig)
    flagged = [
        {'first': pearson.columns[i], 'second': pearson.columns[j],
         'pearson': float(pearson.iloc[i, j]), 'spearman': float(spearman.iloc[i, j])}
        for i in range(len(pearson)) for j in range(i + 1, len(pearson))
        if abs(pearson.iloc[i, j]) > 0.8 or abs(spearman.iloc[i, j]) > 0.8
    ]
    (output_dir / 'collinearity_flags.json').write_text(json.dumps(flagged, indent=2), encoding='utf-8')


# Function guide: export_form_diagnostics is responsible for export form diagnostics.
# Inputs: decisions, output_dir. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def export_form_diagnostics(decisions, output_dir):
    selected = [(name, decision) for name, decision in decisions.items() if decision['selected']]
    fig, ax = plt.subplots(figsize=(10.5, 3.5))
    ax.axis('off')
    rows = [[name, item['transformation'], f"{item['gam_edf']:.2f}",
             f"{item['gam_smooth_p']:.3g}", f"{item['lrt']['p']:.3g}", item['functional_form']]
            for name, item in selected]
    table = ax.table(cellText=rows, colLabels=['Feature', 'Transform', 'GAM edf', 'Smooth p', 'LRT p', 'Form'],
                     loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)
    plt.tight_layout()
    fig.savefig(Path(output_dir) / 'eda_linearity_table.png', dpi=300)
    plt.close(fig)
