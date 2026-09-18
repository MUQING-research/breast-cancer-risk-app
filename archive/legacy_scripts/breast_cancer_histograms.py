# Module guide:
# - Role: Preserve a legacy breast-cancer workflow.
# - Workflow: Retain historical preprocessing, exploratory, or modelling behavior for comparison.
# - Design note: Review archived assumptions before using a routine in a new pipeline.
# breast_cancer_histograms.py
# Histograms for all 30 original features + before/after for transformed features
#
# Color coding (training set):
#   Blue   — LRT linear (19 vars, no transformation needed)
#   Green  — Visually linear (5 vars, kept continuous despite LRT failure)
#   Orange — 1/x transform (2 vars: compactness error, fractal dimension error)
#   Red    — Tertile (4 vars: no transformed histogram per user request)
#
# Outputs:
#   histograms_original.png    -- 6×5 grid, all 30 original distributions
#   histograms_transformed.png -- 2×2 before/after for the 2 inv-transformed vars
#   histograms_log.png         -- 6×5 grid, all 30 after log(1+x)
#
# python breast_cancer_histograms.py

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde, shapiro, skew

from _utils import (CELL_COLORS, VISUAL_LINEAR, TRANSFORM_INV, TERTILE_COLS,
                    setup_matplotlib, load_data)

setup_matplotlib(**{
    'axes.labelsize' : 7,
    'axes.titlesize' : 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'grid.linewidth' : 0.4,
})

import numpy as np

CLR_LRT     = CELL_COLORS[3]
CLR_VISUAL  = CELL_COLORS[2]
CLR_TRANS   = CELL_COLORS[4]
CLR_INV     = CELL_COLORS[5]
CLR_TERTILE = CELL_COLORS[0]

# ── Load & split ───────────────────────────────────────────────────────────────
X, y, X_train, X_test, y_train, y_test = load_data()
N        = len(X_train)
features = list(X_train.columns)

SPECIAL    = set(VISUAL_LINEAR) | set(TRANSFORM_INV) | set(TERTILE_COLS)
LRT_LINEAR = [c for c in features if c not in SPECIAL]

print(f"Training samples : {N}")
print(f"LRT linear       : {len(LRT_LINEAR)}")
print(f"Visually linear  : {len(VISUAL_LINEAR)}")
print(f"1/x transform    : {len(TRANSFORM_INV)}")
print(f"Tertile          : {len(TERTILE_COLS)}")
print(f"Total            : {len(LRT_LINEAR)+len(VISUAL_LINEAR)+len(TRANSFORM_INV)+len(TERTILE_COLS)}")


# ── helpers ────────────────────────────────────────────────────────────────────
N_BINS = 28

# Function guide: draw_hist is responsible for draw hist.
# Inputs: ax, x, color, title, xlabel, alpha_hist. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def draw_hist(ax, x, color, title, xlabel='Value', alpha_hist=0.60):
    """Histogram + KDE on ax."""
    ax.hist(x, bins=N_BINS, density=True,
            color=color, alpha=alpha_hist,
            edgecolor='none')
    try:
        kde  = gaussian_kde(x, bw_method='scott')
        xg   = np.linspace(x.min(), x.max(), 300)
        ax.plot(xg, kde(xg), color=color, lw=1.6, alpha=0.95)
    except Exception:
        pass
    _, sw_p  = shapiro(x)
    sk       = skew(x)
    med      = np.median(x)
    ax.axvline(med, color='#333333', lw=0.8, ls='--', alpha=0.7)
    info = f"med={med:.3g}  sk={sk:.2f}  SW p={'<0.001' if sw_p < 0.001 else f'{sw_p:.3f}'}"
    ax.text(0.98, 0.97, info,
            transform=ax.transAxes, fontsize=4.8, ha='right', va='top',
            color='#333333',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.6))
    ax.set_title(title, fontsize=6.5, fontweight='bold', color=color, pad=2)
    ax.set_xlabel(xlabel, fontsize=6)
    ax.set_ylabel('Density', fontsize=6)


# ── Figure 1: all 30 original histograms (6 rows × 5 cols) ───────────────────
NROWS, NCOLS = 6, 5
fig1, axes1 = plt.subplots(NROWS, NCOLS, figsize=(15, 18))
axes_flat = axes1.flatten()

for i, col in enumerate(features):
    if col in LRT_LINEAR:
        color, tag = CLR_LRT,     'LRT linear'
    elif col in VISUAL_LINEAR:
        color, tag = CLR_VISUAL,  'visual linear'
    elif col in TRANSFORM_INV:
        color, tag = CLR_TRANS,   '1/x → linear'
    else:
        color, tag = CLR_TERTILE, 'tertile'
    draw_hist(axes_flat[i], X_train[col].values, color, title=f'{col}  [{tag}]')

legend_elements = [
    Patch(facecolor=CLR_LRT,     alpha=0.75, label=f'LRT linear (n={len(LRT_LINEAR)})'),
    Patch(facecolor=CLR_VISUAL,  alpha=0.75, label=f'Visually linear (n={len(VISUAL_LINEAR)})'),
    Patch(facecolor=CLR_TRANS,   alpha=0.75, label=f'1/x transform (n={len(TRANSFORM_INV)})'),
    Patch(facecolor=CLR_TERTILE, alpha=0.75, label=f'Tertile (n={len(TERTILE_COLS)})'),
]
fig1.legend(handles=legend_elements, loc='upper center',
            ncol=4, fontsize=8.5, framealpha=0.85,
            bbox_to_anchor=(0.5, 1.002))
plt.tight_layout()
plt.savefig('histograms_original.png')
plt.close(fig1)
print("\nSaved: histograms_original.png")


# ── Figure 2: before / after 1/x for the 2 transformed variables ─────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(11, 7))

for row, col in enumerate(TRANSFORM_INV):
    x_orig = X_train[col].values
    x_inv  = 1.0 / np.where(np.abs(x_orig) < 1e-12, 1e-12, x_orig)
    draw_hist(axes2[row, 0], x_orig, CLR_TRANS,
              title=f'{col}  [original]', xlabel=col)
    draw_hist(axes2[row, 1], x_inv, CLR_INV,
              title=f'1 / ({col})  [after 1/x transform]', xlabel=f'1 / {col}')

for j, lbl in enumerate(['Original', 'After 1/x transform']):
    axes2[0, j].set_title(
        f'{axes2[0, j].get_title()}\n',
        fontsize=6.5, fontweight='bold',
        color=CLR_TRANS if j == 0 else CLR_INV, pad=2)

legend2 = [
    Patch(facecolor=CLR_TRANS, alpha=0.75, label='Original'),
    Patch(facecolor=CLR_INV,   alpha=0.75, label='After 1/x transform'),
]
fig2.legend(handles=legend2, loc='upper right', fontsize=9, framealpha=0.85)
plt.tight_layout()
plt.savefig('histograms_transformed.png')
plt.close(fig2)
print("Saved: histograms_transformed.png")


# ── Figure 3: log1p-transformed histograms for all 30 features ───────────────
CLR_LOG  = '#555555'
X_log    = np.log1p(X_train.values)

fig3, axes3 = plt.subplots(NROWS, NCOLS, figsize=(15, 18))
axes3_flat = axes3.flatten()

for i, col in enumerate(features):
    x_log = X_log[:, i]
    _, sw_p = shapiro(x_log)
    sk_val  = skew(x_log)
    med_val = np.median(x_log)

    ax = axes3_flat[i]
    ax.hist(x_log, bins=N_BINS, density=True,
            color=CLR_LOG, alpha=0.60, edgecolor='none')
    try:
        kde = gaussian_kde(x_log, bw_method='scott')
        xg  = np.linspace(x_log.min(), x_log.max(), 300)
        ax.plot(xg, kde(xg), color=CLR_LOG, lw=1.6, alpha=0.95)
    except Exception:
        pass
    ax.axvline(med_val, color=CELL_COLORS[7], lw=0.9, ls='--', alpha=0.8)
    info = (f"med={med_val:.3g}  sk={sk_val:.2f}  "
            f"SW p={'<0.001' if sw_p < 0.001 else f'{sw_p:.3f}'}")
    ax.text(0.98, 0.97, info,
            transform=ax.transAxes, fontsize=4.8, ha='right', va='top',
            color='#333333',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.6))
    ax.set_title(f'log(1+{col})', fontsize=6.5, fontweight='bold',
                 color=CLR_LOG, pad=2)
    ax.set_xlabel('log(1 + x)', fontsize=6)
    ax.set_ylabel('Density', fontsize=6)

plt.tight_layout()
plt.savefig('histograms_log.png')
plt.close(fig3)
print("Saved: histograms_log.png")
