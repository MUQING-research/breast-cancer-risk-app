# Module guide:
# - Role: Preserve a legacy breast-cancer workflow.
# - Workflow: Retain historical preprocessing, exploratory, or modelling behavior for comparison.
# - Design note: Review archived assumptions before using a routine in a new pipeline.
# breast_cancer_lasso.py
# LASSO logistic regression — three preprocessing pipelines comparison
#
# Pipeline 1 (P1): raw features, no transformation
# Pipeline 2 (P2): 1/x for 2 vars; keep original for remaining 28 (incl. 4 non-linear)
# Pipeline 3 (P3): 1/x for 2 vars; tertile dummies for 4 non-linear vars; 24 original
#
# Scaling: RobustScaler on continuous features
# Model:   LASSO logistic regression (LogisticRegressionCV, l1, 10-fold CV, AUC)
#
# Outputs:
#   scaling_decision.png  -- distribution inspection + normality test + scaler comparison
#   lasso_cv_curve.png    -- 10-fold CV AUC curve for each pipeline
#   lasso_comparison.png  -- ROC curves + performance bar chart
#   lasso_path.png        -- regularization paths for all 3 pipelines
#   lasso_results.csv     -- test-set metrics for all 3 pipelines
#
# python breast_cancer_lasso.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

from _utils import (CELL_COLORS, setup_matplotlib, load_data, get_tertile_cuts,
                    build_p1, build_p2, build_p3, scale_fit_transform, evaluate)

setup_matplotlib()

CLR_P1  = CELL_COLORS[0]
CLR_P2  = CELL_COLORS[3]
CLR_P3  = CELL_COLORS[2]
CLR_REF = CELL_COLORS[5]


# ── Step 1  Load & split ──────────────────────────────────────────────────────
X, y, X_train, X_test, y_train, y_test = load_data()
y_tr = y_train.values.astype(int)
y_te = y_test.values.astype(int)

print("=" * 70)
print("Step 1  Data loaded and split")
print("=" * 70)
print(f"  Train : {len(y_tr)}  (malignant {sum(y_tr==0)}, benign {sum(y_tr==1)})")
print(f"  Test  : {len(y_te)}   (malignant {sum(y_te==0)},  benign {sum(y_te==1)})")

tertile_cuts = get_tertile_cuts(X_train)


# ── Step 2  Scaling decision ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Step 2  Scaling decision (Shapiro-Wilk + distribution inspection)")
print("=" * 70)

sw_pvals = {}
for col in X_train.columns:
    _, p = stats.shapiro(X_train[col].values)
    sw_pvals[col] = p

n_normal   = sum(p >= 0.05 for p in sw_pvals.values())
n_nonnorm  = len(sw_pvals) - n_normal
ranges_raw = X_train.max() - X_train.min()

print(f"  Shapiro-Wilk (n={len(y_tr)}): {n_nonnorm}/30 features non-normal (p<0.05)")
print(f"  Raw feature range: {ranges_raw.min():.5f} – {ranges_raw.max():.1f}  "
      f"(ratio {ranges_raw.max()/ranges_raw.min():.0f}x)")
print(f"  Decision: RobustScaler")
print(f"    Rationale: most features non-normal with right skew and outliers;")
print(f"    MinMaxScaler compresses IQR when outliers are present;")
print(f"    RobustScaler (median/IQR) is more robust than StandardScaler for skewed data.")

bp_kw_raw = dict(
    vert=True, patch_artist=True, widths=0.55,
    boxprops=dict(facecolor='#D6E4F7', color=CLR_P2, linewidth=0.7),
    medianprops=dict(color=CLR_P1, linewidth=1.4),
    whiskerprops=dict(color='#555555', linewidth=0.6),
    capprops=dict(color='#555555', linewidth=0.6),
    flierprops=dict(marker='o', color='#AAAAAA', markersize=1.8, alpha=0.5),
)
bp_kw_std = dict(**bp_kw_raw)
bp_kw_mm  = dict(
    **{k: v for k, v in bp_kw_raw.items() if k not in ('boxprops',)},
    boxprops=dict(facecolor='#FCEAE8', color=CLR_P1, linewidth=0.7),
)

fig1, axes1 = plt.subplots(2, 2, figsize=(15, 9))
tick_kw   = dict(fontsize=5, rotation=90)
cols_list = list(X_train.columns)

ax = axes1[0, 0]
ax.boxplot(X_train.values, **bp_kw_raw)
ax.set_yscale('log')
ax.set_xticks(range(1, 31))
ax.set_xticklabels(cols_list, **tick_kw)
ax.set_ylabel('Value (log scale)', fontsize=7)
ax.set_title('Raw features (log scale)\nRange ratio up to '
             f'{ranges_raw.max()/ranges_raw.min():.0f}x — scaling required',
             fontsize=8, fontweight='bold')

ax = axes1[0, 1]
sorted_items = sorted(sw_pvals.items(), key=lambda x: x[1])
sorted_cols  = [k for k, _ in sorted_items]
sorted_pv    = [v for _, v in sorted_items]
bar_colors   = [CLR_P3 if p >= 0.05 else CLR_P1 for p in sorted_pv]
ax.bar(range(len(sorted_cols)), sorted_pv, color=bar_colors, width=0.8, alpha=0.85)
ax.axhline(0.05, color='#333333', lw=1.0, ls='--', label='p = 0.05')
ax.set_xticks(range(len(sorted_cols)))
ax.set_xticklabels(sorted_cols, **tick_kw)
ax.set_ylabel('Shapiro-Wilk p-value', fontsize=7)
ax.set_title(
    f'Shapiro-Wilk normality test  (n={len(y_tr)})\n'
    f'{n_nonnorm}/30 non-normal (red, p<0.05) — {n_normal}/30 normal (green)',
    fontsize=8, fontweight='bold')
ax.legend(fontsize=7, framealpha=0.7)

ax = axes1[1, 0]
X_std = RobustScaler().fit_transform(X_train)
ax.boxplot(X_std, **bp_kw_std)
ax.axhline(0, color=CLR_REF, lw=0.7, ls='--')
ax.set_xticks(range(1, 31))
ax.set_xticklabels(cols_list, **tick_kw)
ax.set_ylabel('(x − median) / IQR', fontsize=7)
ax.set_title('After RobustScaler (median=0, IQR=1)\nStable with skewed data & outliers — selected',
             fontsize=8, fontweight='bold', color=CLR_P3)

ax = axes1[1, 1]
X_mm = MinMaxScaler().fit_transform(X_train)
ax.boxplot(X_mm, **bp_kw_mm)
ax.set_xticks(range(1, 31))
ax.set_xticklabels(cols_list, **tick_kw)
ax.set_ylabel('Normalized value [0, 1]', fontsize=7)
ax.set_title('After MinMaxScaler [0, 1]\nOutliers compress IQR — not recommended',
             fontsize=8, fontweight='bold', color=CLR_P1)

plt.tight_layout()
plt.savefig('scaling_decision.png')
plt.close(fig1)
print("\n  Saved: scaling_decision.png")


# ── Step 3  Build & scale pipelines ──────────────────────────────────────────
X_tr_p1, X_te_p1 = scale_fit_transform(build_p1(X_train), build_p1(X_test))
X_tr_p2, X_te_p2 = scale_fit_transform(build_p2(X_train), build_p2(X_test))
X_tr_p3, X_te_p3 = scale_fit_transform(
    build_p3(X_train, tertile_cuts), build_p3(X_test, tertile_cuts),
    dummy_suffixes=['_T2', '_T3'])

print("\n" + "=" * 70)
print("Step 3  Pipelines built and scaled")
print("=" * 70)
print(f"  P1 (raw)          : {X_tr_p1.shape[1]} features")
print(f"  P2 (transformed)  : {X_tr_p2.shape[1]} features")
print(f"  P3 (trans+tertile): {X_tr_p3.shape[1]} features")


# ── Step 4  LASSO (10-fold stratified CV, AUC, λ1se rule) ────────────────────
print("\n" + "=" * 70)
print("Step 4  LASSO — 10-fold CV, λ1se rule (AUC)")
print("=" * 70)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
Cs = np.logspace(-4, 2, 60)


# Function guide: fit_lasso_1se is responsible for fit lasso 1se.
# Inputs: X_tr, y_tr, label. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def fit_lasso_1se(X_tr, y_tr, label):
    model_cv = LogisticRegressionCV(
        Cs=Cs, penalty='l1', solver='liblinear',
        cv=cv, scoring='roc_auc', max_iter=5000, random_state=42,
    )
    model_cv.fit(X_tr.values, y_tr)

    cv_scores = list(model_cv.scores_.values())[0]
    n_folds   = cv_scores.shape[0]
    mean_auc  = cv_scores.mean(axis=0)
    se_auc    = cv_scores.std(axis=0, ddof=1) / np.sqrt(n_folds)

    idx_min = int(np.argmax(mean_auc))
    C_min   = Cs[idx_min]
    nz_min  = int((model_cv.coef_[0] != 0).sum())

    threshold = mean_auc[idx_min] - se_auc[idx_min]
    idx_1se   = int(np.where(mean_auc >= threshold)[0][0])
    C_1se     = Cs[idx_1se]

    model = LogisticRegression(
        penalty='l1', solver='liblinear', C=C_1se,
        max_iter=5000, random_state=42,
    )
    model.fit(X_tr.values, y_tr)
    n_nz = int((model.coef_[0] != 0).sum())

    print(f"  {label:<22}  λmin C={C_min:.5f} ({nz_min} vars)"
          f"  →  λ1se C={C_1se:.5f} ({n_nz} vars)")

    return model, C_1se, C_min, mean_auc, se_auc


model_p1, C1se_p1, Cmin_p1, auc_mean_p1, auc_se_p1 = fit_lasso_1se(X_tr_p1, y_tr, 'P1 (raw)')
model_p2, C1se_p2, Cmin_p2, auc_mean_p2, auc_se_p2 = fit_lasso_1se(X_tr_p2, y_tr, 'P2 (transformed)')
model_p3, C1se_p3, Cmin_p3, auc_mean_p3, auc_se_p3 = fit_lasso_1se(X_tr_p3, y_tr, 'P3 (trans+tertile)')


# ── CV curve figure ───────────────────────────────────────────────────────────
log_Cs   = np.log10(Cs)
fig_cv, axes_cv = plt.subplots(1, 3, figsize=(16, 4.5))

cv_items = [
    ('P1: Raw',             CLR_P1, auc_mean_p1, auc_se_p1, C1se_p1, Cmin_p1),
    ('P2: Transformed',     CLR_P2, auc_mean_p2, auc_se_p2, C1se_p2, Cmin_p2),
    ('P3: Trans + Tertile', CLR_P3, auc_mean_p3, auc_se_p3, C1se_p3, Cmin_p3),
]

for ax, (lbl, clr, m_auc, s_auc, c1se, cmin) in zip(axes_cv, cv_items):
    ax.fill_between(log_Cs, m_auc - s_auc, m_auc + s_auc, color=clr, alpha=0.18)
    ax.plot(log_Cs, m_auc, color=clr, lw=1.8)
    ax.axvline(np.log10(cmin), color='#555555', lw=1.0, ls='--',
               label=f'λmin  C={cmin:.4f}')
    ax.axvline(np.log10(c1se), color=clr, lw=1.2, ls=':',
               label=f'λ1se  C={c1se:.4f}')
    ax.axhline(m_auc[np.argmax(m_auc)] - s_auc[np.argmax(m_auc)],
               color='#AAAAAA', lw=0.7, ls='--')
    ax.set_xlabel('log₁₀(C)', fontsize=8)
    ax.set_ylabel('CV AUC (mean ± 1SE)', fontsize=8)
    ax.set_title(lbl, fontsize=9, fontweight='bold', color=clr)
    ax.legend(fontsize=7.5, framealpha=0.85)

plt.tight_layout()
plt.savefig('lasso_cv_curve.png')
plt.close(fig_cv)
print("\n  Saved: lasso_cv_curve.png")


# ── Step 5  Test-set evaluation ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("Step 5  Test-set evaluation")
print("=" * 70)

res = {}
for key, model, Xte, label in [
    ('P1', model_p1, X_te_p1, 'P1: Raw'),
    ('P2', model_p2, X_te_p2, 'P2: Transformed'),
    ('P3', model_p3, X_te_p3, 'P3: Trans + Tertile'),
]:
    r = evaluate(model, Xte, y_te, label)
    r['C']          = model.C
    r['n_nonzero']  = int((model.coef_[0] != 0).sum())
    r['n_features'] = Xte.shape[1]
    res[key] = r
    print(f"  {label}")
    print(f"    AUC={r['auc']:.4f}  Acc={r['acc']:.4f}  Sens={r['sens']:.4f}  "
          f"Spec={r['spec']:.4f}  PPV={r['ppv']:.4f}  F1={r['f1']:.4f}")
    print(f"    Selected features: {r['n_nonzero']}/{r['n_features']}  "
          f"| C={r['C']:.5f}")


# ── Step 6  Figure: ROC curves + performance bar chart ───────────────────────
COLORS = {'P1': CLR_P1, 'P2': CLR_P2, 'P3': CLR_P3}
LABELS = {'P1': 'P1: Raw', 'P2': 'P2: Transformed', 'P3': 'P3: Trans + Tertile'}

fig2, (ax_roc, ax_bar) = plt.subplots(1, 2, figsize=(14, 5.5))

for key, r in res.items():
    ax_roc.plot(r['fpr'], r['tpr'], color=COLORS[key], lw=2.0, alpha=0.9,
                label=f"{LABELS[key]}  AUC={r['auc']:.4f}")
ax_roc.plot([0, 1], [0, 1], color=CLR_REF, lw=0.9, ls='--', label='Chance')
ax_roc.set_xlabel('1 – Specificity (FPR)', fontsize=9)
ax_roc.set_ylabel('Sensitivity (TPR)', fontsize=9)
ax_roc.set_title('ROC Curves — Test Set', fontsize=10, fontweight='bold')
ax_roc.legend(fontsize=8.5, loc='lower right', framealpha=0.85)
ax_roc.set_xlim(-0.02, 1.02)
ax_roc.set_ylim(-0.02, 1.02)

metrics = ['auc', 'acc', 'sens', 'spec', 'ppv', 'f1']
mlabels = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'F1']
x     = np.arange(len(metrics))
width = 0.26

for i, (key, r) in enumerate(res.items()):
    vals = [r[m] for m in metrics]
    bars = ax_bar.bar(x + (i - 1) * width, vals, width,
                      label=LABELS[key], color=COLORS[key],
                      alpha=0.85, edgecolor='none')
    for bar, val in zip(bars, vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.006,
                    f'{val:.3f}',
                    ha='center', va='bottom', fontsize=5.5, rotation=90,
                    color='#333333')

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(mlabels, fontsize=9)
ax_bar.set_ylim(0, 1.18)
ax_bar.set_ylabel('Score', fontsize=9)
ax_bar.set_title('Performance Metrics — Test Set', fontsize=10, fontweight='bold')
ax_bar.legend(fontsize=8.5, loc='upper right', framealpha=0.85)
ax_bar.axhline(1.0, color='#DDDDDD', lw=0.6, ls=':')

for key, r in res.items():
    ax_bar.text(
        len(metrics) - 0.5 + (list(res.keys()).index(key) - 1) * width,
        0.02,
        f"n={r['n_nonzero']}/{r['n_features']}",
        ha='center', va='bottom', fontsize=5.5, color=COLORS[key], rotation=90)

plt.tight_layout()
plt.savefig('lasso_comparison.png')
plt.close(fig2)
print("\n  Saved: lasso_comparison.png")


# ── Step 7  LASSO regularization paths (all 3 pipelines) ─────────────────────
print("\n" + "=" * 70)
print("Step 7  LASSO regularization paths")
print("=" * 70)

C_path  = np.logspace(-4, 2, 120)
log10_C = np.log10(C_path)

PANEL_COLORS = {
    'P1: Raw'            : CLR_P1,
    'P2: Transformed'    : CLR_P2,
    'P3: Trans + Tertile': CLR_P3,
}

path_data = [
    ('P1: Raw',             X_tr_p1, Cmin_p1, C1se_p1),
    ('P2: Transformed',     X_tr_p2, Cmin_p2, C1se_p2),
    ('P3: Trans + Tertile', X_tr_p3, Cmin_p3, C1se_p3),
]

fig_path, axes_path = plt.subplots(3, 1, figsize=(14, 15))

for ax, (label, X_tr, c_min, c_1se) in zip(axes_path, path_data):
    print(f"  {label} ...", end=' ', flush=True)
    coefs = np.zeros((len(C_path), X_tr.shape[1]))
    for i, c in enumerate(C_path):
        m = LogisticRegression(penalty='l1', solver='liblinear', C=c, max_iter=5000)
        m.fit(X_tr.values, y_tr)
        coefs[i] = m.coef_[0]
    print("done")

    feat_names = list(X_tr.columns)
    idx_opt    = np.argmin(np.abs(C_path - c_1se))
    nz_mask    = coefs[idx_opt] != 0
    nz_idx     = np.where(nz_mask)[0]
    n_nz       = nz_mask.sum()
    clr        = PANEL_COLORS[label]
    cmap       = cm.get_cmap('tab20', max(n_nz, 1))

    for fi in range(len(feat_names)):
        if not nz_mask[fi]:
            ax.plot(log10_C, coefs[:, fi], color='#DDDDDD', lw=0.7, alpha=0.7, zorder=1)

    legend_handles = []
    for k, fi in enumerate(nz_idx):
        line, = ax.plot(log10_C, coefs[:, fi], color=cmap(k),
                        lw=1.5, alpha=0.92, zorder=2, label=feat_names[fi])
        legend_handles.append(line)

    ax.axvline(np.log10(c_min), color='#555555', lw=1.1, ls='--', zorder=3)
    ax.axvline(np.log10(c_1se), color=clr,       lw=1.3, ls=':',  zorder=3)
    ax.axhline(0, color='#AAAAAA', lw=0.6)

    ax.set_xlabel(r'$\log_{10}(C)$   ←  stronger regularization  |  weaker  →', fontsize=8)
    ax.set_ylabel('Coefficient', fontsize=8)
    ax.set_title(
        f'{label}  —  LASSO path  ({n_nz}/{len(feat_names)} features selected at λ1se)',
        fontsize=9, fontweight='bold', color=clr)
    ax.set_xlim(log10_C.min(), log10_C.max())

    lmin_handle = plt.Line2D([0], [0], color='#555555', lw=1.1, ls='--',
                              label=f'λmin  C={c_min:.4f}')
    l1se_handle = plt.Line2D([0], [0], color=clr, lw=1.3, ls=':',
                              label=f'λ1se  C={c_1se:.4f}  ({n_nz} vars)')
    leg = ax.legend(
        handles=legend_handles + [lmin_handle, l1se_handle],
        title=f'Selected (n={n_nz}) + λ lines', title_fontsize=7,
        bbox_to_anchor=(1.01, 1), loc='upper left',
        fontsize=6.5, framealpha=0.85, handlelength=1.5,
        borderpad=0.6, labelspacing=0.4)
    leg.get_title().set_fontweight('bold')

plt.tight_layout()
plt.savefig('lasso_path.png')
plt.close(fig_path)
print("\n  Saved: lasso_path.png")


# ── Step 8  Export results ────────────────────────────────────────────────────
rows = []
for key, r in res.items():
    rows.append({
        'pipeline'            : key,
        'description'         : r['label'],
        'n_features_total'    : r['n_features'],
        'n_features_selected' : r['n_nonzero'],
        'optimal_C'           : round(r['C'], 6),
        'AUC'                 : round(r['auc'],  4),
        'Accuracy'            : round(r['acc'],  4),
        'Sensitivity'         : round(r['sens'], 4),
        'Specificity'         : round(r['spec'], 4),
        'PPV'                 : round(r['ppv'],  4),
        'F1'                  : round(r['f1'],   4),
    })

results_df = pd.DataFrame(rows)
results_df.to_csv('lasso_results.csv', index=False, encoding='utf-8-sig')

print("\n" + "=" * 70)
print("Step 8  Results")
print("=" * 70)
print(results_df.to_string(index=False))
print("\n  Saved: lasso_results.csv")

print("\n" + "=" * 70)
print("Complete")
print("=" * 70)
print("  scaling_decision.png  -- distribution + normality test + scaler comparison")
print("  lasso_cv_curve.png    -- 10-fold CV AUC curve")
print("  lasso_comparison.png  -- ROC curves + performance metrics")
print("  lasso_path.png        -- regularization paths (3 pipelines)")
print("  lasso_results.csv     -- test-set metrics for all 3 pipelines")
