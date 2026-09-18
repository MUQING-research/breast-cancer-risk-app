# Module guide:
# - Role: Preserve a legacy breast-cancer workflow.
# - Workflow: Retain historical preprocessing, exploratory, or modelling behavior for comparison.
# - Design note: Review archived assumptions before using a routine in a new pipeline.
# breast_cancer_pipeline.py
# Two-stage pipeline: LASSO feature selection → plain logistic regression
#
# Stage 1 — Feature selection
#   Pipeline: RobustScaler → LogisticRegressionCV (L1, 5-fold CV, AUC)
#   C selected by λ1se rule (most regularized C within 1 SE of λmin)
#
# Stage 2 — Final model
#   Pipeline: RobustScaler → LogisticRegression (no penalty)
#   Fitted on training set using only the LASSO-selected features
#
# No variable transformation — raw features only
#
# Outputs:
#   pipeline_lasso_path.png -- LASSO regularization path (λmin + λ1se)
#   pipeline_cv_curve.png   -- 5-fold CV AUC curve, λmin and λ1se
#   pipeline_results.png    -- ROC curve + coefficient bar chart
#   pipeline_results.csv    -- test-set metrics + selected features
#
# python breast_cancer_pipeline.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, roc_curve,
                              confusion_matrix, f1_score)

from _utils import CELL_COLORS, setup_matplotlib, load_data

setup_matplotlib()

CLR_MAIN = CELL_COLORS[3]
CLR_1SE  = CELL_COLORS[7]
CLR_MIN  = CELL_COLORS[8]
CLR_REF  = CELL_COLORS[5]


# ── Step 1  Load & split ──────────────────────────────────────────────────────
X, y, X_train, X_test, y_train, y_test = load_data()
y_tr = y_train.values.astype(int)
y_te = y_test.values.astype(int)

print("=" * 65)
print("Step 1  Data loaded")
print("=" * 65)
print(f"  Train : {len(y_tr)}  |  Test : {len(y_te)}")
print(f"  Features : {X.shape[1]}")


# ── Step 2  Stage 1 — LASSO pipeline (feature selection) ─────────────────────
print("\n" + "=" * 65)
print("Step 2  Stage 1: RobustScaler + LASSO  (5-fold CV, λ1se)")
print("=" * 65)

Cs = np.logspace(-4, 2, 60)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe_select = Pipeline([
    ('scaler', RobustScaler()),
    ('lasso',  LogisticRegressionCV(
        Cs=Cs, penalty='l1', solver='liblinear',
        cv=cv, scoring='roc_auc',
        max_iter=5000, random_state=42,
    )),
])
pipe_select.fit(X_train.values, y_tr)

cv_scores = list(pipe_select.named_steps['lasso'].scores_.values())[0]
n_folds   = cv_scores.shape[0]
mean_auc  = cv_scores.mean(axis=0)
se_auc    = cv_scores.std(axis=0, ddof=1) / np.sqrt(n_folds)

idx_min   = int(np.argmax(mean_auc))
C_min     = Cs[idx_min]
threshold = mean_auc[idx_min] - se_auc[idx_min]
idx_1se   = int(np.where(mean_auc >= threshold)[0][0])
C_1se     = Cs[idx_1se]

nz_min = int((pipe_select.named_steps['lasso'].coef_[0] != 0).sum())
print(f"  λmin : C={C_min:.5f}  ({nz_min} features)")

pipe_lasso_1se = Pipeline([
    ('scaler', RobustScaler()),
    ('lasso',  LogisticRegression(
        penalty='l1', solver='liblinear', C=C_1se,
        max_iter=5000, random_state=42,
    )),
])
pipe_lasso_1se.fit(X_train.values, y_tr)

lasso_coef    = pipe_lasso_1se.named_steps['lasso'].coef_[0]
selected_mask = lasso_coef != 0
selected_cols = X_train.columns[selected_mask].tolist()
n_selected    = len(selected_cols)

print(f"  λ1se : C={C_1se:.5f}  ({n_selected} features selected)")
print(f"\n  Selected features (sorted by |coefficient|):")
nz_pairs = sorted(zip(selected_cols, lasso_coef[selected_mask]),
                  key=lambda x: abs(x[1]), reverse=True)
for feat, coef in nz_pairs:
    print(f"    {feat:<35} {coef:>+.4f}")


# ── Step 3  Stage 2 — plain LR pipeline on selected features ─────────────────
print("\n" + "=" * 65)
print("Step 3  Stage 2: RobustScaler + plain LR  (selected features only)")
print("=" * 65)

X_train_sel = X_train[selected_cols].values
X_test_sel  = X_test[selected_cols].values

pipe_final = Pipeline([
    ('scaler', RobustScaler()),
    ('lr',     LogisticRegression(penalty=None, solver='lbfgs', max_iter=5000)),
])
pipe_final.fit(X_train_sel, y_tr)
print("  Fitted on training set")

lr_coef = pipe_final.named_steps['lr'].coef_[0]
lr_int  = pipe_final.named_steps['lr'].intercept_[0]
print(f"\n  Coefficients (plain LR, RobustScaler):")
print(f"  {'Feature':<35} {'LASSO coef':>12} {'Plain LR coef':>14}")
print("  " + "-" * 63)
for feat, lc, rc in zip(selected_cols, lasso_coef[selected_mask], lr_coef):
    print(f"  {feat:<35} {lc:>+12.4f} {rc:>+14.4f}")
print(f"  {'Intercept':<35} {'':>12} {lr_int:>+14.4f}")


# ── Step 4  Test-set evaluation ───────────────────────────────────────────────
print("\n" + "=" * 65)
print("Step 4  Test-set evaluation")
print("=" * 65)

prob = pipe_final.predict_proba(X_test_sel)[:, 1]
pred = pipe_final.predict(X_test_sel)
auc  = roc_auc_score(y_te, prob)
acc  = accuracy_score(y_te, pred)
tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
sens = tp / (tp + fn)
spec = tn / (tn + fp)
ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
f1   = f1_score(y_te, pred)
fpr, tpr, _ = roc_curve(y_te, prob)

print(f"  AUC={auc:.4f}  Acc={acc:.4f}  Sens={sens:.4f}  "
      f"Spec={spec:.4f}  PPV={ppv:.4f}  F1={f1:.4f}")


# ── Step 5  LASSO regularization path ────────────────────────────────────────
print("\n" + "=" * 65)
print("Step 5  LASSO regularization path")
print("=" * 65)

X_tr_scaled = pipe_lasso_1se.named_steps['scaler'].transform(X_train.values)
feat_names  = list(X_train.columns)

C_path     = np.logspace(-4, 2, 120)
log_C      = np.log10(C_path)
path_coefs = np.zeros((len(C_path), X_train.shape[1]))

for i, c in enumerate(C_path):
    m = LogisticRegression(penalty='l1', solver='liblinear', C=c, max_iter=5000)
    m.fit(X_tr_scaled, y_tr)
    path_coefs[i] = m.coef_[0]

print("  Path computed")

n_sel   = len(selected_cols)
cmap    = cm.get_cmap('tab10', n_sel)
sel_idx = [feat_names.index(f) for f in selected_cols]

fig_path, ax_path = plt.subplots(figsize=(12, 5.5))

for fi in range(len(feat_names)):
    if fi not in sel_idx:
        ax_path.plot(log_C, path_coefs[:, fi],
                     color='#DDDDDD', lw=0.7, alpha=0.8, zorder=1)

legend_lines = []
for k, fi in enumerate(sel_idx):
    col = cmap(k)
    line, = ax_path.plot(log_C, path_coefs[:, fi],
                         color=col, lw=1.8, alpha=0.92, zorder=2,
                         label=feat_names[fi])
    legend_lines.append(line)

ax_path.axvline(np.log10(C_min), color=CLR_MIN, lw=1.1, ls='--', zorder=3)
ax_path.axvline(np.log10(C_1se), color=CLR_1SE, lw=1.3, ls=':', zorder=3)
ax_path.axhline(0, color='#BBBBBB', lw=0.6)

ax_path.set_xlabel(r'$\log_{10}(C)$   ←  stronger regularization  |  weaker  →', fontsize=8)
ax_path.set_ylabel('Coefficient', fontsize=8)
ax_path.set_title(
    f'LASSO Regularization Path  (RobustScaler, n_train={len(y_tr)})\n'
    'Colored = selected at λ1se  |  Grey = shrunk to zero',
    fontsize=9, fontweight='bold')
ax_path.set_xlim(log_C.min(), log_C.max())

lmin_h = plt.Line2D([0], [0], color=CLR_MIN, lw=1.1, ls='--',
                     label=f'λmin  C={C_min:.4f}  ({nz_min} vars)')
l1se_h = plt.Line2D([0], [0], color=CLR_1SE, lw=1.3, ls=':',
                     label=f'λ1se  C={C_1se:.4f}  ({n_sel} vars)')
leg = ax_path.legend(
    handles=legend_lines + [lmin_h, l1se_h],
    bbox_to_anchor=(1.01, 1), loc='upper left',
    fontsize=7.5, framealpha=0.85,
    handlelength=1.5, borderpad=0.6, labelspacing=0.4,
    title=f'Selected (n={n_sel}) + λ lines', title_fontsize=7.5)
leg.get_title().set_fontweight('bold')

plt.tight_layout()
plt.savefig('pipeline_lasso_path.png')
plt.close(fig_path)
print("  Saved: pipeline_lasso_path.png")


# ── Step 6  CV curve with λmin and λ1se ──────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(8, 4.5))

ax1.fill_between(np.log10(Cs), mean_auc - se_auc, mean_auc + se_auc,
                 color=CLR_MAIN, alpha=0.18, label='±1 SE')
ax1.plot(np.log10(Cs), mean_auc, color=CLR_MAIN, lw=1.8, label='Mean CV AUC')
ax1.axvline(np.log10(C_min), color=CLR_MIN, lw=1.1, ls='--',
            label=f'λmin  C={C_min:.4f}  ({nz_min} vars)')
ax1.axvline(np.log10(C_1se), color=CLR_1SE, lw=1.2, ls=':',
            label=f'λ1se  C={C_1se:.4f}  ({n_selected} vars)')
ax1.axhline(threshold, color=CLR_1SE, lw=0.6, ls='--', alpha=0.5)

ax1.set_xlabel(r'$\log_{10}(C)$   ←  stronger regularization  |  weaker  →', fontsize=8)
ax1.set_ylabel('CV AUC (mean ± 1 SE)', fontsize=8)
ax1.set_title('Stage 1 — LASSO feature selection\n'
              '5-fold CV AUC curve  (RobustScaler + L1)',
              fontsize=9, fontweight='bold')
ax1.legend(fontsize=8, framealpha=0.85)

plt.tight_layout()
plt.savefig('pipeline_cv_curve.png')
plt.close(fig1)
print("\n  Saved: pipeline_cv_curve.png")


# ── Step 7  ROC + coefficients figure ────────────────────────────────────────
fig2, (ax_roc, ax_coef) = plt.subplots(1, 2, figsize=(13, 5))

ax_roc.plot(fpr, tpr, color=CLR_MAIN, lw=2.0,
            label=f'Plain LR  AUC={auc:.4f}')
ax_roc.plot([0, 1], [0, 1], color=CLR_REF, lw=0.9, ls='--', label='Chance')
ax_roc.set_xlabel('1 – Specificity (FPR)', fontsize=9)
ax_roc.set_ylabel('Sensitivity (TPR)', fontsize=9)
ax_roc.set_title('ROC Curve — Test Set', fontsize=10, fontweight='bold')
ax_roc.legend(fontsize=9, loc='lower right', framealpha=0.85)
ax_roc.set_xlim(-0.02, 1.02)
ax_roc.set_ylim(-0.02, 1.02)

order   = np.argsort(lr_coef)
feats_o = [selected_cols[i] for i in order]
coefs_o = lr_coef[order]
colors  = [CLR_1SE if c < 0 else CLR_MAIN for c in coefs_o]
y_pos   = np.arange(len(feats_o))

ax_coef.barh(y_pos, coefs_o, color=colors, alpha=0.82,
             edgecolor='none')
ax_coef.axvline(0, color='#AAAAAA', lw=0.8)
ax_coef.set_yticks(y_pos)
ax_coef.set_yticklabels(feats_o, fontsize=8)
ax_coef.set_xlabel('Coefficient', fontsize=9)
ax_coef.set_title(f'Plain LR coefficients\n({n_selected} LASSO-selected features, RobustScaler)',
                  fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('pipeline_results.png')
plt.close(fig2)
print("  Saved: pipeline_results.png")


# ── Step 8  Export ────────────────────────────────────────────────────────────
rows_feat = [{'feature': f, 'lasso_coef': round(lc, 4), 'lr_coef': round(rc, 4)}
             for f, (_, lc), rc in zip(selected_cols, nz_pairs,
                                       [lr_coef[selected_cols.index(f)]
                                        for f, _ in nz_pairs])]
rows_metric = [{
    'C_lasso_1se': round(C_1se, 6),
    'n_selected' : n_selected,
    'AUC'        : round(auc,  4),
    'Accuracy'   : round(acc,  4),
    'Sensitivity': round(sens, 4),
    'Specificity': round(spec, 4),
    'PPV'        : round(ppv,  4),
    'F1'         : round(f1,   4),
}]

with open('pipeline_results.csv', 'w', encoding='utf-8-sig', newline='') as f:
    pd.DataFrame(rows_metric).to_csv(f, index=False)
    f.write('\n')
    pd.DataFrame(rows_feat).to_csv(f, index=False)
print("  Saved: pipeline_results.csv")

print("\n" + "=" * 65)
print("Complete")
print("=" * 65)
