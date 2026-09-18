# Module guide:
# - Role: Preserve a legacy breast-cancer workflow.
# - Workflow: Retain historical preprocessing, exploratory, or modelling behavior for comparison.
# - Design note: Review archived assumptions before using a routine in a new pipeline.
# breast_cancer_logistic.py
# Logistic regression comparison on the 7 LASSO-selected variables
#
# Models:
#   M1 — plain logistic regression (no penalty)
#   M2 — L2 penalized logistic regression (Ridge, C via 10-fold CV AUC)
#
# Variables (selected by LASSO λ1se, all raw continuous, no transformation):
#   worst area, worst concave points, worst texture, area error,
#   worst concavity, worst symmetry, worst smoothness
#
# Scaling: RobustScaler (fit on train, apply to test)
#
# Outputs:
#   logistic_comparison.png  -- ROC curves + coefficient bar chart
#   logistic_results.csv     -- test-set metrics
#
# python breast_cancer_logistic.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

from _utils import CELL_COLORS, setup_matplotlib, load_data, evaluate

setup_matplotlib()

SELECTED_VARS = [
    'worst area', 'worst concave points', 'worst texture',
    'area error', 'worst concavity', 'worst symmetry', 'worst smoothness',
]

CLR_M1  = CELL_COLORS[3]
CLR_M2  = CELL_COLORS[0]
CLR_REF = CELL_COLORS[5]


# ── Step 1  Load, split, select 7 features ────────────────────────────────────
X, y, X_train, X_test, y_train, y_test = load_data()
y_tr = y_train.values.astype(int)
y_te = y_test.values.astype(int)

X_tr7 = X_train[SELECTED_VARS].copy()
X_te7 = X_test[SELECTED_VARS].copy()

print("=" * 65)
print("Step 1  Data")
print("=" * 65)
print(f"  Train : {len(y_tr)}  |  Test : {len(y_te)}")
print(f"  Features : {SELECTED_VARS}")


# ── Step 2  Scale ─────────────────────────────────────────────────────────────
scaler = RobustScaler()
X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr7), columns=SELECTED_VARS)
X_te_s = pd.DataFrame(scaler.transform(X_te7),    columns=SELECTED_VARS)


# ── Step 3  Fit models ────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Step 2  Model fitting")
print("=" * 65)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

m1 = LogisticRegression(penalty=None, solver='lbfgs', max_iter=5000)
m1.fit(X_tr_s.values, y_tr)
print("  M1 (plain LR)  : fitted")

m2_cv = LogisticRegressionCV(
    Cs=np.logspace(-4, 2, 60), penalty='l2', solver='lbfgs',
    cv=cv, scoring='roc_auc', max_iter=5000, random_state=42)
m2_cv.fit(X_tr_s.values, y_tr)
C_l2 = m2_cv.C_[0]
m2 = LogisticRegression(penalty='l2', solver='lbfgs', C=C_l2, max_iter=5000)
m2.fit(X_tr_s.values, y_tr)
print(f"  M2 (L2 Ridge)  : fitted  optimal C={C_l2:.5f}")


# ── Step 4  Evaluate ──────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Step 3  Test-set evaluation")
print("=" * 65)

res_m1 = evaluate(m1, X_te_s, y_te, 'M1: Plain LR (no penalty)')
print(f"  {res_m1['label']}")
print(f"    AUC={res_m1['auc']:.4f}  Acc={res_m1['acc']:.4f}  Sens={res_m1['sens']:.4f}  "
      f"Spec={res_m1['spec']:.4f}  PPV={res_m1['ppv']:.4f}  F1={res_m1['f1']:.4f}")

res_m2 = evaluate(m2, X_te_s, y_te, f'M2: L2 Ridge  (C={C_l2:.5f})')
print(f"  {res_m2['label']}")
print(f"    AUC={res_m2['auc']:.4f}  Acc={res_m2['acc']:.4f}  Sens={res_m2['sens']:.4f}  "
      f"Spec={res_m2['spec']:.4f}  PPV={res_m2['ppv']:.4f}  F1={res_m2['f1']:.4f}")


# ── Step 5  Coefficients ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Step 4  Coefficients (RobustScaler applied)")
print("=" * 65)
print(f"  {'Feature':<30} {'M1 (plain)':>12} {'M2 (L2)':>12}")
print("  " + "-" * 56)
for feat, c1, c2 in zip(SELECTED_VARS, m1.coef_[0], m2.coef_[0]):
    print(f"  {feat:<30} {c1:>+12.4f} {c2:>+12.4f}")
print(f"\n  Intercept                      {m1.intercept_[0]:>+12.4f} "
      f"{m2.intercept_[0]:>+12.4f}")


# ── Step 6  Figure ────────────────────────────────────────────────────────────
fig, (ax_roc, ax_coef) = plt.subplots(1, 2, figsize=(13, 5.5))

for res, clr, lw in [(res_m1, CLR_M1, 2.0), (res_m2, CLR_M2, 1.8)]:
    ax_roc.plot(res['fpr'], res['tpr'], color=clr, lw=lw, alpha=0.9,
                label=f"{res['label']}  AUC={res['auc']:.4f}")
ax_roc.plot([0, 1], [0, 1], color=CLR_REF, lw=0.9, ls='--', label='Chance')
ax_roc.set_xlabel('1 – Specificity (FPR)', fontsize=9)
ax_roc.set_ylabel('Sensitivity (TPR)', fontsize=9)
ax_roc.set_title('ROC Curves — Test Set', fontsize=10, fontweight='bold')
ax_roc.legend(fontsize=8.5, loc='lower right', framealpha=0.85)
ax_roc.set_xlim(-0.02, 1.02)
ax_roc.set_ylim(-0.02, 1.02)

x     = np.arange(len(SELECTED_VARS))
width = 0.35
ax_coef.bar(x - width/2, m1.coef_[0], width,
            label='M1: Plain LR', color=CLR_M1, alpha=0.82,
            edgecolor='none')
ax_coef.bar(x + width/2, m2.coef_[0], width,
            label=f'M2: L2 Ridge (C={C_l2:.4f})', color=CLR_M2, alpha=0.82,
            edgecolor='none')
ax_coef.axhline(0, color='#AAAAAA', lw=0.7)
ax_coef.set_xticks(x)
short_names = [v.replace('worst ', 'w.').replace(' error', ' err')
               for v in SELECTED_VARS]
ax_coef.set_xticklabels(short_names, rotation=30, ha='right', fontsize=7.5)
ax_coef.set_ylabel('Coefficient', fontsize=9)
ax_coef.set_title('Coefficients — 7 selected variables\n(RobustScaler)',
                  fontsize=9, fontweight='bold')
ax_coef.legend(fontsize=8.5, framealpha=0.85)

plt.tight_layout()
plt.savefig('logistic_comparison.png')
plt.close(fig)
print("\n  Saved: logistic_comparison.png")


# ── Step 7  Export ────────────────────────────────────────────────────────────
metric_keys = ['auc', 'acc', 'sens', 'spec', 'ppv', 'f1']
rows = []
for res, penalty, C_val in [
    (res_m1, 'none', None),
    (res_m2, 'l2',   round(C_l2, 6)),
]:
    rows.append({'model': res['label'], 'penalty': penalty, 'C': C_val,
                 **{k: round(res[k], 4) for k in metric_keys}})
pd.DataFrame(rows).to_csv('logistic_results.csv', index=False, encoding='utf-8-sig')
print("  Saved: logistic_results.csv")

print("\n" + "=" * 65)
print("Complete")
print("=" * 65)
