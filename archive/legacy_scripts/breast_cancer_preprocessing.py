# Module guide:
# - Role: Preserve a legacy breast-cancer workflow.
# - Workflow: Retain historical preprocessing, exploratory, or modelling behavior for comparison.
# - Design note: Review archived assumptions before using a routine in a new pipeline.
# breast_cancer_preprocessing.py
# Data preprocessing pipeline for breast cancer logistic regression
#
# Processing decisions (LRT alpha=0.10, confirmed by visual inspection):
#   - 19 variables : passed LRT -> continuous (original)
#   -  5 variables : failed LRT, but judged linear visually -> continuous (original)
#   -  2 variables : 1/x transformation -> continuous
#   -  4 variables : tertile categorization -> T1/T2/T3
#
# Outputs:
#   linearity_original.png     -- empirical logit(P), all 30 original variables
#   linearity_log.png          -- empirical logit(P), all 30 after log(1+x)
#   lrt_statistics.png         -- LRT results table with final treatment
#   lrt_statistics_log.png     -- LRT comparison table (original vs log)
#   preprocessing_overview.png  -- 1x5 panel: 1/x scatter + 4 GAM curves
#   transformation_decisions.csv -- processing decisions for all 30 variables
#
# python breast_cancer_preprocessing.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from patsy import dmatrix
from scipy import stats
from pygam import LogisticGAM, s

from _utils import (CELL_COLORS, VISUAL_LINEAR, TRANSFORM_INV, TERTILE_COLS,
                    setup_matplotlib, load_data, get_tertile_cuts, build_p3)

setup_matplotlib(**{'axes.titlesize': 8})

ALPHA     = 0.10
SPLINE_DF = 4
N_BINS    = 10

CLR_LINEAR    = CELL_COLORS[3]
CLR_NONLINEAR = CELL_COLORS[0]
CLR_VISUAL    = CELL_COLORS[2]
CLR_FIT       = CELL_COLORS[8]
CLR_REF       = CELL_COLORS[5]


# ── Step 1  Load & split ──────────────────────────────────────────────────────
X, y, X_train, X_test, y_train, y_test = load_data()
y_arr = y_train.values.astype(int)

print("=" * 70)
print("Step 1  Data loaded and split (80/20, stratified)")
print("=" * 70)
print(f"  Total : {len(y)}")
print(f"  Train : {len(y_train)}  (malignant {sum(y_train==0)}, benign {sum(y_train==1)})")
print(f"  Test  : {len(y_test)}   (malignant {sum(y_test==0)},  benign {sum(y_test==1)})")


# ── Step 2  LRT linearity test ────────────────────────────────────────────────
# Function guide: log_likelihood is responsible for log likelihood.
# Inputs: model, Xf, yv. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def log_likelihood(model, Xf, yv):
    p = np.clip(model.predict_proba(Xf)[:, 1], 1e-12, 1 - 1e-12)
    return float(np.sum(yv * np.log(p) + (1 - yv) * np.log(1 - p)))


# Function guide: lrt_linearity is responsible for perform linearity.
# Inputs: x_vals, y_vals. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def lrt_linearity(x_vals, y_vals):
    yv    = np.asarray(y_vals)
    Xl    = x_vals.reshape(-1, 1)
    m_lin = LogisticRegression(C=1e6, max_iter=2000, solver='lbfgs')
    m_lin.fit(Xl, yv)
    ll_lin = log_likelihood(m_lin, Xl, yv)
    basis  = np.asarray(
        dmatrix(f"cr(x, df={SPLINE_DF}) - 1", {"x": x_vals}), dtype=float)
    m_spl  = LogisticRegression(C=1e6, max_iter=2000, solver='lbfgs')
    m_spl.fit(basis, yv)
    ll_spl = log_likelihood(m_spl, basis, yv)
    stat   = max(2 * (ll_spl - ll_lin), 0.0)
    p      = stats.chi2.sf(stat, df=SPLINE_DF - 1)
    return stat, p


print("\n" + "=" * 70)
print(f"Step 2  LRT linearity test  (linear GLM vs natural cubic spline, alpha={ALPHA})")
print("=" * 70)

lrt_records = []
for col in X_train.columns:
    stat, p = lrt_linearity(X_train[col].values.astype(float), y_arr)
    lrt_records.append(dict(
        feature    = col,
        lrt_chi2   = round(stat, 3),
        p_value    = round(p,    4),
        lrt_linear = 'Yes' if p >= ALPHA else 'No',
    ))

lrt_df   = pd.DataFrame(lrt_records)
lrt_pval = dict(zip(lrt_df['feature'], lrt_df['p_value']))
lrt_chi2 = dict(zip(lrt_df['feature'], lrt_df['lrt_chi2']))

pd.set_option('display.max_rows', 35)
pd.set_option('display.width', 120)
print(lrt_df.to_string(index=False))
print(f"\n  Linear (LRT)  : {(lrt_df['lrt_linear']=='Yes').sum()} features")
print(f"  Non-linear    : {(lrt_df['lrt_linear']=='No').sum()} features")


# ── Step 3  Figure A: empirical logit(P) -- all 30 original variables ─────────
# Function guide: emp_logit_scatter is responsible for perform logit scatter.
# Inputs: ax, x, y, title, p_val, dot_color, status_label. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def emp_logit_scatter(ax, x, y, title='', p_val=None, dot_color=None, status_label=None):
    order  = np.argsort(x)
    xs, ys = x[order], np.asarray(y, int)[order]
    grps   = np.array_split(np.arange(len(xs)), N_BINS)
    xm     = np.array([xs[g].mean() for g in grps])
    logits = np.array([
        np.log((ys[g].sum() + 0.5) / (len(g) - ys[g].sum() + 0.5))
        for g in grps
    ])
    c  = np.polyfit(xm, logits, 1)
    xl = np.linspace(xm.min(), xm.max(), 200)

    if dot_color is None:
        dot_color = CLR_LINEAR if (p_val is None or p_val >= ALPHA) else CLR_NONLINEAR
    if status_label is None:
        status_label = 'Linear' if (p_val is None or p_val >= ALPHA) else 'Non-linear'
    p_str = f'  p={p_val:.4f}' if p_val is not None else ''

    ax.scatter(xm, logits, color=dot_color, s=22, zorder=3,
               edgecolors='none')
    ax.plot(xl, np.polyval(c, xl), color=CLR_FIT, lw=1.0, alpha=0.9)
    ax.axhline(0, color=CLR_REF, lw=0.6, ls='--')
    ax.set_title(f'{title}\n{status_label}{p_str}', fontsize=6,
                 color=dot_color, pad=2)
    ax.set_ylabel('logit(P)', fontsize=5.5)
    ax.tick_params(labelsize=5)


print("\n" + "=" * 70)
print("Step 3  Figure A: empirical logit(P), original variables")
print("=" * 70)

fig_a, axes_a = plt.subplots(5, 6, figsize=(15, 11))
for idx, col in enumerate(X_train.columns):
    pv = lrt_pval[col]
    if col in VISUAL_LINEAR:
        clr, lbl = CLR_VISUAL, 'Visual OK'
    elif pv >= ALPHA:
        clr, lbl = CLR_LINEAR, 'Linear'
    else:
        clr, lbl = CLR_NONLINEAR, 'Non-linear'
    emp_logit_scatter(axes_a.flat[idx],
                      X_train[col].values.astype(float),
                      y_arr, title=col, p_val=pv,
                      dot_color=clr, status_label=lbl)

plt.tight_layout()
plt.savefig('linearity_original.png')
plt.close(fig_a)
print("  Saved: linearity_original.png")


# ── Step 4  Figure B: LRT statistics table ────────────────────────────────────
print("\n" + "=" * 70)
print("Step 4  Figure B: LRT statistics table")
print("=" * 70)

# Function guide: treatment_label is responsible for perform label.
# Inputs: col, lrt_linear. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def treatment_label(col, lrt_linear):
    if col in TRANSFORM_INV:  return 'Continuous (1/x)'
    if col in TERTILE_COLS:   return 'Categorical (tertile)'
    if col in VISUAL_LINEAR:  return 'Continuous (visual)'
    return 'Continuous (original)'

ROW_FC = {
    'Continuous (original)' : '#EDF3FC',
    'Continuous (visual)'   : '#E8F5E9',
    'Continuous (1/x)'      : '#FFF8E1',
    'Categorical (tertile)' : '#FCEAE8',
}

fig_b, ax_b = plt.subplots(figsize=(14, 11))
ax_b.axis('off')
fig_b.patch.set_facecolor('white')

cell_text = []
for _, row in lrt_df.iterrows():
    pv  = row['p_value']
    sig = '***' if pv < 0.001 else ('**' if pv < 0.01 else ('*' if pv < 0.05 else 'ns'))
    cell_text.append([
        row['feature'],
        f"{row['lrt_chi2']:.3f}",
        f"{pv:.4f} ({sig})",
        row['lrt_linear'],
        treatment_label(row['feature'], row['lrt_linear']),
    ])

col_labels = ['Feature', 'LRT chi2', 'p-value', f'Linear (alpha={ALPHA})', 'Final treatment']
tbl = ax_b.table(cellText=cell_text, colLabels=col_labels,
                 loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1.0, 1.5)

for j in range(len(col_labels)):
    c = tbl[(0, j)]
    c.set_facecolor(CELL_COLORS[3])
    c.get_text().set_color('white')
    c.get_text().set_fontweight('bold')

for i, (_, row) in enumerate(lrt_df.iterrows()):
    fc = ROW_FC[treatment_label(row['feature'], row['lrt_linear'])]
    for j in range(len(col_labels)):
        tbl[(i + 1, j)].set_facecolor(fc)

tbl.auto_set_column_width(list(range(len(col_labels))))
ax_b.set_title(f'LRT linearity assessment  (alpha={ALPHA})',
               fontsize=11, fontweight='bold', pad=10)
fig_b.text(
    0.5, 0.01,
    'ns p>=0.10  * p<0.05  ** p<0.01  *** p<0.001  |  '
    'Blue=original  Green=visual override  Yellow=1/x  Red=tertile',
    ha='center', fontsize=7.5, style='italic', color=CELL_COLORS[8])
plt.savefig('lrt_statistics.png')
plt.close(fig_b)
print("  Saved: lrt_statistics.png")


# ── Step 5  log(1+x) transformation → LRT linearity test ─────────────────────
print("\n" + "=" * 70)
print(f"Step 5  log(1+x) → LRT linearity test  (alpha={ALPHA})")
print("=" * 70)

X_train_log = X_train.copy()
for col in X_train_log.columns:
    X_train_log[col] = np.log1p(X_train_log[col].values.astype(float))

lrt_log_records = []
for col in X_train_log.columns:
    stat, p = lrt_linearity(X_train_log[col].values.astype(float), y_arr)
    lrt_log_records.append({
        'feature'   : col,
        'lrt_chi2'  : round(stat, 3),
        'p_value'   : round(p,    4),
        'lrt_linear': 'Yes' if p >= ALPHA else 'No',
    })

lrt_log_df   = pd.DataFrame(lrt_log_records)
lrt_log_pval = dict(zip(lrt_log_df['feature'], lrt_log_df['p_value']))

orig_pass    = set(lrt_df.loc[lrt_df['lrt_linear']        == 'Yes', 'feature'])
log_pass     = set(lrt_log_df.loc[lrt_log_df['lrt_linear'] == 'Yes', 'feature'])
newly_linear = sorted(log_pass - orig_pass)
lost_linear  = sorted(orig_pass - log_pass)

print(lrt_log_df.to_string(index=False))
print(f"\n  Original  : {len(orig_pass)}/30 linear")
print(f"  After log : {len(log_pass)}/30 linear")
if newly_linear:
    print(f"  Newly linear after log ({len(newly_linear)}): {', '.join(newly_linear)}")
if lost_linear:
    print(f"  Lost linearity after log ({len(lost_linear)}): {', '.join(lost_linear)}")


# ── Step 5a  Figure: empirical logit(P) after log(1+x) ───────────────────────
CLR_NEW = CELL_COLORS[5]

fig_alog, axes_alog = plt.subplots(5, 6, figsize=(15, 11))

for idx, col in enumerate(X_train_log.columns):
    pv = lrt_log_pval[col]
    if pv >= ALPHA:
        clr = CLR_NEW if col in newly_linear else CLR_LINEAR
        lbl = 'Linear (new)' if col in newly_linear else 'Linear'
    else:
        clr, lbl = CLR_NONLINEAR, 'Non-linear'
    emp_logit_scatter(axes_alog.flat[idx],
                      X_train_log[col].values.astype(float),
                      y_arr,
                      title=f'log(1+{col})', p_val=pv,
                      dot_color=clr, status_label=lbl)

plt.tight_layout()
plt.savefig('linearity_log.png')
plt.close(fig_alog)
print("\n  Saved: linearity_log.png")


# ── Step 5b  Figure: LRT comparison table (original vs log) ──────────────────
merged_lrt = lrt_df.merge(lrt_log_df, on='feature', suffixes=('_orig', '_log'))

fig_blog, ax_blog = plt.subplots(figsize=(16, 11))
ax_blog.axis('off')
fig_blog.patch.set_facecolor('white')

cell_log = []
for _, row in merged_lrt.iterrows():
    col   = row['feature']
    pv_o  = row['p_value_orig']
    pv_l  = row['p_value_log']
    lin_o = row['lrt_linear_orig']
    lin_l = row['lrt_linear_log']
    sig_o = ('***' if pv_o < 0.001 else '**' if pv_o < 0.01 else
              '*'   if pv_o < 0.05  else 'ns')
    sig_l = ('***' if pv_l < 0.001 else '**' if pv_l < 0.01 else
              '*'   if pv_l < 0.05  else 'ns')
    change = ('No→Yes' if col in newly_linear else
               'Yes→No' if col in lost_linear else '—')
    cell_log.append([
        col,
        f"{row['lrt_chi2_orig']:.3f}", f"{pv_o:.4f} ({sig_o})", lin_o,
        f"{row['lrt_chi2_log']:.3f}",  f"{pv_l:.4f} ({sig_l})", lin_l,
        change,
    ])

col_labels_log = [
    'Feature',
    'chi2 (orig)', 'p (orig)', 'Linear (orig)',
    'chi2 (log)',  'p (log)',  'Linear (log)',
    'Change',
]
tbl_log = ax_blog.table(cellText=cell_log, colLabels=col_labels_log,
                         loc='center', cellLoc='center')
tbl_log.auto_set_font_size(False)
tbl_log.set_fontsize(8.0)
tbl_log.scale(1.0, 1.5)

for j in range(len(col_labels_log)):
    c = tbl_log[(0, j)]
    c.set_facecolor(CELL_COLORS[3])
    c.get_text().set_color('white')
    c.get_text().set_fontweight('bold')

ROW_FC_LOG = {'No→Yes': '#E8F5E9', 'Yes→No': '#FCEAE8', '—': '#F5F5F5'}
for i, (_, row) in enumerate(merged_lrt.iterrows()):
    col    = row['feature']
    change = ('No→Yes' if col in newly_linear else
               'Yes→No' if col in lost_linear else '—')
    for j in range(len(col_labels_log)):
        tbl_log[(i + 1, j)].set_facecolor(ROW_FC_LOG[change])

tbl_log.auto_set_column_width(list(range(len(col_labels_log))))
ax_blog.set_title(
    f'LRT linearity — original vs log(1+x)  (alpha={ALPHA})',
    fontsize=11, fontweight='bold', pad=10)
fig_blog.text(
    0.5, 0.01,
    f'Green: newly linear after log  |  Red: lost linearity  |  '
    f'Grey: unchanged  |  alpha={ALPHA}',
    ha='center', fontsize=7.5, style='italic', color='#555555')
plt.savefig('lrt_statistics_log.png')
plt.close(fig_blog)
print("  Saved: lrt_statistics_log.png")


# ── Step 6  Build processed feature matrices ──────────────────────────────────
print("\n" + "=" * 70)
print("Step 6  Build processed feature matrices")
print("=" * 70)

tertile_cuts = get_tertile_cuts(X_train)
X_train_proc = build_p3(X_train, tertile_cuts)
X_test_proc  = build_p3(X_test,  tertile_cuts)

print(f"  X_train_proc : {X_train_proc.shape}")
print(f"  X_test_proc  : {X_test_proc.shape}")
print(f"\n  Columns ({len(X_train_proc.columns)}):")
for col in X_train_proc.columns:
    print(f"    {col}")


# ── Step 7  Export transformation decisions ───────────────────────────────────
print("\n" + "=" * 70)
print("Step 7  Export transformation_decisions.csv")
print("=" * 70)

summary_rows = []
for _, row in lrt_df.iterrows():
    col = row['feature']
    if col in TRANSFORM_INV:
        treatment = 'continuous_transformed'
        transform = '1/x'
        q1_out, q2_out = '', ''
        note = 'Visual inspection confirmed linearity after 1/x transformation'
    elif col in TERTILE_COLS:
        treatment = 'categorical_tertile'
        transform = 'tertile (T1/T2/T3)'
        q1_out, q2_out = tertile_cuts[col]
        note = 'Non-linear pattern confirmed by LRT + GAM; categorized as tertiles'
    elif col in VISUAL_LINEAR:
        treatment = 'continuous_original'
        transform = 'none'
        q1_out, q2_out = '', ''
        note = 'LRT p<0.10 but judged linear by visual inspection of logit plot'
    else:
        treatment = 'continuous_original'
        transform = 'none'
        q1_out, q2_out = '', ''
        note = 'Passed LRT linearity test (p>=0.10)'

    summary_rows.append(dict(
        feature         = col,
        lrt_chi2        = row['lrt_chi2'],
        lrt_p_value     = row['p_value'],
        lrt_linear      = row['lrt_linear'],
        final_treatment = treatment,
        transform       = transform,
        tertile_q1      = q1_out,
        tertile_q2      = q2_out,
        note            = note,
    ))

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('transformation_decisions.csv', index=False, encoding='utf-8-sig')
print(summary_df[['feature', 'lrt_linear', 'final_treatment', 'transform']].to_string(index=False))
print("\n  Saved: transformation_decisions.csv")


# ── Step 8  Figure: 1×5 web-app panel ────────────────────────────────────────
print("\n" + "=" * 70)
print("Step 8  Figure: 1x5 preprocessing overview (web app)")
print("=" * 70)

fig_e, axes_e = plt.subplots(1, 5, figsize=(20, 4))

# Col 1: compactness error (1/x) empirical logit scatter
ax0    = axes_e[0]
col_tx = 'compactness error'
x_orig = X_train[col_tx].values.astype(float)
x_inv  = 1.0 / np.where(np.abs(x_orig) < 1e-12, 1e-12, x_orig)
chi2_t, pv_t = lrt_linearity(x_inv, y_arr)

order  = np.argsort(x_inv)
xs, ys = x_inv[order], y_arr[order]
grps   = np.array_split(np.arange(len(xs)), N_BINS)
xm     = np.array([xs[g].mean() for g in grps])
logits = np.array([
    np.log((ys[g].sum() + 0.5) / (len(g) - ys[g].sum() + 0.5))
    for g in grps
])
c  = np.polyfit(xm, logits, 1)
xl = np.linspace(xm.min(), xm.max(), 200)

ax0.scatter(xm, logits, color=CLR_LINEAR, s=28, zorder=3)
ax0.plot(xl, np.polyval(c, xl), color=CLR_FIT, lw=1.2, alpha=0.9)
ax0.axhline(0, color=CLR_REF, lw=0.7, ls='--')
ax0.set_xlabel('1/x (compactness error)', fontsize=7.5)
ax0.set_ylabel('logit(P)', fontsize=8)
ax0.set_title('compactness error\n1/x transformation', fontsize=8, fontweight='bold')
ax0.text(0.97, 0.04,
         f'Original: chi2={lrt_chi2[col_tx]:.2f}, p={lrt_pval[col_tx]:.4f}\n'
         f'After 1/x: chi2={chi2_t:.2f}, p={pv_t:.4f}',
         transform=ax0.transAxes, fontsize=6.5, ha='right', va='bottom',
         color='#333333',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F4FF',
                   edgecolor='#AABBD0', lw=0.6))

# Col 2-5: GAM smooth curves for 4 tertile variables
for i, col in enumerate(TERTILE_COLS):
    ax  = axes_e[i + 1]
    x   = X_train[[col]].values
    gam = LogisticGAM(s(0)).gridsearch(x, y_arr, progress=False)
    XX  = gam.generate_X_grid(term=0, n=300)
    pdep, confi = gam.partial_dependence(term=0, X=XX, width=0.95)
    gam_p = gam.statistics_['p_values'][0]

    q1, q2 = tertile_cuts[col]

    ax.plot(XX[:, 0], pdep, color=CLR_LINEAR, lw=1.5)
    ax.fill_between(XX[:, 0], confi[:, 0], confi[:, 1],
                    color=CLR_LINEAR, alpha=0.15, label='95% CI')
    ax.axhline(0, color=CLR_REF, lw=0.7, ls='--')
    for cut in [q1, q2]:
        ax.axvline(cut, color='#C62828', lw=1.0, ls=':', alpha=0.85)

    ax.set_xlabel(col, fontsize=7.5)
    ax.set_ylabel('Partial effect (log-odds)', fontsize=7.5)
    ax.set_title(f'{col}\nGAM smooth  →  tertile', fontsize=8, fontweight='bold')
    ax.legend(fontsize=6.5, framealpha=0.7, loc='upper right')
    ax.text(0.97, 0.04,
            f'LRT: chi2={lrt_chi2[col]:.2f}, p={lrt_pval[col]:.4f}\n'
            f'GAM smooth p={gam_p:.4f}\n'
            f'Cuts: {q1:.4f}, {q2:.4f}',
            transform=ax.transAxes, fontsize=6.5, ha='right', va='bottom',
            color='#333333',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF8F8',
                      edgecolor='#D0AAAA', lw=0.6))
    print(f"  {col:<30}  GAM smooth p={gam_p:.4f}")

plt.tight_layout()
plt.savefig('preprocessing_overview.png', bbox_inches='tight')
plt.close(fig_e)
print("  Saved: preprocessing_overview.png")


print("\n" + "=" * 70)
print("Preprocessing complete")
print("=" * 70)
print("  linearity_original.png     -- logit(P), all 30 original variables")
print("  linearity_log.png          -- logit(P), all 30 after log(1+x)")
print("  lrt_statistics.png         -- LRT results table")
print("  lrt_statistics_log.png     -- LRT comparison table (original vs log)")
print("  preprocessing_overview.png  -- 1x5 web-app figure")
print("  transformation_decisions.csv -- processing decisions, all 30 variables")
