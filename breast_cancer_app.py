"""
breast_cancer_app.py — Breast Cancer Classification
Two-stage pipeline: (1) LASSO (λ1se, 5-fold CV) ; (2) Plain Logistic Regression
Wisconsin Breast Cancer Dataset (sklearn) · N=569 · 30 features
Cell Press visual style · Research & educational use only
"""
from __future__ import annotations
import io, pickle, warnings, threading, os, tempfile
from collections import Counter
import requests
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import stats

from shiny import App, reactive, render, ui
import json
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler, SplineTransformer
from sklearn.linear_model import (LogisticRegression, LogisticRegressionCV,
                                   LinearRegression)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, accuracy_score, roc_curve,
                              confusion_matrix, f1_score, brier_score_loss)
from sklearn.calibration import calibration_curve as _sklearn_cal_curve

warnings.filterwarnings("ignore")

if os.name == "nt":
    _TMP_ROOT = Path(__file__).parent / ".cache" / "tmp"
    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(_TMP_ROOT)

    class _SafeTemporaryDirectory:
        def __init__(self, suffix=None, prefix=None, dir=None,
                     ignore_cleanup_errors=True):
            self.name = tempfile.mkdtemp(
                suffix=suffix or "",
                prefix=prefix or "tmp",
                dir=dir or str(_TMP_ROOT),
            )

        def __enter__(self):
            return self.name

        def __exit__(self, exc_type, exc, tb):
            return False

        def cleanup(self):
            return None

    tempfile.TemporaryDirectory = _SafeTemporaryDirectory

# ── Palette ───────────────────────────────────────────────────────────────────
CELL_COLORS = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
    "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85",
]
CLR_MAL   = CELL_COLORS[0]
CLR_BEN   = CELL_COLORS[1]
CLR_TRAIN = CELL_COLORS[3]
CLR_TEST  = CELL_COLORS[4]
CLR_1SE   = CELL_COLORS[7]
CLR_MIN   = CELL_COLORS[8]
CLR_REF   = CELL_COLORS[5]
_MUTED    = CELL_COLORS[8]
_NAVY     = CELL_COLORS[3]
_ORANGE   = CELL_COLORS[4]
_EDGE     = CELL_COLORS[5]
_FAINT    = CELL_COLORS[6]

plt.rcParams.update({
    "font.family":         "Arial",
    "font.sans-serif":     ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans", "sans-serif"],
    "font.size":           9.0,
    "axes.titlesize":      10.0,
    "axes.titleweight":    "bold",
    "axes.linewidth":      0.8,
    "axes.spines.top":     True,
    "axes.spines.right":   True,
    "axes.grid":           False,
    "xtick.direction":     "out",
    "ytick.direction":     "out",
    "xtick.major.size":    3.0,
    "ytick.major.size":    3.0,
    "xtick.major.width":   0.8,
    "ytick.major.width":   0.8,
    "xtick.labelsize":     8.0,
    "ytick.labelsize":     8.0,
    "axes.labelsize":      9.0,
    "axes.labelpad":       3,
    "lines.linewidth":     1.0,
    "lines.markersize":    3.6,
    "legend.fontsize":     8.0,
    "legend.frameon":      False,
    "legend.framealpha":   0.9,
    "legend.edgecolor":    _EDGE,
    "legend.borderpad":    0.35,
    "legend.labelspacing": 0.22,
    "figure.facecolor":    "white",
    "axes.facecolor":      "white",
    "figure.dpi":          300,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "pdf.fonttype":        42,
    "ps.fonttype":         42,
})

# ── 1. Model training ────────────────────────────────────────────────────────
_BUNDLE_PATH = Path(__file__).parent / "bc_bundle.pkl"


def _hosmer_lemeshow(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    """Hosmer-Lemeshow goodness-of-fit test (decile-of-risk)."""
    q = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    q[-1] += 1e-8
    bins = np.digitize(y_prob, q[1:-1])
    chi2 = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.sum() == 0:
            continue
        n_b  = mask.sum()
        obs  = float(y_true[mask].sum())
        exp  = float(y_prob[mask].sum())
        nobs = n_b - obs
        nexp = n_b - exp
        if exp  > 1e-10: chi2 += (obs  - exp)  ** 2 / exp
        if nexp > 1e-10: chi2 += (nobs - nexp) ** 2 / nexp
    df = n_bins - 2
    return float(chi2), float(1 - stats.chi2.cdf(chi2, df)), df


def _compute_vif(X_sc: np.ndarray, names: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for i, nm in enumerate(names):
        y_ = X_sc[:, i]
        X_ = np.delete(X_sc, i, axis=1)
        r2 = LinearRegression().fit(X_, y_).score(X_, y_)
        result[nm] = 1.0 / max(1e-10, 1.0 - r2)
    return result


def _lrt_one(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = x.reshape(-1, 1)
    m_lin = LogisticRegression(penalty=None, solver="lbfgs", max_iter=3000)
    m_lin.fit(x, y)
    p_lin = np.clip(m_lin.predict_proba(x)[:, 1], 1e-15, 1 - 1e-15)
    ll_lin = float(np.sum(y * np.log(p_lin) + (1 - y) * np.log(1 - p_lin)))

    sp = SplineTransformer(n_knots=2, degree=3, include_bias=False)
    Xs = sp.fit_transform(x)
    m_sp = LogisticRegression(penalty=None, solver="lbfgs", max_iter=3000)
    m_sp.fit(Xs, y)
    p_sp = np.clip(m_sp.predict_proba(Xs)[:, 1], 1e-15, 1 - 1e-15)
    ll_sp = float(np.sum(y * np.log(p_sp) + (1 - y) * np.log(1 - p_sp)))

    chi2 = max(0.0, 2 * (ll_sp - ll_lin))
    df   = max(1, Xs.shape[1] - 1)
    p    = float(1 - stats.chi2.cdf(chi2, df))
    return chi2, p


def _train_and_build() -> dict:
    """Train full pipeline; return bundle dict — no raw DataFrames."""
    _seed = 42
    np.random.seed(_seed)

    _data = load_breast_cancer()
    X_all = pd.DataFrame(_data.data, columns=_data.feature_names)
    y_all = pd.Series(_data.target, name="target")   # 0=malignant, 1=benign
    _n_malignant = int((y_all == 0).sum())
    _n_benign    = int((y_all == 1).sum())

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=_seed, stratify=y_all)
    _y_tr = y_train.values.astype(int)
    _y_te = y_test.values.astype(int)

    _N_TOTAL = len(y_all)
    _N_TRAIN = len(_y_tr)
    _N_TEST  = len(_y_te)

    # Stage 1 — LASSO with 5-fold CV
    _CS  = np.logspace(-4, 2, 60)
    _cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=_seed)
    _pipe_cv = Pipeline([
        ("scaler", RobustScaler()),
        ("lasso",  LogisticRegressionCV(
            Cs=_CS, penalty="l1", solver="liblinear",
            cv=_cv, scoring="roc_auc", max_iter=5000, random_state=_seed,
        )),
    ])
    _pipe_cv.fit(X_train.values, _y_tr)

    _cv_scores = list(_pipe_cv.named_steps["lasso"].scores_.values())[0]
    _mean_auc  = _cv_scores.mean(axis=0)
    _se_auc    = _cv_scores.std(axis=0, ddof=1) / np.sqrt(_cv_scores.shape[0])
    _idx_min   = int(np.argmax(_mean_auc))
    _c_min     = _CS[_idx_min]
    _thr_1se   = _mean_auc[_idx_min] - _se_auc[_idx_min]
    _idx_1se   = int(np.where(_mean_auc >= _thr_1se)[0][0])
    _c_1se     = _CS[_idx_1se]
    _nz_min    = int((_pipe_cv.named_steps["lasso"].coef_[0] != 0).sum())

    # Refit at λ1se
    _pipe_lasso = Pipeline([
        ("scaler", RobustScaler()),
        ("lasso",  LogisticRegression(
            penalty="l1", solver="liblinear", C=_c_1se,
            max_iter=5000, random_state=_seed,
        )),
    ])
    _pipe_lasso.fit(X_train.values, _y_tr)

    _lc        = _pipe_lasso.named_steps["lasso"].coef_[0]
    _sel_mask  = _lc != 0
    _sel_cols  = X_train.columns[_sel_mask].tolist()
    _n_sel     = len(_sel_cols)
    _feat_names = list(X_train.columns)
    _sel_idx   = [_feat_names.index(f) for f in _sel_cols]
    _lasso_coef = {f: float(_lc[i]) for f, i in zip(_sel_cols, np.where(_sel_mask)[0])}

    # Stage 2 — plain LR on selected features
    X_tr_sel = X_train[_sel_cols].values
    X_te_sel = X_test[_sel_cols].values
    _pipe_lr = Pipeline([
        ("scaler", RobustScaler()),
        ("lr",     LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000)),
    ])
    _pipe_lr.fit(X_tr_sel, _y_tr)

    _lr_coef = _pipe_lr.named_steps["lr"].coef_[0]
    _lr_int  = float(_pipe_lr.named_steps["lr"].intercept_[0])
    _lr_coef_map = {f: float(_lr_coef[k]) for k, f in enumerate(_sel_cols)}

    _prob_train = _pipe_lr.predict_proba(X_tr_sel)[:, 1]
    _prob_test  = _pipe_lr.predict_proba(X_te_sel)[:, 1]
    _auc_train  = roc_auc_score(_y_tr, _prob_train)
    _auc_test   = roc_auc_score(_y_te, _prob_test)
    _fpr_tr, _tpr_tr, _ = roc_curve(_y_tr, _prob_train)
    _fpr_te, _tpr_te, _ = roc_curve(_y_te, _prob_test)

    _pr05 = (_prob_test >= 0.5).astype(int)
    _tn, _fp, _fn, _tp = confusion_matrix(_y_te, _pr05).ravel()
    _acc05  = accuracy_score(_y_te, _pr05)
    _sens05 = _tp / (_tp + _fn)
    _spec05 = _tn / (_tn + _fp)
    _ppv05  = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0.0
    _npv05  = _tn / (_tn + _fn) if (_tn + _fn) > 0 else 0.0
    _f1_05  = f1_score(_y_te, _pr05)

    _brier_train = float(brier_score_loss(_y_tr, _prob_train))
    _brier_test  = float(brier_score_loss(_y_te, _prob_test))
    _null_brier  = float(np.mean((_y_te - float(_y_tr.mean())) ** 2))

    _cal_frac, _cal_mean = _sklearn_cal_curve(
        _y_te, _prob_test, n_bins=10, strategy="quantile")
    _hl_chi2, _hl_p, _hl_df = _hosmer_lemeshow(_y_te, _prob_test)

    _X_tr_sc_sel = _pipe_lr.named_steps["scaler"].transform(X_tr_sel)
    _vif = _compute_vif(_X_tr_sc_sel, _sel_cols)

    # LASSO regularisation path
    _X_scaled   = _pipe_lasso.named_steps["scaler"].transform(X_train.values)
    _C_PATH     = np.logspace(-4, 2, 120)
    _LOG_C      = np.log10(_C_PATH)
    _PATH_COEFS = np.zeros((len(_C_PATH), X_train.shape[1]))
    for _i, _c in enumerate(_C_PATH):
        _m = LogisticRegression(penalty="l1", solver="liblinear", C=_c, max_iter=5000)
        _m.fit(_X_scaled, _y_tr)
        _PATH_COEFS[_i] = _m.coef_[0]

    # Feature ranges (for sliders)
    _feat_ranges: dict[str, tuple[float, float, float]] = {}
    _train_medians: dict[str, float] = {}
    for f in _sel_cols:
        v = X_all[f].values
        _train_medians[f] = float(np.median(X_train[f].values))
        _feat_ranges[f] = (float(v.min()), float(v.max()), _train_medians[f])

    # LRT linearity test
    _lrt: dict[str, dict] = {}
    for _f in _sel_cols:
        _chi2, _p = _lrt_one(X_train[_f].values, _y_tr)
        _lrt[_f] = {"chi2": _chi2, "p": _p, "linear": _p >= 0.10}

    # Precompute linearity scatter data (keeps raw X_train out of bundle)
    _lrt_plot: dict[str, dict] = {}
    for feat in _sel_cols:
        x    = X_train[feat].values
        cuts = np.unique(np.percentile(x, np.linspace(0, 100, 11)))
        bidx = np.digitize(x, cuts[1:-1])
        mids, logits = [], []
        for b in np.unique(bidx):
            msk = bidx == b
            if msk.sum() < 5:
                continue
            p = np.clip(_y_tr[msk].mean(), 0.01, 0.99)
            mids.append(float(x[msk].mean()))
            logits.append(float(np.log(p / (1 - p))))
        _lrt_plot[feat] = {"mids": mids, "logits": logits}

    return dict(
        pipe_lasso=_pipe_lasso, pipe_lr=_pipe_lr,
        CS=_CS, MEAN_AUC=_mean_auc, SE_AUC=_se_auc,
        C_MIN=_c_min, C_1SE=_c_1se, THR_1SE=_thr_1se,
        NZ_MIN=_nz_min, N_SEL=_n_sel,
        SEL_MASK=_sel_mask, SEL_COLS=_sel_cols,
        FEAT_NAMES=_feat_names, SEL_IDX=_sel_idx, LASSO_COEF=_lasso_coef,
        LR_COEF=_lr_coef, LR_INT=_lr_int, LR_COEF_MAP=_lr_coef_map,
        PROB_TRAIN=_prob_train, PROB_TEST=_prob_test,
        y_tr=_y_tr, y_te=_y_te,
        AUC_TRAIN=_auc_train, AUC_TEST=_auc_test,
        FPR_TR=_fpr_tr, TPR_TR=_tpr_tr, FPR_TE=_fpr_te, TPR_TE=_tpr_te,
        ACC05=_acc05, SENS05=_sens05, SPEC05=_spec05,
        PPV05=_ppv05, NPV05=_npv05, F1_05=_f1_05,
        BRIER_TRAIN=_brier_train, BRIER_TEST=_brier_test, NULL_BRIER=_null_brier,
        CAL_FRAC=_cal_frac, CAL_MEAN=_cal_mean,
        HL_CHI2=_hl_chi2, HL_P=_hl_p, HL_DF=_hl_df,
        VIF=_vif, LRT=_lrt, LRT_PLOT_DATA=_lrt_plot,
        LOG_C=_LOG_C, PATH_COEFS=_PATH_COEFS, C_PATH=_C_PATH,
        FEAT_RANGES=_feat_ranges, TRAIN_MEDIANS=_train_medians,
        N_TOTAL=_N_TOTAL, N_TRAIN=_N_TRAIN, N_TEST=_N_TEST,
        n_malignant=_n_malignant, n_benign=_n_benign,
    )


print("=" * 56, flush=True)
print("Breast Cancer App — initialising", flush=True)
print("=" * 56, flush=True)

SEED = 42

if _BUNDLE_PATH.exists():
    print("  Loading bundle ...", flush=True)
    with open(_BUNDLE_PATH, "rb") as _f:
        _B = pickle.load(_f)
else:
    print("  Training from scratch ...", flush=True)
    _B = _train_and_build()
    with open(_BUNDLE_PATH, "wb") as _f:
        pickle.dump(_B, _f)
    print(f"  Bundle saved: {_BUNDLE_PATH}", flush=True)

pipe_lasso   = _B["pipe_lasso"]
pipe_lr      = _B["pipe_lr"]
CS           = _B["CS"]
MEAN_AUC     = _B["MEAN_AUC"]
SE_AUC       = _B["SE_AUC"]
C_MIN        = _B["C_MIN"]
C_1SE        = _B["C_1SE"]
THR_1SE      = _B["THR_1SE"]
NZ_MIN       = _B["NZ_MIN"]
N_SEL        = _B["N_SEL"]
SEL_MASK     = _B["SEL_MASK"]
SEL_COLS     = _B["SEL_COLS"]
FEAT_NAMES   = _B["FEAT_NAMES"]
SEL_IDX      = _B["SEL_IDX"]
LASSO_COEF   = _B["LASSO_COEF"]
LR_COEF      = _B["LR_COEF"]
LR_INT       = _B["LR_INT"]
LR_COEF_MAP  = _B["LR_COEF_MAP"]
PROB_TRAIN   = _B["PROB_TRAIN"]
PROB_TEST    = _B["PROB_TEST"]
y_tr         = _B["y_tr"]
y_te         = _B["y_te"]
AUC_TRAIN    = _B["AUC_TRAIN"]
AUC_TEST     = _B["AUC_TEST"]
FPR_TR       = _B["FPR_TR"]
TPR_TR       = _B["TPR_TR"]
FPR_TE       = _B["FPR_TE"]
TPR_TE       = _B["TPR_TE"]
ACC05        = _B["ACC05"]
SENS05       = _B["SENS05"]
SPEC05       = _B["SPEC05"]
PPV05        = _B["PPV05"]
NPV05        = _B["NPV05"]
F1_05        = _B["F1_05"]
BRIER_TRAIN  = _B["BRIER_TRAIN"]
BRIER_TEST   = _B["BRIER_TEST"]
NULL_BRIER   = _B["NULL_BRIER"]
CAL_FRAC     = _B["CAL_FRAC"]
CAL_MEAN     = _B["CAL_MEAN"]
HL_CHI2      = _B["HL_CHI2"]
HL_P         = _B["HL_P"]
HL_DF        = _B["HL_DF"]
VIF          = _B["VIF"]
LRT          = _B["LRT"]
LRT_PLOT_DATA = _B["LRT_PLOT_DATA"]
LOG_C        = _B["LOG_C"]
PATH_COEFS   = _B["PATH_COEFS"]
C_PATH       = _B["C_PATH"]
FEAT_RANGES  = _B["FEAT_RANGES"]
_train_centers = getattr(pipe_lr.named_steps["scaler"], "center_", None)
if "TRAIN_MEDIANS" in _B:
    TRAIN_MEDIANS = {k: float(v) for k, v in _B["TRAIN_MEDIANS"].items()}
elif _train_centers is not None and len(_train_centers) == len(SEL_COLS):
    TRAIN_MEDIANS = {
        feat: float(center) for feat, center in zip(SEL_COLS, _train_centers)
    }
else:
    TRAIN_MEDIANS = {feat: float(FEAT_RANGES[feat][2]) for feat in SEL_COLS}
N_TOTAL      = _B["N_TOTAL"]
N_TRAIN      = _B["N_TRAIN"]
N_TEST       = _B["N_TEST"]
n_malignant  = _B["n_malignant"]
n_benign     = _B["n_benign"]

print(f"  λmin C={C_MIN:.5f} ({NZ_MIN} vars)  λ1se C={C_1SE:.5f} ({N_SEL} vars)",
      flush=True)
print(f"  AUC  train={AUC_TRAIN:.4f}  test={AUC_TEST:.4f}", flush=True)
print(f"  Brier train={BRIER_TRAIN:.4f}  test={BRIER_TEST:.4f}  null={NULL_BRIER:.4f}",
      flush=True)
print(f"  HL test  chi2={HL_CHI2:.2f}  p={HL_P:.3f}  df={HL_DF}", flush=True)
print("  Ready.", flush=True)


def _threshold_metrics(y_true: np.ndarray, p_benign: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Evaluate a decision rule while treating malignancy as the positive class."""
    y_malignant = (np.asarray(y_true) == 0).astype(int)
    p_malignant = 1.0 - np.asarray(p_benign, dtype=float)
    pred_malignant = (p_malignant >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_malignant, pred_malignant, labels=[0, 1]).ravel()
    acc = (tp + tn) / len(y_malignant)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) > 0 else 0.0
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "acc": float(acc), "sens": float(sens), "spec": float(spec),
        "ppv": float(ppv), "npv": float(npv), "f1": float(f1),
    }


TEST_METRICS_05 = _threshold_metrics(y_te, PROB_TEST, 0.5)


# ── 2. Figure helpers ────────────────────────────────────────────────────────
def _sid(feat: str) -> str:
    """Slider input ID from feature name."""
    return "f_" + feat.replace(" ", "_")


def _fig_buf(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _P(ax, letter, fs=10.0):
    """Cell-style panel label: bold letter above top-left via set_title."""
    ax.set_title(letter, fontsize=fs, fontweight="bold",
                 loc="left", pad=3, color=_NAVY)


def _style_axis(ax):
    """Apply closed-box Cell-style axes to one matplotlib axis."""
    ax.grid(False)
    ax.tick_params(direction="out", length=3.0, width=0.8, colors=_NAVY, labelsize=8.0)
    ax.xaxis.label.set_color(_NAVY)
    ax.yaxis.label.set_color(_NAVY)
    ax.xaxis.label.set_size(9.0)
    ax.yaxis.label.set_size(9.0)
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_color(_EDGE)
        ax.spines[sp].set_linewidth(0.8)


def _autoscale(fig, scale=0.65, target_w=7.0):
    """Keep each figure's designed aspect ratio.

    The previous implementation recomputed height from the tight bounding box
    and multiplied it by a compression factor. With Cell-style fonts this made
    pre-rendered panels look vertically squashed in the app.
    """
    return fig


def _make_feat_sel_fig():
    """A: LASSO regularisation path · B: 5-fold CV AUC"""
    fig  = plt.figure(figsize=(7.0, 3.5))
    gs   = fig.add_gridspec(1, 5, wspace=0.42)

    # ── A: LASSO regularisation path ────────────────────────────────────────
    ax_p = fig.add_subplot(gs[0, 0:3])
    _style_axis(ax_p)
    for fi in range(len(FEAT_NAMES)):
        if fi not in SEL_IDX:
            ax_p.plot(LOG_C, PATH_COEFS[:, fi], color=_EDGE,
                      lw=0.8, alpha=0.22, zorder=1)
    for k, fi in enumerate(SEL_IDX):
        ax_p.plot(LOG_C, PATH_COEFS[:, fi], color=CELL_COLORS[k % len(CELL_COLORS)],
                  lw=1.0, alpha=0.90, zorder=2, label=FEAT_NAMES[fi])
    ax_p.axvline(np.log10(C_MIN), color=_MUTED, lw=0.8, ls="--", zorder=3,
                 label=f"lambda_min ({NZ_MIN})")
    ax_p.axvline(np.log10(C_1SE), color=CLR_1SE, lw=0.8, ls=":", zorder=3,
                 label=f"lambda_1se ({N_SEL})")
    ax_p.axhline(0, color=_EDGE, lw=0.8, alpha=0.35, zorder=0)
    ax_p.set_xlabel(r"$\log_{10}(C)$")
    ax_p.set_ylabel("Coefficient")
    ax_p.set_xlim(LOG_C.min(), LOG_C.max())
    _P(ax_p, "A")
    leg = ax_p.legend(loc="lower right", ncol=2, handlelength=1.1,
                       title=f"n = {N_SEL} selected",
                       fontsize=8.0, title_fontsize=8.0)
    leg.get_title().set_fontweight("bold")

    # ── B: 5-fold CV AUC ────────────────────────────────────────────────────
    ax_cv = fig.add_subplot(gs[0, 3:5])
    _style_axis(ax_cv)
    ax_cv.fill_between(np.log10(CS), MEAN_AUC - SE_AUC, MEAN_AUC + SE_AUC,
                       color=CLR_TRAIN, alpha=0.13)
    ax_cv.plot(np.log10(CS), MEAN_AUC, color=CLR_TRAIN, lw=1.0)
    ax_cv.axvline(np.log10(C_MIN), color=_MUTED, lw=0.8, ls="--",
                  label=f"lambda_min ({NZ_MIN})")
    ax_cv.axvline(np.log10(C_1SE), color=CLR_1SE, lw=0.8, ls=":",
                  label=f"lambda_1se ({N_SEL})")
    ax_cv.axhline(THR_1SE, color=CLR_1SE, lw=0.8, ls="--", alpha=0.45)
    ax_cv.set_xlabel(r"$\log_{10}(C)$")
    ax_cv.set_ylabel("5-fold CV AUC")
    ax_cv.yaxis.set_major_formatter(plt.FormatStrFormatter("%.3f"))
    _P(ax_cv, "B")
    ax_cv.legend(loc="lower left", fontsize=8.0)

    fig.tight_layout(pad=0.8)
    return _autoscale(fig)


def _make_perf_fig():
    """A: ROC curve · B: LR coefficients"""
    fig = plt.figure(figsize=(7.0, 3.5))
    gs  = fig.add_gridspec(1, 2, wspace=0.34)

    # ── A: ROC curves ───────────────────────────────────────────────────────
    ax_roc = fig.add_subplot(gs[0, 0])
    _style_axis(ax_roc)
    ax_roc.plot([0, 1], [0, 1], color=_EDGE, lw=0.8, ls="--", alpha=0.65, zorder=1)
    ax_roc.plot(FPR_TR, TPR_TR, color=CLR_TRAIN, lw=1.0, alpha=0.9,
                label=f"Train  AUC={AUC_TRAIN:.3f}", zorder=2)
    ax_roc.plot(FPR_TE, TPR_TE, color=CLR_TEST, lw=1.0, alpha=0.9, ls="--",
                label=f"Test   AUC={AUC_TEST:.3f}", zorder=3)
    ax_roc.set_xlabel("1 – Specificity")
    ax_roc.set_ylabel("Sensitivity")
    ax_roc.set_xlim(-0.01, 1.01); ax_roc.set_ylim(-0.01, 1.01)
    _P(ax_roc, "A")
    ax_roc.legend(loc="lower right", fontsize=8.0)

    # ── B: LR coefficients ──────────────────────────────────────────────────
    ax_coef = fig.add_subplot(gs[0, 1])
    _style_axis(ax_coef)
    _ord  = np.argsort(LR_COEF)
    _fo   = [SEL_COLS[i] for i in _ord]
    _co   = LR_COEF[_ord]
    _clrs = [CLR_MAL if c < 0 else CLR_BEN for c in _co]
    ax_coef.barh(range(len(_fo)), _co, color=_clrs, alpha=0.78,
                 edgecolor="none", height=0.58)
    ax_coef.axvline(0, color=_EDGE, lw=0.8, alpha=0.65)
    ax_coef.set_yticks(range(len(_fo)))
    ax_coef.set_yticklabels(_fo, fontsize=8.0)
    ax_coef.set_xlabel("LR Coefficient")
    _P(ax_coef, "B")
    ax_coef.legend(
        handles=[Patch(facecolor=CLR_BEN, alpha=0.78, label="↑ Benign"),
                 Patch(facecolor=CLR_MAL, alpha=0.78, label="↑ Malignant")],
        loc="lower right", fontsize=8.0)

    fig.tight_layout(pad=0.6)
    return _autoscale(fig)


def _make_linearity_fig():
    """A–G: log-odds linearity check (LRT) for each selected feature"""
    n_cols = 4 if len(SEL_COLS) > 4 else max(1, len(SEL_COLS))
    n_rows = int(np.ceil(len(SEL_COLS) / n_cols))
    fig = plt.figure(figsize=(7.0, 2.2 * n_rows))
    gs  = fig.add_gridspec(n_rows, n_cols, wspace=0.42, hspace=0.58)

    for k, feat in enumerate(SEL_COLS):
        r, c = divmod(k, n_cols)
        ax   = fig.add_subplot(gs[r, c])
        _style_axis(ax)
        mids   = LRT_PLOT_DATA[feat]["mids"]
        logits = LRT_PLOT_DATA[feat]["logits"]
        ax.scatter(mids, logits, color=CLR_TRAIN, s=22, zorder=3,
                   alpha=0.82, edgecolors="none")
        if len(mids) > 2:
            z  = np.polyfit(mids, logits, 1)
            xl = np.linspace(min(mids), max(mids), 100)
            ax.plot(xl, np.polyval(z, xl), color=_EDGE, lw=1.0, ls="--")
        r_lrt  = LRT[feat]
        linear = r_lrt["linear"]
        t_col  = _NAVY if linear else CLR_1SE
        verdict = "linear" if linear else "non-linear"
        ax.text(0.97, 0.97, f"p={r_lrt['p']:.3f}  {verdict}",
                transform=ax.transAxes, fontsize=7.2, va="top", ha="right",
                color=t_col, style="italic",
                bbox=dict(facecolor="white", edgecolor="none",
                           alpha=0.75, pad=0.5))
        ax.set_xlabel(feat.replace("worst ", ""), labelpad=2)
        ax.set_ylabel("Log-odds", labelpad=2)
        _P(ax, "ABCDEFG"[k])
        ax.tick_params(labelsize=8.0)

    for k in range(len(SEL_COLS), n_rows * n_cols):
        r, c = divmod(k, n_cols)
        fig.add_subplot(gs[r, c]).set_visible(False)
    fig.tight_layout(pad=0.55)
    return _autoscale(fig)


def _make_vif_fig():
    """Single panel: variance inflation factor for selected features"""
    vif_vals   = [VIF[f] for f in SEL_COLS]
    vif_colors = [CLR_1SE if v > 10 else (_ORANGE if v > 5 else CLR_BEN)
                  for v in vif_vals]
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    _style_axis(ax)
    ax.barh(range(len(SEL_COLS)), vif_vals, color=vif_colors,
            alpha=0.78, edgecolor="none", height=0.52)
    ax.axvline(5.0,  color=_ORANGE, lw=0.8, ls="--", alpha=0.8,
               label="VIF=5  (moderate)")
    ax.axvline(10.0, color=CLR_1SE, lw=0.8, ls="--", alpha=0.8,
               label="VIF=10  (severe)")
    ax.set_yticks(range(len(SEL_COLS)))
    ax.set_yticklabels(SEL_COLS, fontsize=8.0)
    ax.set_xlabel("Variance Inflation Factor")
    _P(ax, "A")
    ax.legend(loc="lower right", fontsize=8.0)
    _vmax = max(vif_vals) if vif_vals else 1.0
    ax.set_xlim(0, _vmax * 1.22)
    for i, v in enumerate(vif_vals):
        ax.text(v + _vmax * 0.012, i, f"{v:.2f}", va="center",
                fontsize=8.0, color=_NAVY)
    fig.tight_layout(pad=0.6)
    return _autoscale(fig)


# Pre-render static figures
_FEAT_SEL_BYTES = _fig_buf(_make_feat_sel_fig())
_PERF_BYTES     = _fig_buf(_make_perf_fig())
_LIN_BYTES      = _fig_buf(_make_linearity_fig())
_VIF_BYTES      = _fig_buf(_make_vif_fig())
print("  Static figures rendered.", flush=True)


# ── 3. Analytics: global visitor map ─────────────────────────────────────────
# Runtime integrations are optional. The app should still start when these
# variables are absent, with visit logging and geo-enrichment disabled.
_IPINFO_TOKEN = os.environ.get("IPINFO_TOKEN", "").strip()
_SUPABASE_URL = os.environ.get(
    "SUPABASE_URL", "https://vmivonjwubhbvlufxkdd.supabase.co"
).strip()
# Supabase anon keys are public client credentials; table access is governed by RLS.
_SUPABASE_KEY = os.environ.get(
    "SUPABASE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZtaXZvbmp3dWJoYnZsdWZ4a2RkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODMwODMyMDQsImV4cCI6MjA5ODY1OTIwNH0.IHH8dPFWYCkZ7eFhIx5zHY0QGMi1_pDM1ebBfzoHha0",
).strip()
_ANALYTICS_CONFIGURED = bool(
    _SUPABASE_URL and _SUPABASE_KEY
)
_ANALYTICS_STATE = {"error": None}
_APP_NAME_BC  = "breast-cancer-classifier"

_COUNTRY_NAMES = {
    "AF":"Afghanistan","AL":"Albania","DZ":"Algeria","AR":"Argentina",
    "AU":"Australia","AT":"Austria","BD":"Bangladesh","BE":"Belgium",
    "BR":"Brazil","BG":"Bulgaria","CA":"Canada","CL":"Chile",
    "CN":"China","CO":"Colombia","HR":"Croatia","CZ":"Czech Republic",
    "DK":"Denmark","EG":"Egypt","FI":"Finland","FR":"France",
    "DE":"Germany","GH":"Ghana","GR":"Greece","HK":"Hong Kong",
    "HU":"Hungary","IN":"India","ID":"Indonesia","IR":"Iran",
    "IQ":"Iraq","IE":"Ireland","IL":"Israel","IT":"Italy",
    "JP":"Japan","JO":"Jordan","KZ":"Kazakhstan","KE":"Kenya",
    "KR":"South Korea","KW":"Kuwait","LB":"Lebanon","MY":"Malaysia",
    "MX":"Mexico","MA":"Morocco","NL":"Netherlands","NZ":"New Zealand",
    "NG":"Nigeria","NO":"Norway","PK":"Pakistan","PE":"Peru",
    "PH":"Philippines","PL":"Poland","PT":"Portugal","QA":"Qatar",
    "RO":"Romania","RU":"Russia","SA":"Saudi Arabia","SG":"Singapore",
    "ZA":"South Africa","ES":"Spain","SE":"Sweden","CH":"Switzerland",
    "TW":"Taiwan","TH":"Thailand","TN":"Tunisia","TR":"Turkey",
    "UA":"Ukraine","AE":"United Arab Emirates","GB":"United Kingdom",
    "US":"United States","VN":"Vietnam","YE":"Yemen","ZW":"Zimbabwe",
}


def _country_name(code: str) -> str:
    return _COUNTRY_NAMES.get((code or "").upper(), code or "")


def _lookup_ip_location(ip: str):
    if not ip or ip in ("127.0.0.1", "::1"):
        return None, None, None, None
    try:
        if _IPINFO_TOKEN:
            response = requests.get(
                f"https://ipinfo.io/{ip}/json",
                params={"token": _IPINFO_TOKEN},
                timeout=4,
            )
            if response.status_code != 200:
                return None, None, None, None
            data = response.json()
            loc = data.get("loc", "")
            lat, lon = map(float, loc.split(",")) if loc else (None, None)
            return data.get("country"), data.get("city"), lat, lon

        response = requests.get(f"https://ipwho.is/{ip}", timeout=4)
        data = response.json() if response.status_code == 200 else {}
        if not data.get("success"):
            return None, None, None, None
        return (
            data.get("country_code"),
            data.get("city"),
            data.get("latitude"),
            data.get("longitude"),
        )
    except Exception:
        return None, None, None, None


def _sb_headers():
    return {
        "apikey":        _SUPABASE_KEY,
        "Authorization": f"Bearer {_SUPABASE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=minimal",
    }


def _log_visit_bc(country, city, lat, lon):
    if not (_SUPABASE_URL and _SUPABASE_KEY):
        return
    if lat is None or lon is None:
        return
    try:
        requests.post(
            f"{_SUPABASE_URL}/rest/v1/visits",
            headers=_sb_headers(),
            json={"app_name": _APP_NAME_BC, "country": country,
                  "city": city, "lat": lat, "lon": lon},
            timeout=5,
        )
    except Exception:
        pass


def _delete_null_visits_bc():
    if not (_SUPABASE_URL and _SUPABASE_KEY):
        return
    try:
        hdrs = {k: v for k, v in _sb_headers().items() if k != "Prefer"}
        requests.delete(
            f"{_SUPABASE_URL}/rest/v1/visits",
            headers=hdrs,
            params={"app_name": f"eq.{_APP_NAME_BC}", "lat": "is.null"},
            timeout=5,
        )
    except Exception:
        pass


def _fetch_visits_bc():
    if not (_SUPABASE_URL and _SUPABASE_KEY):
        _ANALYTICS_STATE["error"] = "configuration"
        return []
    try:
        hdrs = {k: v for k, v in _sb_headers().items() if k != "Prefer"}
        r = requests.get(
            f"{_SUPABASE_URL}/rest/v1/visits",
            headers=hdrs,
            params={"app_name": f"eq.{_APP_NAME_BC}",
                    "select": "country,city,lat,lon"},
            timeout=5,
        )
        if r.status_code == 200:
            data = r.json()
            _ANALYTICS_STATE["error"] = None
            return data if isinstance(data, list) else []
        _ANALYTICS_STATE["error"] = f"http-{r.status_code}"
        return []
    except Exception:
        _ANALYTICS_STATE["error"] = "connection"
        return []


_WORLD_GEO_PATH_BC = Path(__file__).parent / "world.geojson"
_WORLD_GEO_BC = None


def _load_world_geo_bc():
    global _WORLD_GEO_BC
    if _WORLD_GEO_BC is None and _WORLD_GEO_PATH_BC.exists():
        with open(_WORLD_GEO_PATH_BC, encoding="utf-8") as _f:
            _WORLD_GEO_BC = json.load(_f)
    return _WORLD_GEO_BC


threading.Thread(target=_delete_null_visits_bc, daemon=True).start()


def _make_visit_map_bc(user_lat=None, user_lon=None):
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    visits = _fetch_visits_bc()
    valid  = [v for v in visits if v.get("lat") is not None and v.get("lon") is not None]
    lats = [v["lat"] for v in valid]
    lons = [v["lon"] for v in valid]

    city_counts = Counter(
        (v.get("city") or "").strip()
        for v in valid
        if (v.get("city") or "").strip()
    )
    label_names = {city for city, _ in city_counts.most_common(4)}
    seen_cities: set[str] = set()
    city_labels = []
    for v in valid:
        city = (v.get("city") or "").strip()
        if city and city in label_names and city not in seen_cities:
            seen_cities.add(city)
            city_labels.append((v["lat"], v["lon"], city))

    fig, ax = plt.subplots(figsize=(7.0, 3.5), facecolor="white")
    ax.set_facecolor("white")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-70, 85)
    _style_axis(ax)

    geo = _load_world_geo_bc()
    if geo:
        patches = []
        for feat in geo.get("features", []):
            geom = feat.get("geometry") or {}
            gtype = geom.get("type", "")
            coords = geom.get("coordinates", [])
            try:
                if gtype == "Polygon":
                    pts = np.array(coords[0])[:, :2]
                    patches.append(Polygon(pts, closed=True))
                elif gtype == "MultiPolygon":
                    for poly in coords:
                        pts = np.array(poly[0])[:, :2]
                        patches.append(Polygon(pts, closed=True))
            except Exception:
                pass
        if patches:
            pc = PatchCollection(
                patches, facecolor=_FAINT, edgecolor=_EDGE,
                linewidth=0.4, alpha=0.35, zorder=1,
            )
            ax.add_collection(pc)

    if lons:
        ax.scatter(lons, lats, s=28, color=CLR_BEN, alpha=0.8,
                   marker="o", zorder=3, linewidths=0,
                   label=f"Visitors (n={len(lons)})")
        for i, (lat_c, lon_c, name) in enumerate(city_labels):
            if lon_c < -100:
                dx = 5
            elif lon_c > 105:
                dx = -4
            else:
                dx = 5 if i % 2 == 0 else -4
            dy = 4 if i % 3 == 0 else (-7 if i % 3 == 1 else 9)
            ax.annotate(name, xy=(lon_c, lat_c),
                        xytext=(dx, dy), textcoords="offset points",
                        ha="left" if dx >= 0 else "right",
                        fontsize=7.0, color=_NAVY, zorder=5, clip_on=False)

    if user_lat is not None and user_lon is not None:
        ax.scatter([user_lon], [user_lat], s=32, color=CLR_MAL,
                   marker="o", zorder=4, linewidths=0, label="You")

    ax.set_xticks([])
    ax.set_yticks([])

    if lons or user_lat is not None:
        ax.legend(fontsize=8.0, loc="lower left", frameon=False)

    fig.tight_layout(pad=0.6)
    return fig


# ── 4. CSS ───────────────────────────────────────────────────────────────────
_CSS = """
:root{
  --red:#E64B35;--blue:#4DBBD5;--teal:#00A087;--navy:#3C5488;--salmon:#F39B7F;
  --lav:#8491B4;--mint:#91D1C2;--crimson:#DC0000;--brown:#7E6148;--tan:#B09C85;
  --ink:var(--navy);--accent:var(--blue);--accent-2:var(--teal);--surface:#FFFFFF;
  --panel:#F7FBFD;--line:rgba(60,84,136,.16);--line-strong:rgba(60,84,136,.28);
  --muted:rgba(60,84,136,.78);--shadow-sm:0 10px 28px rgba(60,84,136,.08);
  --shadow-lg:0 24px 60px rgba(60,84,136,.12);--r:18px;--rs:12px;
  --font:'Arial','Helvetica','Liberation Sans','DejaVu Sans',sans-serif;
}
html,body{
  height:100%;
  font-family:var(--font);
  font-size:15px;
  font-variant-numeric:tabular-nums;
  background:#F7FBFD;
  color:var(--ink);
  -webkit-font-smoothing:antialiased;
}
.navbar{
  background:rgba(255,255,255,.96)!important;
  border-bottom:1px solid var(--line)!important;
  padding:.88rem 1.8rem;
  box-shadow:var(--shadow-sm);
  backdrop-filter:blur(14px);
  position:relative;
}
.navbar::after{
  content:"";
  position:absolute;
  left:0;right:0;bottom:-1px;height:4px;
  background:var(--accent);
}
.navbar-brand{color:var(--ink)!important;font-weight:800;font-size:1.02rem;letter-spacing:.18px;}
.bslib-sidebar-layout>.sidebar{
  background:rgba(255,255,255,.95)!important;
  border-right:1px solid var(--line)!important;
  box-shadow:var(--shadow-sm);
  overflow-y:auto;
  height:100%;
  padding:1.1rem 1.2rem 1.5rem;
}
.sec{
  font-size:.74rem;
  font-weight:800;
  color:var(--ink);
  text-transform:uppercase;
  letter-spacing:1.6px;
  margin:1.4rem 0 .8rem;
  padding:0 0 .35rem .8rem;
  border-left:5px solid var(--accent-2);
  line-height:1.35;
}
.sec:first-child{margin-top:.25rem;}
.form-label{font-size:.95rem;font-weight:700;color:var(--ink);margin-bottom:.42rem;display:block;}
.form-control,.form-select{
  font-size:.86rem;
  border:1px solid rgba(60,84,136,.18);
  border-radius:var(--rs);
  background:var(--surface);
  padding:.7rem .86rem;
  min-height:2.95rem;
  color:var(--ink);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.7);
  transition:border-color .16s,box-shadow .16s,background .16s,transform .16s;
}
.form-control:focus,.form-select:focus{
  border-color:var(--accent);
  background:white;
  box-shadow:0 0 0 4px rgba(77,187,213,.18);
  outline:none;
}
.btn-primary{
  background:var(--accent)!important;
  border:none!important;
  border-radius:14px!important;
  font-size:.88rem!important;
  font-weight:700!important;
  letter-spacing:.1px;
  padding:.84rem 1rem!important;
  box-shadow:0 12px 24px rgba(60,84,136,.16);
}
.btn-primary:hover{transform:translateY(-1px);}
.btn-sm{font-size:.92rem!important;padding:.68rem 1rem!important;}
.irs{height:44px;}.irs-with-grid{height:60px;}
.irs-line{height:6px!important;background:rgba(132,145,180,.22)!important;border:none!important;border-radius:999px;top:28px;}
.irs-bar,.irs-bar-edge{height:6px!important;background:var(--accent)!important;border:none!important;top:28px;}
.irs-bar-edge{width:6px!important;}
.irs-handle{
  width:18px!important;height:18px!important;background:white!important;
  border:3px solid var(--accent)!important;border-radius:50%!important;top:22px!important;
  box-shadow:0 4px 12px rgba(77,187,213,.28)!important;cursor:grab;
}
.irs-handle:hover,.irs-handle.state_hover{transform:scale(1.16)!important;}
.irs-single{background:var(--ink)!important;color:white;font-size:.78rem;font-weight:700;padding:4px 9px;border-radius:999px;}
.irs-single::before{border-top-color:var(--ink)!important;}
.irs-from,.irs-to{display:none!important;}
.irs-grid-pol{background:rgba(132,145,180,.38)!important;}
.irs-grid-text{font-size:.72rem!important;color:rgba(60,84,136,.72)!important;}
.bslib-sidebar-layout>.main{padding:clamp(18px,2.6vw,30px)!important;}
.card-body{padding:clamp(14px,1.7vw,20px)!important;}
.page-title{
  font-size:clamp(1.55rem,1.9vw,2rem);
  font-weight:800;
  color:var(--ink);
  display:inline-block;
  margin:.15rem 0 .3rem;
  letter-spacing:-.03em;
  position:relative;
}
.page-title::after{
  content:"";
  display:block;
  height:4px;
  width:min(100%, 22rem);
  background:var(--accent);
  border-radius:999px;
  margin-top:.6rem;
}
.page-subtitle{
  color:var(--muted);
  font-size:.84rem;
  margin-bottom:1.25rem;
  line-height:1.62;
  max-width:70rem;
}
.infobar{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
  gap:14px;
  margin-bottom:18px;
}
.chip{
  background:rgba(255,255,255,.92);
  border:1px solid var(--line);
  border-top:4px solid var(--accent);
  border-radius:16px;
  padding:.88rem 1rem;
  display:flex;
  flex-direction:column;
  gap:.25rem;
  box-shadow:var(--shadow-sm);
}
.chip:nth-child(2n){border-top-color:var(--teal);}
.chip:nth-child(3n){border-top-color:var(--navy);}
.chip:nth-child(4n){border-top-color:var(--salmon);}
.chip-lbl{font-size:.62rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.9px;}
.chip-val{font-size:1.0rem;font-weight:800;color:var(--ink);white-space:nowrap;}
.card{
  border:1px solid var(--line)!important;
  border-radius:var(--r)!important;
  box-shadow:var(--shadow-sm)!important;
  background:rgba(255,255,255,.93)!important;
  overflow:hidden;
  margin-bottom:16px;
  position:relative;
}
.card::before{
  content:"";
  display:block;
  height:4px;
  background:var(--accent);
}
.card-header{
  background:rgba(255,255,255,.88)!important;
  border-bottom:1px solid rgba(60,84,136,.10)!important;
  color:var(--ink)!important;
  font-weight:800;
  font-size:.88rem;
  letter-spacing:.14px;
  padding:1rem 1.2rem;
}
.plot-frame{
  width:100%;
  height:clamp(300px,36vw,440px);
  display:flex;
  align-items:center;
  justify-content:center;
  overflow:hidden;
}
.plot-frame.plot-map{height:clamp(260px,31vw,390px);}
.plot-frame.plot-wide{height:clamp(320px,34vw,440px);}
.plot-frame .shiny-plot-output{width:100%!important;height:100%!important;}
.plot-frame .shiny-plot-output img,.plot-frame .shiny-plot-output canvas{
  width:100%!important;height:100%!important;max-width:100%!important;
  max-height:100%!important;object-fit:contain!important;object-position:center center!important;
}
.equal-card{height:100%;display:flex;flex-direction:column;}
.equal-card .card-body{flex:1 1 auto;min-height:0;display:flex;flex-direction:column;}
.plot-frame,.result-frame{flex:1 1 auto;min-height:0;}
.result-frame{display:flex;flex-direction:column;justify-content:flex-start;}
.result-frame.result-map{min-height:clamp(260px,31vw,390px);}
.result-frame.result-gauge{min-height:clamp(300px,36vw,390px);}
.figure-frame{
  width:100%;
  overflow:hidden;
  display:flex;
  align-items:center;
  justify-content:center;
  padding:.2rem 0 .35rem;
}
.figure-frame .shiny-image-output,.figure-frame .shiny-image-output img{
  width:100%!important;max-width:100%!important;height:auto!important;object-fit:contain!important;
}
.nav-tabs{border:0!important;margin-bottom:18px;gap:10px;flex-wrap:wrap;}
.nav-tabs .nav-link{
  color:var(--muted)!important;
  font-size:.84rem;
  font-weight:700;
  border:1px solid rgba(60,84,136,.14)!important;
  padding:.68rem 1.1rem;
  margin-bottom:0;
  border-radius:999px;
  background:rgba(255,255,255,.82)!important;
  box-shadow:0 6px 18px rgba(60,84,136,.05);
  transition:color .14s,background .14s,transform .14s,box-shadow .14s,border-color .14s;
}
.nav-tabs .nav-link:hover{
  color:var(--ink)!important;
  background:rgba(77,187,213,.10)!important;
  border-color:rgba(77,187,213,.32)!important;
  transform:translateY(-1px);
}
.nav-tabs .nav-link.active{
  color:var(--ink)!important;
  font-weight:800;
  border-color:transparent!important;
  background:rgba(77,187,213,.18)!important;
  box-shadow:var(--shadow-sm);
}
.prob-gauge{text-align:center;padding:1rem 0 .85rem;}
.prob-num{font-size:clamp(2.3rem,3.4vw,3rem);font-weight:800;line-height:1.0;}
.prob-label{font-size:.8rem;color:var(--muted);margin-top:.4rem;}
.prob-bar-wrap{
  background:rgba(132,145,180,.18);
  border-radius:999px;
  height:14px;
  margin:14px 22px 10px;
  overflow:hidden;
}
.prob-bar-fill{height:100%;border-radius:999px;transition:width .3s;}
.class-badge{
  display:inline-block;
  padding:.52rem 1.2rem;
  border-radius:999px;
  font-size:.84rem;
  font-weight:800;
  letter-spacing:.2px;
  margin-top:10px;
}
.tbl{width:100%;border-collapse:collapse;font-size:.82rem;}
.tbl th{
  text-align:left;color:var(--ink);font-weight:700;padding:.72rem .7rem;
  border-bottom:2px solid rgba(60,84,136,.12);font-size:.64rem!important;text-transform:uppercase;
  letter-spacing:.65px;white-space:nowrap;
}
.tbl td{
  padding:.72rem .7rem;border-bottom:1px solid rgba(132,145,180,.14);
  font-variant-numeric:tabular-nums;font-size:.82rem!important;
}
.tbl td.num{text-align:right;font-weight:700;}
.tbl tr:hover td{background:rgba(145,209,194,.12);}
.cm-wrap{
  display:grid;
  grid-template-columns:140px 1fr 1fr;
  grid-template-rows:44px 1fr 1fr;
  gap:6px;
  margin:10px 0;
}
.cm-corner{background:transparent;}
.cm-col-hdr,.cm-row-hdr{
  display:flex;align-items:center;justify-content:center;
  font-size:.7rem;font-weight:800;color:var(--ink);
  background:rgba(145,209,194,.18);border-radius:var(--rs);padding:0 8px;
}
.cm-cell{
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  border-radius:var(--rs);padding:18px 10px;text-align:center;
}
.cm-cell .cm-n{font-size:1.6rem;font-weight:800;line-height:1.05;}
.cm-cell .cm-desc{font-size:.68rem;color:rgba(60,84,136,.78);margin-top:5px;line-height:1.35;}
.disclaimer{
  color:var(--muted);
  font-size:.74rem;
  margin-top:10px;
  padding-top:10px;
  border-top:1px solid var(--line);
  text-align:center;
  line-height:1.65;
}
.methods h4{
  font-size:.68rem;font-weight:800;color:var(--accent);
  text-transform:uppercase;letter-spacing:1.2px;margin:16px 0 8px;
  padding-left:10px;border-left:4px solid var(--accent-2);
}
.methods p{font-size:.82rem;line-height:1.58;margin:0 0 12px;}
.bslib-page-fill{height:100dvh!important;}
@media (max-width: 900px){
  html,body{font-size:14px;}
  .bslib-sidebar-layout>.sidebar{padding:1rem .95rem 1.3rem;}
  .bslib-sidebar-layout>.main{padding:14px!important;}
  .page-title{font-size:clamp(1.55rem,7vw,2rem);}
  .page-subtitle{font-size:.9rem;}
  .plot-frame{height:clamp(250px,72vw,360px);}
  .plot-frame.plot-map{height:clamp(220px,64vw,320px);}
  .plot-frame.plot-wide{height:clamp(260px,74vw,360px);}
  .result-frame.result-map{min-height:clamp(220px,64vw,320px);}
  .result-frame.result-gauge{min-height:clamp(250px,72vw,360px);}
  .nav-tabs .nav-link{padding:.5rem .82rem;font-size:.8rem;}
  .cm-wrap{grid-template-columns:98px 1fr 1fr;grid-template-rows:38px 1fr 1fr;}
}
"""

_CSS += """
:root{--r:8px;--rs:8px;}
.card,.chip{border-radius:8px!important;}
.btn-primary{border-radius:8px!important;}
.hero-copy{margin-bottom:1rem;}
.hero-kicker{
  font-size:.68rem;
  font-weight:800;
  color:var(--accent-2);
  text-transform:uppercase;
  letter-spacing:1.1px;
  margin-bottom:.35rem;
}
.summary-grid{
  display:grid;
  grid-template-columns:repeat(4,minmax(0,1fr));
  gap:12px;
  margin:0 0 18px;
}
.summary-tile{
  position:relative;
  overflow:hidden;
  background:rgba(255,255,255,.9);
  border:1px solid var(--line);
  border-radius:8px;
  padding:14px 14px 12px;
  box-shadow:0 8px 22px rgba(60,84,136,.08);
}
.summary-tile::before{
  content:"";
  position:absolute;
  left:0;
  top:0;
  bottom:0;
  width:4px;
  background:var(--tile);
}
.summary-tile.accent-blue{--tile:var(--blue);}
.summary-tile.accent-teal{--tile:var(--teal);}
.summary-tile.accent-navy{--tile:var(--navy);}
.summary-tile.accent-salmon{--tile:var(--salmon);}
.summary-tile.accent-crimson{--tile:var(--crimson);}
.summary-label{
  font-size:.66rem;
  font-weight:800;
  color:var(--muted);
  text-transform:uppercase;
  letter-spacing:.8px;
}
.summary-value{
  font-size:1.24rem;
  font-weight:800;
  color:var(--ink);
  line-height:1.1;
  margin-top:8px;
}
.summary-detail{
  font-size:.78rem;
  color:var(--muted);
  line-height:1.48;
  margin-top:7px;
}
.section-head{
  display:flex;
  flex-direction:column;
  gap:4px;
  margin:0 0 14px;
}
.section-eyebrow{
  font-size:.66rem;
  font-weight:800;
  color:var(--accent-2);
  text-transform:uppercase;
  letter-spacing:.9px;
}
.section-title{
  margin:0;
  font-size:1.08rem;
  font-weight:800;
  color:var(--ink);
}
.section-copy{
  margin:0;
  max-width:60rem;
  font-size:.82rem;
  line-height:1.55;
  color:var(--muted);
}
.note-grid{
  display:grid;
  grid-template-columns:repeat(3,minmax(0,1fr));
  gap:12px;
  margin:0 0 18px;
}
.note-block{
  background:rgba(255,255,255,.72);
  border:1px dashed var(--line-strong);
  border-radius:8px;
  padding:12px 14px;
}
.note-title{
  font-size:.72rem;
  font-weight:800;
  color:var(--ink);
  text-transform:uppercase;
  letter-spacing:.65px;
}
.note-copy{
  margin:6px 0 0;
  font-size:.78rem;
  line-height:1.5;
  color:var(--muted);
}
@media (max-width: 1100px){
  .summary-grid,.note-grid{grid-template-columns:repeat(2,minmax(0,1fr));}
}
@media (max-width: 700px){
  .summary-grid,.note-grid{grid-template-columns:1fr;}
}
"""

_CSS += """
*{letter-spacing:0!important;}
html,body{background:#FFFFFF;}
.navbar{
  background:#FFFFFF!important;
  box-shadow:none;
  backdrop-filter:none;
  padding:.78rem 1.35rem;
}
.navbar::after{height:3px;background:var(--red);}
.bslib-sidebar-layout>.sidebar{
  background:#FFFFFF!important;
  box-shadow:none;
}
.bslib-sidebar-layout>.main{background:#FFFFFF;}
.page-title{font-size:1.7rem;letter-spacing:0;}
.page-title::after{height:3px;border-radius:0;background:var(--red);}
.card,.summary-tile,.chip{
  background:#FFFFFF!important;
  border:1px solid var(--line-strong)!important;
  box-shadow:none!important;
  border-radius:8px!important;
}
.card::before{height:3px;background:var(--blue);}
.card-header{background:#FFFFFF!important;padding:.82rem 1rem;}
.summary-tile{border-top:3px solid var(--tile)!important;}
.summary-tile::before{display:none;}
.note-block{
  background:#FFFFFF;
  border:0;
  border-left:3px solid var(--mint);
  border-radius:0;
  padding:9px 12px;
}
.nav-tabs .nav-link{box-shadow:none;}
.btn-primary{box-shadow:none!important;}
@media (max-width:700px){
  .page-title{font-size:1.45rem;}
  .navbar{padding:.68rem .85rem;}
}
"""


# ── 4. UI ────────────────────────────────────────────────────────────────────

def _slider_step(lo: float, hi: float) -> float:
    rng = hi - lo
    if rng == 0:
        return 0.001
    mag = 10 ** (np.floor(np.log10(rng)) - 2)
    return round(float(mag), 10)


def _make_inputs():
    items = []
    for f in SEL_COLS:
        lo, hi, _ = FEAT_RANGES[f]
        med = TRAIN_MEDIANS[f]
        step     = _slider_step(lo, hi)
        decimals = max(0, -int(np.floor(np.log10(step)))) if step > 0 else 4
        items.append(
            ui.input_numeric(_sid(f), f,
                             value=round(med, decimals),
                             min=round(lo, decimals),
                             max=round(hi, decimals),
                             step=step)
        )
    return items


def _summary_tile(label: str, value: str, detail: str, accent: str) -> ui.Tag:
    return ui.tags.div(
        ui.tags.div(label, class_="summary-label"),
        ui.tags.div(value, class_="summary-value"),
        ui.tags.div(detail, class_="summary-detail"),
        class_=f"summary-tile {accent}",
    )


def _section_head(kicker: str, title: str, copy: str) -> ui.Tag:
    return ui.tags.div(
        ui.tags.div(kicker, class_="section-eyebrow"),
        ui.tags.h4(title, class_="section-title"),
        ui.tags.p(copy, class_="section-copy"),
        class_="section-head",
    )


def _note_block(title: str, copy: str) -> ui.Tag:
    return ui.tags.div(
        ui.tags.div(title, class_="note-title"),
        ui.tags.p(copy, class_="note-copy"),
        class_="note-block",
    )


app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.tags.div("Prediction Inputs", class_="sec"),
        *_make_inputs(),
        ui.input_action_button(
            "submit", "Run Prediction",
            class_="btn btn-primary w-100",
            style="margin-top:10px;font-weight:600;",
        ),
        ui.tags.div("Model Snapshot", class_="sec"),
        ui.tags.div(
            ui.tags.span("■ Plain Logistic Regression",
                         style=f"color:{CLR_BEN};font-weight:600;font-size:.80rem;"),
            style="line-height:2;",
        ),
        ui.tags.p(
            f"Train {N_TRAIN} / Test {N_TEST} · {N_SEL} retained features",
            style=f"font-size:.72rem;color:{_MUTED};margin:4px 0 0;",
        ),
        width=300,
    ),

    ui.tags.style(_CSS),
    ui.tags.div(
        ui.tags.div("Diagnosis operations dashboard", class_="hero-kicker"),
        ui.tags.h3(
            "Breast Cancer Diagnosis Command Center",
            class_="page-title",
        ),
        ui.tags.p(
            f"Wisconsin Breast Cancer Dataset · N={N_TOTAL} · "
            f"LASSO lambda_1se retained {N_SEL} of {len(FEAT_NAMES)} features before plain logistic regression. "
            f"Held-out AUC {AUC_TEST:.3f} · Brier {BRIER_TEST:.3f} · "
            f"stratified split train {N_TRAIN} / test {N_TEST}.",
            class_="page-subtitle",
        ),
        class_="hero-copy",
    ),
    ui.tags.div(
        _summary_tile(
            "Cohort",
            f"{N_TOTAL}",
            f"{n_benign} benign / {n_malignant} malignant cases",
            "accent-blue",
        ),
        _summary_tile(
            "Feature funnel",
            f"{N_SEL} retained",
            f"from {len(FEAT_NAMES)} raw predictors at lambda_1se",
            "accent-teal",
        ),
        _summary_tile(
            "Generalization",
            f"{AUC_TEST:.3f}",
            f"train AUC {AUC_TRAIN:.3f} / test AUC {AUC_TEST:.3f}",
            "accent-navy",
        ),
        _summary_tile(
            "Calibration",
            f"{BRIER_TEST:.3f}",
            f"HL p={HL_P:.3f} · null Brier {NULL_BRIER:.3f}",
            "accent-salmon",
        ),
        class_="summary-grid",
    ),

    ui.navset_tab(
        ui.nav_panel(
            "Dashboard",
            _section_head(
                "Prediction desk",
                "Single-patient assessment",
                "The main dashboard keeps the patient probability, feature context, and live usage intelligence in one analyst-oriented view.",
            ),
            ui.output_ui("pred_chips"),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Predicted Probability"),
                    ui.tags.div(
                        ui.output_ui("pred_gauge"),
                        class_="result-frame result-gauge",
                    ),
                    class_="equal-card",
                ),
                ui.card(
                    ui.card_header("Feature Values vs. Training Median"),
                    ui.tags.div(ui.output_ui("feat_table"), class_="result-frame result-gauge"),
                    class_="equal-card",
                ),
                col_widths=[5, 7],
            ),
            ui.tags.div(
                _note_block(
                    "Classification rule",
                    "The default interpretation uses P(malignant) >= 0.50. Threshold Lab lets you examine what changes when that operational cutoff moves.",
                ),
                _note_block(
                    "Feature context",
                    "Each entered value is benchmarked against the training-set median of the retained features so direction of shift is explicit.",
                ),
                _note_block(
                    "Use boundary",
                    "This tool is intended for research and educational review. It exposes model behavior but does not replace clinical assessment.",
                ),
                class_="note-grid",
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Global Visitor Map"),
                    ui.tags.div(
                        ui.output_plot("visit_map", width="100%", height="100%"),
                        class_="plot-frame plot-map",
                    ),
                    class_="equal-card",
                ),
                ui.card(
                    ui.card_header("Visit Statistics"),
                    ui.tags.div(ui.output_ui("visit_stats"), class_="result-frame result-map"),
                    class_="equal-card",
                ),
                col_widths=[8, 4],
            ),
            ui.tags.p(
                f"Prediction uses the {N_SEL} LASSO-selected features scaled using "
                "RobustScaler (fitted on the training set). For research and educational use only.",
                class_="disclaimer",
            ),
        ),

        ui.nav_panel(
            "Batch Queue",
            _section_head(
                "Batch scoring",
                "Upload, validate, and export cohort predictions",
                "This view is structured like a queue: validate the file on the left, inspect the first rows on the right, then export a scored table.",
            ),
            ui.tags.div(
                _note_block(
                    "CSV schema",
                    "The uploaded file must include the retained feature columns used by the deployed bundle.",
                ),
                _note_block(
                    "Inline review",
                    "Only the first 20 rows are shown inline so obvious column or value problems are visible before download.",
                ),
                _note_block(
                    "Output contract",
                    "Downloaded predictions keep the original feature columns and append probability and class outputs.",
                ),
                class_="note-grid",
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Upload CSV"),
                    ui.input_file("batch_file", None,
                                  accept=[".csv"],
                                  placeholder="Choose CSV file…"),
                    ui.tags.p(
                        "CSV must contain columns: " +
                        ", ".join(f'"{c}"' for c in SEL_COLS),
                        style=f"font-size:.76rem;color:{_MUTED};margin:6px 0 0;",
                    ),
                    ui.output_ui("batch_status"),
                    ui.download_button("batch_download", "Download Predictions",
                                       class_="btn btn-primary btn-sm",
                                       style="margin-top:8px;"),
                ),
                ui.card(
                    ui.card_header("Prediction Preview (first 20 rows)"),
                    ui.output_ui("batch_table"),
                ),
                col_widths=[4, 8],
            ),
        ),

        ui.nav_panel(
            "Threshold Lab",
            _section_head(
                "Threshold tuning",
                "Inspect the cost of moving the decision cutoff",
                "The training-set score distribution makes overlap between benign and malignant predictions visible before you commit to a more aggressive or conservative rule.",
            ),
            ui.tags.div(
                _note_block(
                    "Why it matters",
                    "Changing the threshold trades sensitivity against specificity. The confusion matrix and metric panel update together so the tradeoff stays explicit.",
                ),
                _note_block(
                    "Reference set",
                    "This laboratory view is anchored on training predictions to isolate the effect of the cutoff itself.",
                ),
                _note_block(
                    "Default baseline",
                    "Headline performance elsewhere still references the standard 0.50 rule unless stated otherwise.",
                ),
                class_="note-grid",
            ),
            ui.card(
                ui.card_header("Training Set — Predicted Probability Distribution"),
                ui.tags.div(
                    ui.output_plot("hist_img", width="100%", height="100%"),
                    class_="plot-frame plot-wide",
                ),
            ),
            ui.card(
                ui.card_header("Threshold Selection"),
                ui.input_slider("threshold", "Decision Threshold",
                                min=0.01, max=0.99, value=0.50, step=0.01),
                ui.tags.p(
                    "Classified as malignant when P(malignant) >= threshold; "
                    "benign otherwise.",
                    style=f"font-size:.76rem;color:{_MUTED};margin-bottom:8px;",
                ),
                ui.layout_columns(
                    ui.output_ui("cm_display"),
                    ui.output_ui("metrics_table"),
                    col_widths=[5, 7],
                ),
            ),
        ),

        ui.nav_panel(
            "Model Evidence",
            _section_head(
                "Evidence stack",
                "Feature selection, discrimination, calibration, and diagnostics",
                "This tab turns the modeling workflow into a compact evidence sequence: headline metrics first, coefficient audit second, then the pre-rendered figures supporting the saved bundle.",
            ),
            ui.tags.div(
                _summary_tile(
                    "lambda_1se",
                    f"{C_1SE:.4g}",
                    f"{N_SEL} retained / {len(FEAT_NAMES)} raw features",
                    "accent-crimson",
                ),
                _summary_tile(
                    "Brier skill",
                    f"{1 - BRIER_TEST / NULL_BRIER:.3f}",
                    f"test Brier {BRIER_TEST:.3f} vs null {NULL_BRIER:.3f}",
                    "accent-teal",
                ),
                _summary_tile(
                    "Reference rule",
                    "P(malignant) >= 0.50",
                    f"test accuracy {TEST_METRICS_05['acc']:.3f} · malignant F1 {TEST_METRICS_05['f1']:.3f}",
                    "accent-blue",
                ),
                _summary_tile(
                    "Split",
                    f"{N_TRAIN} / {N_TEST}",
                    "train / test cases, stratified",
                    "accent-salmon",
                ),
                class_="summary-grid",
            ),
            ui.output_ui("perf_metrics_table"),
            ui.output_ui("coef_table"),
            ui.card(
                ui.card_header("Feature Selection — LASSO Regularisation"),
                ui.tags.div(
                    ui.output_image("fig_feat_sel", width="100%", height="auto"),
                    class_="figure-frame",
                ),
            ),
            ui.card(
                ui.card_header("Model Performance"),
                ui.tags.div(
                    ui.output_image("fig_perf", width="100%", height="auto"),
                    class_="figure-frame",
                ),
            ),
            ui.card(
                ui.card_header("Linearity Assessment (LRT, alpha = 0.10)"),
                ui.tags.div(
                    ui.output_image("fig_linearity", width="100%", height="auto"),
                    class_="figure-frame",
                ),
            ),
            ui.card(
                ui.card_header("Collinearity (VIF)"),
                ui.tags.div(
                    ui.output_image("fig_vif", width="100%", height="auto"),
                    class_="figure-frame",
                ),
            ),
        ),

        ui.nav_panel(
            "Methods",
            _section_head(
                "Reproducibility",
                "Bundle, diagnostics, and modeling assumptions",
                "All preprocessing, feature retention, scaling, and diagnostics shown here are loaded from the saved deployment bundle.",
            ),
            ui.output_ui("methods_panel"),
        ),

        id="main_tab",
    ),

    title="Breast Cancer Dashboard",
    fillable=True,
)


# ── 5. Server ────────────────────────────────────────────────────────────────

def server(input, output, session):

    # ── Visit logging ─────────────────────────────────────────────────────────
    _user_loc = {"lat": None, "lon": None}
    try:
        _hdrs = session.http_conn.headers
        _ip = (
            _hdrs.get("x-forwarded-for") or
            _hdrs.get("x-real-ip") or ""
        ).split(",")[0].strip()
    except Exception:
        _ip = ""

    def _do_log(ip: str) -> None:
        country, city, lat, lon = _lookup_ip_location(ip)
        if lat is not None and lon is not None:
            _user_loc["lat"] = lat
            _user_loc["lon"] = lon
        _log_visit_bc(country, city, lat, lon)

    threading.Thread(target=_do_log, args=(_ip,), daemon=True).start()

    # Auto-refresh the map/stats a few times after load so the visitor's own
    # just-logged visit (written asynchronously above) appears without a
    # manual page refresh. Bounded to a handful of ticks, then stops.
    _refresh_tick = reactive.value(0)
    _refresh_n    = {"c": 0}

    @reactive.effect
    def _auto_refresh():
        _refresh_n["c"] += 1
        if _refresh_n["c"] <= 3:
            reactive.invalidate_later(3)
            _refresh_tick.set(_refresh_n["c"])

    @render.plot
    def visit_map():
        _refresh_tick.get()  # re-render when the tick advances
        return _make_visit_map_bc(_user_loc["lat"], _user_loc["lon"])

    @render.ui
    def visit_stats():
        from collections import Counter
        _refresh_tick.get()  # re-render when the tick advances
        visits = _fetch_visits_bc()
        total  = len(visits)
        counts = Counter(
            f"{v.get('city')}, {_country_name(v.get('country'))}" if v.get("country") else v.get("city")
            for v in visits if v.get("city")
        )
        top    = counts.most_common(10)
        rows   = "".join(
            f"<tr><td style='font-size:.70rem;'>{i+1}. {c}</td>"
            f"<td class='num' style='font-size:.70rem;'>{n}</td></tr>"
            for i, (c, n) in enumerate(top)
        )
        empty_message = (
            "Visit analytics are temporarily unavailable."
            if _ANALYTICS_STATE["error"]
            else "No visits recorded yet."
        )
        empty = (
            f'<p style="font-size:.64rem;color:{_MUTED};padding:8px 0;">'
            f"{empty_message}</p>"
            if not top else ""
        )
        return ui.HTML(f"""
<div style="padding:4px 6px;">
  <div style="text-align:center;margin-bottom:12px;">
    <div style="font-size:1.9rem;font-weight:700;color:{_NAVY};">{total}</div>
    <div style="font-size:.62rem;color:{_MUTED};text-transform:uppercase;
                letter-spacing:.8px;">Total Visits</div>
  </div>
  <table class="tbl" style="width:100%;">
    <thead><tr>
      <th>City</th>
      <th style="text-align:right;">Visits</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
  {empty}
</div>
""")

    # ── Individual prediction ─────────────────────────────────────────────────

    @reactive.calc
    def _submitted_inputs() -> dict:
        input.submit()
        with reactive.isolate():
            return {f: float(input[_sid(f)]()) for f in SEL_COLS}

    @reactive.calc
    def _patient_prob() -> float:
        vals = _submitted_inputs()
        X    = pd.DataFrame([vals])[SEL_COLS].values
        return float(pipe_lr.predict_proba(X)[0, 1])

    @render.ui
    def pred_chips():
        p_ben = _patient_prob()
        p_mal = 1.0 - p_ben
        cls  = "Malignant" if p_mal >= 0.5 else "Benign"
        clr  = CLR_MAL if p_mal >= 0.5 else CLR_BEN
        chips = (
            f'<div class="chip"><span class="chip-lbl">P(Malignant)</span>'
            f'<span class="chip-val" style="color:{CLR_MAL};">{p_mal*100:.1f}%</span></div>'
            f'<div class="chip"><span class="chip-lbl">P(Benign)</span>'
            f'<span class="chip-val" style="color:{CLR_BEN};">{p_ben*100:.1f}%</span></div>'
            f'<div class="chip"><span class="chip-lbl">Classification</span>'
            f'<span class="chip-val" style="color:{clr};">{cls}</span></div>'
        )
        return ui.HTML(f'<div class="infobar">{chips}</div>')

    @render.ui
    def pred_gauge():
        p_ben = _patient_prob()
        p_mal = 1.0 - p_ben
        cls  = "Malignant" if p_mal >= 0.5 else "Benign"
        clr  = CLR_MAL if p_mal >= 0.5 else CLR_BEN
        pct  = p_mal * 100
        badge_bg = "#EDF2FA" if p_mal >= 0.5 else "#EAF8FA"
        return ui.HTML(f"""
<div class="prob-gauge">
  <div class="prob-num" style="color:{clr};">{pct:.1f}%</div>
  <div class="prob-label">Predicted probability of malignancy</div>
  <div class="prob-bar-wrap">
    <div class="prob-bar-fill"
         style="width:{pct:.1f}%;background:{clr};"></div>
  </div>
  <div>
    <span class="class-badge"
          style="background:{badge_bg};color:{clr};border:2px solid {clr};">
      {cls}
    </span>
  </div>
  <div style="font-size:.68rem;color:{_MUTED};margin-top:10px;">
    Classification threshold: 0.50
  </div>
</div>
""")

    @render.ui
    def feat_table():
        vals = _submitted_inputs()
        rows = ""
        for f in SEL_COLS:
            val  = vals[f]
            med  = TRAIN_MEDIANS[f]
            diff = val - med
            sign = "↑" if diff > 0 else ("↓" if diff < 0 else "—")
            col  = CLR_1SE if diff > 0 else CLR_TRAIN
            rows += (
                f"<tr><td style='font-size:.72rem;'>{f}</td>"
                f"<td class='num'>{val:.4g}</td>"
                f"<td class='num'>{med:.4g}</td>"
                f"<td class='num' style='color:{col};font-weight:700;'>"
                f"{sign} {abs(diff):.4g}</td></tr>"
            )
        return ui.HTML(f"""
<div style="overflow-x:auto;margin-top:4px;">
  <table class="tbl">
    <thead><tr>
      <th>Feature</th><th>Value</th><th>Train Median</th><th>Δ</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>
<p style="font-size:.68rem;color:{_MUTED};margin-top:6px;">
  A negative LR coefficient indicates that higher feature values are associated with malignancy.
</p>
""")

    # ── Batch prediction ──────────────────────────────────────────────────────

    @reactive.calc
    def _batch_df():
        fi = input.batch_file()
        if not fi:
            return None, "No file uploaded."
        try:
            df = pd.read_csv(fi[0]["datapath"])
        except Exception as e:
            return None, f"Read error: {e}"
        missing = [c for c in SEL_COLS if c not in df.columns]
        if missing:
            return None, f"Missing columns: {', '.join(missing)}"
        X_b  = df[SEL_COLS].values
        p_benign = pipe_lr.predict_proba(X_b)[:, 1]
        p_malignant = 1.0 - p_benign
        df["P_malignant"] = p_malignant.round(4)
        df["P_benign"]    = p_benign.round(4)
        df["Prediction"]  = np.where(p_malignant >= 0.5, "Malignant", "Benign")
        return df, f"OK — {len(df)} rows processed."

    @render.ui
    def batch_status():
        _, msg = _batch_df()
        ok  = msg and msg.startswith("OK")
        col = CLR_BEN if ok else _MUTED
        return ui.HTML(
            f'<p style="font-size:.80rem;color:{col};margin:6px 0 0;">{msg or ""}</p>'
        )

    @render.ui
    def batch_table():
        df, msg = _batch_df()
        if df is None:
            return ui.HTML(
                f'<p style="color:{_MUTED};font-size:.82rem;padding:8px;">'
                "Upload a valid CSV file to see predictions.</p>"
            )
        show = df.head(20)
        cols = list(show.columns)
        thead = "".join(f"<th>{c}</th>" for c in cols)
        tbody = ""
        for _, row in show.iterrows():
            cells = ""
            for c in cols:
                v = row[c]
                if c == "Prediction":
                    col = CLR_BEN if v == "Benign" else CLR_MAL
                    cells += (f'<td style="font-weight:700;color:{col};">{v}</td>')
                elif c in ("P_benign", "P_malignant"):
                    cells += f'<td class="num">{v:.4f}</td>'
                else:
                    cells += f"<td>{v}</td>"
            tbody += f"<tr>{cells}</tr>"
        return ui.HTML(f"""
<div style="overflow-x:auto;max-height:420px;overflow-y:auto;">
  <table class="tbl">
    <thead><tr>{thead}</tr></thead>
    <tbody>{tbody}</tbody>
  </table>
</div>
""")

    @render.download(filename="breast_cancer_predictions.csv")
    def batch_download():
        df, msg = _batch_df()
        if df is None:
            yield ""
        else:
            yield df.to_csv(index=False)

    # ── Decision threshold ────────────────────────────────────────────────────

    @render.plot
    def hist_img():
        thr  = float(input.threshold())
        p_malignant = 1.0 - PROB_TRAIN
        fig, ax = plt.subplots(figsize=(7.0, 3.5))
        _style_axis(ax)
        bins = np.linspace(0, 1, 26)
        ax.hist(p_malignant[y_tr == 0], bins=bins, color=CLR_MAL, alpha=0.65,
                label="Malignant", edgecolor="none")
        ax.hist(p_malignant[y_tr == 1], bins=bins, color=CLR_BEN, alpha=0.65,
                label="Benign",    edgecolor="none")
        ax.axvline(thr, color=_NAVY, lw=1.0, ls="--",
                   label=f"Threshold = {thr:.2f}", zorder=3)
        ax.set_xlabel("P(Malignant)")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8.0, frameon=False, loc="upper center", ncol=3)
        fig.tight_layout(pad=0.8)
        return fig

    @render.ui
    def cm_display():
        thr  = float(input.threshold())
        metrics = _threshold_metrics(y_tr, PROB_TRAIN, thr)
        tn = metrics["tn"]
        fp = metrics["fp"]
        fn = metrics["fn"]
        tp = metrics["tp"]
        return ui.HTML(f"""
<div style="margin:8px 0 12px;">
  <p style="font-size:.70rem;color:{_MUTED};margin-bottom:6px;">
    Threshold = {thr:.2f} · Positive = Malignant risk · Training set (n={N_TRAIN})
  </p>
  <div class="cm-wrap">
    <div class="cm-corner"></div>
    <div class="cm-col-hdr">Predicted<br>Malignant</div>
    <div class="cm-col-hdr">Predicted<br>Benign</div>
    <div class="cm-row-hdr">Actual<br>Malignant</div>
    <div class="cm-cell" style="background:#EEF3FA;">
      <span class="cm-n" style="color:{CLR_MAL};">{tp}</span>
      <span class="cm-desc">True Positive<br>(detected malignant)</span>
    </div>
    <div class="cm-cell" style="background:#F2F6FB;">
      <span class="cm-n" style="color:{CLR_REF};">{fn}</span>
      <span class="cm-desc">False Negative<br><b>missed cancer!</b></span>
    </div>
    <div class="cm-row-hdr">Actual<br>Benign</div>
    <div class="cm-cell" style="background:#EEF3FA;">
      <span class="cm-n" style="color:{CLR_1SE};">{fp}</span>
      <span class="cm-desc">False Positive<br>(benign flagged malignant)</span>
    </div>
    <div class="cm-cell" style="background:#EAF8FA;">
      <span class="cm-n" style="color:{CLR_BEN};">{tn}</span>
      <span class="cm-desc">True Negative<br>(correct benign)</span>
    </div>
  </div>
</div>
""")

    @render.ui
    def metrics_table():
        thr  = float(input.threshold())
        metrics = _threshold_metrics(y_tr, PROB_TRAIN, thr)
        rows_html = "".join(
            f"<tr><td>{lbl}</td><td class='num'>{val:.4f}</td></tr>"
            for lbl, val in [
                ("Accuracy",                           metrics["acc"]),
                ("Sensitivity — Malignant detection",  metrics["sens"]),
                ("Specificity — Benign exclusion",     metrics["spec"]),
                ("PPV (Malignant precision)",          metrics["ppv"]),
                ("NPV",                                metrics["npv"]),
                ("F1 (Malignant)",                     metrics["f1"]),
            ]
        )
        return ui.HTML(f"""
<div style="overflow-x:auto;">
  <table class="tbl">
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
<p style="font-size:.72rem;color:{_MUTED};margin-top:6px;">
  Evaluated on training set (n={N_TRAIN}). This threshold view treats malignancy as the positive class.
</p>
""")

    # ── Results figures (separate panels) ────────────────────────────────────

    @render.image
    def fig_feat_sel():
        return {"src": _bytes_to_tmp(_FEAT_SEL_BYTES, "feat_sel"),
                "alt": "Feature Selection"}

    @render.image
    def fig_perf():
        return {"src": _bytes_to_tmp(_PERF_BYTES, "perf"),
                "alt": "Model Performance"}

    @render.image
    def fig_linearity():
        return {"src": _bytes_to_tmp(_LIN_BYTES, "linearity"),
                "alt": "Linearity Assessment"}

    @render.image
    def fig_vif():
        return {"src": _bytes_to_tmp(_VIF_BYTES, "vif"),
                "alt": "Collinearity VIF"}

    # ── Coefficient table (kept for Methods panel reference) ──────────────────

    @render.ui
    def coef_table():
        rows = ""
        order = np.argsort(np.abs(LR_COEF))[::-1]
        for i in order:
            f   = SEL_COLS[i]
            lc  = LASSO_COEF[f]
            rc  = LR_COEF[i]
            rows += (
                f"<tr><td>{f}</td>"
                f"<td class='num' style='color:{CLR_1SE};'>{lc:+.4f}</td>"
                f"<td class='num' style='color:{CLR_BEN};'>{rc:+.4f}</td></tr>"
            )
        return ui.HTML(f"""
<div class="card">
  <div class="card-header">LASSO (Stage 1) vs. Plain LR (Stage 2) Coefficients</div>
  <div class="card-body">
    <div style="overflow-x:auto;">
      <table class="tbl">
        <thead><tr>
          <th>Feature</th>
          <th style="color:{CLR_1SE};">LASSO Coefficient (λ₁ₛₑ)</th>
          <th style="color:{CLR_BEN};">Plain LR Coefficient</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    <p style="font-size:.74rem;color:{_MUTED};margin-top:8px;">
      LASSO coefficients are shrunk toward zero (biased). Plain LR on the same
      {N_SEL} features provides unbiased estimates. A negative coefficient indicates that
      higher feature values are associated with malignancy.
    </p>
  </div>
</div>
""")

    # ── Performance metrics table (Performance & Calibration tab) ─────────────

    @render.ui
    def perf_metrics_table():
        hl_verdict = "good calibration" if HL_P >= 0.05 else "poor calibration"
        rows_html = "".join(
            f"<tr><td>{lbl}</td><td class='num'>{val:.4f}</td></tr>"
            for lbl, val in [
                ("AUC (Train)",                   AUC_TRAIN),
                ("AUC (Test)",                    AUC_TEST),
                ("Brier Score (Train)",            BRIER_TRAIN),
                ("Brier Score (Test)",             BRIER_TEST),
                ("Null Brier (prevalence model)",  NULL_BRIER),
                ("Accuracy (threshold = 0.50)",    TEST_METRICS_05["acc"]),
                ("Sensitivity — Malignant detection", TEST_METRICS_05["sens"]),
                ("Specificity — Benign exclusion", TEST_METRICS_05["spec"]),
                ("PPV (Malignant precision)",      TEST_METRICS_05["ppv"]),
                ("NPV",                            TEST_METRICS_05["npv"]),
                ("F1 (Malignant)",                 TEST_METRICS_05["f1"]),
            ]
        )
        return ui.HTML(f"""
<div class="card" style="margin-top:10px;">
  <div class="card-header">
    Model Performance Summary — Test Set (threshold = 0.50)
  </div>
  <div class="card-body">
    <table class="tbl">
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    <p style="font-size:.74rem;color:{_MUTED};margin-top:8px;">
      Brier score: mean squared error of probability predictions
      (0 = perfect, 1 = worst; null model = {NULL_BRIER:.4f}).
      Hosmer-Lemeshow χ²={HL_CHI2:.2f}, p={HL_P:.3f}, df={HL_DF}
      → {hl_verdict} (p≥0.05 indicates adequate calibration).
    </p>
  </div>
</div>
""")

    # ── Methods panel ─────────────────────────────────────────────────────────

    @render.ui
    def methods_panel():
        lin_rows = "".join(
            f"<tr><td><code>{f}</code></td>"
            f"<td class='num'>{TRAIN_MEDIANS[f]:.4g}</td>"
            f"<td class='num'>{LRT[f]['chi2']:.2f}</td>"
            f"<td class='num'>{LRT[f]['p']:.3f}</td>"
            f"<td>{'Linear ✓' if LRT[f]['linear'] else 'Non-linear ✗'}</td></tr>"
            for f in SEL_COLS
        )
        vif_rows = "".join(
            f"<tr><td><code>{f}</code></td>"
            f"<td class='num'>{VIF[f]:.3f}</td>"
            f"<td>{'<5 Acceptable' if VIF[f] < 5 else ('5–10 Moderate' if VIF[f] < 10 else '>10 Severe')}</td></tr>"
            for f in SEL_COLS
        )
        hl_verdict = "good calibration" if HL_P >= 0.05 else "poor calibration"
        return ui.HTML(f"""
<div class="methods">

  <div class="card" style="margin-bottom:14px;">
    <div class="card-header">Linearity Assessment (LRT, α = 0.10)</div>
    <div class="card-body" style="padding:14px!important;">
      <table class="tbl">
        <thead><tr>
          <th>Feature</th><th>Train Median</th>
          <th>LRT χ²</th><th>p-value</th><th>Decision</th>
        </tr></thead>
        <tbody>{lin_rows}</tbody>
      </table>
      <p style="font-size:.74rem;color:{_MUTED};margin-top:8px;">
        LRT: linear logistic GLM vs cubic spline (sklearn SplineTransformer,
        n_knots=2, degree=3, df_extra=3). Reject linearity at α=0.10.
      </p>
    </div>
  </div>

  <div class="card" style="margin-bottom:14px;">
    <div class="card-header">Collinearity Assessment (VIF)</div>
    <div class="card-body" style="padding:14px!important;">
      <table class="tbl">
        <thead><tr>
          <th>Feature</th><th>VIF</th><th>Verdict</th>
        </tr></thead>
        <tbody>{vif_rows}</tbody>
      </table>
      <p style="font-size:.74rem;color:{_MUTED};margin-top:8px;">
        VIF computed on RobustScaler-transformed training features.
        VIF = 1/(1−R²) from regressing each feature on the others.
        VIF &lt; 5: acceptable · 5–10: moderate · &gt;10: severe multicollinearity.
      </p>
    </div>
  </div>

  <div class="card" style="margin-bottom:14px;">
    <div class="card-header">Model Performance — Test Set (threshold = 0.50)</div>
    <div class="card-body" style="padding:14px!important;">
      <table class="tbl">
        <thead><tr><th>Metric</th><th>Value</th></tr></thead>
        <tbody>
          <tr><td>AUC (Train / Test)</td>
              <td class='num'>{AUC_TRAIN:.4f} / {AUC_TEST:.4f}</td></tr>
          <tr><td>Brier Score (Train / Test)</td>
              <td class='num'>{BRIER_TRAIN:.4f} / {BRIER_TEST:.4f}</td></tr>
          <tr><td>Null Brier (prevalence model)</td>
              <td class='num'>{NULL_BRIER:.4f}</td></tr>
          <tr><td>Hosmer-Lemeshow χ² (df={HL_DF})</td>
              <td class='num'>{HL_CHI2:.2f}  p={HL_P:.3f}</td></tr>
          <tr><td>Accuracy</td><td class='num'>{TEST_METRICS_05["acc"]:.4f}</td></tr>
          <tr><td>Sensitivity — Malignant detection</td><td class='num'>{TEST_METRICS_05["sens"]:.4f}</td></tr>
          <tr><td>Specificity — Benign exclusion</td><td class='num'>{TEST_METRICS_05["spec"]:.4f}</td></tr>
          <tr><td>PPV (Malignant precision)</td><td class='num'>{TEST_METRICS_05["ppv"]:.4f}</td></tr>
          <tr><td>NPV</td><td class='num'>{TEST_METRICS_05["npv"]:.4f}</td></tr>
          <tr><td>F1 (Malignant)</td><td class='num'>{TEST_METRICS_05["f1"]:.4f}</td></tr>
        </tbody>
      </table>
      <p style="font-size:.74rem;color:{_MUTED};margin-top:8px;">
        Calibration: HL test {hl_verdict} (χ²={HL_CHI2:.2f}, p={HL_P:.3f}).
        Brier skill = 1 − Brier/NullBrier =
        {1 - BRIER_TEST/NULL_BRIER:.3f}.
      </p>
    </div>
  </div>

  <div class="card">
    <div class="card-header">Pipeline Description</div>
    <div class="card-body" style="padding:14px!important;">
      <h4>Dataset</h4>
      <p>Wisconsin Breast Cancer Dataset (sklearn, n = {N_TOTAL}). Features: 30
      nuclear morphology measurements derived from digitised fine-needle aspirate
      images. Outcome: malignant (n = {n_malignant}) vs benign
      (n = {n_benign}). 80/20 stratified train/test split
      (random seed = 42).</p>

      <h4>Scaling</h4>
      <p>Features were scaled with RobustScaler (subtract median, divide by IQR),
      fitted exclusively on the training set to prevent data leakage. RobustScaler is
      preferred over StandardScaler given the right-skewed distributions typical of
      nuclear morphology measurements.</p>

      <h4>Stage 1 — LASSO Feature Selection</h4>
      <p>L1-penalised logistic regression (liblinear solver) evaluated over 60
      log-spaced values of C ∈ [10⁻⁴, 10²] using 5-fold stratified
      cross-validation (criterion: AUC). Feature selection follows the λ₁ₛₑ rule:
      the most regularised model whose mean CV AUC remains within one standard
      error of the best model. Result: {N_SEL} features selected at
      C = {C_1SE:.5f}  (λ_min: {NZ_MIN} features at C = {C_MIN:.5f}).</p>

      <h4>Stage 2 — Plain Logistic Regression</h4>
      <p>Unpenalised logistic regression (lbfgs solver) re-estimated on the
      {N_SEL} LASSO-selected features. Refitting with plain LR removes L1
      shrinkage bias and yields interpretable, unbiased coefficient estimates.
      All features are entered on the RobustScaler-transformed scale without
      further transformation.</p>

      <h4>Model Diagnostics</h4>
      <p>Linearity was assessed via a likelihood-ratio test (LRT) comparing a
      linear logistic GLM against a cubic spline alternative (df_extra = 3).
      All {N_SEL} selected features pass the linearity test at α = 0.10, supporting the
      log-odds linearity assumption. Collinearity was quantified by the variance
      inflation factor (VIF = 1 / (1 − R²)), computed on the scaled training set.</p>

      <h4>Calibration</h4>
      <p>The Hosmer-Lemeshow test (10 quantile-based risk groups, df = {HL_DF})
      yields χ² = {HL_CHI2:.2f}, p = {HL_P:.3f} → {hl_verdict}. The Brier score
      on the test set is {BRIER_TEST:.4f}, compared with the null (prevalence-based)
      model score of {NULL_BRIER:.4f}, giving a Brier skill score of
      {1 - BRIER_TEST/NULL_BRIER:.3f}.</p>
    </div>
  </div>

</div>
""")


# ── 6. Utility — bytes → temp file for render.image ──────────────────────────
import tempfile, os

_TMP_FILES: dict[str, str] = {}


def _bytes_to_tmp(data: bytes, key: str) -> str:
    if key not in _TMP_FILES:
        fd, path = tempfile.mkstemp(suffix=".png", prefix=f"bc_{key}_")
        os.close(fd)
        with open(path, "wb") as fh:
            fh.write(data)
        _TMP_FILES[key] = path
    return _TMP_FILES[key]


app = App(app_ui, server)
