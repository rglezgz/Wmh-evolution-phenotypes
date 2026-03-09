# -*- coding: utf-8 -*-
"""
WMH lesion patterns
GLM (frequency) + OLS subject-level

FDR updates:
- GLM: FDR (BH) within each outcome_frequency (L1/2/3)
- OLS: FDR (BH) within each outcome (∆GM, ∆CSF)
- Stars in plots use p_fdr (NOT raw p)
- Exports include both p and p_fdr
- Vectorized PDF/SVG outputs

Scale updates:
- GLM forest: the 3 panels share the SAME x-axis scale (global CI-based xlim)
- OLS forest: ∆GM and ∆CSF panels share the SAME x-axis scale (global CI-based xlim)

PLOT SIZE updates (compact; consistent width across figures):
- OLS forest: figsize=(7, 2.9)
- GLM forest (3 panels): figsize=(7, 2.9)
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
from statsmodels.stats.multitest import multipletests

# --- Vector export settings ---
mpl.rcParams["pdf.fonttype"] = 42        # TrueType fonts (selectable text)
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"    # keep text as text in SVG
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.pad_inches"] = 0.02


# =========================
# 1) CONFIG
# =========================
INPUT_XLSX = r"C:\Users\rglez\Documents\Ra\Papers\WMH long caracterization\clustering\WMH_clusters_metrics-all_with_clinic_final_CLUSTERING_GLM.xlsx"
SHEET_NAME = "data_with_clusters"

OUT_DIR_BASE = r"C:\Users\rglez\Documents\Ra\Papers\WMH long caracterization\clustering"
RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(OUT_DIR_BASE, f"subject_level_freq_{RUN_TAG}")
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_XLSX = os.path.join(
    OUT_DIR,
    "subject_level_frequency_multivariable_FDR_fullparams_NO_nTotalPredictor.xlsx"
)

# ===== Continuous "main effects" (z-scored + shown in forest) =====
EFFECTS_CONT = [
    "Age",
    "ΔPP", "ΔHR", "ΔWGTKG",
    "Education",
    "APOE_E4_count"
]

# Covariables for GLM
COVARS_CATEG_GLM = ["Sex", "Gpo"]
# ✅ n_total REMOVED from predictors
COVARS_CONT_GLM  = ["log_size_total", "TIV"]

# Plot GLM
PLOT_FILENAME_GLM = "forest_GLM_3clusters_logOR_FDR_NO_nTotalPredictor.png"
TERM_ORDER_PLOT = [
    "APOE_E4_count", "Sex", "Age", "Education",
    "ΔPP", "ΔHR", "ΔWGTKG"
]
TERM_LABELS = {
    "APOE_E4_count": "APOE ε4",
    "Sex": "Sex (Male)",
    "Age": "Age",
    "Education": "Education",
    "ΔPP": "ΔPP",
    "ΔHR": "ΔHR",
    "ΔWGTKG": "ΔWeight"
}

# Plot OLS
PLOT_FILENAME_OLS = "forest_OLS_lesions_stdBeta_FDR.png"

# Compact figure size (same width for both figures)
FIG_W = 7.0
FIG_H = 2.9


# =========================
# 2) HELPERS
# =========================
def apoe_e4_count(geno):
    if pd.isna(geno):
        return np.nan
    s = str(geno).upper().replace(" ", "")
    return float(s.count("E4"))

def assert_unique_col(df, col):
    if list(df.columns).count(col) != 1:
        raise ValueError(f"Problema con columna: {col} (falta o duplicada)")

def safe_to_numeric(df, col):
    assert_unique_col(df, col)
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def _star_from_p(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""

def match_tick_style(ax):
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11, pad=4)
    ax.xaxis.label.set_size(11)
    ax.yaxis.label.set_size(11)
    ax.title.set_size(12)

def _bucket_term(term):
    # For GLM forest: bucketize Sex_* to "Sex"; strip _z
    if term.startswith("Sex_"):
        return "Sex"
    if term.endswith("_z"):
        return term.replace("_z", "")
    return term

def add_fdr_by_group(df, p_col="p", group_cols=(), method="fdr_bh", out_col="p_fdr"):
    """
    Add FDR-corrected p-values within groups.
    Example:
      - GLM: group_cols=("outcome_frequency",)
      - OLS: group_cols=("outcome",)
    """
    df = df.copy()
    df[out_col] = np.nan

    if group_cols is None or len(group_cols) == 0:
        pvals = df[p_col].to_numpy(dtype=float)
        ok = np.isfinite(pvals)
        if ok.sum() > 0:
            _, p_corr, _, _ = multipletests(pvals[ok], method=method)
            df.loc[df.index[ok], out_col] = p_corr
        return df

    for _, sub_idx in df.groupby(list(group_cols)).groups.items():
        pvals = df.loc[sub_idx, p_col].to_numpy(dtype=float)
        ok = np.isfinite(pvals)
        if ok.sum() > 0:
            _, p_corr, _, _ = multipletests(pvals[ok], method=method)
            df.loc[sub_idx[ok], out_col] = p_corr

    return df


# =========================
# 3) GLM functions
# =========================
def fit_binomial_freq_glm(df_subj, success_col, effect_terms_z):
    """
    Binomial GLM on frequency y = n_k / n_total with freq_weights = n_total.
    ✅ n_total is used only as denominator + weights, NOT as a predictor.
    """
    y = (df_subj[success_col] / df_subj["n_total"]).astype(float).to_numpy()

    X = df_subj.copy()
    X = pd.get_dummies(X, columns=COVARS_CATEG_GLM, drop_first=True)

    model_cols = (
        effect_terms_z +
        [f"{c}_z" for c in COVARS_CONT_GLM] +
        [c for c in X.columns if c.startswith("Sex_") or c.startswith("Gpo_")]
    )

    X = sm.add_constant(X[model_cols], has_constant="add").astype(float)

    # ✅ weights = n_total
    w = df_subj["n_total"].astype(float).to_numpy()
    res = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=w).fit(cov_type="HC1")
    return res, X.columns

def make_glm_table(res, cols, outcome_label):
    rows = []
    for term in cols:
        if term == "const":
            continue
        beta = float(res.params[term])
        se   = float(res.bse[term])
        p    = float(res.pvalues[term])
        OR   = float(np.exp(beta))
        lo   = float(np.exp(beta - 1.96 * se))
        hi   = float(np.exp(beta + 1.96 * se))
        rows.append({
            "outcome_frequency": outcome_label,
            "term": term,
            "beta": beta,
            "SE": se,
            "OR": OR,
            "CI95_low": lo,
            "CI95_high": hi,
            "p": p
        })
    return pd.DataFrame(rows)


# =========================
# 4) OLS functions
# =========================
def fit_ols_standardizedY(df_subj, ycol, cont_cols, dummy_cols=("Sex", "Gpo")):
    """
    OLS with standardized Y.
    Continuous predictors are z-scored.
    Sex/Gpo are dummies (drop_first=True) -> baseline is first category (CN if forced).
    """
    X = df_subj.copy()
    X = pd.get_dummies(X, columns=list(dummy_cols), drop_first=True)
    dum_cols = [c for c in X.columns if c.startswith("Sex_") or c.startswith("Gpo_")]

    sc = StandardScaler()
    X_cont_z = pd.DataFrame(
        sc.fit_transform(X[cont_cols].astype(float)),
        columns=[f"{c}_z" for c in cont_cols],
        index=X.index
    )

    X_dum = X[dum_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    Xmat = pd.concat([X_cont_z, X_dum], axis=1)
    Xmat = sm.add_constant(Xmat, has_constant="add")
    Xmat = Xmat.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)

    y_raw = pd.to_numeric(df_subj[ycol], errors="coerce").to_numpy(dtype=float)
    mu = np.nanmean(y_raw)
    sd = np.nanstd(y_raw, ddof=0)
    y_z = (y_raw - mu) / (sd)

    res = sm.OLS(y_z, Xmat).fit(cov_type="HC1")
    return res, Xmat.columns

def make_ols_table(res, cols, outcome):
    rows = []
    for term in cols:
        if term == "const":
            continue
        beta = float(res.params[term])
        se   = float(res.bse[term])
        rows.append({
            "outcome": outcome,
            "term": term,
            "beta_std": beta,
            "SE": se,
            "t": float(res.tvalues[term]),
            "p": float(res.pvalues[term]),
            "CI95_low": float(beta - 1.96 * se),
            "CI95_high": float(beta + 1.96 * se),
            "N": int(res.nobs),
            "R2": float(res.rsquared)
        })
    return pd.DataFrame(rows)

def plot_ols_lesion_forest(ols_df_in, out_dir, filename, outcomes=("∆GM", "∆CSF"), show=True):
    """
    OLS forest: stars by FDR p-values (p_fdr).
    Scale: all outcomes share the SAME x-axis scale (global CI-based xlim).

    FIX: desplaza visualmente L1 (arriba) hacia abajo y L3 (abajo) hacia arriba
         moviendo SOLO los puntos (y_pts), manteniendo ticks fijos (y_ticks)
         y fijando ylim para evitar autoscaling.
    """
    lesion_terms = ["n0_z", "n1_z", "n2_z"]  # orden en el plot (de arriba a abajo después del invert_yaxis)

    labels = {
        "n2_z": "L3 (Count)",
        "n1_z": "L2 (Count)",
        "n0_z": "L0 (Count)",
    }

    dfp = ols_df_in[ols_df_in["term"].isin(lesion_terms)].copy()
    if dfp.empty:
        print("⚠ No se encontraron términos de lesión *_z en OLS.")
        return None, None, None

    # --- GLOBAL XLIM for OLS (same scale for ∆GM and ∆CSF) ---
    dfp_xlim = dfp[dfp["outcome"].isin(outcomes)].copy()
    lo_all = dfp_xlim["CI95_low"].to_numpy(dtype=float)
    hi_all = dfp_xlim["CI95_high"].to_numpy(dtype=float)
    ok_all = np.isfinite(lo_all) & np.isfinite(hi_all)

    if np.any(ok_all):
        x_min = float(np.nanmin(lo_all[ok_all]))
        x_max = float(np.nanmax(hi_all[ok_all]))
        pad_x = 0.20 * (x_max - x_min + 1e-12)
        ols_xlim = (x_min - pad_x, x_max + pad_x)
    else:
        ols_xlim = None

    # ✅ compact size (same width as GLM figure)
    fig, axes = plt.subplots(1, len(outcomes), figsize=(FIG_W, FIG_H * 0.6), sharey=True)
    fig.subplots_adjust(top=0.90, bottom=0.22, wspace=0.30)
    if len(outcomes) == 1:
        axes = [axes]

    # --- Y positioning FIX ---
    y_ticks = np.arange(len(lesion_terms), dtype=float)   # 0,1,2 (ticks fijos)
    y_pts = y_ticks.copy()                                # posiciones reales de los puntos

    delta = 0.25  # prueba 0.20–0.40 para que sea evidente
    # Querías: L1 (arriba) baje un poco y L3 (abajo) suba un poco
    # Con invert_yaxis(): aumentar y -> baja visualmente; disminuir y -> sube visualmente.
    # OJO: tus etiquetas dicen L1=n0_z y L3=n2_z, pero lesion_terms está [n2_z, n1_z, n0_z]
    # Por lo tanto:
    #   índice 0 -> n2_z (L3)
    #   índice 2 -> n0_z (L1)
    y_pts[2] += delta   # L1 (n0_z) baja un poco
    y_pts[0] -= delta   # L3 (n2_z) sube un poco

    pad_y = 0.70  # evita que Matplotlib reescale y “se vea igual”

    for ax, out in zip(axes, outcomes):
        sub = dfp[dfp["outcome"] == out].set_index("term").reindex(lesion_terms)

        beta = sub["beta_std"].to_numpy(dtype=float)
        lo   = sub["CI95_low"].to_numpy(dtype=float)
        hi   = sub["CI95_high"].to_numpy(dtype=float)
        pval = sub["p_fdr"].to_numpy(dtype=float)  # ✅ FDR used for stars

        ok = np.isfinite(beta) & np.isfinite(lo) & np.isfinite(hi)

        if np.any(ok):
            xerr = np.vstack([beta[ok] - lo[ok], hi[ok] - beta[ok]])
            # ✅ usar y_pts (puntos movidos), NO y_ticks
            ax.errorbar(beta[ok], y_pts[ok], xerr=xerr, fmt="o", capsize=3)

        ax.axvline(0.0, linestyle="--", linewidth=1.2, color="black")
        ax.set_title(out)
        ax.set_xlabel("Standardized beta (95% CI)")

        # ✅ ticks fijos (no movidos)
        ax.set_yticks(y_ticks)

        # ✅ fija límites en y para evitar autoscaling (si no, “queda igual”)
        ax.set_ylim(-pad_y, (len(lesion_terms) - 1) + pad_y)

        if ols_xlim is not None:
            ax.set_xlim(ols_xlim)

        if ax is axes[0]:
            ax.set_yticklabels([labels[t] for t in lesion_terms])
        else:
            ax.tick_params(axis="y", labelleft=False, left=False)

        # ✅ estrellas usando y_pts también
        for i in range(len(lesion_terms)):
            if np.isfinite(beta[i]) and np.isfinite(pval[i]):
                s = _star_from_p(float(pval[i]))
                if s:
                    ax.text(beta[i], y_pts[i] - 0.15, s, ha="center", va="bottom", fontsize=12)

        ax.invert_yaxis()
        match_tick_style(ax)

    png_path = os.path.join(out_dir, filename)
    svg_path = os.path.join(out_dir, filename.replace(".png", ".svg"))
    pdf_path = os.path.join(out_dir, filename.replace(".png", ".pdf"))

    fig.savefig(png_path, dpi=300, bbox_inches=None)
    fig.savefig(svg_path, bbox_inches=None)
    fig.savefig(pdf_path, format="pdf", bbox_inches=None)

    print("📊 OLS Plot PNG:", png_path)
    print("📊 OLS Plot SVG:", svg_path)
    print("📊 OLS Plot PDF:", pdf_path)

    if show:
        plt.show()
    plt.close(fig)
    return png_path, svg_path, pdf_path



# =========================
# 5) LOAD + clean duplicates
# =========================
df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME)
dups = df.columns[df.columns.duplicated()].tolist()
if dups:
    print("⚠ Columnas duplicadas:", dups)
    df = df.loc[:, ~df.columns.duplicated()].copy()

# Cluster
assert_unique_col(df, "cluster_gmm_auto")
df["cluster_gmm_auto"] = pd.to_numeric(df["cluster_gmm_auto"], errors="coerce").round().astype("Int64")
print("✅ Conteo clusters:", df["cluster_gmm_auto"].value_counts(dropna=False).to_dict())


# =========================
# 6) SUBJECT-LEVEL BUILD (GLM)
# =========================
ct = df.groupby(["subject_id", "cluster_gmm_auto"]).size().unstack(fill_value=0)
for k in [0, 1, 2]:
    if k not in ct.columns:
        ct[k] = 0
ct = ct[[0, 1, 2]]
ct.columns = ["n0", "n1", "n2"]
ct["n_total"] = ct.sum(axis=1)

need_cols = ["Age", "Sex", "Gpo", "Education", "APOE_GENOTYPE", "ΔPP", "ΔHR", "ΔWGTKG", "TIV"]
for c in need_cols + ["Size", "subject_id", "cluster_gmm_auto"]:
    assert_unique_col(df, c)

subj = df.groupby("subject_id")[need_cols].first()
subj["APOE_E4_count"] = subj["APOE_GENOTYPE"].apply(apoe_e4_count)

size_sum = df.groupby("subject_id")["Size"].sum().rename("size_total")
subj = subj.join(ct).join(size_sum).reset_index()
subj["log_size_total"] = np.log1p(subj["size_total"])

# numeric conversions
numeric_cols = [
    "Age", "Education", "APOE_E4_count",
    "ΔPP", "ΔHR", "ΔWGTKG",
    "log_size_total", 
    "n0", "n1", "n2", "TIV"
]
for c in numeric_cols:
    subj = safe_to_numeric(subj, c)

# categorical cleaning + baseline CN
subj["Gpo"] = subj["Gpo"].astype(str).str.strip()
subj["Sex"] = subj["Sex"].astype(str).str.strip()
subj["Gpo"] = pd.Categorical(subj["Gpo"], categories=["CN", "MCI", "AD", "PD"], ordered=False)

# Required for GLM (note: n_total must be present for denominator/weights, but NOT as predictor)
req_glm = (
    EFFECTS_CONT +
    COVARS_CONT_GLM +
    ["Sex", "Gpo", "n0", "n1", "n2"]
)

d = subj.dropna(subset=req_glm).copy().reset_index(drop=True)

print("N sujetos GLM:", len(d))
if len(d) == 0:
    na_counts = subj[req_glm].isna().sum().sort_values(ascending=False)
    print("⚠ d vacío. NA counts:\n", na_counts)
    raise ValueError("d quedó vacío tras dropna(subset=req_glm).")

# =========================
# 7) Z-score continuous (GLM) - WITHOUT n_total
# =========================
cont_to_z = list(dict.fromkeys(EFFECTS_CONT + COVARS_CONT_GLM))
bad_cols = [c for c in cont_to_z if not pd.api.types.is_numeric_dtype(d[c])]
if bad_cols:
    raise ValueError(f"No numéricas en cont_to_z (no se pueden z-scorear): {bad_cols}")

Z = StandardScaler().fit_transform(d[cont_to_z].astype(float))
for i, c in enumerate(cont_to_z):
    d[f"{c}_z"] = Z[:, i]

effect_terms_z = [f"{c}_z" for c in EFFECTS_CONT]


# =========================
# 8) FIT 3 GLMs
# =========================
glm_tables = []
for label, succ in [("cluster0", "n0"), ("cluster1", "n1"), ("cluster2", "n2")]:
    res, cols = fit_binomial_freq_glm(d, succ, effect_terms_z)
    outlabel = f"{label}: {succ}/n_total"
    glm_tables.append(make_glm_table(res, cols, outlabel))

full_params_df = pd.concat(glm_tables, ignore_index=True)

# --- FDR correction for GLM: per cluster (outcome_frequency) ---
full_params_df = add_fdr_by_group(
    full_params_df,
    p_col="p",
    group_cols=("outcome_frequency",),
    out_col="p_fdr"
)

# main_df for plot: effects + Sex dummies (bucket to "Sex")
sex_dummy_terms = [t for t in full_params_df["term"].unique() if t.startswith("Sex_")]
plot_terms_set = set(effect_terms_z) | set(sex_dummy_terms)

main_df = full_params_df[full_params_df["term"].isin(plot_terms_set)].copy()
main_df["term_plot"] = main_df["term"].apply(_bucket_term)

# one row per term_plot/outcome (keep smallest raw p)
main_df = main_df.sort_values("p").drop_duplicates(subset=["outcome_frequency", "term_plot"], keep="first")


# =========================
# 9) EXPORT GLM (RAW p + FDR)
# =========================
with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    main_df.sort_values(["outcome_frequency", "term_plot"]).to_excel(
        writer, sheet_name="GLM_main_for_plot_FDR", index=False
    )
    full_params_df.sort_values(["outcome_frequency", "p_fdr"]).to_excel(
        writer, sheet_name="GLM_all_params_FDR", index=False
    )


# =========================
# 10) PLOT GLM (3 clusters)  (stars by p_fdr)
# =========================
def _cluster_key(s):
    try:
        return int(s.split(":")[0].replace("cluster", ""))
    except Exception:
        return 999

outcomes = sorted(main_df["outcome_frequency"].unique(), key=_cluster_key)[:3]
panel_titles = [r"L$_1$", r"L$_2$", r"L$_3$"]

# --- GLOBAL XLIM for GLM (same scale across the 3 panels) ---
glm_sub_all = main_df[main_df["outcome_frequency"].isin(outcomes)].copy()
glm_sub_all = glm_sub_all[glm_sub_all["term_plot"].isin(TERM_ORDER_PLOT)].copy()
glm_lo_all = (glm_sub_all["beta"] - 1.96 * glm_sub_all["SE"]).to_numpy(dtype=float)
glm_hi_all = (glm_sub_all["beta"] + 1.96 * glm_sub_all["SE"]).to_numpy(dtype=float)
ok_all = np.isfinite(glm_lo_all) & np.isfinite(glm_hi_all)
if np.any(ok_all):
    x_min = float(np.nanmin(glm_lo_all[ok_all]))
    x_max = float(np.nanmax(glm_hi_all[ok_all]))
    pad = 0.08 * (x_max - x_min + 1e-12)
    glm_xlim = (x_min - pad, x_max + pad)
else:
    glm_xlim = None

# ✅ compact size (same width/height as OLS figure)
fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H), sharey=True)
# A little spacing so 3 panels don't feel cramped in compact width
fig.subplots_adjust(wspace=0.35)

y = np.arange(len(TERM_ORDER_PLOT))

for j, (ax, out) in enumerate(zip(axes, outcomes)):
    sub = main_df[main_df["outcome_frequency"] == out].copy()
    sub = sub.set_index("term_plot").reindex(TERM_ORDER_PLOT)

    beta = sub["beta"].to_numpy(dtype=float)
    lo   = (sub["beta"] - 1.96 * sub["SE"]).to_numpy(dtype=float)
    hi   = (sub["beta"] + 1.96 * sub["SE"]).to_numpy(dtype=float)
    pval = sub["p_fdr"].to_numpy(dtype=float)   # ✅ FDR used for stars

    ok = np.isfinite(beta) & np.isfinite(lo) & np.isfinite(hi)
    if np.any(ok):
        xerr = np.vstack([beta[ok] - lo[ok], hi[ok] - beta[ok]])
        ax.errorbar(beta[ok], y[ok], xerr=xerr, fmt="o", capsize=3)

    ax.axvline(0.0, linestyle="--", linewidth=1.2, color="black")
    ax.set_title(panel_titles[j])
    ax.set_xlabel("log odds ratio")
    ax.set_yticks(y)

    if glm_xlim is not None:
        ax.set_xlim(glm_xlim)

    if j == 0:
        ax.set_yticklabels([TERM_LABELS.get(t, t) for t in TERM_ORDER_PLOT])
    else:
        ax.tick_params(axis="y", labelleft=False, left=False)

    for i in range(len(TERM_ORDER_PLOT)):
        if np.isfinite(beta[i]) and np.isfinite(pval[i]):
            s = _star_from_p(float(pval[i]))
            if s:
                ax.text(beta[i], y[i] - 0.15, s, ha="center", va="bottom", fontsize=12)

    ax.invert_yaxis()
    match_tick_style(ax)

fig.tight_layout()

png_glm = os.path.join(OUT_DIR, PLOT_FILENAME_GLM)
svg_glm = os.path.join(OUT_DIR, PLOT_FILENAME_GLM.replace(".png", ".svg"))
pdf_glm = os.path.join(OUT_DIR, PLOT_FILENAME_GLM.replace(".png", ".pdf"))
fig.savefig(png_glm, dpi=300)
fig.savefig(svg_glm)
fig.savefig(pdf_glm, format="pdf")
plt.show()
plt.close(fig)

print("📊 GLM Plot PNG:", png_glm)
print("📊 GLM Plot SVG:", svg_glm)
print("📊 GLM Plot PDF:", pdf_glm)


# =========================
# 11) SUBJECT-LEVEL OLS BUILD
# =========================
ct_ols = df.groupby(["subject_id", "cluster_gmm_auto"]).size().unstack(fill_value=0)
for k in [0, 1, 2]:
    if k not in ct_ols.columns:
        ct_ols[k] = 0
ct_ols = ct_ols[[0, 1, 2]]
ct_ols.columns = ["n0", "n1", "n2"]

ss_ols = df.groupby(["subject_id", "cluster_gmm_auto"])["Size"].sum().unstack(fill_value=0)
for k in [0, 1, 2]:
    if k not in ss_ols.columns:
        ss_ols[k] = 0.0
ss_ols = ss_ols[[0, 1, 2]]
ss_ols.columns = ["size0", "size1", "size2"]

# --- Total size por sujeto ---
ss_ols["size_total"] = ss_ols[["size0", "size1", "size2"]].sum(axis=1)



need_base = ["Age", "Sex", "Gpo", "Education", "TIV", "∆GM", "∆CSF"]
for c in need_base + ["subject_id"]:
    assert_unique_col(df, c)

subj_base = df.groupby("subject_id")[need_base].first()
subj_ols = subj_base.join(ct_ols).join(ss_ols).reset_index()


subj_ols["log_size_total"] = np.log1p(subj_ols["size_total"])

num_cols = [
    "Age", "Education", "TIV", "∆GM", "∆CSF",
    "n0", "n1", "n2", 
    "log_size_total" 
]
for c in num_cols:
    subj_ols = safe_to_numeric(subj_ols, c)

subj_ols["Gpo"] = subj_ols["Gpo"].astype(str).str.strip()
subj_ols["Sex"] = subj_ols["Sex"].astype(str).str.strip()
subj_ols["Gpo"] = pd.Categorical(subj_ols["Gpo"], categories=["CN", "MCI", "AD", "PD"], ordered=False)

req_ols = [
    "Sex", "Gpo", "Age", "Education", "TIV",
    "n0", "n1", "n2", 
    "log_size_total", 
    "∆GM", "∆CSF", 
]
d_ols = subj_ols.dropna(subset=req_ols).copy().reset_index(drop=True)

print("N sujetos OLS:", len(d_ols))
if len(d_ols) == 0:
    na_counts = subj_ols[req_ols].isna().sum().sort_values(ascending=False)
    print("⚠ d_ols vacío. NA counts:\n", na_counts)
    raise ValueError("d_ols quedó vacío tras dropna(subset=req_ols).")

# =========================
# 12) FIT OLS + FDR + EXPORT + PLOT
# =========================
CONT_COLS_OLS = ["Age", "Education", "TIV", "n0", "n1", "n2", "log_size_total", ]
ols_outcomes = ["∆GM", "∆CSF"]

ols_tables = []
for ycol in ols_outcomes:
    res, cols = fit_ols_standardizedY(d_ols, ycol, cont_cols=CONT_COLS_OLS)
    ols_tables.append(make_ols_table(res, cols, ycol))
ols_df = pd.concat(ols_tables, ignore_index=True)

# --- FDR correction for OLS: per outcome (∆GM separate from ∆CSF) ---
ols_df = add_fdr_by_group(
    ols_df,
    p_col="p",
    group_cols=("outcome",),
    out_col="p_fdr"
)

with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    d_ols.to_excel(writer, sheet_name="OLS_subject_data", index=False)
    ols_df.sort_values(["outcome", "p_fdr"]).to_excel(writer, sheet_name="OLS_params_FDR", index=False)

png_ols, svg_ols, pdf_ols = plot_ols_lesion_forest(
    ols_df, OUT_DIR, filename=PLOT_FILENAME_OLS,
    outcomes=("∆GM", "∆CSF"),
    show=True
)

print("✅ OK ->", OUTPUT_XLSX)
print("📁 Outputs en:", OUT_DIR)
print("📊 GLM Plot PNG:", png_glm)
print("📊 GLM Plot SVG:", svg_glm)
print("📊 GLM Plot PDF:", pdf_glm)
print("📊 OLS Plot PNG:", png_ols)
print("📊 OLS Plot SVG:", svg_ols)
print("📊 OLS Plot PDF:", pdf_ols)
print("N sujetos usados GLM:", len(d))
print("N sujetos usados OLS:", len(d_ols))
