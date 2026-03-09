# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 14:43:33 2026

@author: rglez
"""

# -*- coding: utf-8 -*-
"""
Subject-level frequency models (multivariable) + FDR + full params export
Education como EFECTO (no covariable)
Robusto a columnas duplicadas / tipos raros al leer Excel
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

# =========================
# 1) CONFIG
# =========================
INPUT_XLSX = r"C:\Users\rglez\Documents\Ra\Papers\WMH long caracterization\clustering\WMH_clusters_metrics-all_with_clinic_vars_CLUSTERING_GLM.xlsx"
SHEET_NAME = "data_with_clusters"
OUTPUT_XLSX = r"C:\Users\rglez\Documents\Ra\Papers\WMH long caracterization\clustering\subject_level_frequency_multivariable_EduAsEffect_FDR_fullparams.xlsx"

# Efectos TODOS juntos (incluye Education)
EFFECTS = ["ΔPP", "ΔWGTKG", "ΔMOCA", "Education"]  # + APOE_E4_count se crea abajo

# Covariables fijas (NO incluyen Education)
COVARS_CATEG = ["Sex", "Gpo"]
COVARS_CONT  = ["Age", "log_size_total", "n_total"]  # además de Sex/Gpo

# =========================
# 2) HELPERS
# =========================
def apoe_e4_count(geno):
    """APOE_GENOTYPE -> número de alelos E4 (0,1,2)."""
    if pd.isna(geno):
        return np.nan
    s = str(geno).upper().replace(" ", "")
    return float(s.count("E4"))

def assert_unique_col(df_, colname):
    """Asegura que colname es una columna única (no duplicada)."""
    matches = [c for c in df_.columns if c == colname]
    if len(matches) == 0:
        raise KeyError(f"Falta la columna: {colname}")
    if len(matches) > 1:
        raise ValueError(f"Columna duplicada detectada: {colname}. Renómbrala o elimina duplicados.")
    return True

def safe_to_numeric(df_, colname):
    """Convierte una columna a numérico de forma segura."""
    assert_unique_col(df_, colname)
    df_[colname] = pd.to_numeric(df_[colname], errors="coerce")
    return df_

def fit_binomial_freq_glm(df_subj, success_col, effect_terms_z):
    """
    Binomial GLM sobre proporción success/n_total con freq_weights=n_total.
    Incluye: TODOS los efectos (simultáneo) + covariables base.
    """
    y = (df_subj[success_col] / df_subj["n_total"]).astype(float).to_numpy()

    X = df_subj.copy()
    X = pd.get_dummies(X, columns=COVARS_CATEG, drop_first=True)

    # columnas del modelo
    model_cols = (
        effect_terms_z +
        [f"{c}_z" for c in COVARS_CONT] +
        [c for c in X.columns if c.startswith("Sex_") or c.startswith("Gpo_")]
    )

    X = X[model_cols].copy()
    X = sm.add_constant(X).astype(float)

    w = df_subj["n_total"].astype(float).to_numpy()
    res = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=w).fit(cov_type="HC1")
    return res, X.columns

def make_coef_table(res, cols, outcome_label):
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
# 3) LOAD + fix duplicates
# =========================
df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME)

# Si hay columnas duplicadas, pandas las mantiene. Esto ayuda a detectarlas:
dups = df.columns[df.columns.duplicated()].tolist()
if len(dups) > 0:
    print("⚠ Columnas duplicadas en el Excel:", dups)
    # opción rápida: quedarte con la primera ocurrencia
    df = df.loc[:, ~df.columns.duplicated()].copy()
    print("   -> Se eliminó duplicación quedándose con la primera ocurrencia.")

# =========================
# 4) SUBJECT-LEVEL BUILD
# =========================
# Conteos por sujeto y cluster
ct = df.groupby(["subject_id", "cluster_gmm_auto"]).size().unstack(fill_value=0)
for k in [0, 1, 2]:
    if k not in ct.columns:
        ct[k] = 0
ct = ct[[0, 1, 2]]
ct.columns = ["n0", "n1", "n2"]
ct["n_total"] = ct.sum(axis=1)

need_cols = ["Age", "Sex", "Gpo", "Education", "APOE_GENOTYPE", "ΔPP", "ΔWGTKG", "ΔMOCA"]
for c in need_cols + ["Size", "subject_id", "cluster_gmm_auto"]:
    assert_unique_col(df, c)

subj = df.groupby("subject_id")[need_cols].first()

subj["APOE_E4_count"] = subj["APOE_GENOTYPE"].apply(apoe_e4_count)

size_sum = df.groupby("subject_id")["Size"].agg(size_total="sum")
subj = subj.join(ct).join(size_sum).reset_index()

subj["log_size_total"] = np.log1p(subj["size_total"])

# =========================
# 5) numeric conversions (safe)
# =========================
for c in ["Age", "Education", "ΔPP", "ΔWGTKG", "ΔMOCA", "APOE_E4_count",
          "log_size_total", "n_total", "n0", "n1", "n2"]:
    subj = safe_to_numeric(subj, c)

# Requeridos
req = ["Age", "Sex", "Gpo", "Education", "APOE_E4_count",
       "log_size_total", "n_total", "n0", "n1", "n2", "ΔPP", "ΔWGTKG", "ΔMOCA"]
d = subj.dropna(subset=req).copy().reset_index(drop=True)

# =========================
# 6) Z-SCORE continuous (effects + covars)
# =========================
# efectos continuos (incluye Education + APOE)
cont_effects = ["ΔPP", "ΔWGTKG", "ΔMOCA", "Education", "APOE_E4_count"]
cont_covars  = COVARS_CONT  # Age, log_size_total, n_total
cont_to_z = cont_effects + cont_covars

scaler = StandardScaler()
Z = scaler.fit_transform(d[cont_to_z].astype(float))
for i, c in enumerate(cont_to_z):
    d[f"{c}_z"] = Z[:, i]

# lista final de términos de efectos en z
effect_terms_z = [f"{c}_z" for c in ["ΔPP", "ΔWGTKG", "ΔMOCA", "Education", "APOE_E4_count"]]

# =========================
# 7) Fit 3 models
# =========================
full_params_tables = []
main_effect_tables = []

effect_terms_set = set(effect_terms_z)

for label, succ in [("cluster0", "n0"), ("cluster1", "n1"), ("cluster2", "n2")]:
    res, cols = fit_binomial_freq_glm(d, succ, effect_terms_z)
    outlabel = f"{label}: {succ}/n_total"

    ft = make_coef_table(res, cols, outlabel)
    full_params_tables.append(ft)

    me = ft[ft["term"].isin(effect_terms_set)].copy()
    me["term"] = me["term"].str.replace("_z", "", regex=False)
    main_effect_tables.append(me)

full_params_df = pd.concat(full_params_tables, ignore_index=True)
main_df = pd.concat(main_effect_tables, ignore_index=True)

# =========================
# 8) FDR
# =========================
# FDR dentro de cada outcome (ahora son 6 efectos: 4 deltas + Education + APOE)
main_df["q_FDR_within_outcome"] = np.nan
for out in main_df["outcome_frequency"].unique():
    m = main_df["outcome_frequency"] == out
    main_df.loc[m, "q_FDR_within_outcome"] = multipletests(main_df.loc[m, "p"].values, method="fdr_bh")[1]

# FDR opcional para todos los términos
full_params_df["q_FDR_within_outcome_all_terms"] = np.nan
for out in full_params_df["outcome_frequency"].unique():
    m = full_params_df["outcome_frequency"] == out
    full_params_df.loc[m, "q_FDR_within_outcome_all_terms"] = multipletests(full_params_df.loc[m, "p"].values, method="fdr_bh")[1]

# =========================
# 9) EXPORT
# =========================
with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    main_df.sort_values(["outcome_frequency", "p"]).to_excel(writer, sheet_name="Main_effects_FDR", index=False)
    full_params_df.sort_values(["outcome_frequency", "p"]).to_excel(writer, sheet_name="All_params_including_covariates", index=False)

print("✅ OK ->", OUTPUT_XLSX)
print("N sujetos usados:", len(d))
