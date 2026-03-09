# -*- coding: utf-8 -*-
"""
Auto-clustering on WMH Excel with GLM correction ONLY for clustering features.

Pipeline:
1) NO IMPUTATION (assumes clean data). If NaNs appear -> error (or drop rows if you change flag)
2) GLM/OLS residualization of clustering features (correcting by glm_covars EXACTLY as declared)
3) Z-score scaling of residuals
4) Clustering:
   - GMM: auto-k by min BIC (subset)
5) Post-cluster comparisons use ORIGINAL (UNCORRECTED) variables:
   controlled by compare_vars
6) Save to Excel (multi-sheet) + PNG plots (300 dpi) + interactive 3D HTML (Plotly)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.stats import kruskal, chi2_contingency

import plotly.graph_objects as go
import plotly.io as pio


# ===================== CONFIG =====================

excel_path = r"C:\Users\rglez\Documents\Ra\Papers\WMH long caracterization\WMH_clusters_metrics-all_with_clinic_vars.xlsx"

out_dir = r"C:\Users\rglez\Documents\Ra\Papers\WMH long caracterization\clustering"
base_name = os.path.splitext(os.path.basename(excel_path))[0]
excel_out = os.path.join(out_dir, base_name + "_CLUSTERING_GLM.xlsx")

# Features used for clustering (THESE are corrected via GLM)
requested_cols = [
    "∆T1", "∆GFA","∆QA","∆ISO","∆ha","∆ad","∆fa","∆rd","∆rd1","∆rd2",
    "∆fALFF","∆Hurst","∆Entropy",
    "∆Power slope","∆Autocor", "∆WMH", "Coord-x", "Coord-y", "Coord-z"
]

# Covariates for GLM correction (only applied to requested_cols)
glm_covars = ["Age", "Sex", "WMH_t0", "Size", "Education"]

# Variables to compare AFTER clustering (NOT corrected)
# -> TODO el análisis post-cluster usa ESTA lista (agrega aquí lo que quieras testear)
compare_vars = ["Sex", "Age", "Coord-x", "Coord-y", "Coord-z", "∆WMH", "Gpo", "Size"]

# Speed control
FIT_SUBSET_N = 5000
K_RANGE = list(range(2, 11))
SIL_SAMPLE = 2500

DROP_NA_ROWS = False  # tú dijiste que no hay NaNs; si aparece algo raro, mejor error.


# ===================== HELPERS =====================

def pick_subset(n, subset_n, seed=42):
    rng = np.random.default_rng(seed)
    if n <= subset_n:
        return np.arange(n)
    return rng.choice(n, size=subset_n, replace=False)

def safe_silhouette(X, labels, sample_size=2500, seed=42):
    labels = np.asarray(labels)
    if len(set(labels)) < 2:
        return np.nan
    ss = min(sample_size, len(labels))
    return float(silhouette_score(X, labels, sample_size=ss, random_state=seed))

def plot_pca_scatter(pc2, labels, title, out_png):
    plt.figure()
    plt.scatter(pc2[:, 0], pc2[:, 1], c=labels, s=12)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def plot_curve(x, y, title, xlabel, ylabel, out_png):
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def cluster_summary_table(df_in, label_col):
    """Summary per cluster using ORIGINAL variables (NOT corrected)."""
    df = df_in.copy()
    out = []

    # solo resume variables numéricas si existen y si están en compare_vars
    numeric_candidates = ["Age", "Coord_x", "Coord_y", "Coord_z", "∆WMH"]
    numeric_cols = [c for c in numeric_candidates if (c in df.columns and c in compare_vars)]

    for cl in sorted(df[label_col].dropna().unique()):
        sub = df[df[label_col] == cl]
        n = len(sub)

        row = {"cluster": cl, "N": n}

        # Sex counts SOLO si Sex está en compare_vars
        if "Sex" in compare_vars and "Sex" in sub.columns:
            nF = int((sub["Sex"] == "F").sum())
            nM = int((sub["Sex"] == "M").sum())
            row.update({
                "Sex_F_n": nF,
                "Sex_M_n": nM,
                "Sex_F_%": (100.0 * nF / n) if n else np.nan,
                "Sex_M_%": (100.0 * nM / n) if n else np.nan,
            })

        for c in numeric_cols:
            row[f"{c}_mean"] = float(np.mean(sub[c]))
            row[f"{c}_sd"]   = float(np.std(sub[c], ddof=1))

        out.append(row)

    return pd.DataFrame(out)

def freq_by_gpo(df_in, label_col, gpo_col="Gpo"):
    if (label_col not in df_in.columns) or (gpo_col not in df_in.columns):
        return pd.DataFrame()
    return pd.crosstab(df_in[label_col], df_in[gpo_col], dropna=False)

def chi2_test(df, a, b):
    tab = pd.crosstab(df[a], df[b])
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return None
    chi2, p, dof, _ = chi2_contingency(tab)
    return float(chi2), float(p), int(dof)

def run_tests_original(df_in, label_col):
    """Tests using ORIGINAL variables (NOT corrected), controlled by compare_vars."""
    df = df_in.copy()
    results = []

    # Categóricas: Sex y Gpo (si están en compare_vars)
    for cat in ["Sex", "Gpo"]:
        if cat in compare_vars and cat in df.columns:
            out = chi2_test(df, label_col, cat)
            if out is not None:
                chi2, p, dof = out
                results.append({
                    "label_col": label_col, "variable": cat, "test": "Chi-square",
                    "stat": chi2, "df": dof, "p_value": p
                })

    # Continuas: las que estén en compare_vars
    numeric_candidates = ["Age", "peak_x", "peak_y", "peak_z", "delta_mean_wmh", "size_vox_full"]
    for v in numeric_candidates:
        if v in compare_vars and v in df.columns:
            groups = [df.loc[df[label_col] == k, v].to_numpy()
                      for k in sorted(df[label_col].dropna().unique())]
            if len(groups) >= 2:
                H, p = kruskal(*groups)
                results.append({
                    "label_col": label_col, "variable": v, "test": "Kruskal-Wallis",
                    "stat": float(H), "df": int(len(groups)-1), "p_value": float(p)
                })

    return pd.DataFrame(results)

def rgba_to_plotly_rgba(rgba_tuple):
    r, g, b, a = rgba_tuple
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{a:.3f})"


# ===================== LOAD =====================

print(f"[INFO] Reading Excel: {excel_path}")
df = pd.read_excel(excel_path)

needed = list(set(requested_cols + glm_covars + compare_vars))
missing_needed = [c for c in needed if c not in df.columns]
if missing_needed:
    raise ValueError(f"Missing required columns in Excel: {missing_needed}")

use_cols = [c for c in requested_cols if c in df.columns]
if len(use_cols) < 2:
    raise ValueError("Not enough clustering features found in Excel.")

check_cols = use_cols + glm_covars
na_rows = df[check_cols].isna().any(axis=1)
n_na = int(na_rows.sum())

if n_na > 0:
    msg = f"[ERROR] Found {n_na} rows with NaNs in clustering features/covars. "
    if DROP_NA_ROWS:
        print(msg + "Dropping those rows (DROP_NA_ROWS=True).")
        df = df.loc[~na_rows].reset_index(drop=True)
    else:
        raise ValueError(msg + "Set DROP_NA_ROWS=True to drop them, or fix the Excel.")

print(f"[INFO] Rows after NaN handling: {len(df)}")

X_raw = df[use_cols].to_numpy(dtype=float)


# ===================== GLM / OLS correction (ONLY selected clustering features) =====================

# --- A) define qué features se corrigen por GLM y cuáles NO ---
NO_GLM_COLS = ["Coord_x", "Coord_y", "Coord_z"]  # <-- estas NO se corrigen

glm_features = [c for c in use_cols if c not in NO_GLM_COLS]
raw_features = [c for c in use_cols if c in NO_GLM_COLS]

if len(glm_features) < 2 and len(raw_features) < 2:
    raise ValueError("No quedan suficientes features para clustering.")

print("[INFO] GLM features (corrected):", glm_features)
print("[INFO] RAW features (NOT corrected):", raw_features)

# --- B) matrices ---
X_glm_raw = df[glm_features].to_numpy(dtype=float) if len(glm_features) else None
X_rawpart = df[raw_features].to_numpy(dtype=float) if len(raw_features) else None

# --- C) preparar covariables (igual que antes) ---
cov_df = df[glm_covars].copy()

if "Sex" in glm_covars:
    sex_map = {"F": 0, "M": 1, "f": 0, "m": 1}
    cov_df["Sex"] = cov_df["Sex"].map(sex_map)
    if cov_df["Sex"].isna().any():
        raise ValueError("Found unexpected values in Sex column (not F/M). Fix before running.")

C = cov_df[glm_covars].to_numpy(dtype=float)
Xcov = np.column_stack([np.ones(len(df)), C])

# --- D) GLM residuals SOLO para glm_features ---
if X_glm_raw is not None:
    B = np.linalg.lstsq(Xcov, X_glm_raw, rcond=None)[0]
    X_glm_resid = X_glm_raw - (Xcov @ B)
    X_glm_z = StandardScaler().fit_transform(X_glm_resid)
else:
    X_glm_z = None

# --- E) RAW part: NO residualizar, pero sí z-score para que esté en la misma escala ---
if X_rawpart is not None:
    X_raw_z = StandardScaler().fit_transform(X_rawpart)
else:
    X_raw_z = None

# --- F) combinar para clustering ---
if X_glm_z is not None and X_raw_z is not None:
    X_z = np.column_stack([X_glm_z, X_raw_z])
elif X_glm_z is not None:
    X_z = X_glm_z
elif X_raw_z is not None:
    X_z = X_raw_z
else:
    raise RuntimeError("No hay features para clustering después de separar GLM/RAW.")

# Importante: respetar el ORDEN de glm_covars
C = cov_df[glm_covars].to_numpy(dtype=float)

Xcov = np.column_stack([np.ones(len(df)), C])

B = np.linalg.lstsq(Xcov, X_raw, rcond=None)[0]
X_resid = X_raw - (Xcov @ B)

X_z = StandardScaler().fit_transform(X_resid)


# ===================== SUBSET + clustering =====================

n = len(df)
fit_idx = pick_subset(n, FIT_SUBSET_N, seed=42)
X_fit = X_z[fit_idx]


print(f"[INFO] Total rows: {n} | subset for selection/plots: {len(fit_idx)}")





# ===================== GMM auto-k =====================

print("\n=== [2] Gaussian Mixture (auto-k by min BIC) ===")
gmm_bics = []
gmm_models = {}

for k in K_RANGE:
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        random_state=42,
        n_init=2,
        max_iter=300
    )
    gmm.fit(X_fit)
    bic = gmm.bic(X_fit)
    gmm_bics.append(bic)
    gmm_models[k] = gmm
    print(f"  k={k:2d} | BIC={bic:.0f}")

best_k_gmm = K_RANGE[int(np.argmin(gmm_bics))]
gmm_best = gmm_models[best_k_gmm]

labels_gmm_full = gmm_best.predict(X_z)
labels_gmm_fit  = gmm_best.predict(X_fit)
gmm_probmax_full = gmm_best.predict_proba(X_z).max(axis=1)

print(f"[GMM] Best k = {best_k_gmm} (min BIC)")

plot_curve(K_RANGE, gmm_bics,
           title=f"GMM: BIC vs k (best k={best_k_gmm})",
           xlabel="k", ylabel="Bayesian Information Criterion",
           out_png=os.path.join(out_dir, base_name + "_GMM_bic.png"))




# ===================== OUTPUT DF =====================

df_out = df.copy()
df_out["cluster_gmm_auto"] = labels_gmm_full
df_out["cluster_gmm_probmax"] = gmm_probmax_full






# ===================== ANALYSIS (original variables only) =====================


gmm_summary    = cluster_summary_table(df_out, "cluster_gmm_auto")

# Frecuencia por Gpo solo si Gpo está en compare_vars

gmm_gpo_freq    = freq_by_gpo(df_out, "cluster_gmm_auto", "Gpo") if "Gpo" in compare_vars else pd.DataFrame()


gmm_tests    = run_tests_original(df_out, "cluster_gmm_auto")
tests_all = pd.concat([gmm_tests], ignore_index=True)


# ===================== SAVE EXCEL =====================

with pd.ExcelWriter(excel_out, engine="openpyxl") as writer:
    df_out.to_excel(writer, index=False, sheet_name="data_with_clusters")
    
    gmm_summary.to_excel(writer, index=False, sheet_name="gmm_cluster_summary")

    
    if not gmm_gpo_freq.empty:
        gmm_gpo_freq.to_excel(writer, sheet_name="gmm_freq_by_Gpo")

    tests_all.to_excel(writer, index=False, sheet_name="cluster_tests")

print("\n[DONE] Saved:", excel_out)
print("[DONE] Plots saved in:", out_dir)
print("[DONE] GMM best k:", best_k_gmm)


# ===================== PLOTS: mismos colores (viridis) + LEYENDA L₁,L₂,... =====================


def make_cluster_color_map(labels_or_uniq, cmap_name="viridis", pos_min=0.10, pos_max=0.95):
    """
    Devuelve dict: {cluster_label(int): rgba(tuple)} con colores consistentes.

    Regla: L₁ (cluster 0) empieza en amarillo (pos_max),
           y el resto va hacia colores más bajos del colormap.
    """
    uniq = np.array(sorted(np.unique(np.asarray(labels_or_uniq).astype(int))))
    # posiciones del colormap para cada cluster (mismo tamaño que uniq)
    positions = np.linspace(pos_max, pos_min, len(uniq))  # <-- importante: empieza ALTO (amarillo)
    cmap = cm.get_cmap(cmap_name)
    return {int(lab): cmap(float(pos)) for lab, pos in zip(uniq, positions)}




def to_subscript(n: int) -> str:
    return str(n).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))

def lab_text(lab: int) -> str:
    # labels 0..k-1 -> L₁..Lₖ
    return f"L{to_subscript(int(lab) + 1)}"

pc3 = PCA(n_components=3, random_state=42).fit_transform(X_fit)

labels = np.asarray(labels_gmm_fit).astype(int)
uniq = np.array(sorted(np.unique(labels)))
cluster_color = make_cluster_color_map(uniq, cmap_name="viridis", pos_min=0.10, pos_max=0.95)





views = [(20, 35), (20, 120), (20, 210), (60, 35)]
for elev, azim in views:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    for lab in uniq:
        m = labels == lab
        ax.scatter(pc3[m, 0], pc3[m, 1], pc3[m, 2],
                   s=10, alpha=0.65, color=cluster_color[int(lab)],
                   label=lab_text(lab))

    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title(f"GMM clusters (subset) - k={len(uniq)} | elev={elev}, azim={azim}")
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc="upper right", title="Clusters", fontsize=9)

    out_png = os.path.join(out_dir, f"{base_name}_GMM_PCA3D_e{elev}_a{azim}.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print("saved:", out_png, "| exists?", os.path.exists(out_png))

    plt.show()
    plt.close(fig) 

pairs = [(0, 1, "PC1", "PC2"), (0, 2, "PC1", "PC3"), (1, 2, "PC2", "PC3")]
for a, b, xa, xb in pairs:
    fig, ax = plt.subplots(figsize=(7, 5))

    for lab in uniq:
        m = labels == lab
        ax.scatter(pc3[m, a], pc3[m, b], s=12, alpha=0.65,
                   color=cluster_color[int(lab)], label=lab_text(lab))

    ax.set_xlabel(xa); ax.set_ylabel(xb)
    ax.set_title(f"GMM (subset) - {xa} vs {xb} | k={len(uniq)}")
    ax.legend(title="Clusters", fontsize=9)

    out_png = os.path.join(out_dir, f"{base_name}_GMM_{xa}_vs_{xb}.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print("saved:", out_png, "| exists?", os.path.exists(out_png))

    plt.show()
    plt.close(fig)

# ===================== FRECUENCIAS por Gpo (STACKED BAR) - ROBUSTO =====================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

label_col = "cluster_gmm_auto"
gpo_col   = "Gpo"

# 0) checks básicos
if label_col not in df_out.columns:
    raise ValueError(f"Falta {label_col} en df_out")
if gpo_col not in df_out.columns:
    raise ValueError(f"Falta {gpo_col} en df_out")

# Asegurar carpeta
os.makedirs(out_dir, exist_ok=True)
print("out_dir (abs) =", os.path.abspath(out_dir))

# 1) tabla conteos
tab = pd.crosstab(df_out[label_col], df_out[gpo_col], dropna=False)
print("tab shape:", tab.shape)
print("tab index (clusters):", tab.index.tolist())
print("tab columns (Gpo):", tab.columns.tolist())

if tab.shape[0] == 0 or tab.shape[1] == 0:
    raise ValueError("La tabla de contingencia quedó vacía (revisa Gpo / clusters).")

# 2) ordenar columnas si quieres
desired_order = ["CN", "MCI", "AD", "PD"]
present = [g for g in desired_order if g in tab.columns]
others  = [g for g in tab.columns if g not in present]
group_order = present + others
tab = tab[group_order]

# 3) porcentajes
col_sums = tab.sum(axis=0)
print("sum por grupo:", col_sums.to_dict())

# ojo: si algún grupo suma 0, esa columna quedará NaN
tab_pct = tab.div(col_sums.replace(0, np.nan), axis=1) * 100

# si todo es NaN -> no hay datos válidos
if np.isfinite(tab_pct.to_numpy()).sum() == 0:
    raise ValueError("tab_pct quedó todo NaN (algún grupo con sum=0 o datos inválidos).")

# 4) colores: usa tu mapping existente cluster_color (asegura claves int)
# si no existe, lo creo desde los clusters presentes en tab.index
if "cluster_color" not in globals():
    from matplotlib import cm
    uniq_here = np.array(sorted(tab.index.astype(int)))
    cmap = cm.get_cmap("YlGn")  # o viridis/cividis/inferno
    pos = np.linspace(0.95, 0.20, len(uniq_here))  # empieza amarillo
    cluster_color = {int(lab): cmap(float(p)) for lab, p in zip(uniq_here, pos)}

# 5) plot (usa fig.savefig)
fig, ax = plt.subplots(figsize=(1.8 * len(tab_pct.columns) + 2.5, 6))
bottom = np.zeros(len(tab_pct.columns), dtype=float)

# apilado: de cluster alto a bajo (si quieres)
for lab in sorted(tab_pct.index.astype(int), reverse=True):
    vals = tab_pct.loc[lab].to_numpy(dtype=float)
    ax.bar(tab_pct.columns, vals, bottom=bottom,
           color=cluster_color[int(lab)],
           label=lab_text(int(lab)))
    bottom += np.nan_to_num(vals, nan=0.0)

ax.set_ylim(0, 100)
ax.set_ylabel("Frequency (% within group)", fontsize=14)
ax.tick_params(axis="x", rotation=0, labelsize=12)
ax.tick_params(axis="y", labelsize=12)

# leyenda
handles, leg = ax.get_legend_handles_labels()
ax.legend(handles[::-1], leg[::-1], title="Clusters",
          bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

fig.tight_layout()

out_png = os.path.join(out_dir, f"{base_name}_GMM_cluster_frequencies_by_group.png")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
print("saved:", os.path.abspath(out_png), "| exists?", os.path.exists(out_png))

# opcional: mostrar en pantalla
plt.show()
plt.close(fig)

# ===================== INTERACTIVE 3D (HTML) =====================

def to_subscript(n: int) -> str:
    return str(n).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))

def lab_text(lab: int) -> str:
    return f"L{to_subscript(int(lab) + 1)}"

fig = go.Figure()
for lab in uniq:
    m = labels == lab
    fig.add_trace(go.Scatter3d(
        x=pc3[m, 0], y=pc3[m, 1], z=pc3[m, 2],
        mode="markers",
        name=lab_text(lab),
        marker=dict(
            size=3,
            opacity=0.65,
            color=rgba_to_plotly_rgba(cluster_color[int(lab)])  # <- MISMO mapping
        )
    ))

fig.update_layout(
    title=f"GMM clusters (subset) - k={len(uniq)} (interactive)",
    scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
    legend=dict(title="Clusters")
)

html_out = os.path.join(out_dir, f"{base_name}_GMM_PCA3D_interactive.html")
pio.write_html(fig, file=html_out, auto_open=False, include_plotlyjs="cdn")
print("[DONE] Interactive 3D saved:", html_out)



# =========================================================================================================
# ===================== GMM: TABLA DE FRECUENCIAS por Gpo + STATS + PLOT (L₁,L₂,...) =====================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import chi2_contingency, norm as normal_dist
from statsmodels.stats.multitest import multipletests

label_col = "cluster_gmm_auto"
gpo_col = "Gpo"

# -------- helpers: L₁, L₂, ... --------
def to_subscript(n: int) -> str:
    return str(n).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))

def lab_text(lab) -> str:
    # asume labels 0..k-1
    return f"L{to_subscript(int(lab) + 1)}"


# -------------------- 1) Contingency table (counts) --------------------
tab = pd.crosstab(df_out[label_col], df_out[gpo_col], dropna=False)

# Orden deseado de grupos (si existen)
desired_order = ["CN", "MCI", "AD", "PD"]
present = [g for g in desired_order if g in tab.columns]
others = [g for g in tab.columns if g not in present]
group_order = present + others
tab = tab[group_order]

# -------------------- 2) % dentro de cada grupo (columna) --------------------
tab_pct = tab.div(tab.sum(axis=0), axis=1) * 100

# -------------------- 3) Colores viridis consistentes por label (con reorden k=3) --------------------
# labels de clusters (asumimos numéricos)
uniq_full = np.array(sorted(tab.index.astype(int)))

cmap = cm.get_cmap("viridis")

if len(uniq_full) == 3:
    # viridis: bajo=violeta, medio=verde, alto=amarillo
    pos = {2: 0.10, 0: 0.60, 1: 0.95}
    cluster_color_full = {lab: cmap(pos[int(lab)]) for lab in uniq_full}
else:
    # fallback estándar
    norm_colors = plt.Normalize(vmin=float(np.min(uniq_full)), vmax=float(np.max(uniq_full)))
    cluster_color_full = {lab: cmap(norm_colors(float(lab))) for lab in uniq_full}

# -------------------- 4) Estadísticas: chi-square global + Cramer's V --------------------
obs = tab.values
chi2, p, dof, expected = chi2_contingency(obs, correction=False)

N = obs.sum()
k = min(tab.shape[0] - 1, tab.shape[1] - 1)
cramers_v_global = np.sqrt(chi2 / (N * k)) if (N > 0 and k > 0) else np.nan

global_summary = pd.DataFrame(
    {
        "chi2": [float(chi2)],
        "df": [int(dof)],
        "p": [float(p)],
        "N": [int(N)],
        "Cramers_V": [float(cramers_v_global)],
    }
)

# -------------------- 5) Pairwise comparisons + Effect size (Cramer's V) --------------------
pairs = []
stats = []
cols = tab.columns.tolist()

for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        g1, g2 = cols[i], cols[j]

        sub = tab[[g1, g2]].values
        x2, pp, dfp, _ = chi2_contingency(sub, correction=False)

        N2 = sub.sum()
        k2 = min(sub.shape[0] - 1, sub.shape[1] - 1)
        cramers_v_pair = np.sqrt(x2 / (N2 * k2)) if (N2 > 0 and k2 > 0) else np.nan

        pairs.append((g1, g2))
        stats.append((float(x2), int(dfp), float(pp), float(cramers_v_pair), int(N2)))

pair_df = pd.DataFrame(
    stats,
    columns=["chi2", "df", "p", "Cramers_V", "N"],
    index=[f"{a} vs {b}" for a, b in pairs],
)
pair_df["p_fdr"] = multipletests(pair_df["p"].values, method="fdr_bh")[1]
pair_df["p_bonf"] = multipletests(pair_df["p"].values, method="bonferroni")[1]
pair_df = pair_df.sort_values("p_fdr")

# -------------------- 6) Residuales estandarizados por celda + p-values por celda (FDR todas las celdas) --------------------
exp = expected
row_sum = obs.sum(axis=1, keepdims=True)
col_sum = obs.sum(axis=0, keepdims=True)

row_prop = row_sum / N if N else np.nan
col_prop = col_sum / N if N else np.nan

with np.errstate(divide="ignore", invalid="ignore"):
    std_res = (obs - exp) / np.sqrt(exp * (1 - row_prop) * (1 - col_prop))

std_res_df = pd.DataFrame(std_res, index=tab.index, columns=tab.columns)

p_cell = 2 * normal_dist.sf(np.abs(std_res))
p_cell_adj = multipletests(np.ravel(p_cell), method="fdr_bh")[1].reshape(p_cell.shape)

p_cell_df = pd.DataFrame(p_cell, index=tab.index, columns=tab.columns)
p_cell_fdr_df = pd.DataFrame(p_cell_adj, index=tab.index, columns=tab.columns)

print("[DONE] Chi-square global:", global_summary.to_dict(orient="records")[0])

# -------------------- 7) PLOT: barras apiladas (%) por grupo con leyenda L₁,L₂,... --------------------
plot_df = tab_pct.copy()

fig, ax = plt.subplots(figsize=(1.8 * len(group_order) + 2.5, 6))

bottom = np.zeros(len(plot_df.columns), dtype=float)

# Mantengo TU orden de apilado (NO lo cambio)
for lab in sorted(plot_df.index.astype(int), reverse=True):
    vals = plot_df.loc[lab].values.astype(float)
    c = cluster_color_full[int(lab)]

    ax.bar(plot_df.columns, vals, bottom=bottom, label=lab_text(lab), color=c)
    bottom += vals

ax.set_ylim(0, 100)

# Texto más claro y grande
ax.set_ylabel("Frequency (% within group)", fontsize=16)
ax.tick_params(axis="x", rotation=0, labelsize=16)
ax.tick_params(axis="y", labelsize=14)

# Leyenda invertida (solo la leyenda; NO cambia el plot)
handles, labels_legend = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels_legend[::-1],
          title="Clusters",
          fontsize=14,
          title_fontsize=13,
          bbox_to_anchor=(1.02, 1),
          loc="upper left",
          frameon=False)

plt.tight_layout()

os.makedirs(out_dir, exist_ok=True)
out_png = os.path.join(out_dir, "GMM_cluster_frequencies_by_group.png")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()

print(f"[DONE] Plot guardado en: {out_png}")


# -------------------- 8) Guardar Excel (con nombres L₁,L₂,... como extra) --------------------
# OJO: para stats mantenemos index numérico; agregamos una versión "bonita" para lectura humana
tab_named = tab.copy()
tab_named.index = [lab_text(i) for i in tab.index.astype(int)]

tab_pct_named = tab_pct.copy()
tab_pct_named.index = [lab_text(i) for i in tab_pct.index.astype(int)]

std_res_named = std_res_df.copy()
std_res_named.index = [lab_text(i) for i in std_res_df.index.astype(int)]

p_cell_named = p_cell_df.copy()
p_cell_named.index = [lab_text(i) for i in p_cell_df.index.astype(int)]

p_cell_fdr_named = p_cell_fdr_df.copy()
p_cell_fdr_named.index = [lab_text(i) for i in p_cell_fdr_df.index.astype(int)]

out_xlsx = os.path.join(out_dir, "GMM_cluster_stats.xlsx")
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
    global_summary.to_excel(writer, sheet_name="global_stats", index=False)
    pair_df.to_excel(writer, sheet_name="pairwise_stats")

    tab.to_excel(writer, sheet_name="counts_numeric")                 # original
    tab_named.to_excel(writer, sheet_name="counts_L")                 # L₁, L₂, ...
    tab_pct.to_excel(writer, sheet_name="percent_by_group_numeric")
    tab_pct_named.to_excel(writer, sheet_name="percent_by_group_L")

    std_res_df.to_excel(writer, sheet_name="std_residuals_numeric")
    std_res_named.to_excel(writer, sheet_name="std_residuals_L")

    p_cell_df.to_excel(writer, sheet_name="p_cell_numeric")
    p_cell_named.to_excel(writer, sheet_name="p_cell_L")

    p_cell_fdr_df.to_excel(writer, sheet_name="p_cell_fdr_numeric")
    p_cell_fdr_named.to_excel(writer, sheet_name="p_cell_fdr_L")

print(f"[DONE] Excel guardado en: {out_xlsx}")



# ============================================================
# CLUSTER_GMM_PROBMAX: mean + SEM por cluster + BARPLOT + EXCEL SEPARADO
# - NO depende de cluster_color_full (lo recrea)
# - Guarda un Excel aparte + un PNG chiquito
# Requiere que ya existan:
#   df_out, out_dir, base_name
# y que df_out tenga:
#   "cluster_gmm_auto" y "cluster_gmm_probmax"
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# ---------------- CONFIG ----------------
LABEL_COL = "cluster_gmm_auto"
PROB_COL  = "cluster_gmm_probmax"

# outputs
os.makedirs(out_dir, exist_ok=True)
probmax_xlsx = os.path.join(out_dir, f"{base_name}_GMM_probmax_by_cluster.xlsx")
probmax_png  = os.path.join(out_dir, f"{base_name}_GMM_probmax_by_cluster.png")

# ---------------- CHECKS ----------------
missing_cols = [c for c in [LABEL_COL, PROB_COL] if c not in df_out.columns]
if missing_cols:
    raise ValueError(f"Faltan columnas en df_out: {missing_cols}")

if df_out[[LABEL_COL, PROB_COL]].isna().any().any():
    # Si quieres permitir NaNs, cambia esto por un dropna
    bad_n = int(df_out[[LABEL_COL, PROB_COL]].isna().any(axis=1).sum())
    raise ValueError(f"Hay {bad_n} filas con NaN en {LABEL_COL}/{PROB_COL}. Arregla o filtra antes.")

# ---------------- SUMMARY TABLE ----------------
tmp = df_out[[LABEL_COL, PROB_COL]].copy()
tmp[LABEL_COL] = tmp[LABEL_COL].astype(int)

summary = (
    tmp.groupby(LABEL_COL)[PROB_COL]
       .agg(["count", "mean", "std"])
       .rename(columns={"count": "N", "mean": "probmax_mean", "std": "probmax_sd"})
       .reset_index()
       .sort_values(LABEL_COL)
       .reset_index(drop=True)
)

summary["probmax_sem"] = summary["probmax_sd"] / np.sqrt(summary["N"].clip(lower=1))

# ---------------- COLORS (viridis consistente por label) ----------------
clusters_prob = summary[LABEL_COL].to_numpy()
cmap_v = cm.get_cmap("viridis")

# normalización por min/max label (igual que venías usando en otros plots)
norm_v = plt.Normalize(vmin=float(np.min(clusters_prob)), vmax=float(np.max(clusters_prob)))
bar_colors = [cmap_v(norm_v(float(c))) for c in clusters_prob]

# ---------------- PLOT (chiquito y sencillo) ----------------
plt.figure(figsize=(3.4 + 0.35 * len(clusters_prob), 3.2))

x = np.arange(len(clusters_prob))
y = summary["probmax_mean"].to_numpy(float)
yerr = summary["probmax_sem"].to_numpy(float)

plt.bar(x, y, yerr=yerr, capsize=3, color=bar_colors)
plt.xticks(x, [str(int(c)) for c in clusters_prob], rotation=0)
plt.xlabel("Cluster")
plt.ylabel("Mean max posterior prob")
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig(probmax_png, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print("[DONE] Probmax barplot:", probmax_png)

# ---------------- SAVE EXCEL SEPARADO ----------------
with pd.ExcelWriter(probmax_xlsx, engine="openpyxl") as writer:
    summary.to_excel(writer, index=False, sheet_name="probmax_summary")
    # opcional: guardar distribución completa
    tmp.to_excel(writer, index=False, sheet_name="probmax_all_rows")

print("[DONE] Probmax Excel:", probmax_xlsx)


# ===================== BUBBLE MATRIX (SIG ONLY) - STANDALONE =====================
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# ---------------- CONFIG ----------------
P_COL = "p_fdr"
ALPHA = 0.05

# "chi2" or "cramers_v"
SIZE_MODE = "cramers_v"   # <-- cambia a "cramers_v" si quieres effect size

# Drop categories (optional)
DROP_ROWS = {"CN"}
DROP_COLS = {"PD"}

# Bubble sizes
S_MIN, S_MAX = 650, 4200

# Colorbar endpoints (p_end = 10^-V_MAX)
V_MAX = 6.0

# Manual legend placement
LEG_BASE_X = 0.70
LEG_BASE_Y = -0.20
LEG_GAP_PT = 30.0       # <-- más separación (sube/baja aquí)
LEG_SCALE = 0.85
BOTTOM_MARGIN = 0.30
LEG_EDGE = "#666666"

# χ² legend values shown as (chi2-5) => 10,20,70
CHI2_LEG = np.array([15, 25, 75], dtype=float)
CHI2_TEXT_SHIFT = 5.0

# V legend values (si SIZE_MODE="cramers_v")
V_LEG = np.array([0.10, 0.20, 0.40], dtype=float)
V_TEXT_FMT = "{:.2f}"


# ---------------- REQUIRED OBJECTS CHECK ----------------
missing = []
for name in ["pair_df", "out_dir", "base_name"]:
    if name not in globals():
        missing.append(name)
if missing:
    raise NameError(f"Faltan variables en tu sesión: {missing}. "
                    "Este script asume que ya corriste el pipeline y existen pair_df, out_dir, base_name.")

# pair_df debe tener chi2 y p-values
for col in ["chi2", P_COL]:
    if col not in pair_df.columns:
        raise ValueError(f"pair_df no tiene la columna requerida: '{col}'")


# ---------------- HELPERS ----------------
def get_pair_row(a, b, pair_df):
    k1 = f"{a} vs {b}"
    k2 = f"{b} vs {a}"
    if k1 in pair_df.index:
        return pair_df.loc[k1]
    if k2 in pair_df.index:
        return pair_df.loc[k2]
    return None

def area_to_diameter_pt(s_area):
    # scatter uses area in pt^2
    return 2.0 * np.sqrt(np.array(s_area, dtype=float) / np.pi)

def map_to_sizes(raw, smin, smax):
    raw = np.asarray(raw, float)
    m = np.isfinite(raw)
    if m.sum() == 0:
        return np.array([], float), (np.nan, np.nan, np.nan)
    x = raw[m]
    lo, hi = (np.percentile(x, [5, 95]) if len(x) > 1 else (x.min(), x.max() + 1e-9))
    denom = (hi - lo) if (hi - lo) > 1e-12 else 1.0
    z = np.clip((raw - lo) / denom, 0, 1)
    return (smin + (smax - smin) * z), (lo, hi, denom)

def ensure_group_order(pair_df, df_out=None, group_col="Gpo"):
    # Prefer df_out groups if available
    if df_out is not None and group_col in df_out.columns:
        return list(pd.unique(df_out[group_col].dropna()))
    # else parse from pair_df index "A vs B"
    groups = []
    for idx in pair_df.index.astype(str):
        if " vs " in idx:
            a, b = idx.split(" vs ")
            groups.extend([a.strip(), b.strip()])
    # unique preserving order
    seen = set()
    out = []
    for g in groups:
        if g not in seen:
            out.append(g)
            seen.add(g)
    return out

def cramers_v_from_df(df_out, g1, g2, cluster_col="cluster_gmm_auto", group_col="Gpo"):
    from scipy.stats import chi2_contingency
    sub = df_out[df_out[group_col].isin([g1, g2])]
    tab = pd.crosstab(sub[cluster_col], sub[group_col])
    if g1 not in tab.columns or g2 not in tab.columns or tab.shape[0] < 2:
        return np.nan
    obs = tab[[g1, g2]].values
    chi2, p, dof, exp = chi2_contingency(obs, correction=False)
    n = obs.sum()
    k = min(obs.shape[0] - 1, obs.shape[1] - 1)
    if n <= 0 or k <= 0:
        return np.nan
    return float(np.sqrt(chi2 / (n * k)))


# ---------------- GET GROUP ORDER (NO group_order needed) ----------------
import pandas as pd  # needed for ensure_group_order

df_out_local = globals().get("df_out", None)  # optional
groups_all = ensure_group_order(pair_df, df_out=df_out_local, group_col="Gpo")

# apply drops
rows = [g for g in groups_all if g not in DROP_ROWS]
cols = [g for g in groups_all if g not in DROP_COLS]
R, C = len(rows), len(cols)

if R == 0 or C == 0:
    raise ValueError("Después de DROP_ROWS/DROP_COLS, no quedan filas/columnas para plotear.")


# ---------------- COLLECT SIGNIFICANT CELLS ----------------
xs, ys, pvals = [], [], []
chi2vals, vvals = [], []

for i in range(R):
    for j in range(C):
        if i >= j:
            rr, cc = rows[i], cols[j]
            if rr == cc:
                continue
            r = get_pair_row(rr, cc, pair_df)
            if r is None:
                continue
            pv = float(r[P_COL])
            if np.isfinite(pv) and pv < ALPHA:
                xs.append(j); ys.append(i)
                pvals.append(pv)
                chi2vals.append(float(r["chi2"]))

                vv = np.nan
                if "cramers_v" in pair_df.columns:
                    vv = float(r["cramers_v"]) if "cramers_v" in r.index else np.nan
                # if missing, try compute from df_out
                if (not np.isfinite(vv)) and (SIZE_MODE == "cramers_v") and (df_out_local is not None):
                    try:
                        vv = cramers_v_from_df(df_out_local, rr, cc)
                    except Exception:
                        vv = np.nan
                vvals.append(vv)

pvals = np.array(pvals, float)
chi2vals = np.array(chi2vals, float)
vvals = np.array(vvals, float)

if len(pvals) == 0:
    print("[WARN] No hay pares significativos (p < ALPHA). Plot sin burbujas.")


# ---------------- SIZE DRIVER ----------------
if SIZE_MODE == "cramers_v" and np.any(np.isfinite(vvals)):
    size_label = "V"
    driver_raw = np.clip(vvals, 0, None)        # V in [0,1]
elif SIZE_MODE == "chi2" and np.any(np.isfinite(chi2vals)):
    size_label = "χ²"
    driver_raw = np.sqrt(np.clip(chi2vals, 0, None))  # stabilize
else:
    # fallback
    size_label = "χ²"
    driver_raw = np.sqrt(np.clip(chi2vals, 0, None)) if len(chi2vals) else np.array([])

sizes, norm_params = map_to_sizes(driver_raw, S_MIN, S_MAX)


# ---------------- COLORS (by p) ----------------
val_color = -np.log10(np.clip(pvals, 1e-300, 1.0)) if len(pvals) else np.array([])
cmap = colors.LinearSegmentedColormap.from_list("white_to_red", ["#ffffff", "#ff0000"])
vmin = -np.log10(10)
vmax = V_MAX
norm = colors.Normalize(vmin=vmin, vmax=vmax)
colors_rgba = cmap(norm(np.clip(val_color, vmin, vmax))) if len(val_color) else np.array([])


# ---------------- PLOT ----------------
fig, ax = plt.subplots(figsize=(7.2, 6.0))

ax.set_xlim(-0.5, C - 0.5)
ax.set_ylim(R - 0.5, -0.5)

ax.set_xticks(np.arange(C))
ax.set_yticks(np.arange(R))
ax.set_xticklabels(cols, rotation=35, ha="right")
ax.set_yticklabels(rows)

ax.set_aspect("equal", adjustable="box")

ax.set_xticks(np.arange(-.5, C, 1), minor=True)
ax.set_yticks(np.arange(-.5, R, 1), minor=True)
ax.grid(which="minor", color="#dddddd", linestyle="-", linewidth=1)
ax.tick_params(which="minor", bottom=False, left=False)
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_title("")
ax.set_xlabel("")
ax.set_ylabel("")

if len(xs) > 0:
    ax.scatter(xs, ys, s=sizes, c=colors_rgba, edgecolors="none", linewidths=0)

# ---- colorbar endpoints ----
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
p_start = ALPHA
p_end = 10 ** (-vmax)
cbar.set_ticks([vmin, vmax])
cbar.set_ticklabels([f"{p_start:g}", f"{p_end:g}"])
cbar.ax.xaxis.set_ticks_position("top")
cbar.ax.xaxis.set_label_position("top")
cbar.set_label("")
cbar.ax.set_title("")


# ---------------- MANUAL SIZE LEGEND (no overlap, equal gaps) ----------------
# Build legend circles with the SAME normalization as plot
if size_label == "χ²":
    leg_driver = np.sqrt(np.clip(CHI2_LEG, 0, None))
    leg_sizes, _ = map_to_sizes(leg_driver, S_MIN, S_MAX)
    leg_sizes = leg_sizes * LEG_SCALE

    leg_text = [f"{int(v - CHI2_TEXT_SHIFT)}" for v in CHI2_LEG]
    title_txt = "χ²"
    title_index = 1  # middle
elif size_label == "V":
    leg_driver = np.clip(V_LEG, 0, None)
    # normalize legend using the observed driver_raw distribution (better)
    m = np.isfinite(driver_raw)
    if m.sum() > 0:
        lo = np.percentile(driver_raw[m], 5)
        hi = np.percentile(driver_raw[m], 95) if m.sum() > 1 else (driver_raw[m].max() + 1e-9)
        denom = (hi - lo) if (hi - lo) > 1e-12 else 1.0
        z = np.clip((leg_driver - lo) / denom, 0, 1)
        leg_sizes = (S_MIN + (S_MAX - S_MIN) * z) * LEG_SCALE
    else:
        leg_sizes = (S_MIN + (S_MAX - S_MIN) * np.linspace(0.2, 0.8, len(V_LEG))) * LEG_SCALE

    leg_text = [V_TEXT_FMT.format(v) for v in V_LEG]
    title_txt = "V"
    title_index = 1
else:
    leg_sizes = None

if leg_sizes is not None:
    # spacing by diameters + constant gap
    diam = area_to_diameter_pt(leg_sizes)
    x_pt = np.zeros(len(leg_sizes))
    x_pt[0] = 0.0
    for k in range(1, len(leg_sizes)):
        x_pt[k] = x_pt[k-1] + (diam[k-1]/2 + diam[k]/2) + LEG_GAP_PT

    # convert pt -> axes fraction using axis width
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    ax_w_pt = bbox.width * 72.0
    x_ax = x_pt / ax_w_pt

    xs_leg = LEG_BASE_X + x_ax
    y_leg = LEG_BASE_Y

    for x, s_area, t in zip(xs_leg, leg_sizes, leg_text):
        ax.scatter([x], [y_leg], s=[s_area],
                   facecolors="none", edgecolors=LEG_EDGE, linewidths=1.2,
                   transform=ax.transAxes, clip_on=False)
        ax.text(x, y_leg, t, ha="center", va="center",
                fontsize=9, color="#444444",
                transform=ax.transAxes, clip_on=False)

    ax.text(xs_leg[title_index], y_leg + 0.10, title_txt,
            ha="center", va="center",
            fontsize=11, color="#444444",
            transform=ax.transAxes, clip_on=False)

plt.tight_layout()
plt.subplots_adjust(bottom=BOTTOM_MARGIN)

out_png = os.path.join(
    out_dir,
    f"{base_name}_GMM_pairwise_frecuency.png"
)
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close()
print("[DONE] Bubble-matrix saved:", out_png)



# =============================================================== 
# ===================== CORRELATIONS (matrix + p-values + FDR + heatmap) =====================
# -*- coding: utf-8 -*-
"""
LOWER-TRIANGLE SPEARMAN HEATMAPS (ALL + BY CLUSTER)
- Variables: delta_mean_wmh + métricas
- Solo triángulo inferior (parte superior + diagonal principal VACÍAS)
- Color: divergente azul (neg) -> blanco (0) -> rojo (pos)
- Escala fija: [-0.5, 0.5]
- Asteriscos: significancia (por defecto p_FDR dentro de cada bloque)

Requiere que existan en tu sesión:
  - df_out (DataFrame final con clusters)
  - out_dir (carpeta salida)
  - base_name (string para nombres)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

# ---------------- CONFIG ----------------
vars_corr = [
    "∆WMH",
    "∆T1", "∆GFA", "∆QA", "∆ISO", "∆ha", "∆ad", "∆fa", "∆rd", "∆rd1", "∆rd2",
    "∆fALFF", "∆Hurst", "∆Entropy",
    "∆Power slope", "∆Autocor",
]

# partial Spearman (residualize) o Spearman crudo
DO_PARTIAL = True
covars = glm_covars

# Columna de cluster
CLUSTER_COL = "cluster_gmm_auto"   # o "cluster_kmeans_auto"

# Significancia
ALPHA = 0.05
USE_FDR = True
FDR_METHOD = "fdr_bh"

# Escala y colormap
VMIN, VMAX = -0.5, 0.5
CMAP = "bwr"  # azul-blanco-rojo (neg=azul, pos=rojo)

# Outputs
os.makedirs(out_dir, exist_ok=True)
xlsx_out = os.path.join(out_dir, f"{base_name}_lowertri_spearman_ALL_and_clusters.xlsx")

# ---------------- HELPERS ----------------
def _encode_sex(series):
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    sex_map = {"F": 0, "M": 1, "f": 0, "m": 1}
    out = series.map(sex_map)
    if out.isna().any():
        bad = series[out.isna()].unique()
        raise ValueError(f"Unexpected Sex values: {bad}. Esperado F/M.")
    return out.astype(float)

def residualize_matrix(Y, cov_df):
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    C = cov_df.to_numpy(dtype=float)
    Xcov = np.column_stack([np.ones(len(cov_df)), C])  # intercept + covars
    B = np.linalg.lstsq(Xcov, Y, rcond=None)[0]
    return Y - (Xcov @ B)

def stars_from_p(p):
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""

def compute_spearman_block(df_block, label="ALL"):
    """
    Devuelve df_r, df_p, df_padj (solo triángulo inferior + diag definidos; resto NaN).
    Ajuste FDR SOLO sobre las celdas del triángulo inferior.
    """
    use_cols = vars_corr + (covars if DO_PARTIAL else [])
    miss = [c for c in use_cols if c not in df_block.columns]
    if miss:
        raise ValueError(f"[{label}] Faltan columnas: {miss}")

    if df_block[use_cols].isna().any().any():
        bad_n = int(df_block[use_cols].isna().any(axis=1).sum())
        raise ValueError(f"[{label}] Hay {bad_n} filas con NaN en vars/covars. Arregla o filtra.")

    X = df_block[vars_corr].to_numpy(dtype=float)

    if DO_PARTIAL:
        cov_df = df_block[covars].copy()
        if "Sex" in covars:
            cov_df["Sex"] = _encode_sex(cov_df["Sex"])
        X = residualize_matrix(X, cov_df)

    P = len(vars_corr)
    R = np.full((P, P), np.nan, float)
    PVAL = np.full((P, P), np.nan, float)

    for i in range(P):
        R[i, i] = 1.0
        PVAL[i, i] = 0.0
        for j in range(i):  # SOLO triángulo inferior
            r, p = spearmanr(X[:, i], X[:, j])
            R[i, j] = float(r)
            PVAL[i, j] = float(p)

    PADJ = PVAL.copy()
    if USE_FDR:
        tri = np.tril_indices(P, k=-1)
        pvec = PVAL[tri]
        padj = multipletests(pvec, method=FDR_METHOD)[1]
        PADJ[tri] = padj

    df_r = pd.DataFrame(R, index=vars_corr, columns=vars_corr)
    df_p = pd.DataFrame(PVAL, index=vars_corr, columns=vars_corr)
    df_padj = pd.DataFrame(PADJ, index=vars_corr, columns=vars_corr)
    return df_r, df_p, df_padj

def plot_lower_triangle(df_r, df_p_use, title=None, out_png=None,
                        drop_row="∆WMH", drop_col="∆Autocor",
                        thr_text_white=0.35,
                        tick_label_size=14,
                        cell_text_size=12,
                        cbar_label_size=14,
                        cbar_tick_size=12):
    """
    Heatmap rectangular (filas != columnas):
      - Quita SOLO la fila `drop_row`
      - Quita SOLO la columna `drop_col`
      - Muestra triángulo inferior + diagonal
      - Texto blanco si |r| > thr_text_white
      - SIN título si title=None
    """

    rows = [v for v in df_r.index.tolist() if v != drop_row]
    cols = [v for v in df_r.columns.tolist() if v != drop_col]

    df_rp = df_r.loc[rows, cols]
    df_pp = df_p_use.loc[rows, cols]

    mat = df_rp.to_numpy(dtype=float)
    Pr, Pc = mat.shape

    fig, ax = plt.subplots(figsize=(0.6 * Pc + 6, 0.6 * Pr + 5))

    rr = np.arange(Pr)[:, None]
    cc = np.arange(Pc)[None, :]
    mask = cc > rr
    mat_plot = mat.copy()
    mat_plot[mask] = np.nan

    im = ax.imshow(mat_plot, vmin=VMIN, vmax=VMAX, cmap=CMAP, aspect="equal")

    ax.set_xticks(np.arange(Pc))
    ax.set_yticks(np.arange(Pr))
    ax.set_xticklabels(df_rp.columns.tolist(), rotation=45, ha="right",
                       fontsize=tick_label_size)
    ax.set_yticklabels(df_rp.index.tolist(),
                       fontsize=tick_label_size)

    if title is not None:
        ax.set_title(title, fontsize=18)

    # grid
    ax.set_xticks(np.arange(-.5, Pc, 1), minor=True)
    ax.set_yticks(np.arange(-.5, Pr, 1), minor=True)
    ax.grid(which="minor", color="#dddddd", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    # anotaciones
    for i in range(Pr):
        for j in range(Pc):
            if j <= i:
                r = df_rp.iat[i, j]
                p = df_pp.iat[i, j]
                if np.isfinite(r):
                    txt_color = "white" if abs(r) > thr_text_white else "black"
                    ax.text(j, i, f"{r:.2f}{stars_from_p(p)}",
                            ha="center", va="center",
                            fontsize=cell_text_size,
                            color=txt_color)

    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman r", fontsize=cbar_label_size)
    cbar.ax.tick_params(labelsize=cbar_tick_size)

    plt.tight_layout()
    if out_png is not None:
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()



# ---------------- REQUIRED OBJECTS CHECK ----------------
missing_globals = [name for name in ["df_out", "out_dir", "base_name"] if name not in globals()]
if missing_globals:
    raise NameError(
        f"Faltan variables en tu sesión: {missing_globals}. "
        "Este script asume que ya corriste tu pipeline y existen df_out, out_dir, base_name."
    )

# ---------------- RUN: ALL SUBJECTS ----------------
df_r_all, df_p_all, df_padj_all = compute_spearman_block(df_out, label="ALL")
p_used_all = df_padj_all if USE_FDR else df_p_all

title_cov = f" | covars={','.join(covars)}" if DO_PARTIAL else ""
title_p = "p_FDR" if USE_FDR else "p"
png_all = os.path.join(out_dir, f"{base_name}_lowertri_spearman_ALL.png")

plot_lower_triangle(
    df_r_all, p_used_all,
    out_png=png_all
)
print("[DONE] ALL heatmap:", png_all)

# ---------------- RUN: BY CLUSTER ----------------
if CLUSTER_COL not in df_out.columns:
    raise ValueError(f"No existe {CLUSTER_COL} en df_out. Revisa el nombre de tu columna de clusters.")

clusters = sorted(pd.unique(df_out[CLUSTER_COL].dropna()))
sheets = {
    "ALL_r": df_r_all,
    "ALL_p": df_p_all,
    ("ALL_pFDR" if USE_FDR else "ALL_p_used"): p_used_all
}

for cl in clusters:
    block = df_out[df_out[CLUSTER_COL] == cl].copy()
    if len(block) < 5:
        print(f"[WARN] Cluster {cl}: N={len(block)} muy pequeño, salto.")
        continue

    df_r, df_p, df_padj = compute_spearman_block(block, label=f"cluster_{cl}")
    p_used = df_padj if USE_FDR else df_p

    png_cl = os.path.join(out_dir, f"{base_name}_lowertri_spearman_{CLUSTER_COL}_{cl}.png")
    plot_lower_triangle(
        df_r, p_used,
        out_png=png_cl
    )
    print(f"[DONE] Cluster {cl} heatmap:", png_cl)

    # Excel sheets (<=31 chars)
    sheets[f"cl{cl}_r"[:31]] = df_r
    sheets[f"cl{cl}_p"[:31]] = df_p
    sheets[(f"cl{cl}_pFDR" if USE_FDR else f"cl{cl}_pused")[:31]] = p_used

# ---------------- SAVE EXCEL ----------------
with pd.ExcelWriter(xlsx_out, engine="openpyxl") as writer:
    for name, df_sheet in sheets.items():
        df_sheet.to_excel(writer, sheet_name=str(name)[:31])

print("[DONE] Excel guardado en:", xlsx_out)


# ============================================================
# ML CLASIFICACION + ROC (OOF) PARA TODOS LOS SUJETOS (REPEATED CV)
# - LightGBM multiclass
# - 5-fold x 20 repeticiones (OOF mean)
# - ROC global + (opcional) ROC por Gpo
# - Feature importance (gain)
# - SHAP: guarda BAR + BEESWARM (summary_plot) de forma ROBUSTA
# - Limpia nombres quitando num__/cat__
# ============================================================

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, auc
from matplotlib.cm import get_cmap

# ---------------- LightGBM + SHAP ----------------
try:
    from lightgbm import LGBMClassifier
except Exception as e:
    raise ImportError("No puedo importar lightgbm. Instala con: pip install lightgbm") from e

try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False
    warnings.warn("No se pudo importar shap. SHAP se omitirá. Instala con: pip install shap")


# ============================================================
# ---------------- CONFIG (AJUSTA ESTO) ----------------
# ============================================================
# Deben existir ANTES de correr:
# df_out: DataFrame con features + target + columnas de reporte
# out_dir: carpeta salida
# base_name: prefijo para archivos de salida

TARGET = "cluster_gmm_auto"
GPO_COL = "Gpo"          # SOLO para reportes (NO feature)
SUBJ_COL = "subject_id"  # SOLO para split si hay repetidos (NO feature)

FEATURES = [ "∆T1", "∆GFA","∆QA","∆ISO","∆ha","∆ad","∆fa","∆rd","∆rd1","∆rd2",
"∆fALFF","∆Hurst","∆Entropy",
"∆Power slope","∆Autocor", "∆WMH", "Coord-x", "Coord-y", "Coord-z", 
"Age", "Sex", "WMH_t0", "Size", "Education"
]

N_SPLITS = 5
N_REPEATS = 20
BASE_SEED = 42
RANDOM_STATE = 42

# Bootstrap ROC
N_BOOT = 1000
GRID_N = 200
SEED = 42

DO_ROC_BY_GPO = True
DO_FEATURE_IMPORTANCE_AND_SHAP = True

MAX_SHAP_SAMPLES = 1500
SAVE_SHAP_PER_CLASS = True

# Outputs
os.makedirs(out_dir, exist_ok=True)

roc_png_all = os.path.join(out_dir, f"{base_name}_ROC_OOF_ALLsubjects_rep{N_REPEATS}.png")
oof_xlsx = os.path.join(out_dir, f"{base_name}_OOF_predictions_{TARGET}_rep{N_REPEATS}.xlsx")

featimp_xlsx = os.path.join(out_dir, f"{base_name}_FeatureImportance_LGBM_{TARGET}.xlsx")
featimp_png  = os.path.join(out_dir, f"{base_name}_FeatureImportance_LGBM_{TARGET}.png")

shap_dir = os.path.join(out_dir, f"{base_name}_SHAP_{TARGET}")
os.makedirs(shap_dir, exist_ok=True)


# ============================================================
# ---------------- CHECKS ----------------
# ============================================================
needed = [TARGET, GPO_COL, SUBJ_COL] + FEATURES
missing = [c for c in needed if c not in df_out.columns]
if missing:
    raise ValueError(f"Faltan columnas en df_out: {missing}")

df_ml = df_out.copy()
df_ml = df_ml[df_ml[TARGET].notna()].reset_index(drop=True)

X = df_ml[FEATURES].copy()
y = df_ml[TARGET].copy()
groups = df_ml[SUBJ_COL].copy()
gpo_vec = df_ml[GPO_COL].copy()

classes_global = np.array(sorted(pd.unique(y)))
K = len(classes_global)
if K < 2:
    raise ValueError("El target tiene <2 clases. No se puede clasificar.")

print(f"[INFO] N={len(df_ml)} | K={K} clases: {classes_global}")


# ============================================================
# ---------------- PREPROCESS + MODEL ----------------
# ============================================================
cat_cols = [c for c in ["Sex"] if c in FEATURES]
num_cols = [c for c in FEATURES if c not in cat_cols]

try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), num_cols),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe)
        ]), cat_cols),
    ],
    remainder="drop"
)

clf = LGBMClassifier(
    objective="multiclass",
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=0.0,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

pipe = Pipeline(steps=[("prep", preprocess), ("clf", clf)])


# ============================================================
# ---------------- HELPERS ----------------
# ============================================================

def auc_ovr_macro(y_true, proba, labels):
    return roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=labels)

def align_proba_to_global(fold_classes, proba_fold, global_classes):
    out = np.full((proba_fold.shape[0], len(global_classes)), np.nan, float)
    idx_map = {c: i for i, c in enumerate(global_classes)}
    for j, c in enumerate(fold_classes):
        out[:, idx_map[c]] = proba_fold[:, j]
    return out

def strip_prefixes(feat_names):
    cleaned = []
    for f in feat_names:
        s = str(f)
        s = re.sub(r"^(num|cat)__", "", s)
        s = re.sub(r"^(num|cat)__[^_]+__", "", s)  # por si aparece num__pipeline__x
        s = s.replace("onehot__", "")
        s = s.replace("imputer__", "")
        cleaned.append(s)
    return np.array(cleaned, dtype=object)

def get_feature_names_from_preprocess(prep: ColumnTransformer):
    try:
        names = prep.get_feature_names_out()
    except Exception:
        out = []
        for name, trans, cols in prep.transformers_:
            if name == "remainder" and trans == "drop":
                continue
            if hasattr(trans, "get_feature_names_out"):
                try:
                    out.extend(list(trans.get_feature_names_out(cols)))
                except Exception:
                    out.extend([f"{name}__{c}" for c in cols])
            else:
                out.extend([f"{name}__{c}" for c in cols])
        names = np.array(out, dtype=object)

    return strip_prefixes(names)

def build_splits(X, y, groups, n_splits, seed):
    has_repeats = groups.duplicated().any()
    if has_repeats:
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            splits = list(splitter.split(X, y, groups=groups))
            return splits, "StratifiedGroupKFold", True
        except Exception:
            splitter = GroupKFold(n_splits=n_splits)
            splits = list(splitter.split(X, y, groups=groups))
            print("[WARN] No hay StratifiedGroupKFold. Uso GroupKFold (NO estratifica).")
            return splits, "GroupKFold", True
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(X, y))
        return splits, "StratifiedKFold", False

def plot_multiclass_roc_with_se(y_true, proba, classes, title, out_png,
                                auc_macro_value=None, n_boot=1000, grid_n=200, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, float)

    fpr_grid = np.linspace(0, 1, grid_n)
    cmap = get_cmap("viridis", len(classes))

    plt.figure(figsize=(8, 6))

    for i, cl in enumerate(classes):
        y_i = (y_true == cl).astype(int)
        s_i = proba[:, i]
        if len(np.unique(y_i)) < 2:
            continue

        fpr_obs, tpr_obs, _ = roc_curve(y_i, s_i)
        auc_obs = auc(fpr_obs, tpr_obs)

        idx_all = np.arange(len(y_i))
        auc_boot = []
        tpr_boot = []

        for _ in range(n_boot):
            idx = rng.choice(idx_all, size=len(idx_all), replace=True)
            y_b = y_i[idx]
            s_b = s_i[idx]
            if len(np.unique(y_b)) < 2:
                continue
            fpr_b, tpr_b, _ = roc_curve(y_b, s_b)
            auc_boot.append(auc(fpr_b, tpr_b))

            tpr_interp = np.interp(fpr_grid, fpr_b, tpr_b)
            tpr_interp[0] = 0.0
            tpr_boot.append(tpr_interp)

        color = cmap(i)
        if len(auc_boot) < 10:
            plt.plot(fpr_obs, tpr_obs, color=color, lw=2, label=f"{cl}  AUC={auc_obs:.3f}")
            continue

        auc_boot = np.asarray(auc_boot, float)
        tpr_boot = np.asarray(tpr_boot, float)

        auc_se = auc_boot.std(ddof=1)
        tpr_mean = tpr_boot.mean(axis=0)
        tpr_se = tpr_boot.std(axis=0, ddof=1) / np.sqrt(tpr_boot.shape[0])

        plt.plot(fpr_grid, tpr_mean, color=color, lw=2,
                 label=f"{cl}  AUC={auc_obs:.3f} ± {auc_se:.3f} (SE)")
        plt.fill_between(
            fpr_grid,
            np.clip(tpr_mean - tpr_se, 0, 1),
            np.clip(tpr_mean + tpr_se, 0, 1),
            color=color, alpha=0.18, linewidth=0
        )

    plt.plot([0, 1], [0, 1], linestyle="--", lw=1)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")

    if auc_macro_value is not None:
        plt.title(title + f"\nAUC macro OVR (OOF mean) = {auc_macro_value:.3f}")
    else:
        plt.title(title)

    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print("[DONE] ROC guardada:", out_png)

def compute_shap_global_and_per_class(shap_values, K_expected=None):
    if isinstance(shap_values, list):
        per_class = [np.mean(np.abs(sv), axis=0) for sv in shap_values]  # K x (p,)
        shap_global = np.mean(np.vstack(per_class), axis=0)              # (p,)
        return shap_global, per_class

    sv = np.asarray(shap_values)
    if sv.ndim == 3:
        shap_global = np.mean(np.abs(sv), axis=(0, 2))  # (p,)
        per_class = [np.mean(np.abs(sv[:, :, k]), axis=0) for k in range(sv.shape[2])]
        if K_expected is not None and sv.shape[2] != K_expected:
            print(f"[WARN] SHAP devolvió K={sv.shape[2]} clases, esperaba K={K_expected}.")
        return shap_global, per_class
    elif sv.ndim == 2:
        shap_global = np.mean(np.abs(sv), axis=0)
        return shap_global, None
    else:
        raise ValueError(f"Formato shap_values inesperado: shape={sv.shape}")

def save_shap_summary_plot(shap_vals_2d, X_2d, feat_names, out_png, max_display=30):
    """
    Guardado robusto de SHAP summary_plot (beeswarm):
    - SHAP crea su propia figura interna -> usamos plt.gcf() luego y guardamos.
    """
    try:
        # IMPORTANTE: no crear figure antes; dejar que SHAP la cree
        shap.summary_plot(
            shap_vals_2d, X_2d,
            feature_names=feat_names,
            max_display=max_display,
            show=False
        )
        fig = plt.gcf()
        fig.set_size_inches(10, 7)
        fig.tight_layout()
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("[DONE] SHAP summary (beeswarm) guardado:", out_png)
    except Exception as e:
        print("[WARN] No pude guardar SHAP summary_plot:", repr(e))

def save_shap_bar_plot(shap_vals_2d, X_2d, feat_names, out_png, max_display=30):
    """
    Guardado robusto de SHAP summary_plot tipo BAR.
    """
    try:
        shap.summary_plot(
            shap_vals_2d, X_2d,
            feature_names=feat_names,
            max_display=max_display,
            plot_type="bar",
            show=False
        )
        fig = plt.gcf()
        fig.set_size_inches(10, 7)
        fig.tight_layout()
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("[DONE] SHAP summary (bar) guardado:", out_png)
    except Exception as e:
        print("[WARN] No pude guardar SHAP bar summary_plot:", repr(e))


# ============================================================
# ---------------- OOF REPEATED CV ----------------
# ============================================================

proba_oof_sum = np.zeros((len(df_ml), K), dtype=float)
proba_oof_count = np.zeros((len(df_ml), K), dtype=int)

pred_oof_last = np.full((len(df_ml),), None, object)
auc_per_repeat = []
cv_name_last = None

for rep in range(N_REPEATS):
    seed_rep = BASE_SEED + rep
    splits_list, cv_name, has_repeats = build_splits(X, y, groups, N_SPLITS, seed_rep)
    cv_name_last = cv_name

    print(f"[INFO] Repeat {rep+1}/{N_REPEATS} | CV={cv_name} | seed={seed_rep} | subject_id repeats={has_repeats}")

    proba_oof_rep = np.full((len(df_ml), K), np.nan, float)
    pred_oof_rep = np.full((len(df_ml),), None, object)

    for fold, (tr, te) in enumerate(splits_list, start=1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]

        pipe.set_params(clf__random_state=seed_rep)
        pipe.fit(Xtr, ytr)

        proba = pipe.predict_proba(Xte)
        fold_classes = pipe.named_steps["clf"].classes_

        proba_aligned = align_proba_to_global(fold_classes, proba, classes_global)
        proba_oof_rep[te, :] = proba_aligned
        pred_oof_rep[te] = pipe.predict(Xte)

    if not np.isfinite(proba_oof_rep).all():
        raise RuntimeError("OOF proba (rep) tiene NaNs. Suele pasar si algún fold no vio alguna clase.")

    auc_rep = auc_ovr_macro(y, proba_oof_rep, labels=classes_global)
    auc_per_repeat.append(float(auc_rep))
    print(f"[DONE] Repeat {rep+1}: OOF AUC macro OVR = {auc_rep:.4f}")

    mask_finite = np.isfinite(proba_oof_rep)
    proba_oof_sum[mask_finite] += proba_oof_rep[mask_finite]
    proba_oof_count[mask_finite] += 1

    pred_oof_last = pred_oof_rep

with np.errstate(divide="ignore", invalid="ignore"):
    proba_oof_mean = proba_oof_sum / np.maximum(proba_oof_count, 1)

if not np.isfinite(proba_oof_mean).all():
    raise RuntimeError("proba_oof_mean tiene NaNs/inf. Revisa folds con clases faltantes.")

auc_oof_mean = auc_ovr_macro(y, proba_oof_mean, labels=classes_global)
print(f"[DONE] OOF AUC global (macro OVR) usando promedio de {N_REPEATS} repeticiones: {auc_oof_mean:.4f}")
print(f"[INFO] AUC por repetición: mean={np.mean(auc_per_repeat):.4f} | sd={np.std(auc_per_repeat, ddof=1):.4f}")


# ============================================================
# ROC GLOBAL usando OOF mean
# ============================================================

plot_multiclass_roc_with_se(
    y_true=y,
    proba=proba_oof_mean,
    classes=classes_global,
    title=f"ROC by CLUSTER - TODOS | CV={cv_name_last} | {N_REPEATS} repeats",
    out_png=roc_png_all,
    auc_macro_value=auc_oof_mean,
    n_boot=N_BOOT,
    grid_n=GRID_N,
    seed=SEED
)


# ============================================================
# (OPCIONAL) ROC por Gpo SIN RE-ENTRENAR (filtra OOF mean)
# ============================================================

if DO_ROC_BY_GPO:
    for g in pd.unique(gpo_vec.dropna()):
        m = (np.asarray(gpo_vec) == g)
        n = int(m.sum())
        if n < 30 or len(np.unique(np.asarray(y)[m])) < 2:
            print(f"[SKIP] Gpo={g}: N={n} o 1 clase en ese subgrupo.")
            continue
        out_png = os.path.join(out_dir, f"{base_name}_ROC_OOFmean_Gpo_{g}_rep{N_REPEATS}.png")
        plot_multiclass_roc_with_se(
            y_true=np.asarray(y)[m],
            proba=proba_oof_mean[m, :],
            classes=classes_global,
            title=f"ROC by CLUSTER | Gpo={g} | N={n} | {N_REPEATS} repeats",
            out_png=out_png,
            auc_macro_value=None,
            n_boot=N_BOOT,
            grid_n=GRID_N,
            seed=SEED
        )


# ============================================================
# Feature importance + SHAP (modelo FINAL entrenado en TODO)
# ============================================================

if DO_FEATURE_IMPORTANCE_AND_SHAP:
    pipe.set_params(clf__random_state=RANDOM_STATE)
    pipe.fit(X, y)

    prep_fitted = pipe.named_steps["prep"]
    clf_fitted  = pipe.named_steps["clf"]
    feat_names  = get_feature_names_from_preprocess(prep_fitted)

    # --- Feature importance (gain) ---
    booster = clf_fitted.booster_
    imp_gain = booster.feature_importance(importance_type="gain")

    fi = pd.DataFrame({
        "feature": feat_names,
        "importance_gain": imp_gain
    }).sort_values("importance_gain", ascending=False).reset_index(drop=True)

    with pd.ExcelWriter(featimp_xlsx, engine="openpyxl") as writer:
        fi.to_excel(writer, index=False, sheet_name="feature_importance_gain")

    topn = min(30, len(fi))
    plt.figure(figsize=(8, 10))
    plt.barh(fi.loc[:topn-1, "feature"][::-1], fi.loc[:topn-1, "importance_gain"][::-1])
    plt.xlabel("Importance (gain)")
    plt.title(f"LightGBM Feature Importance (gain) - Top {topn}")
    plt.tight_layout()
    plt.savefig(featimp_png, dpi=300, bbox_inches="tight")
    plt.close()

    print("[DONE] Feature importance guardada:", featimp_xlsx)
    print("[DONE] Feature importance plot:", featimp_png)

    # --- SHAP robusto multiclase + plots robustos ---
    if _HAS_SHAP:
        X_tr = prep_fitted.transform(X)

        rng = np.random.default_rng(SEED)
        if X_tr.shape[0] > MAX_SHAP_SAMPLES:
            idx = rng.choice(np.arange(X_tr.shape[0]), size=MAX_SHAP_SAMPLES, replace=False)
            X_sh = X_tr[idx]
        else:
            X_sh = X_tr

        explainer = shap.TreeExplainer(clf_fitted)
        shap_values = explainer.shap_values(X_sh)

        # global + por clase (para excel)
        shap_global, per_class = compute_shap_global_and_per_class(shap_values, K_expected=K)

        shap_imp = pd.DataFrame({
            "feature": feat_names,
            "mean_abs_shap": shap_global
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        shap_xlsx = os.path.join(shap_dir, f"SHAP_importance_{base_name}_{TARGET}.xlsx")
        with pd.ExcelWriter(shap_xlsx, engine="openpyxl") as writer:
            shap_imp.to_excel(writer, index=False, sheet_name="shap_global")
            fi.to_excel(writer, index=False, sheet_name="lgbm_gain_importance")

            if SAVE_SHAP_PER_CLASS and per_class is not None:
                for k_idx, cl in enumerate(classes_global):
                    dfk = pd.DataFrame({
                        "feature": feat_names,
                        "mean_abs_shap": per_class[k_idx]
                    }).sort_values("mean_abs_shap", ascending=False)
                    dfk.to_excel(writer, index=False, sheet_name=f"class_{cl}"[:31])

        print("[DONE] SHAP guardado en:", shap_xlsx)

        # ---- PLOTS TIPO SHAP (los que te faltan) ----
        # Elegimos una matriz 2D de shap values para graficar (clase 0 por defecto).
        if isinstance(shap_values, list):
            shap2d = shap_values[0]
        else:
            sv = np.asarray(shap_values)
            shap2d = sv[:, :, 0] if sv.ndim == 3 else sv

        # 1) Beeswarm (summary_plot clásico)
        shap_bee_png = os.path.join(shap_dir, f"SHAP_beeswarm_{base_name}_{TARGET}.png")
        save_shap_summary_plot(shap2d, X_sh, feat_names, shap_bee_png, max_display=30)

        # 2) Bar (summary_plot plot_type="bar")
        shap_sumbar_png = os.path.join(shap_dir, f"SHAP_summaryBar_{base_name}_{TARGET}.png")
        save_shap_bar_plot(shap2d, X_sh, feat_names, shap_sumbar_png, max_display=30)

        # (Opcional) Beeswarm por clase
        if SAVE_SHAP_PER_CLASS and per_class is not None:
            try:
                for k_idx, cl in enumerate(classes_global):
                    if isinstance(shap_values, list):
                        shap_k = shap_values[k_idx]
                    else:
                        shap_k = np.asarray(shap_values)[:, :, k_idx]
                    outp = os.path.join(shap_dir, f"SHAP_beeswarm_class_{cl}_{base_name}_{TARGET}.png")
                    save_shap_summary_plot(shap_k, X_sh, feat_names, outp, max_display=30)
                print("[DONE] SHAP beeswarm por clase guardados en:", shap_dir)
            except Exception as e:
                print("[WARN] No pude generar beeswarm por clase:", repr(e))

    else:
        print("[INFO] SHAP omitido (no está instalado).")


# ============================================================
# Guardar OOF predictions (debug) + summary
# ============================================================

oof_df = df_ml[[TARGET, GPO_COL, SUBJ_COL]].copy()
oof_df["pred_oof_last_repeat"] = pred_oof_last

for i, c in enumerate(classes_global):
    oof_df[f"proba_oof_mean_{c}"] = proba_oof_mean[:, i]

summary_df = pd.DataFrame([{
    "target": TARGET,
    "cv": cv_name_last,
    "n_splits": N_SPLITS,
    "n_repeats": N_REPEATS,
    "AUC_OOF_macro_OVR_meanProba": float(auc_oof_mean),
    "AUC_per_repeat_mean": float(np.mean(auc_per_repeat)),
    "AUC_per_repeat_sd": float(np.std(auc_per_repeat, ddof=1)) if len(auc_per_repeat) > 1 else 0.0,
    "N": int(len(df_ml)),
    "classes": ", ".join(map(str, classes_global))
}])

auc_rep_df = pd.DataFrame({
    "repeat": np.arange(1, N_REPEATS + 1),
    "auc_oof_macro_ovr": auc_per_repeat
})

with pd.ExcelWriter(oof_xlsx, engine="openpyxl") as writer:
    summary_df.to_excel(writer, index=False, sheet_name="summary")
    auc_rep_df.to_excel(writer, index=False, sheet_name="auc_by_repeat")
    oof_df.to_excel(writer, index=False, sheet_name="oof_mean")

print("[DONE] OOF (mean over repeats) guardado en:", oof_xlsx)
print("[INFO] SHAP dir:", shap_dir)


