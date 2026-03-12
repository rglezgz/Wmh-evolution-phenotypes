# -*- coding: utf-8 -*-
"""
GMM GROUP MAP + COUNT MAPS (3 CLUSTERS) ✅ + PLOTS ROJOS (sin titulo)
- Plots individuales por cluster
- Colorbar empieza en 0 (NO aparece -3)
- Escala tipo WMH_sum_axial:
    * cmap Reds truncado 0.30->1.0
    * vmax = p99 de data>=0 (y al menos 10)
    * threshold=0
- ✅ SWAP de clusters según:
    map_swap = {0:0, 1:2, 2:1}
  (se aplica al cluster del Excel ANTES de mapear a 1..3)
"""

import os, glob, re
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage as ndi

import matplotlib as mpl
from matplotlib import colors


# -------------------------
# Helpers
# -------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def normalize_id(s: str) -> str:
    return str(s).strip().lower().replace(" ", "")

def extract_sub_id_from_filename(path: str):
    name = os.path.basename(path).lower()
    m = re.search(r"(sub-[a-z0-9]+)", name)
    return m.group(1) if m else None

def label_clusters_26(binary):
    structure = ndi.generate_binary_structure(rank=3, connectivity=3)  # 26 vecinos
    labeled, _ = ndi.label(binary.astype(np.uint8), structure=structure)
    return labeled

def same_space(img_a, img_b, affine_tol=1e-3):
    if img_a.shape != img_b.shape:
        return False
    return np.allclose(img_a.affine, img_b.affine, atol=affine_tol, rtol=0)

def sanitize_labels_to_0_3(arr):
    out = np.asarray(arr).astype(np.int16, copy=True)
    valid = (out >= 0) & (out <= 3)
    out[~valid] = 0
    return out

def cluster_colors_L123():
    base = mpl.colormaps.get_cmap("viridis")
    return {1: base(0.95), 2: base(0.55), 3: base(0.15)}

def make_listed_cmap_bg_plus_3clusters_blackbg():
    bg = (0, 0, 0, 1)
    cl = cluster_colors_L123()
    cols = [bg, cl[1], cl[2], cl[3]]
    return colors.ListedColormap(cols, name="clusters_L123_blackbg")


# -------------------------
# Core: build group map + count maps
# -------------------------
def build_group_map(
    excel_path,
    masks_folder,
    ref_path,
    out_dir,
    sheet_index=0,
    subject_col="subject_id",
    accepted_col="accepted",
    peak_x_col="Coord-x",
    peak_y_col="Coord-y",
    peak_z_col="Coord-z",
    type_col="cluster_gmm_auto",
    collision_rule="majority",     # "majority", "first", "max"
    strict_space_check=True,
    affine_tol=1e-3,
    tie_break="max",               # empates majority: "max" o "min"
    map_swap=None                  # ✅ swap clusters 0..2
):
    """
    Excel trae clusters 0..2.
    Aplicamos swap si map_swap != None.
    Guardamos en mapa final:
      0 = background
      1 = cluster0 (L1)
      2 = cluster1 (L2)
      3 = cluster2 (L3)

    Además guarda mapas de conteo por cluster (L1,L2,L3).
    """
    ensure_dir(out_dir)
    df = pd.read_excel(excel_path, sheet_name=sheet_index)

    if accepted_col in df.columns:
        df = df[df[accepted_col] == 1].copy()

    required = [subject_col, peak_x_col, peak_y_col, peak_z_col, type_col]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Falta columna '{c}'. Columnas: {list(df.columns)}")

    df[subject_col] = df[subject_col].apply(normalize_id)
    df[type_col] = pd.to_numeric(df[type_col], errors="coerce")
    if df[type_col].isna().any():
        bad_n = int(df[type_col].isna().sum())
        raise ValueError(f"{type_col} tiene {bad_n} valores NaN/no numéricos.")
    df[type_col] = df[type_col].astype(int)

    if df[type_col].min() < 0 or df[type_col].max() > 2:
        raise ValueError(f"{type_col} debe estar en 0..2. Min={df[type_col].min()} Max={df[type_col].max()}")

    if map_swap is None:
        map_swap = {0: 0, 1: 1, 2: 2}

    ref_img = nib.as_closest_canonical(nib.load(ref_path))

    # Conteos por cluster (L1,L2,L3)
    count_maps = np.zeros(ref_img.shape + (3,), dtype=np.uint16)

    if collision_rule == "majority":
        vote_counts = np.zeros(ref_img.shape + (3,), dtype=np.uint16)
        group_map = None
    else:
        vote_counts = None
        group_map = np.zeros(ref_img.shape, dtype=np.int16)

    mask_files = sorted(
        glob.glob(os.path.join(masks_folder, "*.nii")) +
        glob.glob(os.path.join(masks_folder, "*.nii.gz"))
    )
    if not mask_files:
        raise RuntimeError(f"No encontré NIfTI en {masks_folder}")

    masks_by_subid = {}
    for f in mask_files:
        subid = extract_sub_id_from_filename(f)
        if subid:
            subid = normalize_id(subid)
            if subid not in masks_by_subid:
                masks_by_subid[subid] = f

    print("[INFO] total masks indexadas:", len(masks_by_subid))

    missing_masks = 0
    subjects_used = 0
    skipped_space = 0

    def combine_subj(subj_type_0_3):
        nonlocal group_map, vote_counts, count_maps
        subj_type_0_3 = sanitize_labels_to_0_3(subj_type_0_3)

        # acumular conteos por voxel y cluster
        for v in (1, 2, 3):
            mv = (subj_type_0_3 == v)
            if np.any(mv):
                count_maps[..., v-1][mv] += 1

        # regla de colisión original
        if collision_rule == "first":
            m = subj_type_0_3 > 0
            w = m & (group_map == 0)
            group_map[w] = subj_type_0_3[w]

        elif collision_rule == "max":
            m = subj_type_0_3 > 0
            group_map[m] = np.maximum(group_map[m], subj_type_0_3[m])

        elif collision_rule == "majority":
            for v in (1, 2, 3):
                mv = (subj_type_0_3 == v)
                if np.any(mv):
                    vote_counts[..., v-1][mv] += 1
        else:
            raise ValueError("collision_rule debe ser 'majority', 'first' o 'max'")

    for subid, df_s in df.groupby(subject_col):
        mask_path = masks_by_subid.get(subid)
        if mask_path is None:
            missing_masks += 1
            continue

        subj_img = nib.as_closest_canonical(nib.load(mask_path))

        if not same_space(subj_img, ref_img, affine_tol=affine_tol):
            msg = (f"[SPACE MISMATCH] {subid} => {mask_path}\n"
                   f"  subj shape={subj_img.shape}, ref shape={ref_img.shape}\n"
                   f"  subj affine!=ref affine (tol={affine_tol})")
            if strict_space_check:
                raise RuntimeError(msg)
            else:
                print(msg)
                skipped_space += 1
                continue

        subj_bin = (subj_img.get_fdata() > 0).astype(np.uint8)
        labeled = label_clusters_26(subj_bin)

        subj_type = np.zeros_like(labeled, dtype=np.int16)
        wrote = False

        for _, row in df_s.iterrows():
            x = int(row[peak_x_col]); y = int(row[peak_y_col]); z = int(row[peak_z_col])
            if not (0 <= x < labeled.shape[0] and 0 <= y < labeled.shape[1] and 0 <= z < labeled.shape[2]):
                continue

            lab = int(labeled[x, y, z])
            if lab == 0:
                continue

            # ✅ SWAP aplicado al cluster del Excel (0..2)
            t_raw = int(row[type_col])           # 0..2
            t = map_swap.get(t_raw, t_raw)       # swap (0..2)
            if t < 0 or t > 2:
                continue

            v = t + 1                            # 1..3 (0=bg)
            subj_type[labeled == lab] = v
            wrote = True

        if not wrote:
            continue

        subjects_used += 1
        combine_subj(subj_type)

    if collision_rule == "majority":
        max_counts = vote_counts.max(axis=3)
        has_votes = max_counts > 0
        candidates = (vote_counts == max_counts[..., None]) & has_votes[..., None]

        group_map = np.zeros(ref_img.shape, dtype=np.int16)

        if tie_break == "max":
            for cls in (3, 2, 1):
                m = candidates[..., cls-1]
                group_map[m] = cls
        elif tie_break == "min":
            for cls in (1, 2, 3):
                m = candidates[..., cls-1]
                w = m & (group_map == 0)
                group_map[w] = cls
        else:
            raise ValueError("tie_break debe ser 'max' o 'min'")

    group_map = sanitize_labels_to_0_3(group_map)

    out_nii = os.path.join(out_dir, "WMH_group_by_gmm_type_L123.nii.gz")
    nib.save(nib.Nifti1Image(group_map.astype(np.int16), ref_img.affine, ref_img.header), out_nii)

    out_count_L1 = os.path.join(out_dir, "WMH_count_L1_cluster0.nii.gz")
    out_count_L2 = os.path.join(out_dir, "WMH_count_L2_cluster1.nii.gz")
    out_count_L3 = os.path.join(out_dir, "WMH_count_L3_cluster2.nii.gz")

    nib.save(nib.Nifti1Image(count_maps[..., 0].astype(np.uint16), ref_img.affine, ref_img.header), out_count_L1)
    nib.save(nib.Nifti1Image(count_maps[..., 1].astype(np.uint16), ref_img.affine, ref_img.header), out_count_L2)
    nib.save(nib.Nifti1Image(count_maps[..., 2].astype(np.uint16), ref_img.affine, ref_img.header), out_count_L3)

    print("[OK] Guardado mapa grupal:", out_nii)
    print("[OK] Guardados conteos:", out_count_L1, out_count_L2, out_count_L3)
    print("[INFO] sujetos usados:", subjects_used)
    print("[INFO] sujetos sin máscara:", missing_masks)
    print("[INFO] sujetos saltados por espacio:", skipped_space)
    print("[INFO] collision_rule:", collision_rule)
    print("[INFO] map_swap:", map_swap)

    return out_nii, (out_count_L1, out_count_L2, out_count_L3)


# -------------------------
# Plot: count map per cluster (ROJO) sin titulo, barra empieza en 0
# -------------------------
def plot_count_map_red_like_wmh_sum(count_map_path, out_png, n_slices=12):
    from nilearn import plotting, datasets
    from matplotlib import cm
    from matplotlib import colors as mcolors

    img = nib.as_closest_canonical(nib.load(count_map_path))
    data = np.nan_to_num(img.get_fdata(), nan=0.0).astype(np.float32)

    if not np.any(data > 0):
        raise RuntimeError(f"El mapa {count_map_path} no tiene valores > 0.")

    try:
        bg_img = datasets.load_mni152_template(resolution=2)
    except Exception:
        bg_img = img

    any_mask = (data > 0)
    if np.any(any_mask):
        affine = img.affine
        iz = np.where(any_mask)[2]
        z_min, z_max = int(iz.min()), int(iz.max())
        z_idxs = np.linspace(z_min, z_max, n_slices).astype(int)
        z_mm = [float((affine @ np.array([0, 0, k, 1]))[2]) for k in z_idxs]
    else:
        z_mm = list(np.linspace(-40, 70, n_slices))

    def truncate_cmap(base_cmap, minval=0.30, maxval=1.0, n=256):
        new_colors = base_cmap(np.linspace(minval, maxval, n))
        return mcolors.LinearSegmentedColormap.from_list('trunc_' + base_cmap.name, new_colors)

    cmap_red = truncate_cmap(cm.get_cmap('Reds'), 0.30, 1.0)

    # colorbar desde 0
    vmin = 0.0
    vmax = float(np.percentile(data[data >= vmin], 99)) if np.any(data >= vmin) else 1.0
    vmax = max(vmax, 10.0)

    clean_img = nib.Nifti1Image(data.astype(np.float32), img.affine, img.header)

    d = plotting.plot_stat_map(
        clean_img,
        bg_img=bg_img,
        display_mode='z',
        cut_coords=z_mm,
        threshold=0,
        colorbar=True,
        cmap=cmap_red,
        vmin=vmin,
        vmax=vmax,
        symmetric_cbar=False,
        title=""
    )

    d.savefig(out_png, dpi=300)
    d.close()
    print("[PLOT] Guardado:", out_png)


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    excel_path   = r"F:\long\test\All\SALIDAS\whm_mask_only\WMH_clusters_metrics-all_with_clinic_final_CLUSTERING_GLM.xlsx"
    masks_folder = r"F:\long\test\All\SALIDAS\whm_mask_only\WMH_masks_WMfilter_mean02"
    ref_path     = r"F:\long\test\All\SALIDAS\whm_mask_only\WMH_sum.nii.gz"
    out_dir      = r"F:\long\test\All\SALIDAS\whm_mask_only\GMM_GROUP_PLOT"
    ensure_dir(out_dir)

    # ✅ Tu swap
    map_swap = {0: 0, 1: 2, 2: 1}

    group_map_path, (count_L1, count_L2, count_L3) = build_group_map(
        excel_path=excel_path,
        masks_folder=masks_folder,
        ref_path=ref_path,
        out_dir=out_dir,
        sheet_index=0,
        type_col="cluster_gmm_auto",
        collision_rule="majority",
        tie_break="max",
        strict_space_check=True,
        affine_tol=1e-3,
        map_swap=map_swap
    )

    plot_count_map_red_like_wmh_sum(count_L1, os.path.join(out_dir, "WMH_count_L1_red.png"), n_slices=12)
    plot_count_map_red_like_wmh_sum(count_L2, os.path.join(out_dir, "WMH_count_L2_red.png"), n_slices=12)
    plot_count_map_red_like_wmh_sum(count_L3, os.path.join(out_dir, "WMH_count_L3_red.png"), n_slices=12)
