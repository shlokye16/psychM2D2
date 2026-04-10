"""
OASIS-2 Structural MRI Processing Pipeline
===========================================
Corrected for actual dataset structure observed in OASIS-2 downloads:

  MRI format:   Analyze 7.5 (.hdr + .img pairs) — NOT NIfTI .nii.gz
  Acquisitions: 3–4 mpr-N repeated scans per session → averaged before any processing
  Sessions:     373 total, 150 unique subjects (OAS2_XXXX_MRY folder naming)
  Labels:       Derived from CDR column — NEVER from Group column or folder name
  Parts:        OAS2_RAW_PART1 + OAS2_RAW_PART2 — check both when locating sessions

Setup (run once before this script):
    Download OASIS-2 zips from: https://sites.wustl.edu/oasisbrains/
    Extract to:
        data/raw/oasis2/OAS2_RAW_PART1/
        data/raw/oasis2/OAS2_RAW_PART2/
    Place demographics file at:
        data/metadata/oasis_longitudinal_demographics.xlsx

Install dependencies:
    pip install nibabel nilearn numpy scipy pandas scikit-image torch openpyxl
    # FSL required for skull stripping + registration (external, install separately)
    # https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation
    # If FSL is unavailable, the pipeline falls back to nilearn-only preprocessing.

Label mapping from CDR:
    CDR 0.0  →  "Control"   (label 0)
    CDR 0.5  →  "MCI"       (label 1) — validated MCI proxy in OASIS literature
    CDR 1.0+ →  "AD"        (label 2) — CDR 1.0 and 2.0 merged

Converter subjects (Group == "Converted") are flagged separately — these
subjects started nondemented and converted during the study. They are your
most analytically valuable subgroup for early-detection framing.

Subject numbering: OAS2_0001 through OAS2_0150 (some sessions may be missing).
MRI ID format:     OAS2_XXXX_MRY (maps exactly to folder names in PART1/PART2).
"""

import logging
import shutil
import subprocess
import traceback
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from nilearn import datasets, image
from nilearn.maskers import NiftiLabelsMasker
from scipy.ndimage import zoom

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

RAW_DIR      = Path("../data/raw/oasis2")
MRI_FEAT_DIR = Path("../data/processed/mri_features")
TENSOR_DIR   = Path("../data/processed/mri_tensors")
METADATA_DIR = Path("../data/metadata")
SCRATCH_DIR  = Path("../data/processed/mri_scratch")   # intermediate FSL outputs
LOG_DIR      = Path("logs")

DEMOGRAPHICS_XLSX = METADATA_DIR / "oasis_longitudinal_demographics.xlsx"
MANIFEST_CSV      = METADATA_DIR / "oasis2_manifest.csv"
PROCESSING_LOG    = METADATA_DIR / "oasis2_processing_summary.csv"

# Raw subdirectory names — check both parts for every session
PARTS = ["OAS2_RAW_PART1", "OAS2_RAW_PART2"]

# CNN input shape — (D, H, W) after resize
MRI_TARGET_SHAPE = (128, 128, 128)

# Intensity normalization — clip z-scores beyond this after standardizing
INTENSITY_CLIP_SIGMA = 3.0

# Minimum file size to treat a .img as real data (not a stub)
MIN_BYTES = 50_000

# FSL BET fractional intensity — 0.5 is standard, lower = more aggressive
BET_FRAC = 0.5

# Harvard-Oxford atlas ROI configuration
# Using HO instead of AAL — HO is bundled with nilearn (no external download)
# and already cached if you've run the PEARL-Neuro fMRI pipeline.
#
# Two HO variants:
#   sub-maxprob-thr25-2mm  → subcortical atlas — L/R hippocampus + amygdala
#   cort-maxprob-thr25-2mm → cortical atlas — DMN-adjacent regions
#
# Label strings must match nilearn's atlas.labels list exactly.

HO_SUBCORTICAL_ROIS = [
    "Left Hippocampus",
    "Right Hippocampus",
    "Left Amygdala",
    "Right Amygdala",
]

HO_CORTICAL_ROIS = [
    "Frontal Medial Cortex",
    "Cingulate Gyrus, posterior division",
    "Precuneous Cortex",
    "Middle Temporal Gyrus, temporooccipital part",
    "Angular Gyrus",
    "Parahippocampal Gyrus, posterior division",
]

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

for d in [MRI_FEAT_DIR, TENSOR_DIR, METADATA_DIR, SCRATCH_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "oasis2_pipeline.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def is_real_file(path: Path, min_bytes: int = MIN_BYTES) -> bool:
    """
    Check that a file both exists and has real content (not a stub or zero-byte
    placeholder). OASIS-2 .img files are large (usually >10 MB for a T1w scan),
    so anything below min_bytes is clearly incomplete.
    """
    try:
        return path.exists() and path.stat().st_size >= min_bytes
    except OSError:
        return False


def fsl_available() -> bool:
    """Return True if FSL bet/flirt are on PATH."""
    return (
        shutil.which("bet") is not None and
        shutil.which("flirt") is not None
    )


def cdr_to_label(cdr) -> str:
    """
    Map CDR score to 3-class label.
    Always use CDR column — Group column is NOT equivalent.
    CDR 0.5 == 'Very mild dementia' in OASIS terminology == 'MCI' in ours.
    CDR 1.0 and 2.0 are merged into 'AD'.
    """
    if cdr == 0.0:
        return "Control"
    if cdr == 0.5:
        return "MCI"
    return "AD"


def find_session_raw_dir(mri_id: str) -> Path | None:
    """
    Locate the RAW/ subfolder for a session across both PART1 and PART2.
    Returns the Path if found, None otherwise.
    """
    for part in PARTS:
        candidate = RAW_DIR / part / mri_id / "RAW"
        if candidate.exists():
            return candidate
    return None


def load_and_average_acquisitions(raw_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all mpr-N.nifti.hdr acquisitions in a session's RAW/ folder and
    average them together. This is the standard OASIS protocol — multiple
    acquisitions are collected per session to reduce motion artifact, and
    they must be averaged before any subsequent processing.

    Explicitly filters to mpr-*.nifti.hdr to avoid picking up OLD/ folder
    files or any .log files that live alongside the real data.

    Returns:
        averaged  — float32 numpy array of shape (X, Y, Z)
        affine    — 4x4 affine matrix from the first acquisition
    """
    hdr_files = sorted(raw_dir.glob("mpr-*.nifti.hdr"))
    if not hdr_files:
        raise FileNotFoundError(f"No mpr-*.nifti.hdr files in {raw_dir}")

    # Verify at least the .img counterpart exists and has real content
    valid = []
    for hdr in hdr_files:
        img_path = hdr.with_suffix(".img")
        if is_real_file(img_path):
            valid.append(hdr)
        else:
            log.warning(f"Skipping {hdr.name} — .img missing or too small")

    if not valid:
        raise FileNotFoundError(f"No valid .hdr/.img pairs in {raw_dir}")

    volumes = []
    affine  = None
    for hdr_path in valid:
        # nibabel pairs .hdr with .img automatically — never load .img directly
        img = nib.load(str(hdr_path))
        if affine is None:
            affine = img.affine
        arr = img.get_fdata(dtype=np.float32)
        # OASIS-2 Analyze 7.5 files sometimes load as 4D (X, Y, Z, 1) — squeeze
        # to 3D before averaging. np.squeeze removes ALL size-1 dims safely.
        arr = np.squeeze(arr)
        if arr.ndim != 3:
            raise ValueError(f"Unexpected volume shape after squeeze: {arr.shape}")
        volumes.append(arr)

    averaged = np.mean(volumes, axis=0)
    log.info(f"  Averaged {len(volumes)} acquisition(s) → shape {averaged.shape}")
    return averaged, affine


def skull_strip_fsl(input_nii: Path, output_nii: Path):
    """
    Run FSL BET on the averaged volume to remove non-brain tissue.
    Raises subprocess.CalledProcessError if FSL is not installed or fails.
    """
    subprocess.run(
        ["bet", str(input_nii), str(output_nii), "-f", str(BET_FRAC), "-R"],
        check=True, capture_output=True, text=True,
    )


def register_to_mni_fsl(input_nii: Path, output_nii: Path):
    """
    Register brain-extracted volume to MNI152 2mm standard space using
    FSL FLIRT with 12-DOF affine (standard for cross-subject MRI analysis).
    Uses nilearn's bundled MNI152 template as the registration target.
    """
    from nilearn.datasets import load_mni152_template
    mni_template = load_mni152_template(resolution=2)
    mni_path     = SCRATCH_DIR / "_mni152_2mm_template.nii.gz"
    if not mni_path.exists():
        nib.save(mni_template, str(mni_path))

    subprocess.run(
        ["flirt",
         "-in",       str(input_nii),
         "-ref",      str(mni_path),
         "-out",      str(output_nii),
         "-dof",      "12",
         "-interp",   "trilinear"],
        check=True, capture_output=True, text=True,
    )


def register_to_mni_nilearn(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Fallback MNI registration using nilearn's resample_to_img.
    Less accurate than FSL FLIRT but requires no external dependencies.
    Used automatically when FSL is not available.
    """
    from nilearn.datasets import load_mni152_template
    mni_template = load_mni152_template(resolution=2)
    return image.resample_to_img(img, mni_template, interpolation="continuous")


def normalize_intensity(arr: np.ndarray) -> np.ndarray:
    """
    Z-score normalize the volume and clip to ±INTENSITY_CLIP_SIGMA.
    Applied after skull stripping and registration so only brain voxels
    influence the mean/std calculation.

    Brain voxels are estimated as values above zero (post-skull-strip).
    This avoids background zeros from inflating the mean.
    """
    brain_mask = arr > 0
    if brain_mask.sum() < 1000:
        log.warning("Very few brain voxels — normalization may be unreliable")

    mu    = arr[brain_mask].mean() if brain_mask.any() else arr.mean()
    sigma = arr[brain_mask].std()  if brain_mask.any() else arr.std()

    normalized = (arr - mu) / (sigma + 1e-8)
    normalized = np.clip(normalized, -INTENSITY_CLIP_SIGMA, INTENSITY_CLIP_SIGMA)
    return normalized.astype(np.float32)


def resize_volume(arr: np.ndarray, target: tuple = MRI_TARGET_SHAPE) -> np.ndarray:
    """
    Resize a 3D volume to target shape using trilinear zoom (order=1).
    Order 1 is a good balance — faster than cubic (order=3) with acceptable
    quality for CNN input at 128^3.
    """
    factors = tuple(t / s for t, s in zip(target, arr.shape))
    return zoom(arr, factors, order=1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — BUILD MANIFEST (run once before processing)
# ─────────────────────────────────────────────────────────────────────────────

def build_manifest() -> pd.DataFrame:
    """
    Join the demographics Excel file with the actual filesystem to produce
    a manifest CSV that links every MRI ID to its folder path, label, and
    pre-computed features (nWBV, eTIV, MMSE).

    Run this once before processing. It will tell you:
    - Which sessions are present in PART1 vs PART2
    - Which sessions are missing entirely (download incomplete)
    - How many mpr-N acquisitions each session has
    - Label distribution (Control / MCI / AD)
    - Converter subjects (Group == 'Converted')

    Saves to data/metadata/oasis2_manifest.csv and returns the DataFrame.
    """
    if not DEMOGRAPHICS_XLSX.exists():
        raise FileNotFoundError(
            f"{DEMOGRAPHICS_XLSX} not found.\n"
            "Download oasis_longitudinal_demographics.xlsx from:\n"
            "  https://sites.wustl.edu/oasisbrains/"
        )

    df = pd.read_excel(str(DEMOGRAPHICS_XLSX))

    # Normalize column names — the xlsx sometimes has trailing spaces
    df.columns = df.columns.str.strip()

    # ── Label derivation — CDR column only ───────────────────────────────────
    df["label"]          = df["CDR"].apply(cdr_to_label)
    df["label_int"]      = df["label"].map({"Control": 0, "MCI": 1, "AD": 2})
    df["is_converter"]   = df["Group"].str.strip() == "Converted"

    # ── Handle known missing values ───────────────────────────────────────────
    # 19 sessions missing SES, 2 sessions missing MMSE — document don't drop
    df["ses_imputed"]    = df["SES"].isna()
    df["SES"]            = df["SES"].fillna(df["SES"].median())
    df["mmse_missing"]   = df["MMSE"].isna()

    # ── Verify filesystem presence ────────────────────────────────────────────
    records  = []
    missing  = []

    for _, row in df.iterrows():
        mri_id  = str(row["MRI ID"]).strip()
        raw_dir = find_session_raw_dir(mri_id)

        if raw_dir is None:
            missing.append(mri_id)
            continue

        hdr_files = list(raw_dir.glob("mpr-*.nifti.hdr"))
        img_ok    = [h for h in hdr_files if is_real_file(h.with_suffix(".img"))]

        records.append({
            "mri_id":        mri_id,
            "subject_id":    str(row["Subject ID"]).strip(),
            "visit":         row["Visit"],
            "cdr":           row["CDR"],
            "label":         row["label"],
            "label_int":     row["label_int"],
            "is_converter":  row["is_converter"],
            "group":         str(row["Group"]).strip(),
            "age":           row["Age"],
            "sex":           str(row["M/F"]).strip(),
            "mmse":          row["MMSE"],
            "mmse_missing":  row["mmse_missing"],
            "educ":          row.get("EDUC", np.nan),
            "ses":           row["SES"],
            "ses_imputed":   row["ses_imputed"],
            "nWBV":          row["nWBV"],
            "eTIV":          row["eTIV"],
            "ASF":           row["ASF"],
            "n_acquisitions": len(img_ok),
            "raw_path":      str(raw_dir),
            "part":          next(p for p in PARTS if p in str(raw_dir)),
        })

    manifest = pd.DataFrame(records)
    manifest.to_csv(str(MANIFEST_CSV), index=False)

    log.info(f"Sessions found:     {len(manifest)}")
    log.info(f"Sessions missing:   {len(missing)}")
    log.info(f"Label distribution:\n{manifest['label'].value_counts().to_string()}")
    log.info(f"Converters:         {manifest['is_converter'].sum()} sessions")
    log.info(f"In PART1:           {(manifest['part'] == 'OAS2_RAW_PART1').sum()}")
    log.info(f"In PART2:           {(manifest['part'] == 'OAS2_RAW_PART2').sum()}")
    log.info(f"nWBV sanity check (CDR → mean nWBV — expect 0.740 / 0.721 / 0.702):")
    log.info(manifest.groupby("cdr")["nWBV"].mean().round(4).to_string())

    if missing:
        log.warning(f"Missing sessions (first 10): {missing[:10]}")
        log.warning("Check both PART1 and PART2 zip downloads are complete.")

    log.info(f"Manifest saved → {MANIFEST_CSV}")
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD + PREPROCESS MRI (one session)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_mri_session(mri_id: str) -> nib.Nifti1Image | None:
    """
    Full preprocessing pipeline for one OASIS-2 session:
      1. Locate RAW/ folder across PART1 and PART2
      2. Load + average all mpr-N acquisitions
      3. Skull strip (FSL BET if available, else returns un-stripped — log warning)
      4. Register to MNI152 2mm (FSL FLIRT if available, else nilearn fallback)
      5. Intensity normalize (z-score, clipped to ±3 sigma)

    Returns a preprocessed Nifti1Image in MNI space, or None on failure.
    """
    raw_dir = find_session_raw_dir(mri_id)
    if raw_dir is None:
        log.error(f"[{mri_id}] RAW/ directory not found in PART1 or PART2")
        return None

    scratch = SCRATCH_DIR / mri_id
    scratch.mkdir(parents=True, exist_ok=True)

    # ── Load + average acquisitions ───────────────────────────────────────────
    try:
        averaged, affine = load_and_average_acquisitions(raw_dir)
    except Exception as e:
        log.error(f"[{mri_id}] Acquisition averaging failed: {e}")
        return None

    avg_nii_path = scratch / "averaged.nii.gz"
    nib.save(nib.Nifti1Image(averaged, affine), str(avg_nii_path))

    # ── Skull stripping ───────────────────────────────────────────────────────
    brain_nii_path = scratch / "brain.nii.gz"
    if fsl_available():
        try:
            skull_strip_fsl(avg_nii_path, brain_nii_path)
            log.info(f"[{mri_id}] Skull strip: FSL BET")
        except subprocess.CalledProcessError as e:
            log.warning(f"[{mri_id}] BET failed ({e.stderr.strip()}) — using un-stripped volume")
            shutil.copy(str(avg_nii_path), str(brain_nii_path))
    else:
        log.warning(f"[{mri_id}] FSL not found — skipping skull strip (install FSL for better results)")
        shutil.copy(str(avg_nii_path), str(brain_nii_path))

    # ── MNI registration ──────────────────────────────────────────────────────
    mni_nii_path = scratch / "brain_mni.nii.gz"
    if fsl_available():
        try:
            register_to_mni_fsl(brain_nii_path, mni_nii_path)
            log.info(f"[{mri_id}] Registration: FSL FLIRT (12-DOF affine)")
            registered_img = nib.load(str(mni_nii_path))
        except subprocess.CalledProcessError as e:
            log.warning(f"[{mri_id}] FLIRT failed ({e.stderr.strip()}) — using nilearn fallback")
            brain_img      = nib.load(str(brain_nii_path))
            registered_img = register_to_mni_nilearn(brain_img)
    else:
        log.warning(f"[{mri_id}] FSL not found — using nilearn resampling for MNI registration")
        brain_img      = nib.load(str(brain_nii_path))
        registered_img = register_to_mni_nilearn(brain_img)

    # ── Intensity normalization ───────────────────────────────────────────────
    arr        = registered_img.get_fdata(dtype=np.float32)
    normalized = normalize_intensity(arr)
    final_img  = nib.Nifti1Image(normalized, registered_img.affine)

    log.info(f"[{mri_id}] Preprocessed: shape {normalized.shape}, "
             f"range [{normalized.min():.2f}, {normalized.max():.2f}]")
    return final_img


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — FEATURE EXTRACTION (one session)
# ─────────────────────────────────────────────────────────────────────────────

def extract_mri_features(mri_id: str, preprocessed_img: nib.Nifti1Image,
                          manifest_row: pd.Series) -> dict:
    """
    Extract handcrafted MRI features for one session.

    Two categories:
    (A) Pre-computed from demographics CSV — available immediately, no image needed:
        nWBV, eTIV, ASF, MMSE, Age, EDUC, SES

    (B) Atlas-based ROI volumes from the preprocessed image:
        AAL atlas parcellation via nilearn — hippocampal and DMN-adjacent ROIs.
        Volumes are normalized by eTIV to control for head size.

    Returns a flat dict ready to be saved as one row in a features CSV.
    """
    features = {
        "mri_id":       mri_id,
        "subject_id":   manifest_row["subject_id"],
        "visit":        manifest_row["visit"],
        "label":        manifest_row["label"],
        "label_int":    manifest_row["label_int"],
        "is_converter": manifest_row["is_converter"],
        # ── (A) Pre-computed CSV features — use directly ──────────────────────
        "nWBV":         manifest_row["nWBV"],
        "eTIV":         manifest_row["eTIV"],
        "ASF":          manifest_row["ASF"],
        "mmse":         manifest_row["mmse"],
        "age":          manifest_row["age"],
        "educ":         manifest_row.get("educ", np.nan),
        "ses":          manifest_row["ses"],
    }

    # ── (B) Harvard-Oxford atlas ROI volumes ─────────────────────────────────
    # Using HO instead of AAL — HO atlases are bundled with nilearn and require
    # no external download. The subcortical variant provides L/R hippocampus;
    # the cortical variant provides DMN-adjacent regions (precuneous, PCC, etc.).
    # If the PEARL-Neuro fMRI pipeline has already run, both are already cached.
    all_ho_rois = HO_SUBCORTICAL_ROIS + HO_CORTICAL_ROIS
    try:
        atlas_sub  = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
        atlas_cort = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")

        img_data    = preprocessed_img.get_fdata()
        vox_dims    = np.abs(np.diag(preprocessed_img.affine)[:3])
        vox_vol_mm3 = float(np.prod(vox_dims))
        e_tiv       = float(manifest_row["eTIV"]) if manifest_row["eTIV"] > 0 else 1.0

        def _extract_ho_rois(atlas_obj, roi_list):
            """Resample one HO atlas to subject space and extract requested ROIs."""
            atlas_img_obj = atlas_obj.maps if isinstance(atlas_obj.maps, nib.Nifti1Image) else nib.load(atlas_obj.maps)
            resampled     = image.resample_to_img(
                atlas_img_obj, preprocessed_img, interpolation="nearest"
            )
            atlas_data = resampled.get_fdata()

            # HO labels list is 0-indexed and offset by 1 in the image
            # (label value 0 = background, label value N = labels[N-1])
            label_to_val = {lbl: idx + 1 for idx, lbl in enumerate(atlas_obj.labels)}

            results = {}
            for roi_label in roi_list:
                safe_key = roi_label.replace(" ", "_").replace(",", "").replace(".", "")
                if roi_label not in label_to_val:
                    log.warning(f"[{mri_id}] HO label '{roi_label}' not found — skipping")
                    results[f"vol_{safe_key}"]             = np.nan
                    results[f"vol_{safe_key}_norm"]        = np.nan
                    results[f"mean_intensity_{safe_key}"]  = np.nan
                    continue

                roi_mask = atlas_data == label_to_val[roi_label]
                vol_mm3  = float(roi_mask.sum()) * vox_vol_mm3
                vol_norm = vol_mm3 / e_tiv
                mean_int = float(img_data[roi_mask].mean()) if roi_mask.any() else np.nan

                results[f"vol_{safe_key}"]            = vol_mm3
                results[f"vol_{safe_key}_norm"]       = vol_norm
                results[f"mean_intensity_{safe_key}"] = mean_int
            return results

        features.update(_extract_ho_rois(atlas_sub,  HO_SUBCORTICAL_ROIS))
        features.update(_extract_ho_rois(atlas_cort, HO_CORTICAL_ROIS))

        # ── Derived hippocampal features (safe key names match above) ─────────
        vol_l = features.get("vol_Left_Hippocampus",  np.nan)
        vol_r = features.get("vol_Right_Hippocampus", np.nan)

        if not (np.isnan(vol_l) or np.isnan(vol_r)):
            features["hippo_vol_bilateral"]      = vol_l + vol_r
            features["hippo_vol_bilateral_norm"] = (
                features["vol_Left_Hippocampus_norm"] +
                features["vol_Right_Hippocampus_norm"]
            )
            denom = vol_l + vol_r
            features["hippo_asymmetry_index"] = (
                (vol_l - vol_r) / (denom + 1e-8) if denom > 0 else np.nan
            )
        else:
            features["hippo_vol_bilateral"]      = np.nan
            features["hippo_vol_bilateral_norm"] = np.nan
            features["hippo_asymmetry_index"]    = np.nan

        log.info(f"[{mri_id}] ROI extraction done — "
                 f"hippo bilateral = {features.get('hippo_vol_bilateral', np.nan):.1f} mm³")

    except Exception as e:
        log.error(f"[{mri_id}] ROI feature extraction failed: {e}")
        for roi_label in all_ho_rois:
            safe_key = roi_label.replace(" ", "_").replace(",", "").replace(".", "")
            features[f"vol_{safe_key}"]            = np.nan
            features[f"vol_{safe_key}_norm"]       = np.nan
            features[f"mean_intensity_{safe_key}"] = np.nan
        features["hippo_vol_bilateral"]      = np.nan
        features["hippo_vol_bilateral_norm"] = np.nan
        features["hippo_asymmetry_index"]    = np.nan

    return features


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — SAVE CNN TENSOR (one session)
# ─────────────────────────────────────────────────────────────────────────────

def save_cnn_tensor(mri_id: str, preprocessed_img: nib.Nifti1Image) -> bool:
    """
    Resize the preprocessed volume to MRI_TARGET_SHAPE and save as a
    PyTorch tensor (.pt) for CNN training.

    Output shape: [1, D, H, W] — single-channel 3D volume.
    Saved to data/processed/mri_tensors/{mri_id}/mri_tensor.pt

    Returns True on success, False on failure.
    """
    try:
        arr     = preprocessed_img.get_fdata(dtype=np.float32)
        resized = resize_volume(arr, MRI_TARGET_SHAPE)
        tensor  = torch.tensor(resized).float().unsqueeze(0)  # [1, D, H, W]

        out_dir = TENSOR_DIR / mri_id
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, str(out_dir / "mri_tensor.pt"))

        log.info(f"[{mri_id}] CNN tensor saved: {tuple(tensor.shape)}")
        return True
    except Exception as e:
        log.error(f"[{mri_id}] Tensor save failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — CLEAN UP SCRATCH (optional, per session)
# ─────────────────────────────────────────────────────────────────────────────

def clean_scratch(mri_id: str):
    """
    Remove intermediate FSL files for this session from SCRATCH_DIR.
    The raw .hdr/.img files in OAS2_RAW_PART* are NOT touched — unlike
    DataLad (PEARL-Neuro), OASIS-2 is a direct download so we cannot
    re-fetch on demand. Only remove scratch, keep raw.
    """
    scratch = SCRATCH_DIR / mri_id
    if scratch.exists():
        shutil.rmtree(str(scratch))
        log.info(f"[{mri_id}] Scratch cleaned")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — one session, all steps
# ─────────────────────────────────────────────────────────────────────────────

def process_session(mri_id: str, manifest_row: pd.Series,
                    clean_after: bool = True) -> dict:
    """
    Full pipeline for one OASIS-2 MRI session.

    Steps:
      1. Preprocess MRI (average acquisitions → skull strip → MNI → normalize)
      2. Extract ROI features (AAL atlas + pre-computed CSV features)
      3. Save CNN tensor (128^3 float32)
      4. Optionally clean intermediate scratch files

    A failure in feature extraction does NOT abort tensor saving and vice versa.
    A session is still useful if at least one output is produced.

    Args:
        mri_id:       OASIS-2 MRI ID string, e.g. "OAS2_0001_MR1"
        manifest_row: corresponding row from the manifest DataFrame
        clean_after:  set False during debugging to inspect intermediate files
    """
    status = {
        "mri_id":      mri_id,
        "subject_id":  manifest_row["subject_id"],
        "visit":       manifest_row["visit"],
        "label":       manifest_row["label"],
        "is_converter": manifest_row["is_converter"],
        "timestamp":   datetime.now().isoformat(),
        "preprocess":  "skip",
        "features":    "skip",
        "tensor":      "skip",
        "overall":     "unknown",
        "notes":       "",
    }

    log.info(f"\n{'='*60}\nProcessing {mri_id}  [{manifest_row['label']}]"
             f"  Visit {manifest_row['visit']}\n{'='*60}")

    feat_path   = MRI_FEAT_DIR / f"{mri_id}_features.csv"
    tensor_path = TENSOR_DIR   / mri_id / "mri_tensor.pt"

    if feat_path.exists() and tensor_path.exists():
        log.info(f"[{mri_id}] Already processed — skipping")
        status["preprocess"] = status["features"] = status["tensor"] = "skip"
        status["overall"]    = "already_done"
        _append_session_status(status)
        return status

    # Step 1 — preprocess
    preprocessed = preprocess_mri_session(mri_id)
    if preprocessed is None:
        status["preprocess"] = "failed"
        status["overall"]    = "failed_preprocess"
        status["notes"]      = "MRI preprocessing failed."
        _append_session_status(status)
        return status
    status["preprocess"] = "ok"

    # Step 2 — features (runs even if tensor fails)
    if not feat_path.exists():
        try:
            features = extract_mri_features(mri_id, preprocessed, manifest_row)
            pd.DataFrame([features]).to_csv(str(feat_path), index=False)
            log.info(f"[{mri_id}] Features saved → {feat_path.name}")
            status["features"] = "ok"
        except Exception as e:
            log.error(f"[{mri_id}] Feature extraction crashed: {e}")
            status["features"]  = "failed"
            status["notes"]    += f"Features failed: {e}. "
    else:
        status["features"] = "skip"

    # Step 3 — CNN tensor (independent of features)
    if not tensor_path.exists():
        ok = save_cnn_tensor(mri_id, preprocessed)
        status["tensor"] = "ok" if ok else "failed"
        if not ok:
            status["notes"] += "Tensor failed. "
    else:
        status["tensor"] = "skip"

    # Step 4 — clean scratch
    if clean_after:
        clean_scratch(mri_id)

    feat_ok   = status["features"] in ("ok", "skip")
    tensor_ok = status["tensor"]   in ("ok", "skip")
    status["overall"] = (
        "success" if (feat_ok and tensor_ok)
        else "partial" if (feat_ok or tensor_ok)
        else "failed"
    )

    log.info(f"[{mri_id}] Done — {status['overall']}")
    _append_session_status(status)
    return status


# ─────────────────────────────────────────────────────────────────────────────
# BATCH RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(mri_ids: list, clean_after: bool = True) -> pd.DataFrame:
    """
    Process a list of MRI IDs sequentially. Continues past individual failures.

    Usage:
        manifest = build_manifest()
        run_pipeline(manifest["mri_id"].tolist()[:5], clean_after=False)   # test
        run_pipeline(manifest["mri_id"].tolist(), clean_after=True)         # full run

    For batched runs across sessions:
        batches = [all_ids[i:i+20] for i in range(0, len(all_ids), 20)]
        run_pipeline(batches[BATCH_TO_RUN], clean_after=True)
    """
    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(
            f"{MANIFEST_CSV} not found. Run build_manifest() first."
        )

    manifest = pd.read_csv(str(MANIFEST_CSV))
    manifest_lookup = {row["mri_id"]: row for _, row in manifest.iterrows()}

    statuses = []
    for mid in mri_ids:
        if mid not in manifest_lookup:
            log.error(f"[{mid}] Not in manifest — skipping (run build_manifest() again?)")
            statuses.append({
                "mri_id": mid, "timestamp": datetime.now().isoformat(),
                "overall": "not_in_manifest", "notes": "MRI ID missing from manifest CSV."
            })
            continue
        try:
            statuses.append(process_session(mid, manifest_lookup[mid], clean_after=clean_after))
        except Exception as e:
            log.error(f"[{mid}] Crashed: {e}\n{traceback.format_exc()}")
            statuses.append({
                "mri_id": mid, "timestamp": datetime.now().isoformat(),
                "overall": "crashed", "notes": str(e)
            })

    summary = pd.DataFrame(statuses)
    summary.to_csv(str(PROCESSING_LOG), index=False)
    log.info("\nOASIS-2 Pipeline complete.")
    if "overall" in summary.columns:
        log.info(summary["overall"].value_counts().to_string())
    return summary


def _append_session_status(status: dict):
    """Upsert a session's status row into the rolling processing log CSV."""
    path = PROCESSING_LOG
    row  = pd.DataFrame([status])
    if path.exists():
        df  = pd.read_csv(str(path))
        df  = df[df["mri_id"] != status["mri_id"]]
        row = pd.concat([df, row], ignore_index=True)
    row.to_csv(str(path), index=False)


# ─────────────────────────────────────────────────────────────────────────────
# ASSEMBLE FEATURE MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def assemble_feature_matrix(
    baseline_only: bool = True,
    flag_converters: bool = True,
) -> pd.DataFrame:
    """
    Merge all per-session feature CSVs into one flat matrix.
    Call this after all sessions are processed.

    Args:
        baseline_only:    If True, keep only Visit 1 per subject to avoid
                          data leakage in cross-sectional experiments.
                          Set to False for longitudinal experiments — but then
                          use subject-level train/test splits (never session-level).
        flag_converters:  If True, adds an 'is_converter' boolean column.
                          Converter sessions (subjects who started nondemented
                          and converted during the study) are kept in the matrix
                          but flagged for separate subgroup analysis.

    Returns a DataFrame saved to data/metadata/oasis2_feature_matrix.csv.
    """
    feat_files = sorted(MRI_FEAT_DIR.glob("OAS2_*_features.csv"))
    if not feat_files:
        raise FileNotFoundError(f"No feature CSVs found in {MRI_FEAT_DIR}")

    dfs   = [pd.read_csv(f) for f in feat_files]
    merged = pd.concat(dfs, ignore_index=True)

    if baseline_only:
        merged = merged[merged["visit"] == 1].copy()
        log.info(f"Baseline only (Visit 1): {len(merged)} sessions")

    log.info(f"Label distribution:\n{merged['label'].value_counts().to_string()}")
    if flag_converters:
        log.info(f"Converter sessions: {merged['is_converter'].sum()}")

    # nWBV sanity check — should decrease monotonically with CDR
    log.info("nWBV by label (sanity check — expect Control > MCI > AD):")
    log.info(merged.groupby("label")["nWBV"].mean().round(4).to_string())

    out = METADATA_DIR / "oasis2_feature_matrix.csv"
    merged.to_csv(str(out), index=False)
    log.info(f"Feature matrix: {merged.shape} → {out}")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Step 0 — build manifest (run once, inspect output before processing)
    # manifest = build_manifest()
    # print(manifest[["mri_id", "label", "visit", "is_converter", "nWBV"]].head(10).to_string())

    # Single session test — keep scratch for inspection
    # row = manifest[manifest["mri_id"] == "OAS2_0001_MR1"].iloc[0]
    # process_session("OAS2_0001_MR1", row, clean_after=False)

    # Full run — all sessions
    # run_pipeline(manifest["mri_id"].tolist(), clean_after=True)
    '''
    # Batched run — 20 sessions per batch (recommended for JupyterLab)
    all_ids    = manifest["mri_id"].tolist()
    BATCH_SIZE = 50
    batches    = [all_ids[i:i + BATCH_SIZE] for i in range(0, len(all_ids), BATCH_SIZE)]

    # Change this each session: 0, 1, 2 ... up to len(batches)-1
    BATCH_TO_RUN = 7

    batch = batches[BATCH_TO_RUN]
    print(f"Running batch {BATCH_TO_RUN}: {batch[0]} → {batch[-1]}")
    summary = run_pipeline(batch, clean_after=True)
    print(summary[["mri_id", "label", "overall"]].to_string())
    '''
    '''
    # Assemble feature matrix after all sessions processed
    df = assemble_feature_matrix(baseline_only=True, flag_converters=True)
    print(df[["label", "nWBV", "hippo_vol_bilateral_norm", "hippo_asymmetry_index"]].groupby("label").mean().round(4))
    
    # These vary by subject and are worth keeping
    usable = ["nWBV", "eTIV", "ASF", "mmse", "age", "educ", "ses"] + [c for c in df.columns if c.startswith("mean_intensity_")]

    # Drop the volume columns — they're artifactually uniform in MNI space
    drop = [c for c in df.columns if c.startswith("vol_") or c == "hippo_vol_bilateral" or c == "hippo_asymmetry_index"]
    
    df = pd.read_csv("data/metadata/oasis2_feature_matrix.csv")
    intensity_cols = [c for c in df.columns if c.startswith("mean_intensity_")]
    print(df.groupby("label")[intensity_cols].mean().round(4))
    '''
    import pandas as pd
    df = pd.read_csv("data/metadata/oasis2_feature_matrix.csv")
    intensity_cols = [c for c in df.columns if c.startswith("mean_intensity_")]
    print(df.groupby("label")[intensity_cols].mean().round(4).T)
