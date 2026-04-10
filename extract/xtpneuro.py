"""
PEARL-Neuro Subject Processing Pipeline
========================================
Corrected for actual dataset structure observed in ds004796:

  EEG format:   BrainVision (.vhdr / .eeg / .vmrk) — NOT .edf
  EEG tasks:    task-rest | task-msit | task-sternberg → use task-rest only
  fMRI tasks:   task-rest_dir-AP + task-rest_dir-PA → concatenate both runs
  File state:   DataLad skeleton (10.8 MB symlinks) — actual data is remote,
                fetched per-subject via `datalad get`, dropped after processing.

Subject numbering observed: sub-01 through sub-80 (some may be missing).

Setup (run once before this script):
    cd data/raw/pearl_neuro
    datalad clone https://github.com/OpenNeuroDatasets/ds004796 .

Install dependencies:
    pip install mne mne-bids nilearn numpy scipy pandas
    conda install -c conda-forge datalad
"""

import json
import logging
import subprocess
import traceback
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker
from scipy.signal import coherence, welch

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

RAW_DIR       = Path("../data/raw/pearl_neuro")
EPOCHS_DIR    = Path("../data/processed/eeg_epochs")
EEG_FEAT_DIR  = Path("../data/processed/eeg_features")
FMRI_FEAT_DIR = Path("../data/processed/fmri_features")
METADATA_DIR  = Path("../data/metadata")
LOG_DIR       = Path("logs")

TARGET_SFREQ   = 500
EPOCH_LENGTH_S = 4.0
EPOCH_OVERLAP  = 0.5
ICA_COMPONENTS = 15
REJECT_THRESH  = 150e-6

EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

COHERENCE_PAIRS = [
    ("F3", "P3"), ("F4", "P4"), ("T7", "O1"), ("T8", "O2"),
    ("Fz", "Pz"), ("F3", "F4"), ("P3", "P4"),
]

POSTERIOR_CHANNELS = ["O1", "O2", "P3", "P4", "Pz", "P7", "P8"]
FMRI_DIRECTIONS    = ["AP", "PA"]

DMN_ROI_KEYWORDS = [
    "Cingulate Gyrus, posterior",
    "Frontal Medial Cortex",
    "Angular Gyrus",
    "Precuneous Cortex",
    "Middle Temporal Gyrus, temporooccipital",
]

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

for d in [EPOCHS_DIR, EEG_FEAT_DIR, FMRI_FEAT_DIR, METADATA_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "pipeline.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SYMLINK-SAFE FILE CHECK
# ─────────────────────────────────────────────────────────────────────────────

def is_fetched(path: Path, min_bytes: int = 10_000) -> bool:
    """
    DataLad files are symlinks into .git/annex/objects.
    Before `datalad get`, these symlinks are dangling — the path appears in
    the directory listing but the target file does not exist yet.

    Path.stat() and Path.exists() FOLLOW symlinks, so they raise
    FileNotFoundError on a dangling symlink even though the symlink entry
    itself is present. This is the cause of the original crash.

    This helper uses lstat() (does NOT follow symlinks) to confirm the
    symlink entry exists in the directory, then stat() to check whether
    the actual content is present and large enough to be real data.

    Returns:
        True  — symlink exists AND resolves AND file is >= min_bytes
        False — dangling symlink (not fetched yet) or file too small
    """
    try:
        path.lstat()           # raises if symlink entry itself is missing
    except (FileNotFoundError, OSError):
        return False
    try:
        return path.stat().st_size >= min_bytes  # follows to actual content
    except (FileNotFoundError, OSError):
        return False           # dangling — content not fetched yet


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — INSPECT DATASET (run once, not per subject)
# ─────────────────────────────────────────────────────────────────────────────

def inspect_dataset() -> pd.DataFrame:
    """
    Run this once before processing to understand what subjects and files
    exist in the DataLad skeleton.

    Checks per subject:
    - Which EEG tasks are present (rest / msit / sternberg)
    - Whether the .eeg binary is already fetched or still a dead symlink
    - Which fMRI directions are present (AP / PA) and their fetch status

    Saves to metadata/dataset_inventory.csv and returns the DataFrame.
    """
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"{RAW_DIR} not found.\n"
            "Clone the dataset first:\n"
            f"  datalad clone https://github.com/OpenNeuroDatasets/ds004796 {RAW_DIR}"
        )

    records = []
    for sub_dir in sorted(RAW_DIR.glob("sub-*")):
        sub_id   = sub_dir.name.replace("sub-", "")
        eeg_dir  = sub_dir / "eeg"
        func_dir = sub_dir / "func"
        rec      = {"subject_id": sub_id}

        for task in ["rest", "msit", "sternberg"]:
            vhdr = list(eeg_dir.glob(f"*task-{task}_eeg.vhdr")) if eeg_dir.exists() else []
            eeg  = list(eeg_dir.glob(f"*task-{task}_eeg.eeg"))  if eeg_dir.exists() else []
            rec[f"eeg_{task}_present"]  = bool(vhdr)
            rec[f"eeg_{task}_fetched"] = is_fetched(eeg[0]) if eeg else False

        for task in ["rest", "msit", "sternberg"]:
            for direction in ["AP", "PA"]:
                nii = list(func_dir.glob(f"*task-{task}_dir-{direction}_bold.nii.gz")) if func_dir.exists() else []
                rec[f"fmri_{task}_{direction}_present"] = bool(nii)
                rec[f"fmri_{task}_{direction}_fetched"] = is_fetched(nii[0]) if nii else False

        records.append(rec)

    df  = pd.DataFrame(records)
    out = METADATA_DIR / "dataset_inventory.csv"
    df.to_csv(str(out), index=False)

    log.info(f"Subjects found:         {len(df)}")
    log.info(f"EEG rest present:       {df['eeg_rest_present'].sum()}")
    log.info(f"EEG rest fetched:       {df['eeg_rest_fetched'].sum()}")
    log.info(f"fMRI rest AP present:   {df['fmri_rest_AP_present'].sum()}")
    log.info(f"fMRI rest PA present:   {df['fmri_rest_PA_present'].sum()}")
    log.info(f"Inventory saved →       {out}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def download_subject(subject_id: str) -> bool:
    """
    Fetch resting-state EEG and fMRI files for one subject via DataLad.
    Skips files that are already fetched (symlink resolves to real data > 10 KB).

    Files fetched:
      eeg/  sub-XX_task-rest_eeg.{eeg, vhdr, vmrk}
      func/ sub-XX_task-rest_dir-{AP,PA}_bold.nii.gz

    Note: .json sidecars are plain text files already present in the skeleton.
    """
    sub     = f"sub-{subject_id}"
    targets = [
        f"{sub}/eeg/{sub}_task-rest_eeg.eeg",
        f"{sub}/eeg/{sub}_task-rest_eeg.vhdr",
        f"{sub}/eeg/{sub}_task-rest_eeg.vmrk",
        f"{sub}/func/{sub}_task-rest_dir-AP_bold.nii.gz",
        f"{sub}/func/{sub}_task-rest_dir-PA_bold.nii.gz",
    ]

    log.info(f"[{sub}] Fetching resting-state files")
    all_ok = True

    for target in targets:
        path = RAW_DIR / target

        # Use lstat() to check if symlink entry exists in the skeleton
        # path.exists() follows symlinks and returns False for dangling ones,
        # which incorrectly reports DataLad annex files as missing
        try:
            path.lstat()
        except (FileNotFoundError, OSError):
            log.warning(f"[{sub}] Not in skeleton: {Path(target).name} — skipping")
            continue

        if is_fetched(path):
            log.info(f"[{sub}] Already fetched: {Path(target).name}")
            continue
        try:
            subprocess.run(
                ["datalad", "get", target],
                cwd=str(RAW_DIR), capture_output=True, text=True, check=True,
            )
            log.info(f"[{sub}] Fetched: {Path(target).name}")
        except subprocess.CalledProcessError as e:
            log.error(f"[{sub}] Fetch FAILED: {Path(target).name} — {e.stderr.strip()}")
            all_ok = False

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — EEG PREPROCESSING + FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_eeg_subject(subject_id: str) -> bool:
    """
    Resting-state EEG pipeline for one subject.

    EEG format is BrainVision (.vhdr/.eeg/.vmrk). mne-bids handles this
    transparently via read_raw_bids — the BIDSPath points to the .vhdr and
    MNE loads the binary .eeg data automatically.

    Task filtering: BIDSPath(task="rest") ensures we load task-rest only,
    not msit or sternberg which are also present in this dataset.
    """
    sub          = f"sub-{subject_id}"
    out_epochs   = EPOCHS_DIR   / f"{sub}_task-rest_clean-epo.fif"
    out_features = EEG_FEAT_DIR / f"{sub}_eeg_features.csv"

    if out_epochs.exists() and out_features.exists():
        log.info(f"[{sub}] EEG already processed — skipping")
        return True

    log.info(f"[{sub}] EEG preprocessing (BrainVision, task-rest)")

    # ── Load ──────────────────────────────────────────────────────────────────
    try:
        bids_path = BIDSPath(
            subject=subject_id,
            task="rest",
            suffix="eeg",
            datatype="eeg",
            root=str(RAW_DIR),
        )
        raw = read_raw_bids(bids_path, verbose=False)
        raw.load_data()
    except Exception as e:
        log.error(f"[{sub}] EEG load failed: {e}")
        return False

    log.info(f"[{sub}] Loaded: {raw.info['nchan']} ch, {raw.info['sfreq']:.0f} Hz, {raw.times[-1]:.1f}s")

    # Ensure channels are typed as EEG (BrainVision files sometimes default to misc)
    non_eeg = [ch for ch in raw.ch_names if raw.get_channel_types([ch])[0] != "eeg"]
    if non_eeg:
        raw.set_channel_types({ch: "eeg" for ch in non_eeg})
        log.info(f"[{sub}] Re-typed {len(non_eeg)} channels to EEG")

    # ── Montage ───────────────────────────────────────────────────────────────
    try:
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore", verbose=False)
    except Exception as e:
        log.warning(f"[{sub}] Montage (non-fatal): {e}")

    # ── Filter + resample + reference ────────────────────────────────────────
    raw.filter(l_freq=0.5, h_freq=45.0, method="fir", fir_design="firwin", verbose=False)
    raw.notch_filter(freqs=50.0, verbose=False)   # 50 Hz European power line

    if raw.info["sfreq"] != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ, verbose=False)

    raw.set_eeg_reference("average", projection=True, verbose=False)
    raw.apply_proj()

    # ── ICA ───────────────────────────────────────────────────────────────────
    try:
        ica = mne.preprocessing.ICA(n_components=ICA_COMPONENTS, method="fastica", random_state=42, max_iter=800)
        ica.fit(raw, verbose=False)
        frontal = next((c for c in ["Fp1", "Fp2", "AF3", "AF4"] if c in raw.ch_names), None)
        if frontal:
            eog_idx, _ = ica.find_bads_eog(raw, ch_name=frontal, threshold=3.0, verbose=False)
            ica.exclude = eog_idx
            log.info(f"[{sub}] ICA: excluded {len(eog_idx)} EOG components")
        raw_clean = ica.apply(raw.copy(), verbose=False)
    except Exception as e:
        log.warning(f"[{sub}] ICA failed ({e}) — using filtered signal")
        raw_clean = raw.copy()

    # ── Epoch + reject ────────────────────────────────────────────────────────
    epochs = mne.make_fixed_length_epochs(
        raw_clean, duration=EPOCH_LENGTH_S, overlap=EPOCH_OVERLAP, preload=True, verbose=False
    )
    n_before = len(epochs)
    epochs.drop_bad(reject={"eeg": REJECT_THRESH}, verbose=False)
    log.info(f"[{sub}] Epochs: {n_before} → {len(epochs)} ({n_before - len(epochs)} rejected)")

    if len(epochs) < 5:
        log.error(f"[{sub}] Too few epochs ({len(epochs)}) — excluding from EEG analysis")
        return False

    epochs.save(str(out_epochs), overwrite=True, verbose=False)

    # ── Feature extraction ────────────────────────────────────────────────────
    try:
        features = _extract_eeg_features(epochs, subject_id)
        pd.DataFrame([features]).to_csv(str(out_features), index=False)
        log.info(f"[{sub}] EEG done — {len(features)} features")
    except Exception as e:
        log.error(f"[{sub}] Feature extraction failed: {e}")
        return False

    return True


def _extract_eeg_features(epochs: mne.Epochs, subject_id: str) -> dict:
    data     = epochs.get_data()
    sfreq    = epochs.info["sfreq"]
    ch_names = epochs.ch_names
    features = {"subject_id": subject_id}

    band_powers = _compute_band_powers(data, sfreq)

    # Band powers
    for band_name, bp in band_powers.items():
        for ch_idx, ch_name in enumerate(ch_names):
            features[f"power_{band_name}_{ch_name}"] = float(bp[:, ch_idx].mean())

    # Alpha/theta ratio
    for ch_idx, ch_name in enumerate(ch_names):
        a = band_powers["alpha"][:, ch_idx].mean()
        t = band_powers["theta"][:, ch_idx].mean()
        features[f"alpha_theta_ratio_{ch_name}"] = float(a / (t + 1e-10))

    # Alpha peak frequency (posterior channels)
    post_idx   = [ch_names.index(c) for c in POSTERIOR_CHANNELS if c in ch_names]
    peak_freqs = []
    for epoch in data:
        for ch_idx in post_idx:
            freqs, psd = welch(epoch[ch_idx], fs=sfreq, nperseg=min(int(sfreq * 2), data.shape[-1]))
            mask = (freqs >= 8.0) & (freqs <= 13.0)
            if mask.any():
                peak_freqs.append(float(freqs[mask][psd[mask].argmax()]))
    features["alpha_peak_freq_mean"] = float(np.mean(peak_freqs)) if peak_freqs else np.nan
    features["alpha_peak_freq_std"]  = float(np.std(peak_freqs))  if peak_freqs else np.nan

    # Spectral entropy
    for ch_idx, ch_name in enumerate(ch_names):
        ents = []
        for epoch in data:
            freqs, psd = welch(epoch[ch_idx], fs=sfreq, nperseg=min(int(sfreq * 2), data.shape[-1]))
            pn = psd / (psd.sum() + 1e-10)
            ents.append(float(-np.sum(pn * np.log2(pn + 1e-10))))
        features[f"spectral_entropy_{ch_name}"] = float(np.mean(ents))

    # Alpha coherence
    for ch1, ch2 in COHERENCE_PAIRS:
        if ch1 in ch_names and ch2 in ch_names:
            i1, i2  = ch_names.index(ch1), ch_names.index(ch2)
            coh_vals = []
            for epoch in data:
                freqs, coh = coherence(epoch[i1], epoch[i2], fs=sfreq, nperseg=min(int(sfreq), data.shape[-1]))
                mask = (freqs >= 8.0) & (freqs <= 13.0)
                if mask.any():
                    coh_vals.append(float(coh[mask].mean()))
            features[f"alpha_coherence_{ch1}_{ch2}"] = float(np.mean(coh_vals)) if coh_vals else np.nan

    # Hjorth parameters
    for ch_idx, ch_name in enumerate(ch_names):
        acts, mobs, comps = [], [], []
        for epoch in data:
            sig  = epoch[ch_idx]
            d1, d2 = np.diff(sig), np.diff(np.diff(sig))
            v0, v1, v2 = np.var(sig), np.var(d1), np.var(d2)
            mob = float(np.sqrt(v1 / (v0 + 1e-10)))
            acts.append(float(v0)); mobs.append(mob)
            comps.append(float(np.sqrt(v2 / (v1 + 1e-10)) / (mob + 1e-10)))
        features[f"hjorth_activity_{ch_name}"]   = float(np.mean(acts))
        features[f"hjorth_mobility_{ch_name}"]   = float(np.mean(mobs))
        features[f"hjorth_complexity_{ch_name}"] = float(np.mean(comps))

    return features


def _compute_band_powers(data: np.ndarray, sfreq: float) -> dict:
    n_epochs, n_channels, n_times = data.shape
    nperseg = min(int(sfreq * 2), n_times)
    powers  = {b: np.zeros((n_epochs, n_channels)) for b in EEG_BANDS}
    for ep in range(n_epochs):
        for ch in range(n_channels):
            freqs, psd = welch(data[ep, ch], fs=sfreq, nperseg=nperseg)
            for band, (fmin, fmax) in EEG_BANDS.items():
                mask = (freqs >= fmin) & (freqs <= fmax)
                powers[band][ep, ch] = psd[mask].mean() if mask.any() else 0.0
    return powers


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — fMRI PREPROCESSING + CONNECTIVITY
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_fmri_subject(subject_id: str) -> bool:
    """
    Resting-state fMRI pipeline for one subject.

    PEARL-Neuro has two resting-state fMRI runs with opposite phase-encoding:
      task-rest_dir-AP_bold.nii.gz  (anterior → posterior)
      task-rest_dir-PA_bold.nii.gz  (posterior → anterior)

    Both are extracted using the Harvard-Oxford cortical atlas (48 ROIs,
    built into nilearn — no separate download needed), then concatenated
    along the time axis before computing the Pearson correlation matrix.
    Concatenating both runs approximately doubles the number of timepoints,
    which substantially improves the connectivity estimate reliability.
    """
    sub          = f"sub-{subject_id}"
    out_conn      = FMRI_FEAT_DIR / f"{sub}_connectivity.npy"
    out_timeseries = FMRI_FEAT_DIR / f"{sub}_roi_timeseries.csv"

    if out_conn.exists() and out_timeseries.exists():
        log.info(f"[{sub}] fMRI already processed — skipping")
        return True

    log.info(f"[{sub}] fMRI preprocessing (AP + PA concatenation)")

    func_dir = RAW_DIR / sub / "func"

    # ── Collect available runs ────────────────────────────────────────────────
    fmri_paths, t_r_values = {}, {}
    for direction in FMRI_DIRECTIONS:
        nii  = func_dir / f"{sub}_task-rest_dir-{direction}_bold.nii.gz"
        jsn  = func_dir / f"{sub}_task-rest_dir-{direction}_bold.json"
        if not is_fetched(nii):
            log.warning(f"[{sub}] fMRI {direction}: not available or not fetched")
            continue
        fmri_paths[direction] = str(nii)
        t_r_values[direction] = 2.0
        if jsn.exists():
            try:
                t_r_values[direction] = float(json.loads(jsn.read_text()).get("RepetitionTime", 2.0))
            except Exception:
                pass

    if not fmri_paths:
        log.error(f"[{sub}] No usable fMRI files found")
        return False

    # ── Load atlas (nilearn built-in, no download needed after first call) ────
    try:
        atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    except Exception as e:
        log.error(f"[{sub}] Atlas load failed: {e}")
        return False

    # ── Extract ROI timeseries per run ────────────────────────────────────────
    all_ts, atlas_labels = [], None
    for direction, nii_path in fmri_paths.items():
        try:
            masker = NiftiLabelsMasker(
                labels_img=atlas.maps, standardize=True, detrend=True,
                low_pass=0.1, high_pass=0.01, t_r=t_r_values[direction],
                resampling_target="labels", verbose=0,
            )
            ts = masker.fit_transform(nii_path)
            all_ts.append(ts)
            if atlas_labels is None:
                atlas_labels = atlas.labels[:ts.shape[1]]
            log.info(f"[{sub}] fMRI {direction}: {ts.shape[0]} TRs × {ts.shape[1]} ROIs")
        except Exception as e:
            log.error(f"[{sub}] fMRI {direction} extraction failed: {e}")

    if not all_ts:
        log.error(f"[{sub}] No fMRI timeseries extracted")
        return False

    # ── Concatenate AP + PA ───────────────────────────────────────────────────
    if len(all_ts) == 2 and all_ts[0].shape[1] == all_ts[1].shape[1]:
        combined = np.concatenate(all_ts, axis=0)
        log.info(f"[{sub}] Concatenated: {combined.shape[0]} TRs total")
    else:
        combined = all_ts[0]
        log.info(f"[{sub}] Using single run only")

    # ── Connectivity matrix ───────────────────────────────────────────────────
    try:
        conn = ConnectivityMeasure(kind="correlation").fit_transform([combined])[0]
        np.fill_diagonal(conn, 0)
    except Exception as e:
        log.error(f"[{sub}] Connectivity failed: {e}")
        return False

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(str(out_conn), conn)
    pd.DataFrame(combined, columns=list(atlas_labels)).to_csv(str(out_timeseries), index=False)
    _save_dmn_features(subject_id, conn, list(atlas_labels))
    log.info(f"[{sub}] fMRI done — connectivity + timeseries + DMN features saved")
    return True


def _save_dmn_features(subject_id: str, conn: np.ndarray, labels: list):
    dmn_idx = [i for i, l in enumerate(labels) if any(kw.lower() in l.lower() for kw in DMN_ROI_KEYWORDS)]
    features = {"subject_id": subject_id}
    if len(dmn_idx) >= 2:
        sub_m   = conn[np.ix_(dmn_idx, dmn_idx)]
        upper   = sub_m[np.triu_indices_from(sub_m, k=1)]
        all_up  = conn[np.triu_indices_from(conn, k=1)]
        features.update({
            "dmn_mean_connectivity":    float(upper.mean()),
            "dmn_std_connectivity":     float(upper.std()),
            "dmn_min_connectivity":     float(upper.min()),
            "global_mean_connectivity": float(all_up.mean()),
            "dmn_roi_count":            len(dmn_idx),
        })
    else:
        log.warning(f"sub-{subject_id}: Only {len(dmn_idx)} DMN ROIs — check DMN_ROI_KEYWORDS")
        features.update({k: np.nan for k in ["dmn_mean_connectivity", "dmn_std_connectivity",
                                               "dmn_min_connectivity", "global_mean_connectivity"]})
        features["dmn_roi_count"] = len(dmn_idx)
    pd.DataFrame([features]).to_csv(str(FMRI_FEAT_DIR / f"sub-{subject_id}_dmn_features.csv"), index=False)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — DROP RAW
# ─────────────────────────────────────────────────────────────────────────────

def drop_subject_raw(subject_id: str) -> bool:
    """
    Drop fetched data for this subject's task-rest files to reclaim disk.
    DataLad keeps the symlink skeleton — re-fetch anytime with datalad get.
    Does NOT touch msit or sternberg files.
    """
    sub     = f"sub-{subject_id}"
    targets = [
        f"{sub}/eeg/{sub}_task-rest_eeg.eeg",
        f"{sub}/eeg/{sub}_task-rest_eeg.vhdr",
        f"{sub}/eeg/{sub}_task-rest_eeg.vmrk",
        f"{sub}/func/{sub}_task-rest_dir-AP_bold.nii.gz",
        f"{sub}/func/{sub}_task-rest_dir-PA_bold.nii.gz",
    ]
    for target in targets:
        path = RAW_DIR / target
        if not is_fetched(path):
            continue
        try:
            subprocess.run(["datalad", "drop", target], cwd=str(RAW_DIR), capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            log.warning(f"[{sub}] Drop warning (non-fatal): {Path(target).name}")
    log.info(f"[{sub}] Raw files dropped")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — one subject, four steps
# ─────────────────────────────────────────────────────────────────────────────

def process_subject(subject_id: str, drop_after: bool = True) -> dict:
    """
    Full pipeline for one subject.
    Steps 2 (EEG) and 3 (fMRI) are independent — a failure in one does not
    abort the other. The subject is still useful if at least one modality succeeds.

    Args:
        subject_id:  zero-padded string e.g. "01", "07", "42"
        drop_after:  False during debugging to keep raw files for inspection
    """
    sub    = f"sub-{subject_id}"
    status = {
        "subject_id": subject_id, "timestamp": datetime.now().isoformat(),
        "download": "skip", "eeg": "skip", "fmri": "skip",
        "drop": "skip", "overall": "unknown", "notes": "",
    }

    log.info(f"\n{'='*60}\nProcessing {sub}\n{'='*60}")

    eeg_done  = (EPOCHS_DIR   / f"{sub}_task-rest_clean-epo.fif").exists() and \
                (EEG_FEAT_DIR / f"{sub}_eeg_features.csv").exists()
    fmri_done = (FMRI_FEAT_DIR / f"{sub}_connectivity.npy").exists() and \
                (FMRI_FEAT_DIR / f"{sub}_roi_timeseries.csv").exists()

    if eeg_done and fmri_done:
        log.info(f"[{sub}] Already fully processed — skipping")
        status["overall"] = "already_done"
        _append_status(status); return status

    # Step 1 — download
    ok = download_subject(subject_id)
    status["download"] = "ok" if ok else "failed"
    if not ok:
        status["overall"] = "failed_download"
        status["notes"]   = "DataLad fetch failed"
        _append_status(status); return status

    # Step 2 — EEG
    if not eeg_done:
        ok = preprocess_eeg_subject(subject_id)
        status["eeg"] = "ok" if ok else "failed"
        if not ok: status["notes"] += "EEG failed. "
    else:
        status["eeg"] = "skip"

    # Step 3 — fMRI
    if not fmri_done:
        ok = preprocess_fmri_subject(subject_id)
        status["fmri"] = "ok" if ok else "failed"
        if not ok: status["notes"] += "fMRI failed. "
    else:
        status["fmri"] = "skip"

    # Step 4 — drop
    if drop_after:
        drop_subject_raw(subject_id)
        status["drop"] = "ok"
    else:
        status["drop"] = "kept_debug"

    eeg_ok  = status["eeg"]  in ("ok", "skip")
    fmri_ok = status["fmri"] in ("ok", "skip")
    status["overall"] = "success" if (eeg_ok and fmri_ok) else ("partial" if (eeg_ok or fmri_ok) else "failed")

    log.info(f"[{sub}] Done — {status['overall']}")
    _append_status(status)
    return status


# ─────────────────────────────────────────────────────────────────────────────
# BATCH RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(subject_ids: list, drop_after: bool = True) -> pd.DataFrame:
    """
    Process all subjects sequentially. Continues past individual failures.

    Usage:
        run_pipeline(["01"], drop_after=False)           # single subject test
        run_pipeline([f"{i:02d}" for i in range(1,81)]) # full run
    """
    statuses = []
    for sid in subject_ids:
        try:
            statuses.append(process_subject(sid, drop_after=drop_after))
        except Exception as e:
            log.error(f"[sub-{sid}] Crashed: {e}\n{traceback.format_exc()}")
            statuses.append({"subject_id": sid, "timestamp": datetime.now().isoformat(),
                             "overall": "crashed", "notes": str(e)})

    summary = pd.DataFrame(statuses)
    summary.to_csv(str(METADATA_DIR / "processing_summary.csv"), index=False)
    log.info("\nPipeline complete.")
    if "overall" in summary.columns:
        log.info(summary["overall"].value_counts().to_string())
    return summary


def _append_status(status: dict):
    path = METADATA_DIR / "processing_summary.csv"
    row  = pd.DataFrame([status])
    if path.exists():
        df  = pd.read_csv(path)
        df  = df[df["subject_id"] != status["subject_id"]]
        row = pd.concat([df, row], ignore_index=True)
    row.to_csv(str(path), index=False)


# ─────────────────────────────────────────────────────────────────────────────
# ASSEMBLE FEATURE MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def assemble_feature_matrix(participants_tsv: str = None) -> pd.DataFrame:
    """
    Merge all per-subject feature CSVs into one flat matrix ready for training.
    Call this after all subjects are processed.

    Label derivation — PEARL-Neuro has no pre-made Group column.
    The dataset recruited exclusively at-risk individuals split into two
    distinct genetic risk pathways:

      APOE_risk   — carries APOE ε4 allele (e3/e4, e4/e4, e2/e4)
                    Mechanism: lipid metabolism disruption, amyloid clearance
      PICALM_risk — carries PICALM rs3851179 A allele (G/A or A/A), no APOE ε4
                    Mechanism: clathrin-mediated endocytosis, amyloid trafficking

    This binary split is scientifically grounded and more novel than a
    simple AtRisk vs Control framing — almost no EEG paper compares these
    two distinct genetic AD risk pathways directly.

    Subjects carrying both risk alleles (APOE ε4 AND PICALM A) are assigned
    to APOE_risk since APOE ε4 is the dominant risk factor by effect size.
    """
    eeg_dfs = [pd.read_csv(f) for f in sorted(EEG_FEAT_DIR.glob("sub-*_eeg_features.csv"))]
    dmn_dfs = [pd.read_csv(f) for f in sorted(FMRI_FEAT_DIR.glob("sub-*_dmn_features.csv"))]

    if not eeg_dfs:
        raise FileNotFoundError(f"No EEG feature CSVs in {EEG_FEAT_DIR}")

    merged = pd.concat(eeg_dfs, ignore_index=True)
    if dmn_dfs:
        merged = merged.merge(pd.concat(dmn_dfs, ignore_index=True), on="subject_id", how="outer")

    if participants_tsv:
        pts = pd.read_csv(participants_tsv, sep="\t")

        # One row per participant — TSV has 192 rows for 79 subjects (multi-session)
        pts = pts.drop_duplicates(subset="participant_id", keep="first").copy()

        # Align subject_id format — TSV uses "sub-01", features use "01"
        pts["subject_id"]  = pts["participant_id"].str.replace("sub-", "").str.strip()
        merged["subject_id"] = merged["subject_id"].astype(str).str.zfill(2)

        # ── Genotype flags ────────────────────────────────────────────────────
        # APOE ε4 carrier: haplotype string contains "e4"
        pts["apoe_e4"] = (
            pts["APOE_haplotype"].astype(str).str.lower().str.contains("e4", na=False)
        )
        # PICALM rs3851179: G/G = reference homozygous (non-risk)
        # G/A = heterozygous risk, A/A = homozygous risk
        pts["picalm_a"] = (
            pts["PICALM_rs3851179"].astype(str).str.strip().str.upper() != "G/G"
        )

        # ── Label assignment ──────────────────────────────────────────────────
        # APOE ε4 takes priority — subjects carrying both get APOE_risk
        def assign_label(row):
            if row["apoe_e4"]:
                return "APOE_risk"
            elif row["picalm_a"]:
                return "PICALM_risk"
            else:
                return "Control"   # G/G APOE e3/e3 — should be ~0 in this cohort

        pts["label"]     = pts.apply(assign_label, axis=1)
        pts["label_int"] = pts["label"].map({"APOE_risk": 0, "PICALM_risk": 1, "Control": 2})

        log.info(f"APOE ε4 carriers:    {pts['apoe_e4'].sum()}")
        log.info(f"PICALM A carriers:   {pts['picalm_a'].sum()}")
        log.info(f"APOE_risk:           {(pts['label'] == 'APOE_risk').sum()}")
        log.info(f"PICALM_risk:         {(pts['label'] == 'PICALM_risk').sum()}")
        log.info(f"Control (neither):   {(pts['label'] == 'Control').sum()}")

        keep = [c for c in [
            "subject_id", "label", "label_int", "age", "sex", "education",
            "APOE_haplotype", "PICALM_rs3851179", "apoe_e4", "picalm_a",
            "BDI", "SES", "RPM",
        ] if c in pts.columns]

        merged = merged.merge(pts[keep], on="subject_id", how="left")
        log.info(f"Labels in feature matrix:\n{merged['label'].value_counts().to_string()}")

        # Warn if any processed subjects didn't match the TSV
        unmatched = merged["label"].isna().sum()
        if unmatched:
            log.warning(f"{unmatched} processed subjects not found in participants.tsv")

    out = METADATA_DIR / "pearl_neuro_feature_matrix.csv"
    merged.to_csv(str(out), index=False)
    log.info(f"Feature matrix: {merged.shape} → {out}")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 0 — inspect dataset skeleton (run once)
    # inspect_dataset()

    # Single subject test — keep raw files for inspection
    # process_subject("01", drop_after=False)

    # Full run with batches
    # all_subjects = [f"{i:02d}" for i in range(1, 81)]
    # BATCH_SIZE = 10
    # batches = [all_subjects[i:i+BATCH_SIZE] for i in range(0, len(all_subjects), BATCH_SIZE)]
    # BATCH_TO_RUN = 0   # change each session: 0 → 7
    # batch = batches[BATCH_TO_RUN]
    # print(f"Running batch {BATCH_TO_RUN}: subjects {batch[0]} → {batch[-1]}")
    # summary = run_pipeline(batch, drop_after=True)
    # print(summary[["subject_id", "overall"]].to_string())

    # Assemble feature matrix — run this once all 79 subjects are processed
    df = assemble_feature_matrix("data/raw/pearl_neuro/participants.tsv")
    print(df[["label", "age"]].groupby("label").agg(["count", "mean"]).round(2).to_string())
