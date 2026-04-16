import os
import re
import pickle
import zipfile
import tempfile
import shutil

import numpy as np
from scipy.io import loadmat

# ── Cache location ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_HERE), "data")
CACHE_PATH = os.path.join(DATA_DIR, "data_parsed.pkl")

_EEG_RE = re.compile(r"^(P\d{3})_EEG\.mat$", re.IGNORECASE)
_ECG_RE = re.compile(r"^(P\d{3})_ECG\.mat$", re.IGNORECASE)


def _load_cache():
    """Load the parsed-data cache from disk. Returns empty dict if not found."""
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def _save_cache(data):
    """Persist the parsed-data dict to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(data, f)


def handle_training_upload(zip_path):
    """
    Extract and parse a ZIP archive of EEG/ECG MATLAB files.

    Expected ZIP contents (files may be in subdirectories):
        P###_EEG.mat  — EEG struct: EEG.data (channels × timepoints),
                        EEG.srate, EEG.nbchan, EEG.chanlocs
        P###_ECG.mat  — ECG fields: ECG_i (signal), R_peak (indices), t_int (time)

    Each subject must have both files. Sampling rate is read from EEG.srate.
    New subjects are merged into the existing dataset (existing subjects overwritten).

    Args:
        zip_path (str): Absolute path to the uploaded ZIP file.

    Returns:
        dict: {
            "status": "success" | "error",
            "message": str,
            "num_subjects": int,   # subjects parsed in this upload
            "total_subjects": int  # total subjects now in dataset
        }
    """
    tmpdir = tempfile.mkdtemp()
    try:
        # ── Extract ZIP ────────────────────────────────────────────────────────
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir)

        # ── Discover .mat files ────────────────────────────────────────────────
        eeg_files = {}   # "P023" -> full path
        ecg_files = {}

        for root, _, files in os.walk(tmpdir):
            for fname in files:
                m = _EEG_RE.match(fname)
                if m:
                    eeg_files[m.group(1).upper()] = os.path.join(root, fname)
                    continue
                m = _ECG_RE.match(fname)
                if m:
                    ecg_files[m.group(1).upper()] = os.path.join(root, fname)

        # ── Validate pairs ─────────────────────────────────────────────────────
        eeg_ids = set(eeg_files)
        ecg_ids = set(ecg_files)
        common  = eeg_ids & ecg_ids
        orphans = (eeg_ids | ecg_ids) - common

        if not common:
            raise ValueError(
                "No complete EEG/ECG pairs found in the ZIP. "
                "Each subject needs both P###_EEG.mat and P###_ECG.mat."
            )
        if orphans:
            raise ValueError(
                f"Incomplete pairs for: {', '.join(sorted(orphans))}. "
                "Each subject needs both P###_EEG.mat and P###_ECG.mat."
            )

        # ── Parse each subject ─────────────────────────────────────────────────
        new_data = {}
        for subj_id in sorted(common):

            # EEG ---------------------------------------------------------------
            eeg_mat    = loadmat(eeg_files[subj_id])
            eeg_struct = eeg_mat["EEG"]
            eeg_data   = eeg_struct["data"][0, 0].astype(float)   # (channels, timepoints)
            srate      = float(np.asarray(eeg_struct["srate"][0, 0]).reshape(-1)[0])

            # ECG ---------------------------------------------------------------
            ecg_mat    = loadmat(ecg_files[subj_id])
            ecg_signal = ecg_mat["ECG_i"].flatten().astype(float)
            r_peaks    = ecg_mat["R_peak"].flatten() - 1           # 1-indexed → 0-indexed

            new_data[subj_id] = {
                "EEG":     eeg_data,
                "ECG":     ecg_signal,
                "R_peaks": r_peaks,
                "freq":    srate,
            }

        # ── Merge into cache ───────────────────────────────────────────────────
        cache = _load_cache()
        cache.update(new_data)
        _save_cache(cache)

        n_new   = len(new_data)
        n_total = len(cache)
        return {
            "status":         "success",
            "message":        f"Parsed {n_new} subject(s). Dataset now contains {n_total} subject(s).",
            "num_subjects":   n_new,
            "total_subjects": n_total,
        }

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def get_saved_subjects():
    """
    Return sorted list of subject IDs currently in the training dataset.

    Returns:
        list[str]: e.g. ["P023", "P040", "P103"]
    """
    cache = _load_cache()
    return sorted(cache.keys())


def get_saved_subject_data(subject_ids=None):
    """
    Return cached subject payloads for the requested IDs.

    Args:
        subject_ids (list[str] | None): Subject IDs to include. If None,
            returns all cached subjects.

    Returns:
        dict[str, dict]: Subject data dict in the same format stored in cache.
    """
    cache = _load_cache()

    if subject_ids is None:
        return {sid: cache[sid] for sid in sorted(cache.keys())}

    requested = [sid.upper() for sid in subject_ids]
    missing = [sid for sid in requested if sid not in cache]
    if missing:
        raise ValueError(f"Unknown subject(s) in dataset: {', '.join(sorted(set(missing)))}")

    return {sid: cache[sid] for sid in sorted(set(requested))}


def reset_dataset():
    """
    Delete the parsed-data cache, clearing all subjects from the dataset.

    Returns:
        dict: {"status": "success", "message": str}
    """
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)
        return {"status": "success", "message": "Dataset cleared. All subject data removed."}
    return {"status": "success", "message": "Dataset was already empty."}
