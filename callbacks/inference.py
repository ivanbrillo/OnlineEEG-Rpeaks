import os
import re
import tempfile
import zipfile
import math
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import scipy.io

from callbacks.model import load_model_bundle
from callbacks.preprocessing import get_saved_subject_data
from lib.model import SeizureTransformerImproved
from lib.dataset_utils import bandpass_eeg, create_segments_nonoverlapping, scale_window_standard
from lib.target_generation import compute_R_distance_next
from lib.metrics import extract_peaks_from_distance_transform, evaluate, discrete_score
from lib.plot_utils import plot_inference_window

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
STATIC_PLOTS_DIR = os.path.join(_PROJECT_ROOT, "static", "plots")

# 64-channel electrode subset (must match training.py)
_CHANNELS_64 = [
    'E22', 'E17', 'E9', 'E19', 'E4', 'E26', 'E2', 'E16', 'E12', 'E5',
    'E24', 'E124', 'E27', 'E111', 'E33', 'E122', 'E11', 'E13', 'E112', 'E28',
    'E117', 'E31', 'E105', 'E6', 'E29', 'E110', 'E36', 'E104', 'E41', 'E103',
    'E54', 'E34', 'E123', 'E45', 'E108', 'E46', 'E102', 'E42', 'E93', 'E43',
    'E80', 'E47', 'E98', 'E55', 'E53', 'E86', 'E52', 'E92', 'E51', 'E97',
    'E58', 'E96', 'E62', 'E60', 'E85', 'E65', 'E90', 'E72', 'E70', 'E83',
    'E75', 'E82', 'E57', 'E100'
]

# ── Module-level state ─────────────────────────────────────────────────────────
_inference_results = None   # returned by get_results()
_inference_data = None      # raw arrays for on-demand window plot generation


def _clean_metric_value(value):
    """Convert metric values to JSON-safe floats; return None for NaN/inf/invalid."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _peaks_to_seconds(peaks_samples, fs):
    """Convert sample indices to seconds from the segment start."""
    return [float(p) / float(fs) for p in np.asarray(peaks_samples, dtype=float).ravel().tolist()]


# ── ZIP parsing ────────────────────────────────────────────────────────────────

def _parse_inference_zip(zip_path):
    """
    Extract and parse a ZIP containing P###_EEG.mat and (optionally) P###_ECG.mat files.

    Returns:
        dict[str, dict]: Keyed by "P###". Each value has at minimum:
            {"EEG": ndarray(ch x T), "freq": float, "has_ecg": bool}
        And if ECG is present:
            {"ECG": ndarray(T,), "R_peaks": ndarray(int)}
    """
    tmpdir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmpdir)

    eeg_pat = re.compile(r"(P\d{3})_EEG\.mat$", re.IGNORECASE)
    ecg_pat = re.compile(r"(P\d{3})_ECG\.mat$", re.IGNORECASE)

    eeg_files = {}
    ecg_files = {}

    for root, _, files in os.walk(tmpdir):
        for fname in files:
            fpath = os.path.join(root, fname)
            m = eeg_pat.search(fname)
            if m:
                eeg_files[m.group(1).upper()] = fpath
                continue
            m = ecg_pat.search(fname)
            if m:
                ecg_files[m.group(1).upper()] = fpath

    if not eeg_files:
        raise ValueError("No P###_EEG.mat files found in the ZIP.")

    subjects = {}
    for sid, eeg_path in sorted(eeg_files.items()):
        mat = scipy.io.loadmat(eeg_path, squeeze_me=True, struct_as_record=False)
        eeg_struct = mat["EEG"]
        eeg_data = np.asarray(eeg_struct.data, dtype=float)   # (ch, T)
        srate = float(eeg_struct.srate)

        entry = {"EEG": eeg_data, "freq": srate, "has_ecg": False}

        if sid in ecg_files:
            ecg_mat = scipy.io.loadmat(ecg_files[sid], squeeze_me=True, struct_as_record=False)
            ecg_signal = np.asarray(ecg_mat["ECG_i"], dtype=float).ravel()
            r_peaks_raw = np.asarray(ecg_mat["R_peak"], dtype=int).ravel()
            r_peaks = r_peaks_raw - 1  # MATLAB 1-indexed → 0-indexed
            entry["ECG"] = ecg_signal
            entry["R_peaks"] = r_peaks
            entry["has_ecg"] = True

        subjects[sid] = entry

    return subjects


def _load_inference_subjects_from_cache(subject_ids):
    """
    Load selected subjects from the parsed-data cache (data/data_parsed.pkl).

    Returns:
        dict[str, dict]: Keyed by subject ID, with inference-compatible fields.
    """
    cached = get_saved_subject_data(subject_ids)
    if not cached:
        raise ValueError("No cached subjects available. Upload training data first.")

    subjects = {}
    for sid, row in cached.items():
        eeg = np.asarray(row["EEG"], dtype=float)
        entry = {
            "EEG": eeg,
            "freq": float(row["freq"]),
            "has_ecg": "ECG" in row and "R_peaks" in row,
        }
        if entry["has_ecg"]:
            entry["ECG"] = np.asarray(row["ECG"], dtype=float).ravel()
            entry["R_peaks"] = np.asarray(row["R_peaks"], dtype=int).ravel()
        subjects[sid] = entry
    return subjects


# ── Preprocessing helpers ──────────────────────────────────────────────────────

def _apply_preprocessing(subjects, config):
    """
    Apply channel selection, butterworth filtering and distance transform
    computation in-place. Returns (data dict, frequency, seg_len, eval_params).
    """
    hp = config.get("hyperparams", {})
    n_channels_requested = int(config.get("channels", 128))
    use_butterworth = bool(config.get("butterworth", False))
    wind = hp.get("windowing", {})
    eval_hp = hp.get("evaluation", {})

    # Derive frequency from first subject
    first = next(iter(subjects.values()))
    frequency = float(first["freq"])

    time_window_length = float(wind.get("time_window_length", 10))
    seg_len = int(time_window_length * frequency)

    eval_params = {
        "height_threshold": float(eval_hp.get("height_threshold", -0.4)),
        "min_dist": int(eval_hp.get("min_dist", 200)),
        "tol_ms": float(eval_hp.get("tol_ms", 150)),
        "prominence": float(eval_hp.get("prominence", 0.035)),
        "fs": frequency,
        "window_len": seg_len,
    }

    # ── Channel selection ─────────────────────────────────────────────────────
    if n_channels_requested not in (64, 128):
        raise ValueError(f"Unsupported channel selection: {n_channels_requested}. Use 64 or 128.")

    if n_channels_requested == 64:
        ch_map = hp.get("channels_64_map", _CHANNELS_64)
        ch_indices = [int(ch[1:]) - 1 for ch in ch_map]

        for sid in subjects:
            eeg = np.asarray(subjects[sid]["EEG"])
            n_ch = int(eeg.shape[0])

            if n_ch == 128:
                if max(ch_indices, default=-1) >= n_ch:
                    raise ValueError(
                        f"64-channel map references channel index {max(ch_indices)} but subject {sid} has only {n_ch} channels."
                    )
                subjects[sid]["EEG"] = eeg[ch_indices, :]
            elif n_ch == 64:
                # Subject is already 64-channel; keep as-is.
                continue
            else:
                raise ValueError(
                    f"Subject {sid} has {n_ch} EEG channels. Expected 64 or 128 for 64-channel inference mode."
                )
    else:
        invalid = []
        for sid in subjects:
            n_ch = int(np.asarray(subjects[sid]["EEG"]).shape[0])
            if n_ch != 128:
                invalid.append((sid, n_ch))

        if invalid:
            details = ", ".join(f"{sid}={count}ch" for sid, count in invalid)
            raise ValueError(
                "128-channel inference mode requires all selected subjects to have 128 channels. "
                f"Found: {details}."
            )

    # ── Bandpass filter ───────────────────────────────────────────────────────
    if use_butterworth:
        lowcut = float(config.get("f_min", 0.1))
        highcut = float(config.get("f_max", 40.0))
        for sid in subjects:
            subjects[sid]["EEG"] = bandpass_eeg(
                subjects[sid]["EEG"], frequency, lowcut=lowcut, highcut=highcut
            )

    # ── Distance transform targets ────────────────────────────────────────────
    for sid in subjects:
        if subjects[sid]["has_ecg"]:
            r_peaks = subjects[sid]["R_peaks"]
            sig_len = int(np.asarray(subjects[sid]["ECG"]).shape[-1])
            subjects[sid]["ECG_pulse"] = compute_R_distance_next(r_peaks, sig_len) / frequency

    return frequency, seg_len, eval_params


# ── Main inference entry point ─────────────────────────────────────────────────

def run_inference(model_id, zip_path=None, included_subjects=None, log_fn=None):
    """
    Run the trained model on new evaluation subjects from a ZIP file.

    Evaluation subjects are NOT added to the training dataset.

    Args:
        model_id (str): Identifier of the saved model (from list_available_models)
                        OR an absolute path to a .pt file (manual upload).
        zip_path (str | None): Absolute path to the uploaded ZIP file.
        included_subjects (list[str] | None): Subject IDs selected from the
                existing parsed-data cache.
        log_fn (callable | None): Optional callback to emit a log line in real time.

    Returns:
        dict: {
            "status": "success" | "error",
            "logs": str,
            "summary": dict,
            "plot_files": list[str]
        }
    """
    global _inference_results, _inference_data

    logs = []

    def _log(msg):
        logs.append(msg)
        if log_fn is not None:
            log_fn(msg)

    # ── Load model bundle ─────────────────────────────────────────────────────
    _log("Loading model...")
    bundle = load_model_bundle(model_id)
    config = bundle["config"]
    state_dict = bundle["state_dict"]

    hp = config.get("hyperparams", {})
    model_hp = hp.get("model", {})
    in_channels = int(config.get("_in_channels", 128))
    in_samples = int(config.get("_in_samples", 5000))

    _log(f"Model: in_channels={in_channels}, in_samples={in_samples}")

    # ── Parse ZIP ─────────────────────────────────────────────────────────────
    if included_subjects:
        _log("Loading subjects from cached dataset...")
        subjects = _load_inference_subjects_from_cache(included_subjects)
        _log(f"Loaded {len(subjects)} subject(s) from data_parsed.pkl: {', '.join(sorted(subjects))}")
    else:
        if not zip_path:
            raise ValueError("Either a ZIP file or cached subjects must be provided for inference.")
        _log("Parsing ZIP file...")
        subjects = _parse_inference_zip(zip_path)
        _log(f"Found {len(subjects)} subject(s): {', '.join(sorted(subjects))}")

    subjects_with_ecg = [s for s, d in subjects.items() if d["has_ecg"]]
    if subjects_with_ecg:
        _log(f"ECG available for: {', '.join(sorted(subjects_with_ecg))}")
    else:
        _log("No ECG files found — metrics will not be computed.")

    # ── Preprocess ────────────────────────────────────────────────────────────
    _log("Preprocessing...")
    frequency, seg_len, eval_params = _apply_preprocessing(subjects, config)
    _log(f"Frequency: {frequency} Hz, window: {seg_len} samples ({seg_len/frequency:.1f}s)")

    # ── Reconstruct model ─────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"Device: {device}")

    model = SeizureTransformerImproved(
        in_channels=in_channels,
        in_samples=in_samples,
        dim_feedforward=int(model_hp.get("dim_feedforward", 512)),
        num_layers=int(model_hp.get("num_layers", 8)),
        num_heads=int(model_hp.get("num_heads", 4)),
        drop_rate=float(model_hp.get("drop_rate", 0.1)),
        skip_type=str(model_hp.get("skip_type", "SE")),
        conv_type=str(model_hp.get("conv_type", "default")),
        skip_concat=bool(model_hp.get("skip_concat", False)),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # ── Clear stale plots from any previous inference run ────────────────────
    os.makedirs(STATIC_PLOTS_DIR, exist_ok=True)
    for fname in os.listdir(STATIC_PLOTS_DIR):
        if fname.startswith("inference_") and fname.endswith(".png"):
            try:
                os.remove(os.path.join(STATIC_PLOTS_DIR, fname))
            except OSError:
                pass

    # ── Per-subject inference ─────────────────────────────────────────────────

    all_subject_data = {}   # for _inference_data
    per_subject_metrics = []
    plot_files = []

    with torch.no_grad():
        for sid in sorted(subjects):
            subj = subjects[sid]
            has_ecg = subj["has_ecg"]

            eeg = subj["EEG"]                # (ch, T)
            ecg_raw = subj.get("ECG")        # (T,) or None
            gt_pulse = subj.get("ECG_pulse") # (T,) or None

            # Create a dummy pulse array if no ECG (windowing function requires it)
            if gt_pulse is None:
                dummy_pulse = np.zeros(eeg.shape[1], dtype=np.float32)
            else:
                dummy_pulse = gt_pulse

            if ecg_raw is None:
                dummy_ecg = np.zeros(eeg.shape[1], dtype=np.float32)
            else:
                dummy_ecg = ecg_raw

            X_win, y_win, ecg_win = create_segments_nonoverlapping(
                eeg, dummy_pulse, dummy_ecg, seg_len
            )
            # X_win: (n_win, seg_len, ch)
            X_norm = scale_window_standard(X_win)

            n_win = X_win.shape[0]
            _log(f"  {sid}: {n_win} windows")

            # Forward pass in batches
            batch_size = 16
            pred_list = []
            X_t = torch.from_numpy(X_norm.astype(np.float32))
            for start in range(0, n_win, batch_size):
                xb = X_t[start:start + batch_size].to(device)
                out = model(xb)
                if isinstance(out, tuple):
                    out = out[0]
                pred_list.append(out.cpu().numpy())  # (b, 1, seg_len)

            pred_all = np.concatenate(pred_list, axis=0)  # (n_win, 1, seg_len)
            pred_dist_all = pred_all[:, 0, :]              # (n_win, seg_len)

            gt_dist_all = None
            ecg_win_clean = None

            if has_ecg:
                gt_dist_all = y_win      # (n_win, seg_len) — already distance transform
                ecg_win_clean = ecg_win  # (n_win, seg_len)

                # Per-window metrics
                maes, precisions, recalls, f1s = [], [], [], []
                mrr_errs, prr50_errs, sdrr_errs, rmssd_errs = [], [], [], []

                for i in range(n_win):
                    pred_peaks = extract_peaks_from_distance_transform(
                        pred_dist_all[i], seg_len,
                        eval_params["min_dist"],
                        eval_params["height_threshold"],
                        eval_params["prominence"],
                    )
                    gt_peaks = extract_peaks_from_distance_transform(
                        gt_dist_all[i], seg_len,
                        eval_params["min_dist"],
                        eval_params["height_threshold"],
                        eval_params["prominence"],
                    )
                    res = evaluate(pred_peaks, gt_peaks, f=frequency, window_len=seg_len)
                    _, _, _, rec, prec, f1 = discrete_score(
                        pred_peaks, gt_peaks, frequency, eval_params["tol_ms"]
                    )
                    maes.append(res["mae"])
                    precisions.append(prec)
                    recalls.append(rec)
                    f1s.append(f1)
                    mrr_errs.append(res["mrr_error"])
                    prr50_errs.append(res["prr50_error"])
                    sdrr_errs.append(res["sdrr_error"])
                    rmssd_errs.append(res["rmssd_error"])

                per_subject_metrics.append({
                    "subject_id": sid,
                    "n_segments": n_win,
                    "mae": float(np.mean(maes)),
                    "precision": float(np.mean(precisions)),
                    "recall": float(np.mean(recalls)),
                    "f1": float(np.mean(f1s)),
                    "mrr_error": float(np.mean(mrr_errs)),
                    "prr50_error": float(np.mean(prr50_errs)),
                    "sdrr_error": float(np.mean(sdrr_errs)),
                    "rmssd_error": float(np.mean(rmssd_errs)),
                })

            # Store raw data for on-demand plot generation
            all_subject_data[sid] = {
                "has_ecg": has_ecg,
                "n_windows": n_win,
                "pred_dist": pred_dist_all,
                "gt_dist": gt_dist_all,
                "ecg_signal": ecg_win_clean,
                "eval_params": eval_params,
                "fs": frequency,
            }

            # Generate window-0 plot immediately
            plot_fname = _plot_window(sid, 0, all_subject_data[sid])
            plot_files.append(plot_fname)

    # ── Global metrics ────────────────────────────────────────────────────────
    global_metrics = None
    if per_subject_metrics:
        keys = ["mae", "precision", "recall", "f1",
                "mrr_error", "prr50_error", "sdrr_error", "rmssd_error"]
        global_metrics = {
            k: float(np.mean([s[k] for s in per_subject_metrics])) for k in keys
        }

    # ── Build results dict ────────────────────────────────────────────────────
    subject_details = {
        sid: {
            "n_windows": d["n_windows"],
            "has_ecg": d["has_ecg"],
            "initial_plot": f"inference_{sid}_w0.png",
        }
        for sid, d in all_subject_data.items()
    }

    _inference_results = {
        "global_metrics": global_metrics,
        "per_subject": per_subject_metrics,
        "plots": {
            "per_subject": {
                sid: [f"inference_{sid}_w0.png"] for sid in all_subject_data
            }
        },
        "subject_details": subject_details,
        "eval_params": eval_params,
    }
    _inference_data = {"subjects": all_subject_data, "eval_params": eval_params}

    _log("Inference complete.")
    summary = {
        "subjects": len(subjects),
        "subjects_with_ecg": len(subjects_with_ecg),
        "metrics_available": global_metrics is not None,
    }

    return {
        "status": "success",
        "logs": "\n".join(logs),
        "summary": summary,
        "plot_files": plot_files,
    }


# ── On-demand plot generation (called by AJAX endpoint in app.py) ──────────────

def _plot_window(subject_id, window_idx, subject_data, force=False):
    """
    Generate (or retrieve from cache) the PNG for one subject/window.
    Returns the filename (relative to static/plots/).
    Set force=True to regenerate even if the file already exists.
    """
    plot_fname = f"inference_{subject_id}_w{window_idx}.png"
    plot_path = os.path.join(STATIC_PLOTS_DIR, plot_fname)

    if not force and os.path.exists(plot_path):
        return plot_fname

    d = subject_data
    n_win = d["n_windows"]
    pred_dist = d["pred_dist"][window_idx]
    gt_dist = d["gt_dist"][window_idx] if d["gt_dist"] is not None else None
    ecg_signal = d["ecg_signal"][window_idx] if d["ecg_signal"] is not None else None

    plot_inference_window(
        pred_dist=pred_dist,
        subject_id=subject_id,
        window_idx=window_idx,
        n_windows=n_win,
        save_path=plot_path,
        gt_dist=gt_dist,
        ecg_signal=ecg_signal,
        eval_params=d["eval_params"],
        fs=d["fs"],
    )
    return plot_fname


def generate_window_plot(subject_id, window_idx):
    """
    Public interface for the AJAX endpoint.

    Returns:
        str: Filename relative to static/plots/, or raises ValueError.
    """
    global _inference_data

    if _inference_data is None:
        raise ValueError("No inference data available. Run inference first.")

    subjects = _inference_data["subjects"]
    if subject_id not in subjects:
        raise ValueError(f"Unknown subject: {subject_id}")

    d = subjects[subject_id]
    if window_idx < 0 or window_idx >= d["n_windows"]:
        raise ValueError(
            f"Window index {window_idx} out of range for {subject_id} "
            f"(0–{d['n_windows'] - 1})."
        )

    return _plot_window(subject_id, window_idx, d)


# ── Metric recomputation (no model re-run needed) ─────────────────────────────

def recompute_metrics(new_eval_params):
    """
    Re-extract peaks and recompute metrics using new evaluation hyperparameters,
    without re-running the model. Updates _inference_results and regenerates
    window-0 plots for all subjects.

    Args:
        new_eval_params (dict): Must contain height_threshold, min_dist,
                                tol_ms, prominence. fs and window_len are
                                taken from the stored data (not overridable).
    Raises:
        ValueError: If no inference data is available.
    """
    global _inference_results, _inference_data

    if _inference_data is None:
        raise ValueError("No inference data available. Run inference first.")

    subjects = _inference_data["subjects"]
    stored_ep = _inference_data["eval_params"]

    # Build merged eval_params: user-supplied thresholds, fixed fs/window_len from data
    eval_params = {
        "height_threshold": float(new_eval_params.get("height_threshold",
                                  stored_ep["height_threshold"])),
        "min_dist":         int(new_eval_params.get("min_dist",
                                stored_ep["min_dist"])),
        "tol_ms":           float(new_eval_params.get("tol_ms",
                                  stored_ep["tol_ms"])),
        "prominence":       float(new_eval_params.get("prominence",
                                  stored_ep["prominence"])),
        "fs":               stored_ep["fs"],
        "window_len":       stored_ep["window_len"],
    }

    per_subject_metrics = []
    plot_files = []

    # Delete cached window-0 plots so they are regenerated with new params
    for fname in os.listdir(STATIC_PLOTS_DIR):
        if fname.startswith("inference_") and fname.endswith(".png"):
            try:
                os.remove(os.path.join(STATIC_PLOTS_DIR, fname))
            except OSError:
                pass

    for sid in sorted(subjects):
        d = subjects[sid]
        has_ecg = d["has_ecg"]
        pred_dist_all = d["pred_dist"]
        gt_dist_all = d["gt_dist"]
        frequency = d["fs"]
        seg_len = eval_params["window_len"]
        n_win = d["n_windows"]

        # Update stored eval_params for this subject (used by plot generation)
        d["eval_params"] = eval_params

        if has_ecg and gt_dist_all is not None:
            maes, precisions, recalls, f1s = [], [], [], []
            mrr_errs, prr50_errs, sdrr_errs, rmssd_errs = [], [], [], []

            for i in range(n_win):
                pred_peaks = extract_peaks_from_distance_transform(
                    pred_dist_all[i], seg_len,
                    eval_params["min_dist"],
                    eval_params["height_threshold"],
                    eval_params["prominence"],
                )
                gt_peaks = extract_peaks_from_distance_transform(
                    gt_dist_all[i], seg_len,
                    eval_params["min_dist"],
                    eval_params["height_threshold"],
                    eval_params["prominence"],
                )
                res = evaluate(pred_peaks, gt_peaks, f=frequency, window_len=seg_len)
                _, _, _, rec, prec, f1 = discrete_score(
                    pred_peaks, gt_peaks, frequency, eval_params["tol_ms"]
                )
                maes.append(res["mae"])
                precisions.append(prec)
                recalls.append(rec)
                f1s.append(f1)
                mrr_errs.append(res["mrr_error"])
                prr50_errs.append(res["prr50_error"])
                sdrr_errs.append(res["sdrr_error"])
                rmssd_errs.append(res["rmssd_error"])

            per_subject_metrics.append({
                "subject_id": sid,
                "n_segments": n_win,
                "mae":          float(np.mean(maes)),
                "precision":    float(np.mean(precisions)),
                "recall":       float(np.mean(recalls)),
                "f1":           float(np.mean(f1s)),
                "mrr_error":    float(np.mean(mrr_errs)),
                "prr50_error":  float(np.mean(prr50_errs)),
                "sdrr_error":   float(np.mean(sdrr_errs)),
                "rmssd_error":  float(np.mean(rmssd_errs)),
            })

        # Regenerate window-0 plot with new params
        plot_fname = _plot_window(sid, 0, d, force=True)
        plot_files.append(plot_fname)

    global_metrics = None
    if per_subject_metrics:
        keys = ["mae", "precision", "recall", "f1",
                "mrr_error", "prr50_error", "sdrr_error", "rmssd_error"]
        global_metrics = {
            k: float(np.mean([s[k] for s in per_subject_metrics])) for k in keys
        }

    # Update top-level eval_params in stored data
    _inference_data["eval_params"] = eval_params

    # Preserve subject_details (window counts etc.) from previous run
    subject_details = _inference_results["subject_details"]
    for sid in subject_details:
        subject_details[sid]["initial_plot"] = f"inference_{sid}_w0.png"

    _inference_results = {
        "global_metrics": global_metrics,
        "per_subject": per_subject_metrics,
        "plots": {
            "per_subject": {sid: [f"inference_{sid}_w0.png"] for sid in subjects}
        },
        "subject_details": subject_details,
        "eval_params": eval_params,
    }


def build_inference_export():
    """
    Build a JSON-serializable dictionary with per-subject and per-segment
    R-peak predictions and, when ECG is available, ground truth and metrics.

    Returns:
        dict: JSON-ready export payload.

    Raises:
        ValueError: If no inference data is available.
    """
    global _inference_data, _inference_results

    if _inference_data is None or _inference_results is None:
        raise ValueError("No inference data available. Run inference first.")

    subjects = _inference_data.get("subjects", {})
    eval_params = _inference_data.get("eval_params", {})

    fs = float(eval_params.get("fs", 0.0))
    window_len = int(eval_params.get("window_len", 0))
    if fs <= 0 or window_len <= 0:
        raise ValueError("Inference data is incomplete. Re-run inference.")

    subject_metric_map = {
        row.get("subject_id"): row
        for row in (_inference_results.get("per_subject") or [])
        if row.get("subject_id")
    }

    export_subjects = []
    for sid in sorted(subjects):
        d = subjects[sid]
        n_windows = int(d["n_windows"])
        has_ecg = bool(d["has_ecg"])
        pred_dist_all = d["pred_dist"]
        gt_dist_all = d.get("gt_dist")

        subject_metrics = None
        if has_ecg:
            sm = subject_metric_map.get(sid, {})
            subject_metrics = {
                "mae_s": _clean_metric_value(sm.get("mae")),
                "precision": _clean_metric_value(sm.get("precision")),
                "recall": _clean_metric_value(sm.get("recall")),
                "f1": _clean_metric_value(sm.get("f1")),
                "mrr_error_percent": _clean_metric_value(sm.get("mrr_error")),
                "prr50_error_percent": _clean_metric_value(sm.get("prr50_error")),
                "sdrr_error_percent": _clean_metric_value(sm.get("sdrr_error")),
                "rmssd_error_percent": _clean_metric_value(sm.get("rmssd_error")),
            }

        segments = []
        for idx in range(n_windows):
            pred_peaks = extract_peaks_from_distance_transform(
                pred_dist_all[idx],
                window_len,
                int(eval_params["min_dist"]),
                float(eval_params["height_threshold"]),
                float(eval_params["prominence"]),
            )

            segment_entry = {
                "segment_index": idx,
                "predicted_r_peaks_s": _peaks_to_seconds(pred_peaks, fs),
            }

            if has_ecg and gt_dist_all is not None:
                gt_peaks = extract_peaks_from_distance_transform(
                    gt_dist_all[idx],
                    window_len,
                    int(eval_params["min_dist"]),
                    float(eval_params["height_threshold"]),
                    float(eval_params["prominence"]),
                )
                res = evaluate(pred_peaks, gt_peaks, f=fs, window_len=window_len)
                _, _, _, rec, prec, f1 = discrete_score(
                    pred_peaks, gt_peaks, fs, float(eval_params["tol_ms"])
                )
                segment_entry["true_r_peaks_s"] = _peaks_to_seconds(gt_peaks, fs)
                segment_entry["segment_metrics"] = {
                    "mae_s": _clean_metric_value(res.get("mae")),
                    "precision": _clean_metric_value(prec),
                    "recall": _clean_metric_value(rec),
                    "f1": _clean_metric_value(f1),
                    "mrr_error_percent": _clean_metric_value(res.get("mrr_error")),
                    "prr50_error_percent": _clean_metric_value(res.get("prr50_error")),
                    "sdrr_error_percent": _clean_metric_value(res.get("sdrr_error")),
                    "rmssd_error_percent": _clean_metric_value(res.get("rmssd_error")),
                }

            segments.append(segment_entry)

        export_subjects.append({
            "subject_id": sid,
            "segment_length": {
                "samples": window_len,
                "seconds": float(window_len) / float(fs),
            },
            "subject_metrics": subject_metrics,
            "segments": segments,
        })

    return {
        "format_version": "1.0",
        "eval_params": {
            "height_threshold": float(eval_params["height_threshold"]),
            "min_dist": int(eval_params["min_dist"]),
            "tol_ms": float(eval_params["tol_ms"]),
            "prominence": float(eval_params["prominence"]),
            "fs_hz": fs,
            "window_length_samples": window_len,
            "window_length_seconds": float(window_len) / float(fs),
        },
        "subjects": export_subjects,
    }
