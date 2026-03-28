import os
import pickle
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

from callbacks.preprocessing import _load_cache
from lib.model import SeizureTransformerImproved
from lib.losses import WingLoss
from lib.dataset_utils import bandpass_eeg, process_subjects, scale_window_standard
from lib.target_generation import compute_R_distance_next
from lib.train_utils import train_model
from lib.plot_utils import plot_training_curves
from lib.utils import seed_everything

# ── Module-level state shared with callbacks/model.py ─────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
SAVED_MODELS_DIR = os.path.join(_PROJECT_ROOT, "saved_models")
STATIC_PLOTS_DIR = os.path.join(_PROJECT_ROOT, "static", "plots")

# Holds the most recently trained model so save_trained_model can persist it
_last_trained = {
    "state_dict": None,
    "config": None,
    "in_channels": None,
    "in_samples": None,
}

# 64-channel electrode subset (E54 is the reference)
_CHANNELS_64 = [
    'E22', 'E17', 'E9', 'E19', 'E4', 'E26', 'E2', 'E16', 'E12', 'E5',
    'E24', 'E124', 'E27', 'E111', 'E33', 'E122', 'E11', 'E13', 'E112', 'E28',
    'E117', 'E31', 'E105', 'E6', 'E29', 'E110', 'E36', 'E104', 'E41', 'E103',
    'E54', 'E34', 'E123', 'E45', 'E108', 'E46', 'E102', 'E42', 'E93', 'E43',
    'E80', 'E47', 'E98', 'E55', 'E53', 'E86', 'E52', 'E92', 'E51', 'E97',
    'E58', 'E96', 'E62', 'E60', 'E85', 'E65', 'E90', 'E72', 'E70', 'E83',
    'E75', 'E82', 'E57', 'E100'
]
_CHANNEL_64_INDICES = [int(ch[1:]) - 1 for ch in _CHANNELS_64]


def _make_dataset(subjects, data, seg_len, stage, aug_params, log_fn=None):
    """Process subjects into a TensorDataset."""
    if log_fn:
        log_fn(f"  Building {stage} set from {len(subjects)} subjects...")

    X_list, y_list, ecg_list, ids_list = process_subjects(
        subjects, data, seg_len, stage, **aug_params
    )

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    ECG = np.concatenate(ecg_list, axis=0)
    ids = np.concatenate(ids_list, axis=0)

    X = scale_window_standard(X)

    X_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)
    ECG_t = torch.from_numpy(ECG.astype(np.float32))
    ids_t = torch.from_numpy(ids.astype(np.int64))

    if log_fn:
        log_fn(f"  {stage.capitalize()}: {X_t.shape[0]} segments, shape {tuple(X_t.shape)}")

    return TensorDataset(X_t, y_t, ECG_t, ids_t)


def start_training(config, excluded_subjects, log_fn=None):
    """
    Launch the training pipeline with the given configuration.

    Args:
        config (dict): Training and model hyperparameters, including:
            - model_name (str)
            - channels (int): 64 or 128
            - butterworth (bool): whether to apply Butterworth filter
            - f_min (float): filter lower bound in Hz (if butterworth is True)
            - f_max (float): filter upper bound in Hz (if butterworth is True)
            - hyperparams (dict): arbitrary model/training hyperparameters from JSON editor
        excluded_subjects (list[str]): Subject IDs to exclude (treated as test set).
        log_fn (callable | None): Optional callback to emit a log line during training.

    Returns:
        dict: {"status": str, "logs": str, "metrics": dict}
    """
    def _log(text):
        if log_fn is not None:
            log_fn(text)

    hp = config.get("hyperparams", {})
    general = hp.get("general", {})
    opt = hp.get("optimization", {})
    sched = hp.get("scheduler", {})
    es = hp.get("early_stopping", {})
    model_hp = hp.get("model", {})
    eval_hp = hp.get("evaluation", {})
    wind = hp.get("windowing", {})
    aug = hp.get("augmentation", {})

    # ── Hyperparameters ───────────────────────────────────────────────────────
    SEED = int(general.get("SEED", 1234))
    BATCH_SIZE = int(general.get("BATCH_SIZE", 32))
    NUM_EPOCHS = int(general.get("NUM_EPOCHS", 50))
    VALIDATION_RATIO = float(general.get("VALIDATION_RATIO", 0.3))

    LEARN_RATE = float(opt.get("LEARN_RATE", 1e-3))
    WEIGHT_DECAY = float(opt.get("WEIGHT_DECAY", 1e-4))
    CLIP_NORM = float(opt.get("CLIP_NORM", 1))

    SCHED_FACTOR = float(sched.get("SCHED_FACTOR", 0.7))
    SCHED_PATIENCE = int(sched.get("SCHED_PATIENCE", 5))

    EARLY_STOP_PATIENCE = int(es.get("EARLY_STOP_PATIENCE", 15))
    MIN_EPOCH_IMPROVEMENT = int(es.get("MIN_EPOCH_IMPROVEMENT", 3))

    dim_feedforward = int(model_hp.get("dim_feedforward", 512))
    num_layers = int(model_hp.get("num_layers", 8))
    num_heads = int(model_hp.get("num_heads", 4))
    drop_rate = float(model_hp.get("drop_rate", 0.1))
    skip_type = str(model_hp.get("skip_type", "SE"))
    conv_type = str(model_hp.get("conv_type", "default"))
    skip_concat = bool(model_hp.get("skip_concat", False))

    height_threshold = float(eval_hp.get("height_threshold", -0.4))
    min_dist = int(eval_hp.get("min_dist", 200))
    tol_ms = float(eval_hp.get("tol_ms", 150))
    prominence = float(eval_hp.get("prominence", 0.035))

    time_window_length = float(wind.get("time_window_length", 10))
    overlap_percentage = float(wind.get("overlap_percentage", 0.0))

    use_augmentation = bool(aug.get("use_augmentation", False))
    warp_min = float(aug.get("warp_factor_range_min", 0.85))
    warp_max = float(aug.get("warp_factor_range_max", 1.15))
    warp_factor_range = (warp_min, warp_max)
    n_augmented_per_segment = int(aug.get("n_augmented_per_segment", 5))

    model_name = config.get("model_name", "model")
    use_butterworth = config.get("butterworth", False)
    n_channels_requested = int(config.get("channels", 128))

    # ── Load data ─────────────────────────────────────────────────────────────
    _log("Loading dataset cache...")
    cache = _load_cache()
    if not cache:
        raise ValueError("No subjects in dataset. Upload data first.")

    # Convert string IDs "P023" -> int 23 for internal use
    data = {int(k[1:]): v for k, v in cache.items()}

    # Convert excluded_subjects (UI strings "P023") -> int set; these are dropped entirely
    excluded_int = set(int(s[1:]) for s in excluded_subjects if len(s) >= 2)

    all_subjects = sorted(data.keys())
    usable = [s for s in all_subjects if s not in excluded_int]

    if len(usable) < 2:
        raise ValueError(
            f"Not enough subjects for train/val split. "
            f"Have {len(usable)} usable after excluding {len(excluded_int)}."
        )

    # Shuffle for reproducible split
    random.seed(SEED)
    random.shuffle(usable)

    n_val = max(1, int(len(usable) * VALIDATION_RATIO))
    val_subj = usable[:n_val]
    train_subj = usable[n_val:]

    if not train_subj:
        raise ValueError("No subjects left for training after val split. Reduce VALIDATION_RATIO.")

    _log(f"Subjects — Train: {len(train_subj)}, Val: {len(val_subj)} (excluded: {len(excluded_int)})")
    _log(f"Train: {train_subj}")
    _log(f"Val:   {val_subj}")

    # ── Determine frequency from data ─────────────────────────────────────────
    frequency = float(data[all_subjects[0]]["freq"])
    _log(f"Sampling frequency: {frequency} Hz")

    seg_len = int(time_window_length * frequency)
    train_stride = max(1, int(seg_len * (1.0 - overlap_percentage / 100.0)))

    _log(f"Window: {time_window_length}s = {seg_len} samples, stride: {train_stride}")

    # ── Channel selection ─────────────────────────────────────────────────────
    if n_channels_requested == 64:
        ch_map = hp.get("channels_64_map", _CHANNELS_64)
        ch_indices = [int(ch[1:]) - 1 for ch in ch_map]
        _log(f"Applying 64-channel selection ({len(ch_indices)} channels)...")
        for sid in all_subjects:
            data[sid]["EEG"] = data[sid]["EEG"][ch_indices, :]
        n_channels = len(ch_indices)
    else:
        n_channels = data[all_subjects[0]]["EEG"].shape[0]
    _log(f"EEG channels: {n_channels}")

    # ── Bandpass filter (lowcut/highcut come from UI fields f_min/f_max) ──────
    if use_butterworth:
        lowcut = float(config.get("f_min", 0.1))
        highcut = float(config.get("f_max", 40.0))
        _log(f"Applying bandpass filter: {lowcut}-{highcut} Hz...")
        for sid in all_subjects:
            eeg_unfilt = np.asarray(data[sid]["EEG"], dtype=float)
            data[sid]["EEG_unfilt"] = eeg_unfilt
            data[sid]["EEG"] = bandpass_eeg(eeg_unfilt, frequency, lowcut=lowcut, highcut=highcut)

    # ── Compute ECG pulse (distance transform target) ─────────────────────────
    _log("Computing ECG distance transform targets...")
    for sid in all_subjects:
        r_peaks = data[sid].get("R_peaks", [])
        sig_len = int(np.asarray(data[sid]["ECG"]).shape[-1])
        data[sid]["ECG_pulse"] = compute_R_distance_next(r_peaks, sig_len) / frequency

    # ── Seed ──────────────────────────────────────────────────────────────────
    seed_everything(SEED)

    # ── Build datasets ────────────────────────────────────────────────────────
    aug_seg_len = int(warp_factor_range[1] * seg_len) if use_augmentation else seg_len

    aug_params = {
        "use_augmentation": use_augmentation,
        "aug_seg_len": aug_seg_len,
        "train_stride": train_stride,
        "warp_factor_range": warp_factor_range,
        "n_augmented_per_segment": n_augmented_per_segment,
    }

    _log("Building datasets...")
    train_ds = _make_dataset(train_subj, data, seg_len, "training", aug_params, log_fn=_log)
    val_ds = _make_dataset(val_subj, data, seg_len, "validation", aug_params, log_fn=_log)

    trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"Device: {device}")

    model = SeizureTransformerImproved(
        in_channels=n_channels,
        in_samples=seg_len,
        dim_feedforward=dim_feedforward,
        num_layers=num_layers,
        num_heads=num_heads,
        drop_rate=drop_rate,
        skip_type=skip_type,
        conv_type=conv_type,
        skip_concat=skip_concat,
    ).to(device)

    _log(f"Model: SeizureTransformerImproved | in_channels={n_channels}, in_samples={seg_len}")

    criterion = WingLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARN_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHED_FACTOR, patience=SCHED_PATIENCE
    )

    # ── Eval params ───────────────────────────────────────────────────────────
    eval_params = {
        "height_threshold": height_threshold,
        "min_dist": min_dist,
        "fs": frequency,
        "tol_ms": tol_ms,
        "window_len": seg_len,
        "prominence": prominence,
    }

    # ── Train ─────────────────────────────────────────────────────────────────
    _log("=" * 60)
    _log("Starting training...")
    _log("=" * 60)

    history, best_state = train_model(
        model=model,
        trainloader=trainloader,
        valloader=valloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=NUM_EPOCHS,
        early_stop_patience=EARLY_STOP_PATIENCE,
        clip_norm=CLIP_NORM,
        model_name=model_name,
        eval_params=eval_params,
        save_dir=SAVED_MODELS_DIR,
        verbose=True,
        min_epoch_improvement=MIN_EPOCH_IMPROVEMENT,
        log_fn=log_fn,
    )

    # ── Save training curves plot ─────────────────────────────────────────────
    os.makedirs(STATIC_PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(STATIC_PLOTS_DIR, f"{model_name}_training_curves.png")
    plot_training_curves(history, best_state, save_path=plot_path)
    _log(f"Training curves saved to static/plots/{model_name}_training_curves.png")

    # ── Store trained state for save_trained_model ────────────────────────────
    _last_trained["state_dict"] = {k: v.cpu() for k, v in model.state_dict().items()}
    _last_trained["config"] = config
    _last_trained["in_channels"] = n_channels
    _last_trained["in_samples"] = seg_len

    # ── Final metrics ─────────────────────────────────────────────────────────
    best_ep = best_state["epoch"]
    metrics = {
        "best_epoch": best_ep + 1,
        "best_val_f1": round(best_state["val_f1"], 4),
        "val_f1": round(history["val"].disc_f1[best_ep], 4),
        "val_mae": round(history["val"].mae[best_ep], 4),
        "train_f1": round(history["train"].disc_f1[best_ep], 4),
        "train_mae": round(history["train"].mae[best_ep], 4),
    }

    return {
        "status": "success",
        "logs": "",
        "metrics": metrics,
    }
