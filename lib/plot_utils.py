import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server use
import matplotlib.pyplot as plt

from lib.metrics import extract_peaks_from_distance_transform, evaluate, discrete_score


def plot_training_curves(history, best_state, save_path):
    """
    Plot training curves for loss, F1, and MAE and save to file.

    Args:
        history: Dict with 'train', 'val', 'test' MetricHistory objects
        best_state: Dict with 'epoch' and 'val_f1' keys
        save_path: Absolute path to save the PNG figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    best_epoch = best_state['epoch']
    best_val_f1 = best_state['val_f1']

    # Loss plot
    axes[0].plot(history['train'].loss, label='Train Loss', marker='o', markersize=4)
    axes[0].plot(history['val'].loss, label='Validation Loss', marker='^', markersize=4)
    axes[0].axvline(x=best_epoch, color='red', linestyle='--', linewidth=1.5,
                    label=f'Best Epoch ({best_epoch+1})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss per Epoch')
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    if len(history['train'].loss) > 0:
        max_loss = max(history['train'].loss[:min(3, len(history['train'].loss))])
        axes[0].set_ylim(0, max(0.06, max_loss))

    # F1 plot
    axes[1].plot(history['train'].disc_f1, label='Train F1', marker='o', markersize=4)
    axes[1].plot(history['val'].disc_f1, label='Validation F1', marker='^', markersize=4)
    axes[1].axvline(x=best_epoch, color='red', linestyle='--', linewidth=1.5,
                    label=f'Best Epoch ({best_epoch+1})')
    axes[1].axhline(y=best_val_f1, color='orange', linestyle=':', linewidth=1,
                    label=f'Best F1: {best_val_f1:.4f}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('F1 Score per Epoch')
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    axes[1].set_ylim(0, 1)

    # MAE plot
    axes[2].plot(history['train'].mae, label='Train MAE', marker='o', markersize=4)
    axes[2].plot(history['val'].mae, label='Validation MAE', marker='^', markersize=4)
    axes[2].axvline(x=best_epoch, color='red', linestyle='--', linewidth=1.5,
                    label=f'Best Epoch ({best_epoch+1})')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('MAE')
    axes[2].set_title('Mean Absolute Error per Epoch')
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_inference_window(
    pred_dist,
    subject_id,
    window_idx,
    n_windows,
    save_path,
    gt_dist=None,
    ecg_signal=None,
    eval_params=None,
    fs=500.0,
):
    """
    Generate a peak-detection visualization for one inference window.

    If ground-truth data is provided (gt_dist / ecg_signal), plots two subplots:
      - Top: ECG signal + GT peaks (circles) + predicted peaks (crosses)
      - Bottom: GT and predicted distance transforms with peak markers

    Without ground truth (EEG-only upload), plots one subplot:
      - Predicted distance transform + predicted peaks only

    Args:
        pred_dist (ndarray): Predicted distance transform, shape (seg_len,).
        subject_id (str): E.g. "P001", used in the title.
        window_idx (int): 0-based window index.
        n_windows (int): Total number of windows for this subject.
        save_path (str): Absolute path where the PNG should be written.
        gt_dist (ndarray | None): Ground-truth distance transform, shape (seg_len,).
        ecg_signal (ndarray | None): Raw ECG signal, shape (seg_len,).
        eval_params (dict | None): Keys: height_threshold, min_dist, prominence, tol_ms, fs, window_len.
        fs (float): Sampling frequency in Hz (fallback if not in eval_params).
    """
    if eval_params is None:
        eval_params = {}

    height_threshold = eval_params.get("height_threshold", -0.4)
    min_dist = eval_params.get("min_dist", 200)
    prominence = eval_params.get("prominence", 0.035)
    tol_ms = eval_params.get("tol_ms", 150)
    window_len = eval_params.get("window_len", len(pred_dist))
    fs = eval_params.get("fs", fs)

    pred_peaks = extract_peaks_from_distance_transform(
        pred_dist, window_len, min_dist, height_threshold, prominence
    )

    has_gt = gt_dist is not None
    has_ecg = ecg_signal is not None

    if has_gt:
        gt_peaks = extract_peaks_from_distance_transform(
            gt_dist, window_len, min_dist, height_threshold, prominence
        )
        _, _, _, recall, precision, f1 = discrete_score(pred_peaks, gt_peaks, fs, tol_ms)
        res = evaluate(pred_peaks, gt_peaks, f=fs, window_len=window_len)
        mae_ms = res["mae"] * 1000
    else:
        gt_peaks = None

    t = np.arange(len(pred_dist)) / fs  # time axis in seconds

    n_rows = 2 if has_ecg else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(18, 5 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    # ── Title ────────────────────────────────────────────────────────────────
    if has_gt:
        title = (
            f"{subject_id} — Window {window_idx + 1}/{n_windows} | "
            f"Prec: {precision:.2f}  Rec: {recall:.2f}  F1: {f1:.2f}  MAE: {mae_ms:.1f} ms | "
            f"GT peaks: {len(gt_peaks)}  Pred peaks: {len(pred_peaks)}"
        )
    else:
        title = (
            f"{subject_id} — Window {window_idx + 1}/{n_windows} | "
            f"Pred peaks: {len(pred_peaks)}  (no ECG — metrics unavailable)"
        )
    fig.suptitle(title, fontsize=10)

    plot_idx = 0

    # ── ECG subplot (top) ─────────────────────────────────────────────────────
    if has_ecg:
        ax = axes[plot_idx]
        ax.plot(t, ecg_signal, color="#555555", linewidth=0.7, label="ECG")
        if has_gt and gt_peaks is not None and len(gt_peaks) > 0:
            ax.scatter(
                gt_peaks / fs, ecg_signal[gt_peaks],
                marker="o", s=60, color="#27ae60", zorder=5, label="GT peaks"
            )
        if len(pred_peaks) > 0:
            ax.scatter(
                pred_peaks / fs, ecg_signal[pred_peaks],
                marker="x", s=50, color="#e74c3c", zorder=5, label="Pred peaks"
            )
        ax.set_ylabel("ECG Amplitude")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.25)
        plot_idx += 1

    # ── Distance transform subplot ────────────────────────────────────────────
    ax = axes[plot_idx]
    if has_gt and gt_dist is not None:
        ax.plot(t, gt_dist, color="#2980b9", linewidth=0.9, label="GT dist. transform")
        if gt_peaks is not None and len(gt_peaks) > 0:
            ax.scatter(
                gt_peaks / fs, gt_dist[gt_peaks],
                marker="o", s=60, color="#27ae60", zorder=5, label="GT peaks"
            )
    ax.plot(t, pred_dist, color="#e67e22", linewidth=0.9, label="Pred dist. transform")
    if len(pred_peaks) > 0:
        ax.scatter(
            pred_peaks / fs, pred_dist[pred_peaks],
            marker="x", s=50, color="#e74c3c", zorder=5, label="Pred peaks"
        )
    ax.set_ylabel("Distance Transform")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
