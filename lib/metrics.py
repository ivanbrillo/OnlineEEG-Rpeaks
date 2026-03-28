from collections import defaultdict
from copy import deepcopy
import numpy as np
import torch
import scipy
import pandas as pd


def min_distance_from_pred_to_true(pred_peaks, true_peaks, window_len=5000):
    """
    Match predicted peaks to nearest ground truth peaks.
    Returns list of distances for matched peaks.
    """
    if len(true_peaks) == 0:
        return np.array([])

    distances = []
    for pp in pred_peaks:
        if (pp < window_len / 100 or pp > (window_len - window_len / 100)):
            continue

        dists_to_true = np.abs(true_peaks - pp)
        distances.append(np.min(dists_to_true))
    return np.array(distances)


def clip(value):
    if np.isnan(value):
        return 0
    return np.clip(value, 0, 1)


def evaluate(pred_peaks, true_peaks, f=500, window_len=5000):
    pred_peaks, true_peaks = np.array(pred_peaks), np.array(true_peaks)

    # Fallback for empty peaks
    if len(pred_peaks) == 0 or len(true_peaks) == 0:
        return {k: 0 for k in ["mae", "precision", "recall", "f1"] +
                [f"{m}_{t}" for m in ["mrr", "prr50", "sdrr", "rmssd"] for t in ["pred", "true", "error"]]}

    # MAE
    distances = min_distance_from_pred_to_true(pred_peaks, true_peaks, window_len=window_len)
    mae = clip(np.mean(distances) / f)

    # RR intervals
    rr_pred, rr_true = np.diff(pred_peaks), np.diff(true_peaks)

    def pct_error(pred_val, true_val):
        return 100.0 * abs(pred_val - true_val) / (true_val + 1e-4)

    # mRR (mean RR)
    mrr_pred, mrr_true = np.mean(rr_pred), np.mean(rr_true)
    mrr_error = pct_error(mrr_pred, mrr_true)

    # pRR50 (percentage of successive RR changes > 50ms)
    thr = 0.05 * f
    prr50_pred = 100.0 * np.sum(np.abs(np.diff(rr_pred)) > thr) / max(len(rr_pred) - 1, 1)
    prr50_true = 100.0 * np.sum(np.abs(np.diff(rr_true)) > thr) / max(len(rr_true) - 1, 1)
    prr50_error = abs(prr50_pred - prr50_true)

    # SDRR (standard deviation of RR)
    sdrr_pred, sdrr_true = np.std(rr_pred, ddof=1), np.std(rr_true, ddof=1)
    sdrr_error = pct_error(sdrr_pred, sdrr_true)

    # RMSSD (root mean square of successive differences)
    rmssd_pred = np.sqrt(np.mean(np.diff(rr_pred)**2))
    rmssd_true = np.sqrt(np.mean(np.diff(rr_true)**2))
    rmssd_error = pct_error(rmssd_pred, rmssd_true)

    results = {
        "mae": mae,
        "mrr_pred": mrr_pred, "mrr_true": mrr_true, "mrr_error": mrr_error,
        "prr50_pred": prr50_pred, "prr50_true": prr50_true, "prr50_error": prr50_error,
        "sdrr_pred": sdrr_pred, "sdrr_true": sdrr_true, "sdrr_error": sdrr_error,
        "rmssd_pred": rmssd_pred, "rmssd_true": rmssd_true, "rmssd_error": rmssd_error,
    }

    return results


def discrete_score(pred_peaks, true_peaks, fs=500, tol_ms=75):
    thr = tol_ms / 1000  # convert ms to seconds
    tol_samples = thr * fs

    pred_peaks = np.array(pred_peaks)
    true_peaks = np.array(true_peaks)
    TP, FP, FN = 0, 0, 0

    for j in range(len(true_peaks)):
        loc = np.where(np.abs(pred_peaks - true_peaks[j]) <= tol_samples)[0]

        if j == 0:
            err = np.where((pred_peaks >= 0.5 * fs + tol_samples) &
                           (pred_peaks <= true_peaks[j] - tol_samples))[0]
        elif j == len(true_peaks) - 1:
            err = np.where((pred_peaks >= true_peaks[j] + tol_samples) &
                           (pred_peaks <= 9.5 * fs - tol_samples))[0]
        else:
            err = np.where((pred_peaks >= true_peaks[j] + tol_samples) &
                           (pred_peaks <= true_peaks[j + 1] - tol_samples))[0]

        FP += len(err)

        if len(loc) >= 1:
            TP += 1
            FP += len(loc) - 1
        else:
            FN += 1

    recall = TP / (TP + FN) if TP + FN > 0 else 0
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
    return TP, FP, FN, recall, precision, f1


def extract_peaks_from_distance_transform(dist_transform, window_len, min_distance=300, height_threshold=-0.4, prominence=0.035):
    """
    Extract peak locations from distance transform signal.
    Returns indices of peaks (valleys in the inverted distance transform).
    """
    margin = window_len * 0.01 # exclude 1% of the windows
    idx = np.arange(len(dist_transform))
    surrogate_dist = deepcopy(dist_transform)
    surrogate_dist[(idx < margin) | (idx > (window_len - margin))] = 1

    peaks, _properties = scipy.signal.find_peaks(
        -surrogate_dist,
        distance=min_distance,
        height=height_threshold,
        prominence=prominence
    )
    return peaks


def evaluate_on_loader(dataset_loader, model, device, height_threshold=-0.2, min_dist=200, fs=500, tol_ms=75, window_len=5000, prominence=0.035):
    """Compute peak-based MAE and HRV errors for a given data loader."""
    metrics = {k: [] for k in ['mae', 'mrr_err', 'prr50_err', 'sdrr_err', 'rmssd_err', 'recall', 'prec', 'f1']}

    model.eval()
    with torch.no_grad():
        for x, y, _, _ in dataset_loader:
            x = x.to(device)
            y_pred = (model(x)[0] if isinstance(model(x), tuple) else model(x)).cpu().numpy()
            y_true = y.cpu().numpy()

            for i in range(y_pred.shape[0]):
                pred_peaks = extract_peaks_from_distance_transform(y_pred[i, 0, :], window_len, min_dist, height_threshold, prominence)
                true_peaks = extract_peaks_from_distance_transform(y_true[i, 0, :], window_len, min_dist, height_threshold, prominence)

                res = evaluate(pred_peaks, true_peaks, f=fs, window_len=window_len)
                _, _, _, recall, precision, f1 = discrete_score(pred_peaks, true_peaks, fs, tol_ms)

                metrics['mae'].append(res["mae"])
                metrics['mrr_err'].append(res["mrr_error"])
                metrics['prr50_err'].append(res["prr50_error"])
                metrics['sdrr_err'].append(res["sdrr_error"])
                metrics['rmssd_err'].append(res["rmssd_error"])
                metrics['recall'].append(recall)
                metrics['prec'].append(precision)
                metrics['f1'].append(f1)

    return (np.mean(metrics['mae']), len(metrics['mae']), *[np.mean(metrics[k]) for k in
            ['mrr_err', 'prr50_err', 'sdrr_err', 'rmssd_err', 'prec', 'recall', 'f1']])


def compute_per_subject_metrics(dataset_loader, model, device, min_distance=200, height_threshold=-0.3, fs=500, tol_ms=75, window_len=5000, prominence=0.035):
    """Compute peak-based metrics grouped by subject ID."""
    model.eval()
    subject_data = defaultdict(lambda: {'metrics': [], 'disc_metrics': []})

    with torch.no_grad():
        for x, y, _, subj_ids in dataset_loader:
            y_pred = (model(x.to(device))[0] if isinstance((out := model(x.to(device))), tuple) else out).cpu().numpy()
            y_true, subj_ids = y.cpu().numpy(), subj_ids.cpu().numpy()

            for i in range(y_pred.shape[0]):
                pred_peaks = extract_peaks_from_distance_transform(y_pred[i, 0, :], window_len, min_distance, height_threshold, prominence)
                true_peaks = extract_peaks_from_distance_transform(y_true[i, 0, :], window_len, min_distance, height_threshold, prominence)

                subj_id = int(subj_ids[i])
                subject_data[subj_id]['metrics'].append(evaluate(pred_peaks, true_peaks, f=fs, window_len=window_len))

                _, _, _, recall, precision, f1 = discrete_score(pred_peaks, true_peaks, fs, tol_ms)
                subject_data[subj_id]['disc_metrics'].append({'precision': precision, 'recall': recall, 'f1': f1})

    return {
        subj_id: {
            'Subject_ID': subj_id,
            'Num_Segments': len(data['metrics']),
            'MAE_s': np.mean([m['mae'] for m in data['metrics']]),
            **{f'Disc_{k.upper()}_%': np.mean([m[k] for m in data['disc_metrics']]) * 100
               for k in ['precision', 'recall', 'f1']},
            **{f'{k}_err_%': np.mean([m[f'{k}_error'] for m in data['metrics']])
               for k in ['mrr', 'prr50', 'sdrr', 'rmssd']}
        }
        for subj_id, data in subject_data.items() if len(data['metrics']) > 0
    }


def summary_per_subject(model, testloader, trainloader, valloader, device, eval_params):
    """Compute and display per-subject metrics for all datasets."""
    min_distance = eval_params['min_dist']
    height_threshold = eval_params['height_threshold']
    fs = eval_params['fs']
    tol_ms = eval_params['tol_ms']
    window_len = eval_params['window_len']
    prominence = eval_params["prominence"]

    test_per_subject = compute_per_subject_metrics(testloader, model, device, min_distance=min_distance, height_threshold=height_threshold, fs=fs, tol_ms=tol_ms, window_len=window_len, prominence=prominence)
    train_per_subject = compute_per_subject_metrics(trainloader, model, device, min_distance=min_distance, height_threshold=height_threshold, fs=fs, tol_ms=tol_ms, window_len=window_len, prominence=prominence)
    val_per_subject = compute_per_subject_metrics(valloader, model, device, min_distance=min_distance, height_threshold=height_threshold, fs=fs, tol_ms=tol_ms, window_len=window_len, prominence=prominence)

    df_test = pd.DataFrame.from_dict(test_per_subject, orient='index')
    df_train = pd.DataFrame.from_dict(train_per_subject, orient='index')
    df_val = pd.DataFrame.from_dict(val_per_subject, orient='index')

    return df_train, df_val, df_test
