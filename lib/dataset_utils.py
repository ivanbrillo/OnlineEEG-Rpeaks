import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfiltfilt


def create_segments_nonoverlapping(EEG, pulse, ECG, seg_len):
    """Create non-overlapping segments."""
    C, T = EEG.shape
    n_seg = T // seg_len

    EEG_tc = EEG[:, :n_seg * seg_len].T
    X = EEG_tc.reshape(n_seg, seg_len, C)
    y = pulse[:n_seg * seg_len].reshape(n_seg, seg_len)
    ecg = ECG[:n_seg * seg_len].reshape(n_seg, seg_len)

    return X, y, ecg


def create_segments_sliding(EEG, pulse, ECG, seg_len, stride):
    """Create overlapping segments with sliding window."""
    C, T = EEG.shape
    starts = np.arange(0, T - seg_len + 1, stride, dtype=int)

    X = np.stack([EEG[:, s:s + seg_len].T for s in starts], axis=0)
    y = np.stack([pulse[s:s + seg_len] for s in starts], axis=0)
    ecg = np.stack([ECG[s:s + seg_len] for s in starts], axis=0)

    return X, y, ecg


def apply_time_warp(signal, warp_factor_range, axis=-1):
    """Apply random time warping to a signal."""
    original_len = signal.shape[axis]
    warp_factor = np.random.uniform(*warp_factor_range)
    new_len = int(original_len * warp_factor)

    old_indices = np.linspace(0, original_len - 1, original_len)
    new_indices = np.linspace(0, original_len - 1, new_len)

    if signal.ndim == 1:  # ECG
        interp_func = interp1d(old_indices, signal, kind='linear', fill_value='extrapolate')  # type: ignore
        return interp_func(new_indices)

    # 2D signal: warp along specified axis
    warped = np.zeros((signal.shape[0], new_len))
    for ch in range(signal.shape[0]):
        interp_func = interp1d(old_indices, signal[ch, :], kind='linear', fill_value='extrapolate')  # type: ignore
        warped[ch, :] = interp_func(new_indices)

    return warped


def extract_random_window(signal, target_len, axis=-1):
    """Extract a random window of target_len from signal."""
    max_start = signal.shape[axis] - target_len
    start_idx = np.random.randint(0, max_start + 1)

    if signal.ndim == 1:
        return signal[start_idx:start_idx + target_len]

    return signal[:, start_idx:start_idx + target_len]


def augment_segment(eeg_seg, pulse_seg, ecg_seg, target_len, warp_range, n_aug):
    """Create augmented versions of a segment via time warping."""
    augmented = []
    eeg_seg_t = eeg_seg.T  # (C, T)

    for _ in range(n_aug):
        rng_state = np.random.get_state()
        eeg_warped = apply_time_warp(eeg_seg_t, warp_range, axis=-1)
        np.random.set_state(rng_state)
        pulse_warped = apply_time_warp(pulse_seg, warp_range, axis=-1)
        np.random.set_state(rng_state)
        ecg_warped = apply_time_warp(ecg_seg, warp_range, axis=-1)

        rng_state = np.random.get_state()
        eeg_window = extract_random_window(eeg_warped, target_len, axis=-1)
        np.random.set_state(rng_state)
        pulse_window = extract_random_window(pulse_warped, target_len, axis=-1)
        np.random.set_state(rng_state)
        ecg_window = extract_random_window(ecg_warped, target_len, axis=-1)

        augmented.append({
            'eeg': eeg_window.T,
            'pulse': pulse_window,
            'ecg': ecg_window
        })

    return augmented


def create_training_segments_with_augmentation(subject_data, aug_seg_len, seg_len, stride, warp_range, n_aug):
    """Create training segments with augmentation."""
    EEG = subject_data['EEG']
    pulse = subject_data['ECG_pulse']
    ECG = subject_data['ECG']

    # Create larger segments
    X_large, y_large, ecg_large = create_segments_sliding(EEG, pulse, ECG, aug_seg_len, stride)

    # Extract the original seg_len segments, centered from aug_seg_len segments
    center_offset = (aug_seg_len - seg_len) // 2
    X_orig = X_large[:, center_offset:center_offset + seg_len, :]
    y_orig = y_large[:, center_offset:center_offset + seg_len]
    ecg_orig = ecg_large[:, center_offset:center_offset + seg_len]

    # Create augmented versions
    X_aug_list, y_aug_list, ecg_aug_list = [], [], []
    for i in range(len(X_large)):
        augs = augment_segment(X_large[i], y_large[i], ecg_large[i], seg_len, warp_range, n_aug)
        for aug in augs:
            X_aug_list.append(aug['eeg'])
            y_aug_list.append(aug['pulse'])
            ecg_aug_list.append(aug['ecg'])

    X_all = np.concatenate([X_orig, np.stack(X_aug_list, axis=0)], axis=0)
    y_all = np.concatenate([y_orig, np.stack(y_aug_list, axis=0)], axis=0)
    ecg_all = np.concatenate([ecg_orig, np.stack(ecg_aug_list, axis=0)], axis=0)

    return X_all, y_all, ecg_all


def bandpass_eeg(x, fs, lowcut=0.1, highcut=22.5, order=4):
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfiltfilt(sos, x, axis=-1)


def scale_window_standard(x_win):
    # x_win: (n_segments, window_len, channels) -> z-score per channel per segment
    mean = x_win.mean(axis=1, keepdims=True)
    std = x_win.std(axis=1, keepdims=True) + 1e-8
    return (x_win - mean) / std


def process_subjects(subjects, data_preprocessed, seg_len, stage, **aug_params):
    """Process subjects and create segments."""
    X_list, y_list, ecg_list, ids_list = [], [], [], []

    for subj_id in subjects:
        # Choose segmentation method
        if stage == 'training' and aug_params.get('use_augmentation'):
            X, y, ecg = create_training_segments_with_augmentation(
                data_preprocessed[subj_id],
                aug_params['aug_seg_len'],
                seg_len,
                aug_params['train_stride'],
                aug_params['warp_factor_range'],
                aug_params['n_augmented_per_segment']
            )
        elif stage == 'training':
            X, y, ecg = create_segments_sliding(
                data_preprocessed[subj_id]['EEG'],
                data_preprocessed[subj_id]['ECG_pulse'],
                data_preprocessed[subj_id]['ECG'],
                seg_len, aug_params['train_stride']
            )
        else:
            X, y, ecg = create_segments_nonoverlapping(
                data_preprocessed[subj_id]['EEG'],
                data_preprocessed[subj_id]['ECG_pulse'],
                data_preprocessed[subj_id]['ECG'],
                seg_len
            )

        X_list.append(X)
        y_list.append(y)
        ecg_list.append(ecg)
        ids_list.append(np.full(X.shape[0], subj_id, dtype=np.int32))

    return X_list, y_list, ecg_list, ids_list
