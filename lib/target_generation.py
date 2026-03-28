import numpy as np


def compute_R_distance_next(r_peaks, sig_len):
    r_peaks = np.asarray(r_peaks, dtype=int)
    dist = np.full(sig_len, np.nan, dtype=float)

    for i in range(r_peaks.size - 1):
        a = r_peaks[i]
        b = r_peaks[i + 1]
        if b <= a:
            continue
        seg = np.arange(a, b + 1)
        left = seg - a
        right = b - seg
        min_dist = np.minimum(left, right).astype(float)  # 0..(b-a)/2
        interval = float(b - a)  # interval length in samples
        values = 2.0 * min_dist  # ranges 0 .. interval
        dist[a:b+1] = values

    # before first peak: use interval to next peak
    first = r_peaks[0]
    if first > 0:
        next_interval = float(r_peaks[1] - r_peaks[0]) if r_peaks.size > 1 else float(sig_len)
        seg = np.arange(0, first)
        d = first - seg
        if first > 0:
            values = (d / float(first)) * next_interval
        dist[0:first] = values

    # after last peak: use interval to previous peak
    last = r_peaks[-1]
    if last < sig_len - 1:
        prev_interval = float(r_peaks[-1] - r_peaks[-2]) if r_peaks.size > 1 else float(last + 1)
        seg = np.arange(last + 1, sig_len)
        d = seg - last
        if (sig_len - 1 - last) > 0:
            values = (d / float(sig_len - 1 - last)) * prev_interval
        else:
            values = np.minimum(d, prev_interval)
        dist[last+1:] = values

    return dist
