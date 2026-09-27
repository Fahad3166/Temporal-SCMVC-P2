import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from sampling import sample_indices
from save_indices import save_window_indices


# WISDM was recorded at ~20 Hz. We resample all views onto this grid.
WISDM_TARGET_HZ = 20.0
WISDM_TARGET_DT = 1.0 / WISDM_TARGET_HZ  # 0.05 s


class WISDMTemporalDataset(Dataset):
    def __init__(self, views, labels):
        self.views = views
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        xs = [torch.tensor(v[idx], dtype=torch.float32) for v in self.views]
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return xs, y, idx


def _default_wisdm_root():
    repo_root = Path(__file__).resolve().parents[1]
    return (
        repo_root
        / "data"
        / "wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset"
        / "wisdm-dataset"
        / "raw"
    )


def _default_indices_dir():
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "p2_results" / "indices"


def _parse_line(line: str):
    line = line.strip()
    if not line:
        return None

    line = line.rstrip(";")
    parts = [p.strip() for p in line.split(",")]

    if len(parts) != 6:
        return None

    try:
        subject = int(parts[0])
        activity = parts[1]
        timestamp = int(float(parts[2]))
        x = float(parts[3])
        y = float(parts[4])
        z = float(parts[5])
        return subject, activity, timestamp, x, y, z
    except ValueError:
        return None


def _load_sensor_file(filepath):
    rows = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = _parse_line(line)
            if parsed is not None:
                rows.append(parsed)
    return rows


def _collect_files_from_dir(folder_path):
    files = []

    if not os.path.exists(folder_path):
        print(f"Folder does not exist: {folder_path}")
        return files

    for dirpath, _, filenames in os.walk(folder_path):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            if os.path.isfile(full):
                files.append(full)

    return sorted(files)


def _group_by_subject_activity(rows):
    """
    Group raw rows by (subject, activity). Each group keeps a list of
    (timestamp, x, y, z) tuples.
    """
    grouped = {}
    for subject, activity, timestamp, x, y, z in rows:
        key = (subject, activity)
        grouped.setdefault(key, []).append((timestamp, x, y, z))

    # sort each group by timestamp
    for key in grouped:
        grouped[key].sort(key=lambda t: t[0])

    return grouped


def _resample_group_to_grid(group, target_dt):
    """
    Given a sorted list of (timestamp, x, y, z) for one (subject, activity),
    return (t_seconds, xyz_array) resampled onto a uniform grid of step target_dt.

    Steps:
      1. Zero the timestamps relative to the first sample.
      2. Convert to seconds.
      3. Build the uniform grid from 0 to the max time in seconds.
      4. Linear-interpolate each of x, y, z onto that grid.
    """
    if len(group) < 2:
        return None, None

    t0 = group[0][0]
    # timestamps in WISDM are nanoseconds
    times = np.array([(ts - t0) / 1e9 for ts, _, _, _ in group], dtype=np.float64)
    values = np.array([[x, y, z] for _, x, y, z in group], dtype=np.float32)

    # drop duplicate timestamps (keep first) to keep np.interp happy
    uniq_mask = np.concatenate(([True], np.diff(times) > 0))
    times = times[uniq_mask]
    values = values[uniq_mask]

    if len(times) < 2:
        return None, None

    t_end = times[-1]
    n_steps = int(np.floor(t_end / target_dt)) + 1
    if n_steps < 2:
        return None, None

    grid = np.arange(n_steps, dtype=np.float64) * target_dt

    resampled = np.empty((n_steps, 3), dtype=np.float32)
    for c in range(3):
        resampled[:, c] = np.interp(grid, times, values[:, c]).astype(np.float32)

    return grid, resampled


def _build_aligned_windows(
    view_groups,        # dict: view_name -> {(subject, activity) -> [(ts, x, y, z), ...]}
    window_size,        # number of samples per window
    target_dt,          # seconds per sample after resampling
):
    """
    For each (subject, activity) present in ALL selected views:
      - resample each view onto a uniform grid
      - truncate all views to their common minimum length
      - extract non-overlapping windows of length `window_size`

    Returns:
        aligned: dict keyed by (subject, activity, w_idx) ->
                 { "views": {view_name: np.array[window_size, 3]},
                   "start_sample": int (start index in resampled grid) }
    """
    view_names = list(view_groups.keys())

    # collect common (subject, activity) keys across views
    common_keys = None
    for name in view_names:
        keys = set(view_groups[name].keys())
        common_keys = keys if common_keys is None else (common_keys & keys)
    common_keys = sorted(common_keys)

    aligned = {}

    for key in common_keys:
        subject, activity = key

        per_view_signals = {}
        min_len = None
        ok = True

        for name in view_names:
            group = view_groups[name][key]
            grid, resampled = _resample_group_to_grid(group, target_dt)
            if resampled is None:
                ok = False
                break
            per_view_signals[name] = resampled
            min_len = len(resampled) if min_len is None else min(min_len, len(resampled))

        if not ok or min_len is None or min_len < window_size:
            continue

        num_windows = min_len // window_size

        for w_idx in range(num_windows):
            start = w_idx * window_size
            end = start + window_size

            views_for_key = {
                name: per_view_signals[name][start:end].copy()
                for name in view_names
            }

            aligned[(subject, activity, w_idx)] = {
                "views": views_for_key,
                "start_sample": int(start),
            }

    return aligned


def _standardize_view_windows(v):
    n, t, d = v.shape
    flat = v.reshape(n * t, d)

    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True) + 1e-8

    flat = (flat - mean) / std
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    flat = np.clip(flat, -10.0, 10.0)

    return flat.reshape(n, t, d).astype(np.float32)


def load_wisdm_temporal(
    data_root=None,
    window_size=200,
    max_files_per_view=5,
    selected_view_indices=None,
    max_samples=1000,
    seed=42,
    sample_strategy="stratified",
):
    data_root = Path(data_root) if data_root is not None else _default_wisdm_root()

    folder_map = {
        "phone_accel": os.path.join(data_root, "phone", "accel"),
        "phone_gyro": os.path.join(data_root, "phone", "gyro"),
        "watch_accel": os.path.join(data_root, "watch", "accel"),
        "watch_gyro": os.path.join(data_root, "watch", "gyro"),
    }

    ordered_names = ["phone_accel", "phone_gyro", "watch_accel", "watch_gyro"]

    if selected_view_indices is None:
        selected_view_indices = [0, 1, 2, 3]

    selected_names = [ordered_names[i] for i in selected_view_indices]

    print("\nWISDM folder check:")
    for name, folder in folder_map.items():
        print(f"{name}: {folder}")

    print("\nSelected WISDM views:", selected_names)
    print(f"Resampling all views to common grid: {WISDM_TARGET_HZ} Hz "
          f"(dt={WISDM_TARGET_DT*1000:.2f} ms)")

    # ---- load and group per view ----
    view_groups = {}

    for name in selected_names:
        folder = folder_map[name]
        files = _collect_files_from_dir(folder)

        if len(files) == 0:
            raise FileNotFoundError(
                f"No files found for view '{name}' under {folder}"
            )

        if max_files_per_view is not None:
            files = files[:max_files_per_view]

        rows = []
        for fpath in files:
            rows.extend(_load_sensor_file(fpath))

        view_groups[name] = _group_by_subject_activity(rows)
        print(f"{name}: {len(files)} files, "
              f"{len(view_groups[name])} (subject, activity) groups")

    # ---- aligned windows across all selected views ----
    aligned = _build_aligned_windows(
        view_groups,
        window_size=window_size,
        target_dt=WISDM_TARGET_DT,
    )

    if len(aligned) == 0:
        raise RuntimeError("No aligned windows found across selected WISDM views.")

    print("Aligned windows across selected views:", len(aligned))

    # ---- build label map and materialize tensors ----
    label_map = {}
    next_label = 0

    views = [[] for _ in selected_names]
    labels = []
    subjects = []
    start_samples = []

    for key in sorted(aligned.keys()):
        subject, activity, w_idx = key
        entry = aligned[key]

        if activity not in label_map:
            label_map[activity] = next_label
            next_label += 1

        for i, name in enumerate(selected_names):
            views[i].append(entry["views"][name])

        labels.append(label_map[activity])
        subjects.append(subject)
        start_samples.append(entry["start_sample"])

    views = [np.array(v, dtype=np.float32) for v in views]
    labels = np.array(labels, dtype=np.int64)
    subjects = np.array(subjects, dtype=np.int64)
    start_samples = np.array(start_samples, dtype=np.int64)

    total_windows = len(labels)

    # ---- sampling ----
    if max_samples is not None and total_windows > max_samples:
        indices = sample_indices(labels, max_samples, seed, sample_strategy)
    else:
        indices = np.arange(total_windows, dtype=np.int64)

    views = [v[indices] for v in views]
    labels = labels[indices]
    subjects = subjects[indices]
    start_samples = start_samples[indices]

    # ---- standardize each view ----
    views = [_standardize_view_windows(v) for v in views]

    # ---- save preprocessing indices ----
    save_window_indices(
        dataset="wisdm",
        seed=seed,
        window_starts=start_samples,
        subject_ids=subjects,
        sampled_indices=indices,
        window_length=window_size,
        stride=window_size,  # non-overlapping on the resampled grid
        extra={
            "total_windows": int(total_windows),
            "window_size": int(window_size),
            "max_files_per_view": int(max_files_per_view) if max_files_per_view else -1,
            "selected_views": selected_names,
            "label_map": label_map,
            "resample_hz": float(WISDM_TARGET_HZ),
            "resample_dt_seconds": float(WISDM_TARGET_DT),
            "alignment": "timestamp-resampled to common 20 Hz grid per (subject, activity)",
        },
        save_dir=str(_default_indices_dir()),
    )

    print("\nUnique labels after sampling:", np.unique(labels))
    print("Label map:", label_map)

    for i, v in enumerate(views):
        print(f"View {i+1} shape: {v.shape}")

    dims = [v.shape[2] for v in views]
    view = len(views)
    data_size = len(labels)
    class_num = len(np.unique(labels))

    dataset = WISDMTemporalDataset(views, labels)
    return dataset, dims, view, data_size, class_num