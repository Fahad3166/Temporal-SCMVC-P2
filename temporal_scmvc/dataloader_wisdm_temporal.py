import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from sampling import sample_indices
from save_indices import save_window_indices


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


def _build_windows(file_list, window_size=200, max_files=None):
    """
    Build temporal windows.

    Returns:
        windows: dict keyed by (subject, activity, w_idx) -> (chunk, start_row)
            chunk shape: [window_size, 3]
            start_row:   int, the row index in the raw per-(subject, activity)
                         sequence where this window starts
    """
    windows = {}

    if max_files is not None:
        file_list = file_list[:max_files]

    for filepath in file_list:
        rows = _load_sensor_file(filepath)
        if not rows:
            continue

        by_group = {}

        for subject, activity, timestamp, x, y, z in rows:
            key = (subject, activity)
            by_group.setdefault(key, []).append((timestamp, x, y, z))

        for (subject, activity), seq in by_group.items():
            seq.sort(key=lambda t: t[0])

            arr = np.array([[x, y, z] for _, x, y, z in seq], dtype=np.float32)

            num_windows = len(arr) // window_size

            for w_idx in range(num_windows):
                start = w_idx * window_size
                end = start + window_size
                chunk = arr[start:end]  # [window_size, 3]
                windows[(subject, activity, w_idx)] = (chunk, int(start))

    return windows


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

    sensor_files = {}

    print("\nDetected files per view:")
    for name, folder in folder_map.items():
        files = _collect_files_from_dir(folder)
        sensor_files[name] = files
        print(f"{name}: {len(files)} files")

    print("\nSelected WISDM views:", selected_names)

    sensor_windows = {}

    for name in selected_names:
        files = sensor_files[name]

        if len(files) == 0:
            raise FileNotFoundError(
                f"No files found for view '{name}' under {folder_map[name]}"
            )

        sensor_windows[name] = _build_windows(
            files,
            window_size=window_size,
            max_files=max_files_per_view,
        )

        print(f"{name}: {len(sensor_windows[name])} windows")

    common_keys = None

    for name in selected_names:
        keys = set(sensor_windows[name].keys())
        common_keys = keys if common_keys is None else (common_keys & keys)

    common_keys = sorted(common_keys)

    if len(common_keys) == 0:
        raise RuntimeError("No aligned windows found across selected WISDM views.")

    print("Aligned windows across selected views:", len(common_keys))

    label_map = {}
    next_label = 0

    views = [[] for _ in selected_names]
    labels = []
    subjects = []
    start_rows = []

    for key in common_keys:
        subject, activity, w_idx = key

        if activity not in label_map:
            label_map[activity] = next_label
            next_label += 1

        for i, name in enumerate(selected_names):
            chunk, _start = sensor_windows[name][key]
            views[i].append(chunk)

        labels.append(label_map[activity])
        subjects.append(subject)
        # start row = w_idx * window_size (non-overlapping windows)
        start_rows.append(int(w_idx) * window_size)

    views = [np.array(v, dtype=np.float32) for v in views]
    labels = np.array(labels, dtype=np.int64)
    subjects = np.array(subjects, dtype=np.int64)
    start_rows = np.array(start_rows, dtype=np.int64)

    total_windows = len(labels)

    # ---- sampling ----
    if max_samples is not None and total_windows > max_samples:
        indices = sample_indices(labels, max_samples, seed, sample_strategy)
    else:
        indices = np.arange(total_windows, dtype=np.int64)

    views = [v[indices] for v in views]
    labels = labels[indices]
    subjects = subjects[indices]
    start_rows = start_rows[indices]

    # ---- standardize each view ----
    views = [_standardize_view_windows(v) for v in views]

    # ---- save preprocessing indices ----
    save_window_indices(
        dataset="wisdm",
        seed=seed,
        window_starts=start_rows,
        subject_ids=subjects,
        sampled_indices=indices,
        window_length=window_size,
        stride=window_size,  # non-overlapping by construction
        extra={
            "total_windows": int(total_windows),
            "window_size": int(window_size),
            "max_files_per_view": int(max_files_per_view),
            "selected_views": selected_names,
            "label_map": label_map,
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