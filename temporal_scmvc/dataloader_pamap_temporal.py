import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from sampling import sample_indices
from save_indices import save_window_indices


class PAMAP2TemporalDataset(Dataset):
    def __init__(self, views, labels):
        self.views = views
        self.labels = labels

    def __len__(self):
        return self.views[0].shape[0]

    def __getitem__(self, idx):
        xs = [torch.tensor(v[idx], dtype=torch.float32) for v in self.views]
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return xs, y, idx


def _default_pamap2_path():
    repo_root = Path(__file__).resolve().parents[1]
    return (
        repo_root
        / "data"
        / "pamap2+physical+activity+monitoring"
        / "PAMAP2_Dataset"
        / "Protocol"
    )


def _default_indices_dir():
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "p2_results" / "indices"


def _parse_subject_id(filename):
    """
    PAMAP2 Protocol filenames look like:
        subject101.dat  -> 101
        subject102.dat  -> 102
    Returns int, or -1 if not parseable.
    """
    stem = Path(filename).stem  # "subject101"
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        return -1
    return int(digits)


def create_sliding_windows(X, y, subject_id, window_size=50, stride=25):
    """
    Create sliding windows from sequential PAMAP2 data.
    Keep only windows where all labels are the same.

    Returns:
        X_windows:      (n_win, window_size, n_features)
        y_windows:      (n_win,)
        subj_windows:   (n_win,)  -- constant subject_id per window
        start_positions:(n_win,)  -- start row index within this file
    """
    X_windows = []
    y_windows = []
    subj_windows = []
    start_positions = []

    for start in range(0, len(X) - window_size + 1, stride):
        end = start + window_size
        x_win = X[start:end]
        y_win = y[start:end]

        if np.all(y_win == y_win[0]):
            X_windows.append(x_win)
            y_windows.append(y_win[0])
            subj_windows.append(subject_id)
            start_positions.append(start)

    X_windows = np.asarray(X_windows, dtype=np.float32)
    y_windows = np.asarray(y_windows, dtype=np.int64)
    subj_windows = np.asarray(subj_windows, dtype=np.int64)
    start_positions = np.asarray(start_positions, dtype=np.int64)

    return X_windows, y_windows, subj_windows, start_positions


def _safe_standardize_view(x):
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, -10.0, 10.0)

    return x.astype(np.float32)


def load_pamap2_temporal(
    data_path=None,
    max_samples=5000,
    window_size=50,
    stride=25,
    selected_view_indices=None,
    seed=42,
    sample_strategy="stratified",
):
    data_path = Path(data_path) if data_path is not None else _default_pamap2_path()

    all_sequences = []
    all_labels = []
    all_subjects = []
    all_starts = []

    files = sorted([f for f in os.listdir(data_path) if f.endswith(".dat")])
    if len(files) == 0:
        raise FileNotFoundError(f"No .dat files found in: {data_path}")

    for filename in files:
        subject_id = _parse_subject_id(filename)
        file_path = os.path.join(data_path, filename)
        data = np.loadtxt(file_path)

        # remove activity 0
        data = data[data[:, 1] != 0]

        # labels
        y = data[:, 1].astype(int)

        # remove timestamp and activity column
        X = data[:, 2:]
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        Xw, yw, sw, st = create_sliding_windows(
            X, y, subject_id, window_size=window_size, stride=stride
        )

        if len(Xw) > 0:
            all_sequences.append(Xw)
            all_labels.append(yw)
            all_subjects.append(sw)
            all_starts.append(st)

    if len(all_sequences) == 0:
        raise RuntimeError("No valid PAMAP2 windows were created.")

    X_windows = np.vstack(all_sequences)
    y_windows = np.concatenate(all_labels)
    subject_windows = np.concatenate(all_subjects)
    start_positions = np.concatenate(all_starts)

    print("Windowed PAMAP2 data:", X_windows.shape, y_windows.shape)
    print("Unique subjects:", np.unique(subject_windows).tolist())

    # PAMAP2 feature layout after removing first 2 cols:
    # total = 52 features = 1 heart-rate + 17 hand IMU + 17 chest IMU + 17 ankle IMU
    view1 = X_windows[:, :, 0:1]      # heart rate
    view2 = X_windows[:, :, 1:18]     # hand IMU
    view3 = X_windows[:, :, 18:35]    # chest IMU
    view4 = X_windows[:, :, 35:52]    # ankle IMU

    def reshape_standardize(v):
        n, t, d = v.shape
        v2d = v.reshape(n * t, d)
        v2d = _safe_standardize_view(v2d)
        return v2d.reshape(n, t, d)

    views = [view1, view2, view3, view4]

    if selected_view_indices is not None:
        selected = []
        for idx in selected_view_indices:
            if idx < 0 or idx >= len(views):
                raise ValueError(
                    f"View index {idx} is out of range for {len(views)} views"
                )
            selected.append(views[idx])
        views = selected

    total_windows = len(X_windows)

    # ---- sampling ----
    if max_samples is not None and total_windows > max_samples:
        indices = sample_indices(y_windows, max_samples, seed, sample_strategy)
    else:
        indices = np.arange(total_windows, dtype=np.int64)

    views = [v[indices] for v in views]
    y_windows = y_windows[indices]
    subject_windows = subject_windows[indices]
    start_positions = start_positions[indices]

    # ---- relabel to contiguous class ids (same as before) ----
    labels, y_windows = np.unique(y_windows, return_inverse=True)
    y_windows = y_windows.astype(np.int64)
    print("Unique original labels after sampling:", labels)

    # ---- standardize views ----
    views = [reshape_standardize(v) for v in views]

    # ---- save preprocessing indices ----
    save_window_indices(
        dataset="pamap2",
        seed=seed,
        window_starts=start_positions,
        subject_ids=subject_windows,
        sampled_indices=indices,
        window_length=window_size,
        stride=stride,
        extra={
            "total_windows": int(total_windows),
            "window_size": int(window_size),
            "stride": int(stride),
            "original_labels": labels.tolist(),
        },
        save_dir=str(_default_indices_dir()),
    )

    for i, v in enumerate(views):
        print(f"View {i+1} shape:", v.shape)

    dims = [v.shape[2] for v in views]
    view = len(views)
    data_size = len(y_windows)
    class_num = len(np.unique(y_windows))

    dataset = PAMAP2TemporalDataset(views, y_windows)
    return dataset, dims, view, data_size, class_num