import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from sampling import sample_indices
from save_indices import save_window_indices


class MHEALTHTemporalDataset(Dataset):
    def __init__(self, views, labels):
        self.views = views
        self.labels = labels

    def __len__(self):
        return self.views[0].shape[0]

    def __getitem__(self, idx):
        xs = [torch.tensor(v[idx], dtype=torch.float32) for v in self.views]
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return xs, y, idx


def _default_mhealth_path():
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "data" / "MHEALTHDATASET"


def _default_indices_dir():
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "p2_results" / "indices"


def _parse_subject_id(filename):
    """
    Extract subject number from filenames like:
        mHealth_subject1.log
        mHealth_subject10.log
        subject1.log
    Returns an int, or -1 if not parseable.
    """
    stem = Path(filename).stem  # e.g. "mHealth_subject1"
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        return -1
    return int(digits)


def _load_per_subject(data_path):
    """
    Load each subject's log file separately so that subject identity
    is preserved per row.

    Returns:
        X: (total_rows, num_features)
        y: (total_rows,) int64
        subject_ids: (total_rows,) int64
    """
    X_list = []
    y_list = []
    subj_list = []

    files = sorted(f for f in os.listdir(data_path) if f.endswith(".log"))
    if not files:
        raise FileNotFoundError(f"No .log files found in: {data_path}")

    for filename in files:
        subject_id = _parse_subject_id(filename)
        file_path = os.path.join(data_path, filename)
        raw = np.loadtxt(file_path)

        X_sub = raw[:, :-1].astype(np.float32)
        y_sub = raw[:, -1].astype(np.int64)

        # drop null-label rows (label 0)
        keep = y_sub != 0
        X_sub = X_sub[keep]
        y_sub = y_sub[keep]

        X_list.append(X_sub)
        y_list.append(y_sub)
        subj_list.append(np.full(len(y_sub), subject_id, dtype=np.int64))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    subject_ids = np.concatenate(subj_list, axis=0)
    return X, y, subject_ids


def create_sliding_windows(X, y, subject_ids, window_size=50, stride=25):
    """
    Create sliding windows from sequential data.

    Windows are only kept when:
    - all labels inside the window are identical
    - all rows in the window come from the same subject
      (prevents windows that straddle subject boundaries)

    Returns:
        X_windows:  (num_windows, window_size, num_features)
        y_windows:  (num_windows,)  int64
        subj_windows: (num_windows,) int64
        start_positions: (num_windows,) int64  -- start row index in the
                         concatenated X array
    """
    X_windows = []
    y_windows = []
    subj_windows = []
    start_positions = []

    n = len(X)
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        y_win = y[start:end]
        s_win = subject_ids[start:end]

        # pure-label and pure-subject windows only
        if not np.all(y_win == y_win[0]):
            continue
        if not np.all(s_win == s_win[0]):
            continue

        X_windows.append(X[start:end])
        y_windows.append(int(y_win[0]))
        subj_windows.append(int(s_win[0]))
        start_positions.append(int(start))

    X_windows = np.asarray(X_windows, dtype=np.float32)
    y_windows = np.asarray(y_windows, dtype=np.int64)
    subj_windows = np.asarray(subj_windows, dtype=np.int64)
    start_positions = np.asarray(start_positions, dtype=np.int64)

    return X_windows, y_windows, subj_windows, start_positions


def load_mhealth_temporal(
    data_path=None,
    max_samples=None,
    window_size=50,
    stride=25,
    selected_view_indices=None,
    seed=42,
    sample_strategy="stratified",
):
    data_path = Path(data_path) if data_path is not None else _default_mhealth_path()

    # ---- load with subject identity preserved ----
    X, y, subject_ids = _load_per_subject(data_path)

    # ---- global standardization (as before) ----
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    X = X.astype(np.float32)

    print("Original MHEALTH row data:", X.shape, y.shape)
    print("Unique subjects:", np.unique(subject_ids).tolist())

    # ---- windowing ----
    X_windows, y_windows, subj_windows, start_positions = create_sliding_windows(
        X, y, subject_ids, window_size=window_size, stride=stride
    )
    print("Windowed MHEALTH data:", X_windows.shape, y_windows.shape)

    # ---- view split ----
    view1 = X_windows[:, :, 0:8]     # chest
    view2 = X_windows[:, :, 8:16]    # wrist / arm
    view3 = X_windows[:, :, 16:23]   # ankle
    views = [view1, view2, view3]

    if selected_view_indices is not None:
        selected = []
        for idx in selected_view_indices:
            if idx < 0 or idx >= len(views):
                raise ValueError(
                    f"View index {idx} is out of range for {len(views)} views"
                )
            selected.append(views[idx])
        views = selected

    # ---- sampling ----
    total_windows = len(y_windows)
    if max_samples is not None and total_windows > max_samples:
        indices = sample_indices(y_windows, max_samples, seed, sample_strategy)
    else:
        indices = np.arange(total_windows, dtype=np.int64)

    views = [v[indices] for v in views]
    y_windows = y_windows[indices]
    subj_windows = subj_windows[indices]
    start_positions = start_positions[indices]

    # ---- save preprocessing indices ----
    save_window_indices(
        dataset="mhealth",
        seed=seed,
        window_starts=start_positions,
        subject_ids=subj_windows,
        sampled_indices=indices,
        window_length=window_size,
        stride=stride,
        extra={
            "total_windows": total_windows,
            "window_size": window_size,
            "stride": stride,
        },
        save_dir=str(_default_indices_dir()),
    )

    for i, v in enumerate(views):
        print(f"View {i+1} shape:", v.shape)

    dims = [v.shape[2] for v in views]
    view = len(views)
    data_size = len(y_windows)
    class_num = len(np.unique(y_windows))

    dataset = MHEALTHTemporalDataset(views, y_windows)
    return dataset, dims, view, data_size, class_num