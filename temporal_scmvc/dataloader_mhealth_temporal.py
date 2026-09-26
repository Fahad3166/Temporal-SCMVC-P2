import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from sampling import sample_indices


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


def create_sliding_windows(X, y, window_size=50, stride=25):
    """
    Create sliding windows from sequential data.

    Each window keeps only rows with the same label.
    If a window contains mixed labels, we skip it.

    Args:
        X: shape (num_rows, num_features)
        y: shape (num_rows,)
        window_size: number of time steps per window
        stride: step size between windows

    Returns:
        X_windows: shape (num_windows, window_size, num_features)
        y_windows: shape (num_windows,)
    """
    X_windows = []
    y_windows = []

    for start in range(0, len(X) - window_size + 1, stride):
        end = start + window_size
        x_win = X[start:end]
        y_win = y[start:end]

        # Keep only pure-label windows
        if np.all(y_win == y_win[0]):
            X_windows.append(x_win)
            y_windows.append(y_win[0])

    X_windows = np.array(X_windows, dtype=np.float32)
    y_windows = np.array(y_windows, dtype=np.int64)

    return X_windows, y_windows


def load_mhealth_temporal(
    data_path=None,
    max_samples=None,
    window_size=50,
    stride=25,
    selected_view_indices=None,
    seed=42,
    sample_strategy="stratified"
):
    if data_path is None:
        repo_root = Path(__file__).resolve().parents[1]
        data_path = repo_root / "data" / "MHEALTHDATASET"
    else:
        data_path = Path(data_path)

    all_data = []

    for file in sorted(os.listdir(data_path)):
        if file.endswith(".log"):
            file_path = os.path.join(data_path, file)
            data = np.loadtxt(file_path)
            all_data.append(data)

    if len(all_data) == 0:
        raise FileNotFoundError(f"No .log files found in: {data_path}")

    data = np.vstack(all_data)

    # Last column is label
    X = data[:, :-1]
    y = data[:, -1].astype(int)

    # Remove null label 0
    mask = y != 0
    X = X[mask]
    y = y[mask]

    # Standardize feature columns
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    X = X.astype(np.float32)

    print("Original MHEALTH row data:", X.shape, y.shape)

    # Create temporal windows
    X_windows, y_windows = create_sliding_windows(
        X, y, window_size=window_size, stride=stride
    )

    print("Windowed MHEALTH data:", X_windows.shape, y_windows.shape)

    # Split into 3 views by sensor groups
    # shape: (num_windows, time, features)
    view1 = X_windows[:, :, 0:8]     # chest
    view2 = X_windows[:, :, 8:16]    # wrist / arm
    view3 = X_windows[:, :, 16:23]   # ankle

    views = [view1, view2, view3]

    if selected_view_indices is not None:
        selected = []
        for idx in selected_view_indices:
            if idx < 0 or idx >= len(views):
                raise ValueError(f"View index {idx} is out of range for {len(views)} views")
            selected.append(views[idx])
        views = selected

    if max_samples is not None and len(y_windows) > max_samples:
        indices = sample_indices(y_windows, max_samples, seed, sample_strategy)
        views = [v[indices] for v in views]
        y_windows = y_windows[indices]

    for i, v in enumerate(views):
        print(f"View {i+1} shape: {v.shape}")

    dims = [v.shape[2] for v in views]   # features per time step for each view
    view = len(views)
    data_size = len(y_windows)
    class_num = len(np.unique(y_windows))

    dataset = MHEALTHTemporalDataset(views, y_windows)

    return dataset, dims, view, data_size, class_num
