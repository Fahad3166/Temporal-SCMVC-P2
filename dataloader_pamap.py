import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class MultiViewDataset(Dataset):
    def __init__(self, views, labels):
        self.views = [torch.tensor(v, dtype=torch.float32) for v in views]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        xs = [v[idx] for v in self.views]
        return xs, self.labels[idx], idx


def _safe_standardize(x: np.ndarray) -> np.ndarray:
    """
    Standardize one view safely.
    Also prevents NaN/Inf and clips extreme values.
    """
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, -10.0, 10.0)

    return x.astype(np.float32)


def load_pamap2(
    data_path='/Users/fahad/Documents/MY#Documents/FAHAD ALI/Uni Wien/6th Semester/P-1/SCMVC/data/pamap2+physical+activity+monitoring/PAMAP2_Dataset/Protocol',
    max_samples=30000,
    seed=42
):
    """
    Load PAMAP2 and split into 4 views:
      v1 = heart rate
      v2 = hand IMU
      v3 = chest IMU
      v4 = ankle IMU
    """

    rng = np.random.default_rng(seed)
    all_data = []

    for file in sorted(os.listdir(data_path)):
        if file.endswith(".dat"):
            file_path = os.path.join(data_path, file)
            data = np.loadtxt(file_path)

            # Remove activity 0
            data = data[data[:, 1] != 0]

            # Replace NaN / Inf
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            all_data.append(data)

    if len(all_data) == 0:
        raise FileNotFoundError(f"No .dat files found in: {data_path}")

    data = np.vstack(all_data)

    # Subsample for faster experiments
    if max_samples is not None and data.shape[0] > max_samples:
        indices = rng.choice(data.shape[0], max_samples, replace=False)
        data = data[indices]

    print("Loaded PAMAP2 dataset:", data.shape)

    # Labels
    y = data[:, 1].astype(int)

    # Features: remove timestamp and activity column
    X = data[:, 2:]

    # Safety cleanup
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e6, 1e6)

    # -----------------------------
    # Define views
    # -----------------------------
    # PAMAP2 after removing first 2 cols:
    # [heart rate] + [hand IMU] + [chest IMU] + [ankle IMU]

    v1 = X[:, 0:1]       # heart rate
    v2 = X[:, 1:20]      # hand IMU
    v3 = X[:, 20:40]     # chest IMU
    v4 = X[:, 40:60]     # ankle IMU

    views = [v1, v2, v3, v4]

    # Safe normalization per view
    views = [_safe_standardize(v) for v in views]

    # Final safety check
    for i, v in enumerate(views):
        if np.isnan(v).any() or np.isinf(v).any():
            raise ValueError(f"View {i+1} still contains NaN/Inf after preprocessing.")

    dataset = MultiViewDataset(views, y)

    dims = [v.shape[1] for v in views]
    view = len(views)
    data_size = len(y)
    class_num = len(np.unique(y))

    print("Number of views:", view)
    for i, v in enumerate(views):
        print(f"View {i+1} shape: {v.shape}")

    return dataset, dims, view, data_size, class_num