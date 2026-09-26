import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MHEALTHDataset(Dataset):
    def __init__(self, views, labels):
        self.views = views
        self.labels = labels

    def __len__(self):
        return self.views[0].shape[0]

    def __getitem__(self, idx):
        xs = [torch.tensor(v[idx], dtype=torch.float32) for v in self.views]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return xs, label, idx


def load_mhealth(
    data_path='/Users/fahad/Documents/MY#Documents/FAHAD ALI/Uni Wien/6th Semester/P-1/SCMVC/data/MHEALTHDATASET',
    max_samples=30000
):
    all_data = []

    for file in os.listdir(data_path):
        if file.endswith(".log"):
            file_path = os.path.join(data_path, file)
            data = np.loadtxt(file_path)
            all_data.append(data)

    data = np.vstack(all_data)

    # Last column is label
    X = data[:, :-1]
    y = data[:, -1].astype(int)

    # Remove null label 0
    mask = y != 0
    X = X[mask]
    y = y[mask]

    # Normalize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # Limit samples for faster experiments
    if max_samples is not None and X.shape[0] > max_samples:
        X = X[:max_samples]
        y = y[:max_samples]

    print("Loaded MHEALTH dataset:", X.shape)

    # Create 3 sensor-based views
    view1 = X[:, 0:8]      # chest
    view2 = X[:, 8:16]     # wrist / arm
    view3 = X[:, 16:23]    # ankle

    views = [view1, view2, view3]

    for i, v in enumerate(views):
        print(f"View {i+1} shape:", v.shape)

    dims = [v.shape[1] for v in views]
    view = len(views)
    data_size = X.shape[0]
    class_num = len(np.unique(y))

    dataset = MHEALTHDataset(views, y)

    return dataset, dims, view, data_size, class_num