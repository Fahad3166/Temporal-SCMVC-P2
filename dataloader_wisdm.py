import os
import numpy as np
import torch
from torch.utils.data import Dataset


class WISDMDataset(Dataset):
    def __init__(self, views, labels):
        self.views = views
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        xs = [torch.tensor(v[idx], dtype=torch.float32) for v in self.views]
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return xs, y, idx


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
            # accept every regular file
            if os.path.isfile(full):
                files.append(full)

    return sorted(files)


def _build_windows(file_list, window_size=200, max_files=None):
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
                chunk = arr[start:end]
                flat = chunk.reshape(-1)  # 200 * 3 = 600
                windows[(subject, activity, w_idx)] = flat

    return windows


def load_wisdm(
    data_root="/Users/fahad/Documents/MY#Documents/FAHAD ALI/Uni Wien/6th Semester/P-1/SCMVC/data/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/wisdm-dataset/raw",
    window_size=200,
    max_files_per_view=10,
    selected_view_indices=None,
    max_samples =1000,
):
    # exact folders from your dataset
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
    for k, v in folder_map.items():
        print(f"{k}: {v}")

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
            raise FileNotFoundError(f"No files found for view '{name}' under {folder_map[name]}")
        sensor_windows[name] = _build_windows(
            files,
            window_size=window_size,
            max_files=max_files_per_view,
        )

    common_keys = None
    for name in selected_names:
        keys = set(sensor_windows[name].keys())
        common_keys = keys if common_keys is None else (common_keys & keys)

    common_keys = sorted(common_keys)
    if len(common_keys) == 0:
        raise RuntimeError("No aligned windows found across selected WISDM views.")

    label_map = {}
    next_label = 0

    views = [[] for _ in selected_names]
    labels = []

    for key in common_keys:
        _, activity, _ = key

        if activity not in label_map:
            label_map[activity] = next_label
            next_label += 1

        labels.append(label_map[activity])

        for i, name in enumerate(selected_names):
            views[i].append(sensor_windows[name][key])

    views = [np.array(v, dtype=np.float32) for v in views]
    labels = np.array(labels, dtype=np.int64)

    for i in range(len(views)):
        mean = views[i].mean(axis=0, keepdims=True)
        std = views[i].std(axis=0, keepdims=True) + 1e-8
        views[i] = (views[i] - mean) / std

    dataset = WISDMDataset(views, labels)
    dims = [v.shape[1] for v in views]
    view = len(views)
    data_size = len(labels)
    class_num = len(np.unique(labels))

    print("\nLoaded WISDM dataset:")
    print("  samples:", data_size)
    print("  classes:", class_num)
    for i, v in enumerate(views):
        print(f"  View {i+1} shape: {v.shape}")

    return dataset, dims, view, data_size, class_num