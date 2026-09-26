import os
import numpy as np


def save_window_indices(dataset, seed, window_starts, subject_ids,
                        sampled_indices, window_length, stride,
                        extra=None, save_dir="p2_results/indices"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{dataset}_seed{seed}_indices.npz")

    payload = dict(
        window_starts=np.asarray(window_starts, dtype=np.int64),
        subject_ids=np.asarray(subject_ids),
        sampled_indices=np.asarray(sampled_indices, dtype=np.int64),
        window_length=int(window_length),
        stride=int(stride),
    )
    if extra:
        payload["extra"] = np.array([extra], dtype=object)

    np.savez_compressed(path, **payload)
    print(f"[save_window_indices] wrote {path}")
    return path