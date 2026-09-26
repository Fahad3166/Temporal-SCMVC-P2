import glob
import os
import numpy as np

for f in sorted(glob.glob("../p2_results/indices/*.npz")):
    d = np.load(f, allow_pickle=True)
    n_windows = len(d["window_starts"])
    n_sampled = len(d["sampled_indices"])
    subjects = np.unique(d["subject_ids"])
    win_len = int(d["window_length"])
    stride = int(d["stride"])
    print(f"{os.path.basename(f)}: windows={n_windows}, "
          f"sampled={n_sampled}, subjects={subjects.tolist()}, "
          f"win_len={win_len}, stride={stride}")