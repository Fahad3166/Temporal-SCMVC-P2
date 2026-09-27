import glob
import os
import numpy as np

INDICES_DIR = "../p2_results/indices"

print(f"{'file':<32} {'win':>5} {'stride':>7} {'overlap%':>10} {'min_gap':>9} {'max_gap':>9}")
print("-" * 78)

for f in sorted(glob.glob(os.path.join(INDICES_DIR, "*_indices.npz"))):
    d = np.load(f, allow_pickle=True)
    win = int(d["window_length"])
    stride = int(d["stride"])

    starts = d["window_starts"]
    # how far apart are consecutive window starts? (across the whole sampled set,
    # sorted ascending — this reveals the true stride at the source)
    sorted_starts = np.sort(np.unique(starts))
    if len(sorted_starts) >= 2:
        gaps = np.diff(sorted_starts)
        # exclude gaps that span across subjects / activity boundaries
        # by keeping only small positive gaps (<= window)
        local_gaps = gaps[(gaps > 0) & (gaps <= win)]
        min_gap = int(local_gaps.min()) if len(local_gaps) else -1
        max_gap = int(local_gaps.max()) if len(local_gaps) else -1
    else:
        min_gap = max_gap = -1

    overlap_pct = 100.0 * max(0.0, 1.0 - stride / win)
    print(f"{os.path.basename(f):<32} {win:>5} {stride:>7} {overlap_pct:>9.1f}% "
          f"{min_gap:>9} {max_gap:>9}")