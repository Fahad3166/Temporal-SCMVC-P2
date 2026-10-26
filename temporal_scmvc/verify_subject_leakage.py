import glob
import os
import numpy as np

INDICES_DIR = "../p2_results/indices"
DATASETS = ["har", "mhealth", "pamap2", "wisdm"]
SEEDS = [0, 1, 2]


def load_seed(dataset, seed):
    path = os.path.join(INDICES_DIR, f"{dataset}_seed{seed}_indices.npz")
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True)


def gini(counts):
    """Gini coefficient of a count vector. 0 = perfectly equal, 1 = one holds everything."""
    sorted_c = np.sort(counts.astype(np.float64))
    n = len(sorted_c)
    cum = np.cumsum(sorted_c)
    if cum[-1] == 0:
        return 0.0
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


def summarize(dataset):
    print("=" * 78)
    print(f"Dataset: {dataset.upper()}")
    print("=" * 78)

    for seed in SEEDS:
        d = load_seed(dataset, seed)
        if d is None:
            print(f"  seed {seed}: MISSING")
            continue

        subjects = d["subject_ids"]
        labels = d["labels"]
        total = len(subjects)
        unique, counts = np.unique(subjects, return_counts=True)

        shares = counts / total
        max_share = shares.max()
        min_share = shares.min()
        n_subjects = len(unique)
        top3_idx = np.argsort(counts)[::-1][:3]
        top3 = [(int(unique[i]), int(counts[i]), round(float(shares[i]), 3)) for i in top3_idx]
        g = gini(counts)

        print(f"  seed {seed}: total={total}, subjects={n_subjects}, "
              f"max_share={max_share:.3f}, min_share={min_share:.3f}, Gini={g:.3f}")
        print(f"    top-3 subjects (id, count, share): {top3}")

        # per-class subject coverage
        # how many distinct subjects appear per class? min/median/max
        per_class_subjects = []
        for lab in np.unique(labels):
            mask = labels == lab
            n_subj = len(np.unique(subjects[mask]))
            per_class_subjects.append(n_subj)
        per_class_subjects = np.array(per_class_subjects)
        print(f"    subjects per class: "
              f"min={per_class_subjects.min()}, "
              f"median={int(np.median(per_class_subjects))}, "
              f"max={per_class_subjects.max()}")
        print()


if __name__ == "__main__":
    for ds in DATASETS:
        summarize(ds)