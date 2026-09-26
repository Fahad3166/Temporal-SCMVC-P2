import numpy as np


def sample_indices(labels, max_samples=None, seed=42, strategy="stratified"):
    if max_samples is None or len(labels) <= max_samples:
        return np.arange(len(labels))

    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)

    if strategy == "random":
        return rng.choice(len(labels), size=max_samples, replace=False)

    if strategy == "first":
        return np.arange(max_samples)

    if strategy not in {"stratified", "balanced"}:
        raise ValueError("sample_strategy must be one of: stratified, balanced, random, first")

    unique_labels, counts = np.unique(labels, return_counts=True)
    class_indices = {label: np.flatnonzero(labels == label) for label in unique_labels}

    if strategy == "balanced":
        base = max_samples // len(unique_labels)
        remainder = max_samples % len(unique_labels)
        per_class = {label: min(base, len(class_indices[label])) for label in unique_labels}

        for label in unique_labels[:remainder]:
            if per_class[label] < len(class_indices[label]):
                per_class[label] += 1
    else:
        proportions = counts / counts.sum()
        raw = proportions * max_samples
        base_counts = np.floor(raw).astype(int)

        if max_samples >= len(unique_labels):
            base_counts = np.maximum(base_counts, 1)

        base_counts = np.minimum(base_counts, counts)

        while base_counts.sum() > max_samples:
            candidates = np.where(base_counts > 1)[0]
            if len(candidates) == 0:
                candidates = np.where(base_counts > 0)[0]
            idx = candidates[np.argmax(base_counts[candidates])]
            base_counts[idx] -= 1

        fractional = raw - np.floor(raw)
        order = np.argsort(-fractional)

        while base_counts.sum() < max_samples:
            changed = False
            for idx in order:
                if base_counts[idx] < counts[idx]:
                    base_counts[idx] += 1
                    changed = True
                    if base_counts.sum() == max_samples:
                        break
            if not changed:
                break

        per_class = {
            label: int(base_counts[i])
            for i, label in enumerate(unique_labels)
        }

    selected = []
    for label in unique_labels:
        k = per_class[label]
        if k <= 0:
            continue
        selected.append(rng.choice(class_indices[label], size=k, replace=False))

    selected = np.concatenate(selected)
    rng.shuffle(selected)
    return selected
