import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn import metrics


def safe_normalize(x):
    """Normalize safely to prevent NaN/Inf and exploding values."""
    x = np.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)
    x = np.clip(x, -1e3, 1e3)

    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.where(norm < 1e-8, 1e-8, norm)

    return x / norm


def clustering_metrics(y_true, y_pred):
    acc = clustering_accuracy(y_true, y_pred)
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    pur = purity_score(y_true, y_pred)
    return acc, nmi, ari, pur


def clustering_accuracy(y_true, y_pred):
    from scipy.optimize import linear_sum_assignment

    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_sum_assignment(w.max() - w)
    return sum(w[i, j] for i, j in zip(ind[0], ind[1])) * 1.0 / y_pred.size


def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def valid(model, device, dataset, view, data_size, class_num, eval_h=True, epoch=0):
    model.eval()

    Xs = [[] for _ in range(view)]
    Ys = []

    for xs, y, _ in dataset:
        for v in range(view):
            x_v = xs[v].cpu().numpy() if torch.is_tensor(xs[v]) else np.asarray(xs[v])
            Xs[v].append(x_v)

        if torch.is_tensor(y):
            Ys.append(int(y.item()))
        else:
            Ys.append(int(y))

    Xs = [np.array(x, dtype=np.float32) for x in Xs]
    Y = np.array(Ys, dtype=np.int64)

    Xs_tensor = [torch.tensor(x, dtype=torch.float32, device=device) for x in Xs]

    with torch.no_grad():
        _, zs, rs, H = model(Xs_tensor)

    if eval_h:
        features = H.detach().cpu().numpy()
    else:
        features = torch.cat(rs, dim=1).detach().cpu().numpy()

    features = np.asarray(features, dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=1e3, neginf=-1e3)
    features = np.clip(features, -1e3, 1e3)
    features = safe_normalize(features)

    if not np.isfinite(features).all():
        print(f"Non-finite values detected in features at epoch {epoch}, skipping evaluation")
        return 0, 0, 0

    if np.isnan(features).any():
        print(f"NaN detected in features at epoch {epoch}, skipping evaluation")
        return 0, 0, 0

    feature_std = np.std(features)
    if feature_std < 1e-8:
        print(f"Collapsed features at epoch {epoch}, skipping evaluation")
        return 0, 0, 0

    try:
        kmeans = KMeans(
            n_clusters=class_num,
            n_init=10,
            random_state=42
        )
        y_pred = kmeans.fit_predict(features)

        acc, nmi, ari, pur = clustering_metrics(Y, y_pred)

        print(
            f"Epoch {epoch} The clustering performace: "
            f"ACC = {acc:.4f} NMI = {nmi:.4f} ARI = {ari:.4f} PUR={pur:.4f}"
        )

        return acc, nmi, pur

    except Exception as e:
        print(f"KMeans failed at epoch {epoch}: {e}")
        return 0, 0, 0