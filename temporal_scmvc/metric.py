import numpy as np
import torch
from sklearn.cluster import KMeans
import torch.nn.functional as F
from sklearn import metrics
import warnings

# Suppress sklearn warnings temporarily
#warnings.filterwarnings("ignore", category=RuntimeWarning)
#warnings.filterwarnings("ignore", category=UserWarning)


# ========================
# MORE ROBUST NORMALIZATION
# ========================
def safe_normalize(x):
    """Robust row-wise L2 normalization with multiple safety layers"""
    
    # Layer 1: Convert to float64 for better precision
    if x.dtype == np.float32:
        x = x.astype(np.float64)
    
    # Layer 2: Handle NaN/Inf
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Layer 3: Clip extreme values
    x = np.clip(x, -10.0, 10.0)
    
    # Layer 4: Compute norm with epsilon
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)  # Prevent division by zero
    
    # Layer 5: Normalize
    result = x / norms
    
    # Layer 6: Final clip
    result = np.clip(result, -10.0, 10.0)
    
    # Layer 7: Back to float32 if needed
    if result.dtype == np.float64:
        result = result.astype(np.float32)
    
    return result


def clustering_metrics(y_true, y_pred):
    """Clustering metrics with error handling"""
    
    acc = clustering_accuracy(y_true, y_pred)
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    pur = purity_score(y_true, y_pred)
    
    return acc, nmi, ari, pur


def clustering_accuracy(y_true, y_pred):
    """Hungarian algorithm with safe casting"""
    
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
    """Purity with safe division"""
    
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    
    if contingency_matrix.sum() == 0:
        return 0.0
    
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


# ========================
# IMPROVED VALIDATION
# ========================
def valid(
    model,
    device,
    dataset,
    view,
    data_size,
    class_num,
    eval_h=True,
    epoch=0,
    kmeans_seed=42,
    kmeans_n_init=10,
    return_ari=False,
):
    
    model.eval()
    
    Xs = [[] for _ in range(view)]
    Ys = []
    
    # Collect dataset
    for xs, y, _ in dataset:
        for v in range(view):
            x_v = (xs[v].cpu().numpy() if torch.is_tensor(xs[v]) else np.asarray(xs[v]))
            Xs[v].append(x_v)
        Ys.append(int(y.item()) if torch.is_tensor(y) else int(y))
    
    Xs = [np.array(x, dtype=np.float32) for x in Xs]
    Y = np.array(Ys, dtype=np.int64)
    
    # Convert to tensor
    Xs_tensor = [torch.tensor(x, dtype=torch.float32, device=device) for x in Xs]
    
    # Forward pass with gradient disabled
    with torch.no_grad():
        _, zs, rs, H = model(Xs_tensor)
    
    # 🔥 More robust normalization in torch
    H = F.normalize(H, dim=1, eps=1e-8)
    H = torch.clamp(H, -5.0, 5.0)
    
    rs = [F.normalize(r, dim=1, eps=1e-8) for r in rs]
    rs = [torch.clamp(r, -5.0, 5.0) for r in rs]
    
    # Select features
    if eval_h:
        features = H.detach().cpu().numpy()
    else:
        features = torch.cat(rs, dim=1).detach().cpu().numpy()
    
    # 🔥 Aggressive cleaning
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    features = np.clip(features, -10.0, 10.0)
    features = safe_normalize(features)
    features = np.clip(features, -5.0, 5.0)
    
    # Safety checks
    if not np.isfinite(features).all():
        print(f"WARNING: Non-finite at epoch {epoch}, skipping")
        return (0, 0, 0, 0) if return_ari else (0, 0, 0)
    
    if np.std(features) < 1e-6:
        print(f"WARNING: Collapsed features at epoch {epoch}, skipping")
        return (0, 0, 0, 0) if return_ari else (0, 0, 0)
    
    # 🔥 KMeans with explicit error handling
    try:
        kmeans = KMeans(
            n_clusters=class_num,
            n_init=kmeans_n_init,
            max_iter=300,
            tol=1e-4,
            algorithm="lloyd",
            random_state=kmeans_seed,
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = kmeans.fit_predict(features)
        
        acc, nmi, ari, pur = clustering_metrics(Y, y_pred)
        
        print(f"Epoch {epoch:3d} | ACC: {acc:.4f} | NMI: {nmi:.4f} | ARI: {ari:.4f} | PUR: {pur:.4f}")
        
        return (acc, nmi, ari, pur) if return_ari else (acc, nmi, pur)
        
    except Exception as e:
        print(f"KMeans failed at epoch {epoch}: {str(e)[:100]}")
        return (0, 0, 0, 0) if return_ari else (0, 0, 0)
