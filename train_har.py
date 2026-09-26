import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.functional as F
from network import Network
from metric import valid
import numpy as np
import argparse
import random
from loss2 import ContrastiveLoss
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# =====================
# SEED
# =====================
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)

# =====================
# CONFIG
# =====================
parser = argparse.ArgumentParser(description='HAR Multi-View Clustering')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--learning_rate", default=0.0003, type=float)
parser.add_argument("--weight_decay", default=1e-5, type=float)
parser.add_argument("--pre_epochs", default=200, type=int)
parser.add_argument("--con_epochs", default=100, type=int)   # FIX: was 50, too few
parser.add_argument("--feature_dim", default=64, type=int)
parser.add_argument("--high_feature_dim", default=20, type=int)
parser.add_argument("--temperature", default=1.0, type=float)
parser.add_argument("--num_views", default=2, type=int,
                    help="Number of views to split HAR features into (2, 3, 4, 5, 6, 7)")
parser.add_argument("--data_path", default='./HAR_dataset/UCI HAR Dataset/', type=str)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =====================
# HAR DATASET
# =====================
class HAR(Dataset):
    def __init__(self, data_path, num_views=2, split='train'):
        """
        HAR dataset split into num_views equal views.
        Each sensor group = one view (biologically meaningful split).
        split: 'train' or 'test'
        """
        X = np.loadtxt(data_path + f'{split}/X_{split}.txt')
        y = np.loadtxt(data_path + f'{split}/y_{split}.txt')

        print(f"Loaded HAR {split}: {X.shape}, classes: {len(np.unique(y))}")

        # Standardise
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Split into num_views equal views
        # HAR has 561 features — split as evenly as possible
        total_features = X.shape[1]
        base_size = total_features // num_views
        remainder = total_features % num_views

        self.views = []
        self.dims = []
        start = 0
        for i in range(num_views):
            # Distribute remainder features to first few views
            end = start + base_size + (1 if i < remainder else 0)
            view_data = X[:, start:end]
            self.views.append(view_data.astype(np.float32))
            self.dims.append(view_data.shape[1])
            print(f"  View {i+1}: features {start}-{end-1}, dim={view_data.shape[1]}")
            start = end

        self.y = y.astype(int) - 1  # make 0-indexed (HAR labels are 1-6)
        self.num_views = num_views

    def __len__(self):
        return self.views[0].shape[0]

    def __getitem__(self, idx):
        data = [torch.from_numpy(v[idx]).float() for v in self.views]
        label = torch.tensor(self.y[idx]).long()
        return data, label, torch.tensor(idx).long()


# =====================
# VIEW WEIGHT (MMD-based, paper eq.14)
# =====================
def compute_view_weights(rs, H):
    """
    Compute adaptive view weights using cosine similarity to H.
    Views more similar to H (lower discrepancy) get higher weight.
    This approximates the MMD-based weighting in the paper.
    """
    H_norm = F.normalize(H, dim=1)
    weights = []
    for r in rs:
        r_norm = F.normalize(r, dim=1)
        # Mean cosine similarity between this view and global H
        sim = torch.mean(torch.sum(r_norm * H_norm, dim=1))
        sim = torch.clamp(sim, -1.0, 1.0)
        # Higher similarity = lower discrepancy = higher weight
        weights.append(torch.exp(sim))

    weights = torch.stack(weights)
    weights = weights / (weights.sum() + 1e-8)  # normalise to sum=1
    # KEY CHANGE: sharpen weights (VERY IMPORTANT)
    # weights = torch.softmax(weights * 5.0, dim=0)
    return weights


# =====================
# PRETRAIN
# =====================
def pretrain(model, data_loader, optimizer, view, device, epoch):
    model.train()
    tot_loss = 0.
    criterion = torch.nn.MSELoss()

    for xs, _, _ in data_loader:
        for v in range(view):
            xs[v] = xs[v].to(device)

        optimizer.zero_grad()
        xrs, _, _, _ = model(xs)
        loss = sum(criterion(xs[v], xrs[v]) for v in range(view))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        tot_loss += loss.item()

    avg_loss = tot_loss / len(data_loader)
    if epoch % 20 == 0 or epoch == 1:
        print(f'Pretrain Epoch {epoch}  Loss: {avg_loss:.6f}')
    return avg_loss


# =====================
# CONTRASTIVE TRAIN
# =====================
def contrastive_train(model, data_loader, optimizer, contrastiveloss, view, device, epoch):
    model.train()
    tot_loss = 0.
    mse = torch.nn.MSELoss()

    for xs, _, _ in data_loader:
        for v in range(view):
            xs[v] = xs[v].to(device)

        optimizer.zero_grad()
        xrs, zs, rs, H = model(xs)

        # FIX: normalise ONCE (not twice as before)
        H = F.normalize(H, dim=1)
        rs = [F.normalize(r, dim=1) for r in rs]

        # Compute adaptive view weights (no gradient needed)
        with torch.no_grad():
            w = compute_view_weights(rs, H)

        # Total loss = reconstruction + weighted contrastive per view
        loss = 0
        for v in range(view):
            loss += w[v] * contrastiveloss(H, rs[v])   # FIX: weight applied here
            loss += mse(xs[v], xrs[v])

        if torch.isnan(loss):
            print(f"  Warning: NaN loss at epoch {epoch}, skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        tot_loss += loss.item()

    avg_loss = tot_loss / len(data_loader)
    print(f'Contrastive Epoch {epoch}  Loss: {avg_loss:.6f}')
    return avg_loss


# =====================
# MAIN EXPERIMENT
# =====================
def run_experiment(num_views):
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {num_views} VIEWS")
    print(f"{'='*60}")

    # Load dataset
    dataset = HAR(args.data_path, num_views=num_views, split='train')
    dims = dataset.dims
    data_size = len(dataset)
    class_num = 6  # HAR has 6 activity classes

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # Build model
    model = Network(num_views, dims, args.feature_dim, args.high_feature_dim, device).to(device)
    print(f"\nModel built: {num_views} encoders, dims={dims}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    #  pass weight=None in ContrastiveLoss — weight handled in train loop
    contrastiveloss = ContrastiveLoss(args.batch_size, args.temperature, device).to(device)

    best_acc, best_nmi, best_pur = 0, 0, 0

    # --- PRETRAIN ---
    print(f"\n--- Pretraining ({args.pre_epochs} epochs) ---")
    for epoch in range(1, args.pre_epochs + 1):
        pretrain(model, data_loader, optimizer, num_views, device, epoch)

    # Reduce LR after pretrain
    for g in optimizer.param_groups:
        g['lr'] *= 0.3
    print(f"LR reduced to {args.learning_rate * 0.3:.6f}")

    # --- CONTRASTIVE ---
    print(f"\n--- Contrastive Training ({args.con_epochs} epochs) ---")
    for epoch in range(args.pre_epochs + 1, args.pre_epochs + args.con_epochs + 1):
        contrastive_train(model, data_loader, optimizer, contrastiveloss, num_views, device, epoch)

        # FIX: eval_h=True — use global H for clustering (as paper specifies)
        acc, nmi, pur = valid(
            model, device, dataset, num_views, data_size, class_num,
            eval_h=True, epoch=epoch
        )

        if acc > best_acc:
            best_acc, best_nmi, best_pur = acc, nmi, pur
            os.makedirs('./models', exist_ok=True)
            torch.save(model.state_dict(), f'./models/HAR_{num_views}views.pth')

        if np.isnan(acc) or acc < 0.01:
            print("Early stopping: instability detected")
            break

    print(f"\n>>> RESULT ({num_views} views): ACC={best_acc:.4f}  NMI={best_nmi:.4f}  PUR={best_pur:.4f}")
    return best_acc, best_nmi, best_pur


# =====================
# RUN ALL VIEW CONFIGS
# =====================

if __name__ == '__main__':
    results = {}

    # Change this list to control which views to run
    # Start with just [2] to test, then switch to [2, 3, 4, 5, 6, 7]
    for n_views in [2,3,4,5,6,7]:
        acc, nmi, pur = run_experiment(n_views)
        results[n_views] = {'ACC': acc, 'NMI': nmi, 'PUR': pur}
        setup_seed(42)  # reset seed for fair comparison

    # Summary table — always printed even if only one view ran
    print(f"\n{'='*60}")
    print(f"  SUMMARY — HAR DATASET (UCI)")
    print(f"{'='*60}")
    print(f"{'Views':<8} {'ACC':<10} {'NMI':<10} {'PUR':<10}")
    print(f"{'-'*38}")
    for n_views, r in results.items():
        print(f"{n_views:<8} {r['ACC']:<10.4f} {r['NMI']:<10.4f} {r['PUR']:<10.4f}")
    print(f"{'='*60}")