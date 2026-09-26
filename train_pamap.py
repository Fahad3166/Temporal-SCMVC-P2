import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import random

from network import Network
from metric import valid
from loss import ContrastiveLoss
from dataloader_pamap import load_pamap2


# ======================
# CONFIG
# ======================
parser = argparse.ArgumentParser(description='Train SCMVC on PAMAP2')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--pre_epochs', default=50, type=int)
parser.add_argument('--con_epochs', default=50, type=int)
parser.add_argument('--feature_dim', default=64, type=int)
parser.add_argument('--high_feature_dim', default=20, type=int)
parser.add_argument('--temperature', default=1.0, type=float)
parser.add_argument('--max_samples', default=10000, type=int)
parser.add_argument('--seed', default=42, type=int)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================
# SEED
# ======================
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


setup_seed(args.seed)


# ======================
# LOAD DATA
# ======================
dataset, dims_full, view_full, data_size, class_num = load_pamap2(
    max_samples=args.max_samples,
    seed=args.seed
)


# ======================
# SAFE WEIGHT FUNCTION
# ======================
def compute_view_value(rs, H, view):
    """
    Compute adaptive view weights more safely.
    """
    N = H.shape[0]
    weights = []

    H = F.normalize(H, dim=1)
    global_sim = torch.matmul(H, H.t())
    global_sim = torch.nan_to_num(global_sim, nan=0.0, posinf=1e4, neginf=-1e4)

    for v in range(view):
        r = F.normalize(rs[v], dim=1)

        view_sim = torch.matmul(r, r.t())
        related_sim = torch.matmul(r, H.t())

        view_sim = torch.nan_to_num(view_sim, nan=0.0, posinf=1e4, neginf=-1e4)
        related_sim = torch.nan_to_num(related_sim, nan=0.0, posinf=1e4, neginf=-1e4)

        w_v = (torch.sum(view_sim) + torch.sum(global_sim) - 2 * torch.sum(related_sim)) / (N * N)
        w_v = torch.clamp(w_v, -10.0, 10.0)

        weights.append(torch.exp(-w_v))

    weights = torch.stack(weights)
    weights = weights / (torch.sum(weights) + 1e-8)

    return weights.view(-1)


# ======================
# TRAIN FUNCTIONS
# ======================
def pretrain(model, optimizer, data_loader, view):
    model.train()
    criterion = torch.nn.MSELoss()
    total_loss = 0.0

    for xs, _, _ in data_loader:
        xs = [x.to(device) for x in xs]

        optimizer.zero_grad()
        xrs, _, _, _ = model(xs)

        loss = sum(criterion(xs[v], xrs[v]) for v in range(view))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def contrastive_train(model, optimizer, data_loader, view, contrastive_loss_fn):
    model.train()
    mse = torch.nn.MSELoss()
    total_loss = 0.0

    for xs, _, _ in data_loader:
        xs = [x.to(device) for x in xs]

        optimizer.zero_grad()
        xrs, zs, rs, H = model(xs)

        with torch.no_grad():
            w = compute_view_value(rs, H, view)

        loss = 0.0
        for v in range(view):
            loss = loss + contrastive_loss_fn(H, rs[v], w[v])
            loss = loss + mse(xs[v], xrs[v])

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


# ======================
# VIEW CONFIGS
# ======================
# Better progression:
# 1 view  -> hand only (stronger than heart-rate-only)
# 2 views -> hand + chest
# 3 views -> hand + chest + ankle
# 4 views -> all views
view_configs = {
    1: [1],             # hand IMU
    2: [1, 2],          # hand + chest
    3: [1, 2, 3],       # hand + chest + ankle
    4: [0, 1, 2, 3],    # all views including heart rate
}

results = {}


# ======================
# MAIN LOOP
# ======================
for num_views, indices in view_configs.items():
    print("\n" + "=" * 60)
    print(f"Training with {num_views} view(s): {indices}")
    print("=" * 60)

    dims = [dims_full[i] for i in indices]

    class SubDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            xs, y, original_idx = self.dataset[idx]
            xs = [xs[i] for i in self.indices]
            return xs, y, original_idx

    sub_dataset = SubDataset(dataset, indices)

    data_loader = torch.utils.data.DataLoader(
        sub_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )

    setup_seed(args.seed)

    model = Network(num_views, dims, args.feature_dim, args.high_feature_dim, device).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    contrastive_loss_fn = ContrastiveLoss(args.batch_size, args.temperature, device).to(device)

    best_acc, best_nmi, best_pur = 0.0, 0.0, 0.0

    # -------- Pretrain --------
    for epoch in range(1, args.pre_epochs + 1):
        loss = pretrain(model, optimizer, data_loader, num_views)
        print(f"[Pretrain] Epoch {epoch} Loss: {loss:.6f}")

    # -------- Contrastive training --------
    for epoch in range(1, args.con_epochs + 1):
        loss = contrastive_train(model, optimizer, data_loader, num_views, contrastive_loss_fn)

        acc, nmi, pur = valid(
            model,
            device,
            sub_dataset,
            num_views,
            data_size,
            class_num,
            eval_h=False,
            epoch=epoch
        )

        print(f"[Train] Epoch {epoch} Loss:{loss:.6f} ACC:{acc:.4f}")

        if acc > best_acc:
            best_acc, best_nmi, best_pur = acc, nmi, pur

    results[num_views] = (best_acc, best_nmi, best_pur)


# ======================
# FINAL SUMMARY
# ======================
print("\n" + "=" * 60)
print("SUMMARY — PAMAP2 DATASET")
print("=" * 60)
print("Views    ACC        NMI        PUR")
print("-" * 40)

for v in sorted(results.keys()):
    acc, nmi, pur = results[v]
    print(f"{v:<8}{acc:.4f}     {nmi:.4f}     {pur:.4f}")

print("=" * 60)