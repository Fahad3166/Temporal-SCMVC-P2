import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from network import Network
from metric import valid
import numpy as np
import argparse
import random
from loss import ContrastiveLoss
from dataloader_mhealth import load_mhealth

# ========================
# CONFIG
# ========================
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument("--learning_rate", default=0.0003, type=float)
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--pre_epochs", default=30, type=int)
parser.add_argument("--con_epochs", default=30, type=int)
parser.add_argument("--feature_dim", default=64, type=int)
parser.add_argument("--high_feature_dim", default=20, type=int)
parser.add_argument("--temperature", default=1.0, type=float)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

seed = 42

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(seed)

# ========================
# LOAD DATA
# ========================
dataset_full, dims_full, total_view, data_size, class_num = load_mhealth()

# ========================
# WEIGHT FUNCTION
# ========================
def compute_view_value(rs, H, view):
    N = H.shape[0]
    w = []

    H = torch.nn.functional.normalize(H, dim=1)
    global_sim = torch.matmul(H, H.t())

    for v in range(view):
        r = torch.nn.functional.normalize(rs[v], dim=1)

        view_sim = torch.matmul(r, r.t())
        related_sim = torch.matmul(r, H.t())

        w_v = (torch.sum(view_sim) + torch.sum(global_sim) - 2 * torch.sum(related_sim)) / (N * N)
        w_v = torch.clamp(w_v, -10, 10)

        w.append(torch.exp(-w_v))

    w = torch.stack(w)
    w = w / (torch.sum(w) + 1e-8)
    return w.view(-1)

# ========================
# TRAIN FUNCTIONS
# ========================
def pretrain(model, loader, optimizer, view, epoch):
    mse = torch.nn.MSELoss()
    tot_loss = 0.0

    for xs, _, _ in loader:
        for v in range(view):
            xs[v] = xs[v].to(device)

        optimizer.zero_grad()
        xrs, _, _, _ = model(xs)

        loss = sum(mse(xs[v], xrs[v]) for v in range(view))
        loss.backward()
        optimizer.step()

        tot_loss += loss.item()

    print(f"[Pretrain] Epoch {epoch} Loss:{tot_loss/len(loader):.6f}")


def contrastive_train(model, loader, optimizer, view, contrastiveloss, epoch):
    mse = torch.nn.MSELoss()
    tot_loss = 0.0

    for xs, _, _ in loader:
        for v in range(view):
            xs[v] = xs[v].to(device)

        optimizer.zero_grad()
        xrs, zs, rs, H = model(xs)

        with torch.no_grad():
            w = compute_view_value(rs, H, view)

        loss = 0.0
        for v in range(view):
            loss += contrastiveloss(H, rs[v], w[v])
            loss += mse(xs[v], xrs[v])

        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

    print(f"[Train] Epoch {epoch} Loss:{tot_loss/len(loader):.6f}")


# ========================
# MAIN LOOP (ALL VIEWS)
# ========================
results = []

for num_views in range(1, total_view + 1):

    print("\n" + "="*60)
    print(f"Training with {num_views} view(s)")
    print("="*60)

    selected = list(range(num_views))
    subset_views = [dataset_full.views[i] for i in selected]
    dims = [dims_full[i] for i in selected]

    # dataset wrapper
    class SubDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(dataset_full)

        def __getitem__(self, idx):
            xs = [torch.tensor(v[idx], dtype=torch.float32) for v in subset_views]
            y = torch.tensor(dataset_full.labels[idx], dtype=torch.long)
            return xs, y, idx

    dataset = SubDataset()

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )

    view = num_views

    # model
    setup_seed(seed)
    model = Network(view, dims, args.feature_dim, args.high_feature_dim, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    contrastiveloss = ContrastiveLoss(args.batch_size, args.temperature, device).to(device)

    best_acc, best_nmi, best_pur = 0, 0, 0

    # ===== PRETRAIN =====
    for epoch in range(1, args.pre_epochs + 1):
        pretrain(model, loader, optimizer, view, epoch)

    # ===== TRAIN =====
    for epoch in range(1, args.con_epochs + 1):
        contrastive_train(model, loader, optimizer, view, contrastiveloss, epoch)

        acc, nmi, pur = valid(
            model, device, dataset, view,
            len(dataset), class_num, eval_h=False
        )

        print(f"ACC:{acc:.4f} NMI:{nmi:.4f} PUR:{pur:.4f}")

        if acc > best_acc:
            best_acc, best_nmi, best_pur = acc, nmi, pur

    results.append((num_views, best_acc, best_nmi, best_pur))


# ========================
# FINAL SUMMARY
# ========================
print("\n" + "="*60)
print("SUMMARY — MHEALTH DATASET")
print("="*60)
print("Views    ACC        NMI        PUR")
print("-"*40)

for r in results:
    print(f"{r[0]}       {r[1]:.4f}     {r[2]:.4f}     {r[3]:.4f}")

print("="*60)