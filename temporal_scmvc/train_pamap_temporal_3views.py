import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import random
import argparse
import torch.nn.functional as F

from network_temporal import NetworkTemporal
from dataloader_pamap_temporal import load_pamap2_temporal
from loss2 import ContrastiveLoss
from metric import valid


# ========================
# CONFIG
# ========================
parser = argparse.ArgumentParser(description='Temporal SCMVC on PAMAP2 (3 views)')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--learning_rate', default=0.0003, type=float)
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--pre_epochs', default=20, type=int)
parser.add_argument('--con_epochs', default=20, type=int)
parser.add_argument('--feature_dim', default=64, type=int)
parser.add_argument('--high_feature_dim', default=20, type=int)
parser.add_argument('--temperature', default=1.0, type=float)
parser.add_argument('--window_size', default=50, type=int)
parser.add_argument('--stride', default=25, type=int)
parser.add_argument('--max_samples', default=1000, type=int)
parser.add_argument('--seed', default=42, type=int)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ========================
# SEED
# ========================
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(args.seed)


# ========================
# LOAD DATA
# ========================
dataset_full, dims_full, view_full, data_size, class_num = load_pamap2_temporal(
    window_size=args.window_size,
    stride=args.stride,
    max_samples=args.max_samples
)

# Use only IMU views: hand, chest, ankle
selected_indices = [1, 2, 3]
dims = [dims_full[i] for i in selected_indices]
view = len(selected_indices)

class SubDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, selected_indices):
        self.dataset = dataset
        self.selected_indices = selected_indices

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        xs, y, original_idx = self.dataset[idx]
        xs = [xs[i] for i in self.selected_indices]
        return xs, y, original_idx

dataset = SubDataset(dataset_full, selected_indices)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True
)

print("Selected view indices =", selected_indices)
print("dims =", dims)
print("view =", view)
print("data_size =", data_size)
print("class_num =", class_num)


# ========================
# SAFE VIEW WEIGHT FUNCTION
# ========================
def compute_view_value(rs, H, view):
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


# ========================
# TRAIN FUNCTIONS
# ========================
def pretrain(model, data_loader, optimizer, view, device, epoch):
    model.train()
    mse = torch.nn.MSELoss()
    total_loss = 0.0

    for xs, _, _ in data_loader:
        xs = [x.to(device) for x in xs]

        optimizer.zero_grad()
        xrs, _, _, _ = model(xs)

        loss = sum(mse(xs[v], xrs[v]) for v in range(view))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f"[Pretrain] Epoch {epoch} Loss: {avg_loss:.6f}")
    return avg_loss


def contrastive_train(model, data_loader, optimizer, contrastive_loss_fn, view, device, epoch):
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
            loss = loss + mse(xs[v], xrs[v])

        for v in range(view):
            loss = loss + contrastive_loss_fn(H, rs[v], w[v])

        if torch.isnan(loss):
            print(f"Warning: NaN loss at epoch {epoch}, skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f"[Train] Epoch {epoch} Loss: {avg_loss:.6f}")
    return avg_loss


# ========================
# MAIN
# ========================
if not os.path.exists("./models"):
    os.makedirs("./models")

model = NetworkTemporal(
    view=view,
    input_size=dims,
    feature_dim=args.feature_dim,
    high_feature_dim=args.high_feature_dim,
    device=device,
    seq_len=args.window_size
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay
)

contrastive_loss_fn = ContrastiveLoss(
    args.batch_size,
    args.temperature,
    device
).to(device)

best_acc, best_nmi, best_pur = 0.0, 0.0, 0.0
best_epoch = 0
patience = 5
epochs_no_improve = 0

print("\n===== PRETRAINING =====")
for epoch in range(1, args.pre_epochs + 1):
    pretrain(model, data_loader, optimizer, view, device, epoch)

for g in optimizer.param_groups:
    g["lr"] *= 0.3

print("\n===== CONTRASTIVE TRAINING =====")
for epoch in range(1, args.con_epochs + 1):
    contrastive_train(model, data_loader, optimizer, contrastive_loss_fn, view, device, epoch)

    acc, nmi, pur = valid(
        model, device, dataset, view, data_size, class_num,
        eval_h=True, epoch=epoch
    )

    print(f"[Eval] Epoch {epoch} ACC: {acc:.4f} NMI: {nmi:.4f} PUR: {pur:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_nmi = nmi
        best_pur = pur
        best_epoch = epoch
        epochs_no_improve = 0

        torch.save(model.state_dict(), "./models/pamap_temporal_3views_best.pth")
        print(f"Saved new best model at epoch {epoch}")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"Early stopping triggered at epoch {epoch}")
        break

print("\n===== FINAL BEST RESULTS =====")
print(f"Best Epoch: {best_epoch}")
print(f"Best ACC: {best_acc:.4f}")
print(f"Best NMI: {best_nmi:.4f}")
print(f"Best PUR: {best_pur:.4f}")
print("Best model saved to: ./models/pamap_temporal_3views_best.pth")