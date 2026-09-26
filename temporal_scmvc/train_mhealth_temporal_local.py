import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import random
import argparse
import torch.nn.functional as F

from network_temporal import NetworkTemporal
from dataloader_mhealth_temporal import load_mhealth_temporal
from loss2 import ContrastiveLoss
from metric import valid


# ========================
# CONFIG
# ========================
parser = argparse.ArgumentParser(description="Temporal SCMVC on MHEALTH with Local Contrastive Loss")
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--learning_rate", default=0.0003, type=float)
parser.add_argument("--weight_decay", default=1e-5, type=float)
parser.add_argument("--pre_epochs", default=20, type=int)
parser.add_argument("--con_epochs", default=20, type=int)
parser.add_argument("--feature_dim", default=64, type=int)
parser.add_argument("--high_feature_dim", default=20, type=int)
parser.add_argument("--temperature", default=1.0, type=float)
parser.add_argument("--window_size", default=50, type=int)
parser.add_argument("--stride", default=25, type=int)
parser.add_argument("--local_chunk_size", default=10, type=int)
parser.add_argument("--local_loss_weight", default=0.2, type=float)
parser.add_argument("--max_samples", default=5000, type=int)
parser.add_argument("--seed", default=42, type=int)
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
dataset, dims, view, data_size, class_num = load_mhealth_temporal(
    window_size=args.window_size,
    stride=args.stride,
    max_samples=args.max_samples
)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True
)

print("dims =", dims)
print("view =", view)
print("data_size =", data_size)
print("class_num =", class_num)
print("local_chunk_size =", args.local_chunk_size)
print("local_loss_weight =", args.local_loss_weight)


# ========================
# VIEW WEIGHT FUNCTION
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
# LOCAL CONTRASTIVE HELPERS
# ========================
def split_into_local_chunks(xs, chunk_size):
    time_len = xs[0].size(1)

    if time_len % chunk_size != 0:
        raise ValueError(
            f"window_size={time_len} must be divisible by local_chunk_size={chunk_size}"
        )

    num_chunks = time_len // chunk_size
    chunks = []

    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk_views = [x[:, start:end, :] for x in xs]
        chunks.append(chunk_views)

    return chunks


def compute_local_contrastive_loss(model, xs, contrastive_loss_fn, view, chunk_size):
    chunks = split_into_local_chunks(xs, chunk_size)

    local_losses = []

    for chunk_xs in chunks:
        # We only need rs_local and H_local.
        # Decoder output is ignored here.
        _, _, rs_local, H_local = model(chunk_xs)

        with torch.no_grad():
            w_local = compute_view_value(rs_local, H_local, view)

        chunk_loss = 0.0

        for v in range(view):
            chunk_loss = chunk_loss + contrastive_loss_fn(
                H_local,
                rs_local[v],
                w_local[v]
            )

        local_losses.append(chunk_loss)

    return sum(local_losses) / len(local_losses)


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

        # Global SCMVC branch
        xrs, zs, rs, H = model(xs)

        with torch.no_grad():
            w = compute_view_value(rs, H, view)

        reconstruction_loss = 0.0
        global_contrastive_loss = 0.0

        for v in range(view):
            reconstruction_loss = reconstruction_loss + mse(xs[v], xrs[v])
            global_contrastive_loss = global_contrastive_loss + contrastive_loss_fn(
                H,
                rs[v],
                w[v]
            )

        # Local window-level contrastive branch
        local_contrastive_loss = compute_local_contrastive_loss(
            model=model,
            xs=xs,
            contrastive_loss_fn=contrastive_loss_fn,
            view=view,
            chunk_size=args.local_chunk_size
        )

        loss = (
            reconstruction_loss
            + global_contrastive_loss
            + args.local_loss_weight * local_contrastive_loss
        )

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

print("\n===== CONTRASTIVE TRAINING WITH LOCAL LOSS =====")
for epoch in range(1, args.con_epochs + 1):
    contrastive_train(model, data_loader, optimizer, contrastive_loss_fn, view, device, epoch)

    acc, nmi, pur = valid(
        model,
        device,
        dataset,
        view,
        data_size,
        class_num,
        eval_h=True,
        epoch=epoch
    )

    print(f"[Eval] Epoch {epoch} ACC: {acc:.4f} NMI: {nmi:.4f} PUR: {pur:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_nmi = nmi
        best_pur = pur
        best_epoch = epoch
        epochs_no_improve = 0

        torch.save(model.state_dict(), "./models/mhealth_temporal_local_best.pth")
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
print("Best model saved to: ./models/mhealth_temporal_local_best.pth")