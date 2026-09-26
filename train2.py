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
from dataloader_har import load_data

# =====================
# CONFIG
# =====================
Dataname = 'HAR'

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--learning_rate", default=0.0003, type=float)
parser.add_argument("--weight_decay", default=1e-5, type=float)
parser.add_argument("--pre_epochs", default=200, type=int)
parser.add_argument("--con_epochs", default=50, type=int)
parser.add_argument("--feature_dim", default=64)
parser.add_argument("--high_feature_dim", default=20)
parser.add_argument("--temperature", default=0.5)  # 🔥 lower temp = more stable

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================
# SEED
# =====================
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

seed = 42
setup_seed(seed)


# =====================
# DATA
# =====================
dataset, dims, view, data_size, class_num = load_data(args.dataset)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
)


# =====================
# STABLE VIEW WEIGHT
# =====================

def compute_view_value(rs, H, view):
    N = H.shape[0]

    # normalize first
    H = torch.nn.functional.normalize(H, dim=1)
    rs = [torch.nn.functional.normalize(r, dim=1) for r in rs]

    w = []

    for v in range(view):
        sim = torch.matmul(rs[v], H.t())
        sim = torch.clamp(sim, -10, 10)

        w_v = torch.mean(sim)
        w.append(torch.exp(-w_v))

    w = torch.stack(w)
    w = w / (torch.sum(w) + 1e-8)

    return w.squeeze()

# =====================
# PRETRAIN
# =====================
def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()

    for xs, _, _ in data_loader:
        for v in range(view):
            xs[v] = xs[v].to(device)

        optimizer.zero_grad()

        xrs, _, _, _ = model(xs)

        loss = sum(criterion(xs[v], xrs[v]) for v in range(view))

        loss.backward()

        # 🔥 gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()

        tot_loss += loss.item()

    print(f'Epoch {epoch} Loss:{tot_loss/len(data_loader):.6f}')


# =====================
# CONTRASTIVE TRAIN
# =====================
def contrastive_train(epoch):
    tot_loss = 0.
    mse = torch.nn.MSELoss()

    for xs, _, _ in data_loader:
        for v in range(view):
            xs[v] = xs[v].to(device)

        optimizer.zero_grad()

        xrs, zs, rs, H = model(xs)

        # 🔥 CRITICAL FIX
        H = torch.nn.functional.normalize(H, dim=1)
        rs = [torch.nn.functional.normalize(r, dim=1) for r in rs]

        # 🔥 normalize all features
        H = F.normalize(H, dim=1)
        rs = [F.normalize(r, dim=1) for r in rs]

        with torch.no_grad():
            w = compute_view_value(rs, H, view)

        loss = 0

        for v in range(view):
            loss += contrastiveloss(H, rs[v], w[v])
            loss += mse(xs[v], xrs[v])

        # 🔥 NaN protection
        if torch.isnan(loss):
            print("⚠️ NaN detected. Skipping batch.")
            continue

        loss.backward()

        # 🔥 gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()

        tot_loss += loss.item()

    print(f'Epoch {epoch} Loss:{tot_loss/len(data_loader):.6f}')


# =====================
# TRAIN LOOP
# =====================
if not os.path.exists('./models'):
    os.makedirs('./models')

print("ROUND:1")

model = Network(view, dims, args.feature_dim, args.high_feature_dim, device).to(device)

# 🔥 lower LR for stability
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay
)

contrastiveloss = ContrastiveLoss(args.batch_size, args.temperature, device).to(device)

best_acc, best_nmi, best_pur = 0, 0, 0


# =====================
# PRETRAIN
# =====================
for epoch in range(1, args.pre_epochs + 1):
    pretrain(epoch)


# 🔥 reduce LR after pretrain
for g in optimizer.param_groups:
    g['lr'] *= 0.3


# =====================
# CONTRASTIVE
# =====================
for epoch in range(args.pre_epochs + 1, args.pre_epochs + args.con_epochs + 1):

    contrastive_train(epoch)

    acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=False, epoch=epoch)

    if acc > best_acc:
        best_acc, best_nmi, best_pur = acc, nmi, pur
        torch.save(model.state_dict(), f'./models/{args.dataset}.pth')

    # 🔥 early stop if exploding
    if np.isnan(acc) or acc < 0.01:
        print("⛔ Early stopping (instability detected)")
        break


# =====================
# FINAL RESULT
# =====================
print(f'Best: ACC={best_acc:.4f} NMI={best_nmi:.4f} PUR={best_pur:.4f}')