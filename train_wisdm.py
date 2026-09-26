import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import random
import argparse

from network import Network
from metric import valid
from loss import ContrastiveLoss
from dataloader_wisdm import load_wisdm

parser = argparse.ArgumentParser(description="WISDM SCMVC experiment")
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--learning_rate", default=0.0003, type=float)
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--pre_epochs", default=50, type=int)
parser.add_argument("--con_epochs", default=50, type=int)
parser.add_argument("--feature_dim", default=64, type=int)
parser.add_argument("--high_feature_dim", default=20, type=int)
parser.add_argument("--temperature", default=1.0, type=float)
parser.add_argument("--window_size", default=200, type=int)
parser.add_argument("--max_files_per_view", default=5, type=int)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


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

    if w.dim() == 0:
        w = w.unsqueeze(0)

    return w


def pretrain(model, optimizer, data_loader, view):
    model.train()
    criterion = torch.nn.MSELoss()
    total_loss = 0.0

    for xs, _, _ in data_loader:
        for v in range(view):
            xs[v] = xs[v].to(device)

        optimizer.zero_grad()
        xrs, _, _, _ = model(xs)

        loss = sum(criterion(xs[v], xrs[v]) for v in range(view))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)


def contrastive_train(model, optimizer, data_loader, view, contrastiveloss):
    model.train()
    mse = torch.nn.MSELoss()
    total_loss = 0.0

    for xs, _, _ in data_loader:
        for v in range(view):
            xs[v] = xs[v].to(device)

        optimizer.zero_grad()
        xrs, zs, rs, H = model(xs)

        with torch.no_grad():
            w = compute_view_value(rs, H, view)

        loss = 0.0
        for v in range(view):
            weight = w[v] if w.dim() > 0 else w
            loss += contrastiveloss(H, rs[v], weight)
            loss += mse(xs[v], xrs[v])

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)


def run_experiment(selected_view_indices):
    setup_seed(seed)

    dataset, dims, view, data_size, class_num = load_wisdm(
        data_root="/Users/fahad/Documents/MY#Documents/FAHAD ALI/Uni Wien/6th Semester/P-1/SCMVC/data/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/wisdm-dataset/raw",
        window_size=args.window_size,
        max_files_per_view=args.max_files_per_view,
        selected_view_indices=selected_view_indices,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = Network(view, dims, args.feature_dim, args.high_feature_dim, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    contrastiveloss = ContrastiveLoss(args.batch_size, args.temperature, device).to(device)

    print(model)

    best_acc, best_nmi, best_pur = 0.0, 0.0, 0.0

    for epoch in range(1, args.pre_epochs + 1):
        loss = pretrain(model, optimizer, data_loader, view)
        print(f"[Pretrain] Epoch {epoch} Loss: {loss:.6f}")

    for epoch in range(1, args.con_epochs + 1):
        loss = contrastive_train(model, optimizer, data_loader, view, contrastiveloss)
        acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=False, epoch=epoch)
        print(f"[Train] Epoch {epoch} Loss:{loss:.6f} ACC:{acc:.4f}")

        if acc > best_acc:
            best_acc, best_nmi, best_pur = acc, nmi, pur

    return best_acc, best_nmi, best_pur


def main():
    view_configs = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 2],
        4: [0, 1, 2, 3],
    }

    results = {}

    for num_views, selected in view_configs.items():
        print("\n" + "=" * 60)
        print(f"Training WISDM with {num_views} view(s): {selected}")
        print("=" * 60)

        best_acc, best_nmi, best_pur = run_experiment(selected)
        results[num_views] = (best_acc, best_nmi, best_pur)

    print("\n" + "=" * 60)
    print(" SUMMARY — WISDM DATASET")
    print("=" * 60)
    print("Views    ACC        NMI        PUR")
    print("-" * 40)
    for v in sorted(results.keys()):
        acc, nmi, pur = results[v]
        print(f"{v:<8}{acc:.4f}     {nmi:.4f}     {pur:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()