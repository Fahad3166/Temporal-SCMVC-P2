import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import random
import argparse
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from network_temporal import NetworkTemporal
from metric import valid
from loss2 import ContrastiveLoss
from dataloader_mhealth_temporal import load_mhealth_temporal


parser = argparse.ArgumentParser(description="Temporal SCMVC on MHEALTH")
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--learning_rate", default=0.0001, type=float)
parser.add_argument("--weight_decay", default=1e-5, type=float)
parser.add_argument("--pre_epochs", default=30, type=int)
parser.add_argument("--con_epochs", default=50, type=int)
parser.add_argument("--feature_dim", default=64, type=int)
parser.add_argument("--high_feature_dim", default=20, type=int)
parser.add_argument("--temperature", default=0.5, type=float)
parser.add_argument("--window_size", default=50, type=int)
parser.add_argument("--max_samples", default=10000, type=int)
parser.add_argument("--seed", default=42, type=int)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


setup_seed(args.seed)


def compute_view_value(rs, H, view):
    weights = []
    H_norm = F.normalize(H, dim=1, eps=1e-8)
    
    for v in range(view):
        r_norm = F.normalize(rs[v], dim=1, eps=1e-8)
        sim = torch.mean(torch.sum(H_norm * r_norm, dim=1))
        sim = torch.clamp(sim, -0.9, 0.9)
        w_v = 0.7 + sim
        weights.append(w_v)
    
    weights = torch.stack(weights)
    weights = torch.clamp(weights, 0.2, 2.0)
    weights = weights / (weights.sum() + 1e-8)
    return weights.view(-1)


def pretrain(model, data_loader, optimizer, view, device, epoch):
    model.train()
    mse = torch.nn.MSELoss()
    total_loss = 0.0
    
    for xs, _, _ in data_loader:
        xs = [x.to(device) for x in xs]
        optimizer.zero_grad()
        xrs, _, _, _ = model(xs)
        loss = sum(mse(xs[v], xrs[v]) for v in range(view))
        
        if torch.isnan(loss):
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    print(f"[Pretrain] Epoch {epoch:3d} Loss: {avg_loss:.6f}")
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
        
        reconstruction_loss = sum(mse(xs[v], xrs[v]) for v in range(view))
        
        contrastive_loss = 0.0
        for v in range(view):
            contrastive_loss = contrastive_loss + contrastive_loss_fn(H, rs[v], w[v])
        
        loss = reconstruction_loss + contrastive_loss
        
        if torch.isnan(loss):
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    print(f"[Train] Epoch {epoch:3d} Loss: {avg_loss:.6f}")
    return avg_loss


# Load data using your existing dataloader
print("\n" + "="*50)
print("LOADING MHEALTH DATA")
print("="*50)

dataset, dims, view, data_size, class_num = load_mhealth_temporal(
    max_samples=args.max_samples,
    window_size=args.window_size,
    stride=25
)

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
)

print(f"\nDataset Info:")
print(f"  dims: {dims}")
print(f"  view: {view}")
print(f"  data_size: {data_size}")
print(f"  class_num: {class_num}")
print(f"  window_size: {args.window_size}")

# Create model
model = NetworkTemporal(
    view=view,
    input_size=dims,
    feature_dim=args.feature_dim,
    high_feature_dim=args.high_feature_dim,
    device=device,
    seq_len=args.window_size
).to(device)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=10
)

contrastive_loss_fn = ContrastiveLoss(
    args.batch_size,
    args.temperature,
    device
).to(device)

# Training
best_acc, best_nmi, best_pur = 0.0, 0.0, 0.0
best_epoch = 0

print("\n" + "="*50)
print("PRETRAINING PHASE - MHEALTH TEMPORAL")
print("="*50)
for epoch in range(1, args.pre_epochs + 1):
    pretrain(model, data_loader, optimizer, view, device, epoch)

for g in optimizer.param_groups:
    g["lr"] *= 0.3
print(f"\nLearning rate reduced to: {optimizer.param_groups[0]['lr']:.6f}")

print("\n" + "="*50)
print("CONTRASTIVE TRAINING PHASE - MHEALTH TEMPORAL")
print("="*50)
for epoch in range(1, args.con_epochs + 1):
    contrastive_train(model, data_loader, optimizer, contrastive_loss_fn, view, device, epoch)
    
    acc, nmi, pur = valid(
        model, device, dataset, view, data_size, class_num,
        eval_h=True, epoch=epoch
    )
    
    scheduler.step(acc)
    
    if acc > best_acc:
        best_acc = acc
        best_nmi = nmi
        best_pur = pur
        best_epoch = epoch
        
        os.makedirs("./models", exist_ok=True)
        torch.save(model.state_dict(), "./models/mhealth_temporal_best.pth")
        print(f"  >>> New best model saved (ACC={acc:.4f})")

print("\n" + "="*50)
print("FINAL RESULTS - MHEALTH TEMPORAL")
print("="*50)
print(f"Best Epoch: {best_epoch}")
print(f"Best ACC:  {best_acc:.4f}")
print(f"Best NMI:  {best_nmi:.4f}")
print(f"Best PUR:  {best_pur:.4f}")
print("="*50)

# Compare with original
print("\n" + "="*50)
print("COMPARISON WITH ORIGINAL SCMVC")
print("="*50)
print(f"Original SCMVC (3 views): ACC = 0.9199, NMI = 0.9276")
print(f"Temporal SCMVC (3 views): ACC = {best_acc:.4f}, NMI = {best_nmi:.4f}")
if best_acc > 0.9199:
    print("✓ TEMPORAL IMPROVED OVER BASELINE!")
else:
    print(" Temporal did not beat baseline")
print("="*50)