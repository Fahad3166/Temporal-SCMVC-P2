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
from dataloader_wisdm_temporal import load_wisdm_temporal
from loss2 import ContrastiveLoss
from metric import valid


parser = argparse.ArgumentParser(description="Tuned Temporal SCMVC on WISDM")
parser.add_argument("--batch_size", default=128, type=int)  # Increased
parser.add_argument("--learning_rate", default=0.0001, type=float)  # Lower LR
parser.add_argument("--weight_decay", default=1e-5, type=float)
parser.add_argument("--pre_epochs", default=50, type=int)  # More pretraining
parser.add_argument("--con_epochs", default=50, type=int)
parser.add_argument("--feature_dim", default=128, type=int)  # Larger latent space
parser.add_argument("--high_feature_dim", default=20, type=int)
parser.add_argument("--temperature", default=0.5, type=float)  # Lower temp
parser.add_argument("--window_size", default=200, type=int)
parser.add_argument("--max_samples", default=1000, type=int)
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
    """Improved view weight computation"""
    weights = []
    H_norm = F.normalize(H, dim=1, eps=1e-8)
    
    for v in range(view):
        r_norm = F.normalize(rs[v], dim=1, eps=1e-8)
        # Cosine similarity
        sim = torch.mean(torch.sum(H_norm * r_norm, dim=1))
        sim = torch.clamp(sim, -0.9, 0.9)
        # Map similarity to weight (0.3 to 1.7 range)
        w_v = 0.7 + sim  # More stable range
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
        
        # Add small L2 regularization
        l2_reg = sum(p.norm(2) for p in model.parameters()) * 0.00001
        
        loss = reconstruction_loss + contrastive_loss + l2_reg
        
        if torch.isnan(loss):
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    print(f"[Train] Epoch {epoch:3d} Loss: {avg_loss:.6f}")
    return avg_loss


# Load data
dataset, dims, view, data_size, class_num = load_wisdm_temporal(
    window_size=args.window_size,
    max_files_per_view=5,
    selected_view_indices=[0, 1, 2],
    max_samples=args.max_samples,
    seed=args.seed
)

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
)

print(f"dims={dims}, view={view}, data_size={data_size}, class_num={class_num}")
print(f"feature_dim={args.feature_dim}, temperature={args.temperature}, lr={args.learning_rate}")

# Create model with larger feature dim
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

# Learning rate scheduler
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
print("PRETRAINING PHASE")
print("="*50)
for epoch in range(1, args.pre_epochs + 1):
    pretrain(model, data_loader, optimizer, view, device, epoch)

# Reduce LR after pretraining
for g in optimizer.param_groups:
    g["lr"] *= 0.3
print(f"\nLearning rate reduced to: {optimizer.param_groups[0]['lr']:.6f}")

print("\n" + "="*50)
print("CONTRASTIVE TRAINING PHASE")
print("="*50)
for epoch in range(1, args.con_epochs + 1):
    contrastive_train(model, data_loader, optimizer, contrastive_loss_fn, view, device, epoch)
    
    acc, nmi, pur = valid(
        model, device, dataset, view, data_size, class_num,
        eval_h=True, epoch=epoch
    )
    
    scheduler.step(acc)  # Adjust LR based on ACC
    
    if acc > best_acc:
        best_acc = acc
        best_nmi = nmi
        best_pur = pur
        best_epoch = epoch
        
        os.makedirs("./models", exist_ok=True)
        torch.save(model.state_dict(), "./models/wisdm_temporal_tuned_best.pth")
        print(f"  >>> New best model saved (ACC={acc:.4f})")

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(f"Best Epoch: {best_epoch}")
print(f"Best ACC:  {best_acc:.4f}")
print(f"Best NMI:  {best_nmi:.4f}")
print(f"Best PUR:  {best_pur:.4f}")
print("="*50)