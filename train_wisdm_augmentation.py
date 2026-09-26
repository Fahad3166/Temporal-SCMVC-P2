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

# Add parent path to import original SCMVC
import sys
sys.path.append('/Users/fahad/Documents/MY#Documents/FAHAD ALI/Uni Wien/6th Semester/P-1/SCMVC')

from network import Network
from dataloader_wisdm import load_wisdm
from loss import ContrastiveLoss
from metric import valid


def time_augment(x, aug_type="random"):
    """Simple time-series augmentations"""
    batch_size, dim = x.shape
    x_aug = x.clone()
    
    if aug_type == "noise" or (aug_type == "random" and random.random() < 0.33):
        # Add Gaussian noise
        noise = torch.randn_like(x_aug) * 0.05
        x_aug = x_aug + noise
        
    elif aug_type == "scale" or (aug_type == "random" and random.random() < 0.33):
        # Random scaling
        scale = 0.8 + torch.rand(batch_size, 1) * 0.4
        x_aug = x_aug * scale
        
    elif aug_type == "mask" or (aug_type == "random" and random.random() < 0.33):
        # Random feature masking
        mask_prob = 0.1
        mask = torch.rand_like(x_aug) > mask_prob
        x_aug = x_aug * mask.float()
    
    return x_aug


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--learning_rate", default=0.0001, type=float)
parser.add_argument("--pre_epochs", default=30, type=int)
parser.add_argument("--con_epochs", default=50, type=int)
parser.add_argument("--feature_dim", default=64, type=int)
parser.add_argument("--high_feature_dim", default=20, type=int)
parser.add_argument("--temperature", default=0.5, type=float)
parser.add_argument("--aug_prob", default=0.5, type=float)
parser.add_argument("--max_samples", default=1000, type=int)
parser.add_argument("--seed", default=42, type=int)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def setup_seed(seed):
    torch.manual_seed(seed)
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


def pretrain(model, loader, optimizer, view, device, epoch):
    model.train()
    mse = torch.nn.MSELoss()
    total_loss = 0
    for xs, _, _ in loader:
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
    print(f"[Pretrain] Epoch {epoch:3d} Loss: {total_loss/len(loader):.6f}")


def contrastive_train(model, loader, optimizer, loss_fn, view, device, epoch):
    model.train()
    mse = torch.nn.MSELoss()
    total_loss = 0
    for xs, _, _ in loader:
        xs = [x.to(device) for x in xs]
        
        # Apply augmentations to create positive pairs (70% chance)
        if random.random() < 0.7:
            xs_aug = [time_augment(x, "random") for x in xs]
        else:
            xs_aug = xs
        
        optimizer.zero_grad()
        
        # Original forward
        xrs, zs, rs, H = model(xs)
        
        # Augmented forward (for contrastive)
        _, _, rs_aug, H_aug = model(xs_aug)
        
        with torch.no_grad():
            w = compute_view_value(rs, H, view)
        
        recon_loss = sum(mse(xs[v], xrs[v]) for v in range(view))
        
        # Contrastive loss between original and augmented
        contrast_loss = loss_fn(H, H_aug)
        
        # Cross-view contrastive loss
        cross_loss = sum(loss_fn(H, rs[v], w[v]) for v in range(view))
        
        loss = recon_loss + contrast_loss + cross_loss
        
        if torch.isnan(loss):
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
    
    print(f"[Train] Epoch {epoch:3d} Loss: {total_loss/len(loader):.6f}")


# Load WISDM data (using original dataloader)
print("\n" + "="*50)
print("LOADING WISDM DATA")
print("="*50)

dataset, dims, view, data_size, class_num = load_wisdm(
    max_files_per_view=5,
    selected_view_indices=[0, 1, 2],  # 3 best views (phone_accel, phone_gyro, watch_accel)
    max_samples=args.max_samples
)

loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
)

print(f"\nDataset Info:")
print(f"  dims: {dims}")
print(f"  view: {view}")
print(f"  data_size: {data_size}")
print(f"  class_num: {class_num}")
print(f"  Augmentation probability: {args.aug_prob}")

# Create original SCMVC model (MLP, not temporal)
model = Network(
    view=view,
    input_size=dims,
    feature_dim=args.feature_dim,
    high_feature_dim=args.high_feature_dim,
    device=device
).to(device)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=1e-5
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
loss_fn = ContrastiveLoss(args.batch_size, args.temperature, device).to(device)

best_acc = 0
best_nmi = 0
best_pur = 0

print("\n" + "="*50)
print("PRETRAINING PHASE")
print("="*50)
for epoch in range(1, args.pre_epochs + 1):
    pretrain(model, loader, optimizer, view, device, epoch)

for g in optimizer.param_groups:
    g["lr"] *= 0.3
print(f"\nLearning rate reduced to: {optimizer.param_groups[0]['lr']:.6f}")

print("\n" + "="*50)
print("CONTRASTIVE TRAINING WITH AUGMENTATIONS")
print("="*50)
for epoch in range(1, args.con_epochs + 1):
    contrastive_train(model, loader, optimizer, loss_fn, view, device, epoch)
    
    acc, nmi, pur = valid(
        model, device, dataset, view, data_size, class_num,
        eval_h=True, epoch=epoch
    )
    
    scheduler.step(acc)
    
    if acc > best_acc:
        best_acc = acc
        best_nmi = nmi
        best_pur = pur
        print(f"  >>> New best model! ACC={acc:.4f}, NMI={nmi:.4f}, PUR={pur:.4f}")

print("\n" + "="*50)
print("FINAL RESULTS WITH AUGMENTATIONS")
print("="*50)
print(f"Best ACC:  {best_acc:.4f}")
print(f"Best NMI:  {best_nmi:.4f}")
print(f"Best PUR:  {best_pur:.4f}")
print("="*50)

# Compare with baseline
print("\n" + "="*50)
print("COMPARISON WITH BASELINE")
print("="*50)
print(f"Original SCMVC (baseline from report): ACC = 0.3667, NMI = 0.5042")
print(f"Original SCMVC + Augmentations:        ACC = {best_acc:.4f}, NMI = {best_nmi:.4f}")
if best_acc > 0.3667:
    improvement = (best_acc - 0.3667) / 0.3667 * 100
    print(f"✓ AUGMENTATIONS IMPROVED by {improvement:.1f}%!")
else:
    print(" Augmentations did not improve baseline")
print("="*50)