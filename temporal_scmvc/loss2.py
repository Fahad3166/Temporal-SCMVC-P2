import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

    def forward(self, h_i, h_j, weight=None):
        N = h_i.size(0)
        
        # 🔥 CRITICAL: Ensure inputs are finite first
        h_i = torch.nan_to_num(h_i, nan=0.0, posinf=1.0, neginf=-1.0)
        h_j = torch.nan_to_num(h_j, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 🔥 Normalize with safe epsilon
        h_i = F.normalize(h_i, dim=1, eps=1e-8)
        h_j = F.normalize(h_j, dim=1, eps=1e-8)
        
        # 🔥 Clamp to prevent extreme values
        h_i = torch.clamp(h_i, -5.0, 5.0)
        h_j = torch.clamp(h_j, -5.0, 5.0)
        
        # similarity matrix
        sim = torch.matmul(h_i, h_j.T) / self.temperature
        
        # 🔥 Clamp similarity before exp
        sim = torch.clamp(sim, -50.0, 50.0)
        
        # 🔥 Log-sum-exp trick for numerical stability
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()
        
        # Positive pairs (diagonal)
        pos = torch.exp(torch.diag(sim))
        
        # Negative pairs mask (remove diagonal)
        mask = ~torch.eye(N, device=self.device).bool()
        neg = torch.exp(sim[mask].view(N, -1))
        neg_sum = neg.sum(dim=1)
        
        # Loss
        loss = -torch.log(pos / (neg_sum + 1e-8))
        
        # Remove possible NaN/Inf
        loss = torch.nan_to_num(loss, nan=0.0, posinf=10.0)
        
        loss = loss.mean()
        
        if weight is not None:
            loss = weight * loss
        
        #  Final clamp
        loss = torch.clamp(loss, 0.0, 10.0)
        
        return loss