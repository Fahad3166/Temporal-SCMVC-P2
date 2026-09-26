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

        # 🔥 Normalize embeddings (CRITICAL)
        h_i = F.normalize(h_i, dim=1)
        h_j = F.normalize(h_j, dim=1)

        # similarity matrix
        sim = torch.matmul(h_i, h_j.T) / self.temperature

        # 🔥 stabilize (log-sum-exp trick)
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()
        
        sim_max = torch.clamp(sim_max, -10, 10)
        exp_sim = torch.exp(sim)

        # remove diagonal
        mask = torch.ones_like(exp_sim).fill_diagonal_(0)

        # positives
        pos = torch.exp(torch.diag(sim))

        # denominator (exclude diagonal properly)
        denom = (exp_sim * mask).sum(dim=1) + 1e-8

        loss = -torch.log(pos / denom)

        loss = loss.mean()

        if weight is not None:
            loss = weight * loss

        return loss