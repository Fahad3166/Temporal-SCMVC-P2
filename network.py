import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_l2_normalize(x, dim=1, eps=1e-8):
    x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)
    x = torch.clamp(x, -1e3, 1e3)
    return F.normalize(x, p=2, dim=dim, eps=eps)


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)
        x = torch.clamp(x, -1e3, 1e3)
        return x


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        x = self.decoder(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)
        x = torch.clamp(x, -1e3, 1e3)
        return x


class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, device):
        super(Network, self).__init__()
        self.view = view

        self.encoders = nn.ModuleList([
            Encoder(input_size[v], feature_dim).to(device)
            for v in range(view)
        ])

        self.decoders = nn.ModuleList([
            Decoder(input_size[v], feature_dim).to(device)
            for v in range(view)
        ])

        self.feature_fusion_module = nn.Sequential(
            nn.Linear(self.view * feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, high_feature_dim)
        )

        self.common_information_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim)
        )

    def feature_fusion(self, zs, zs_gradient):
        if zs_gradient:
            fusion_input = torch.cat(zs, dim=1)
        else:
            fusion_input = torch.cat(zs, dim=1).detach()

        fusion_input = torch.nan_to_num(fusion_input, nan=0.0, posinf=1e3, neginf=-1e3)
        fusion_input = torch.clamp(fusion_input, -1e3, 1e3)

        H = self.feature_fusion_module(fusion_input)
        H = safe_l2_normalize(H, dim=1)
        return H

    def forward(self, xs, zs_gradient=True):
        rs = []
        xrs = []
        zs = []

        for v in range(self.view):
            x = xs[v]

            z = self.encoders[v](x)
            xr = self.decoders[v](z)

            r = self.common_information_module(z)
            r = safe_l2_normalize(r, dim=1)

            rs.append(r)
            zs.append(z)
            xrs.append(xr)

        H = self.feature_fusion(zs, zs_gradient)
        return xrs, zs, rs, H