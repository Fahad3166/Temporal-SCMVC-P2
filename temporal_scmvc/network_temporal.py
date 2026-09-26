import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# SAFE NORMALIZATION
# =========================================================
def safe_l2_normalize(x, dim=1, eps=1e-8):
    x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
    x = torch.clamp(x, -10.0, 10.0)
    return F.normalize(x, p=2, dim=dim, eps=eps)


# =========================================================
# WEIGHT INITIALIZATION
# =========================================================
def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(m.weight)

        if m.bias is not None:
            nn.init.zeros_(m.bias)


# =========================================================
# TEMPORAL ENCODER
# =========================================================
class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(TemporalEncoder, self).__init__()

        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)

        self.activation = nn.GELU()

        self.dropout = nn.Dropout(0.2)

        # avg + max pooling
        self.proj = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, feature_dim)
        )

        self.apply(init_weights)

    def forward(self, x):
        """
        x: [batch, time, features]
        """

        x = x.transpose(1, 2)

        x = self.activation(self.bn1(self.conv1(x)))
        x = self.dropout(x)

        x = self.activation(self.bn2(self.conv2(x)))
        x = self.dropout(x)

        x = self.activation(self.bn3(self.conv3(x)))

        avg_pool = torch.mean(x, dim=2)
        max_pool = torch.max(x, dim=2).values

        x = torch.cat([avg_pool, max_pool], dim=1)

        x = self.proj(x)

        # IMPORTANT: bound latent space
        x = torch.tanh(x)

        x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        x = torch.clamp(x, -10.0, 10.0)

        return x


# =========================================================
# TEMPORAL DECODER
# =========================================================
class TemporalDecoder(nn.Module):
    def __init__(self, output_dim, feature_dim, seq_len):
        super(TemporalDecoder, self).__init__()

        self.seq_len = seq_len
        self.output_dim = output_dim

        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(64, 128),
            nn.GELU(),

            nn.Linear(128, seq_len * output_dim)
        )

        self.apply(init_weights)

    def forward(self, z):

        x = self.fc(z)

        x = torch.tanh(x)

        x = x.view(z.size(0), self.seq_len, self.output_dim)

        x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        x = torch.clamp(x, -10.0, 10.0)

        return x


# =========================================================
# NETWORK TEMPORAL
# =========================================================
class NetworkTemporal(nn.Module):
    def __init__(
        self,
        view,
        input_size,
        feature_dim,
        high_feature_dim,
        device,
        seq_len=200
    ):
        super(NetworkTemporal, self).__init__()

        self.view = view
        self.seq_len = seq_len

        # -------------------------------------------------
        # Encoders
        # -------------------------------------------------
        self.encoders = nn.ModuleList([
            TemporalEncoder(input_size[v], feature_dim).to(device)
            for v in range(view)
        ])

        # -------------------------------------------------
        # Decoders
        # -------------------------------------------------
        self.decoders = nn.ModuleList([
            TemporalDecoder(input_size[v], feature_dim, seq_len).to(device)
            for v in range(view)
        ])

        # -------------------------------------------------
        # Fusion module
        # -------------------------------------------------
        self.feature_fusion_module = nn.Sequential(

            nn.Linear(self.view * feature_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(128, high_feature_dim)
        )

        # -------------------------------------------------
        # Common information module
        # -------------------------------------------------
        self.common_information_module = nn.Sequential(

            nn.Linear(feature_dim, high_feature_dim),
            nn.LayerNorm(high_feature_dim)
        )

        self.apply(init_weights)

    # =====================================================
    # FEATURE FUSION
    # =====================================================
    def feature_fusion(self, zs, zs_gradient):

        if zs_gradient:
            fusion_input = torch.cat(zs, dim=1)
        else:
            fusion_input = torch.cat(zs, dim=1).detach()

        fusion_input = torch.nan_to_num(
            fusion_input,
            nan=0.0,
            posinf=10.0,
            neginf=-10.0
        )

        fusion_input = torch.clamp(fusion_input, -10.0, 10.0)

        H = self.feature_fusion_module(fusion_input)

        H = torch.tanh(H)

        H = safe_l2_normalize(H, dim=1)

        return H

    # =====================================================
    # FORWARD
    # =====================================================
    def forward(self, xs, zs_gradient=True):

        rs = []
        xrs = []
        zs = []

        for v in range(self.view):

            z = self.encoders[v](xs[v])

            xr = self.decoders[v](z)

            r = self.common_information_module(z)

            z = safe_l2_normalize(z, dim=1)

            r = safe_l2_normalize(r, dim=1)

            zs.append(z)
            rs.append(r)
            xrs.append(xr)

        H = self.feature_fusion(zs, zs_gradient)

        return xrs, zs, rs, H