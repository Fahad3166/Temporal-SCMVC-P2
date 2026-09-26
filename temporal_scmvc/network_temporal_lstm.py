import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        feature_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
    ):
        super(LSTMEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, feature_dim),
        )

        self.apply(init_weights)
        self._init_lstm_weights()

    def _init_lstm_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x):
        """
        x: [batch, time, features]
        """
        lstm_out, _ = self.lstm(x)

        mean_pool = torch.mean(lstm_out, dim=1)
        max_pool = torch.max(lstm_out, dim=1).values
        pooled = torch.cat([mean_pool, max_pool], dim=1)

        z = self.projection(pooled)
        z = torch.clamp(z, -10.0, 10.0)
        z = torch.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0)
        return z


class LSTMDecoder(nn.Module):
    def __init__(self, output_dim, feature_dim, seq_len):
        super(LSTMDecoder, self).__init__()

        self.seq_len = seq_len
        self.output_dim = output_dim

        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, seq_len * output_dim),
        )

        self.apply(init_weights)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), self.seq_len, self.output_dim)
        x = torch.clamp(x, -10.0, 10.0)
        x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        return x


class NetworkTemporalLSTM(nn.Module):
    def __init__(
        self,
        view,
        input_size,
        feature_dim,
        high_feature_dim,
        device,
        seq_len=200,
        hidden_dim=128,
        num_layers=2,
    ):
        super(NetworkTemporalLSTM, self).__init__()

        self.view = view
        self.seq_len = seq_len

        self.encoders = nn.ModuleList(
            [
                LSTMEncoder(
                    input_dim=input_size[v],
                    feature_dim=feature_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                ).to(device)
                for v in range(view)
            ]
        )

        self.decoders = nn.ModuleList(
            [
                LSTMDecoder(
                    output_dim=input_size[v],
                    feature_dim=feature_dim,
                    seq_len=seq_len,
                ).to(device)
                for v in range(view)
            ]
        )

        self.feature_fusion_module = nn.Sequential(
            nn.Linear(view * feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, high_feature_dim),
        )

        self.common_information_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),
            nn.LayerNorm(high_feature_dim),
        )

        self.apply(init_weights)

    def forward(self, xs):
        rs = []
        xrs = []
        zs = []

        for v in range(self.view):
            z = self.encoders[v](xs[v])
            xr = self.decoders[v](z)
            r = self.common_information_module(z)

            z = torch.clamp(z, -10.0, 10.0)
            r = torch.clamp(r, -10.0, 10.0)

            zs.append(z)
            rs.append(r)
            xrs.append(xr)

        fusion_input = torch.cat(zs, dim=1)
        H = self.feature_fusion_module(fusion_input)
        H = F.normalize(H, dim=1, eps=1e-8)
        H = torch.clamp(H, -10.0, 10.0)
        H = torch.nan_to_num(H, nan=0.0, posinf=10.0, neginf=-10.0)

        return xrs, zs, rs, H
