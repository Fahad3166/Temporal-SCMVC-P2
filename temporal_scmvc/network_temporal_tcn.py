import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# CHOMP
# =========================================================
class Chomp1d(nn.Module):
    """
    Removes extra padding added for causal convolutions
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):

        if self.chomp_size == 0:
            return x

        return x[:, :, :-self.chomp_size].contiguous()


# =========================================================
# TEMPORAL BLOCK
# =========================================================
class TemporalBlock(nn.Module):

    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        dropout=0.3
    ):

        super(TemporalBlock, self).__init__()

        # -------------------------
        # FIRST CONV
        # -------------------------
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )

        self.chomp1 = Chomp1d(padding)

        self.bn1 = nn.BatchNorm1d(n_outputs)

        self.relu1 = nn.ReLU()

        self.dropout1 = nn.Dropout(dropout)

        # -------------------------
        # SECOND CONV
        # -------------------------
        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )

        self.chomp2 = Chomp1d(padding)

        self.bn2 = nn.BatchNorm1d(n_outputs)

        self.relu2 = nn.ReLU()

        self.dropout2 = nn.Dropout(dropout)

        # -------------------------
        # NETWORK
        # -------------------------
        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.bn1,
            self.relu1,
            self.dropout1,

            self.conv2,
            self.chomp2,
            self.bn2,
            self.relu2,
            self.dropout2
        )

        # -------------------------
        # RESIDUAL CONNECTION
        # -------------------------
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1)
            if n_inputs != n_outputs
            else None
        )

        self.final_relu = nn.ReLU()

        self.init_weights()

    # =====================================================
    # INITIALIZATION
    # =====================================================
    def init_weights(self):

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)

        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)

        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight)

    # =====================================================
    # FORWARD
    # =====================================================
    def forward(self, x):

        out = self.net(x)

        res = x if self.downsample is None else self.downsample(x)

        return self.final_relu(out + res)


# =========================================================
# TEMPORAL CONV NET
# =========================================================
class TemporalConvNet(nn.Module):

    def __init__(
        self,
        num_inputs,
        num_channels,
        kernel_size=7,
        dropout=0.3
    ):

        super(TemporalConvNet, self).__init__()

        layers = []

        num_levels = len(num_channels)

        for i in range(num_levels):

            dilation_size = 2 ** i

            in_channels = (
                num_inputs
                if i == 0
                else num_channels[i - 1]
            )

            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    n_inputs=in_channels,
                    n_outputs=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):

        return self.network(x)


# =========================================================
# TCN ENCODER
# =========================================================
class TCNEncoder(nn.Module):

    def __init__(
        self,
        input_dim,
        feature_dim,
        seq_len,
        num_channels=[64, 128, 128, 256],
        kernel_size=7
    ):

        super(TCNEncoder, self).__init__()

        self.seq_len = seq_len
        self.input_dim = input_dim

        # -------------------------------------------------
        # TCN
        # -------------------------------------------------
        self.tcn = TemporalConvNet(
            num_inputs=input_dim,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=0.1
        )

        # -------------------------------------------------
        # PROJECTION HEAD
        # -------------------------------------------------
        self.projection = nn.Sequential(

            nn.Linear(num_channels[-1] * 2, 256),

            nn.BatchNorm1d(256),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256, 128),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, feature_dim)
        )

        self._init_weights()

    # =====================================================
    # INITIALIZATION
    # =====================================================
    def _init_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Linear):

                nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # =====================================================
    # FORWARD
    # =====================================================
    def forward(self, x):
        """
        x shape:
        [batch, seq_len, input_dim]
        """

        # -------------------------------------------------
        # CONV FORMAT
        # [batch, channels, seq_len]
        # -------------------------------------------------
        x = x.transpose(1, 2)

        # -------------------------------------------------
        # TCN
        # -------------------------------------------------
        tcn_out = self.tcn(x)

        # -------------------------------------------------
        # TEMPORAL POOLING
        # -------------------------------------------------
        mean_pool = torch.mean(tcn_out, dim=2)

        max_pool = torch.max(tcn_out, dim=2)[0]

        pooled = torch.cat([mean_pool, max_pool], dim=1)

        # -------------------------------------------------
        # PROJECTION
        # -------------------------------------------------
        out = self.projection(pooled)

        # -------------------------------------------------
        # STABILITY
        # -------------------------------------------------
        out = torch.clamp(out, -10.0, 10.0)

        out = torch.nan_to_num(
            out,
            nan=0.0,
            posinf=10.0,
            neginf=-10.0
        )

        return out


# =========================================================
# TCN DECODER
# =========================================================
class TCNDecoder(nn.Module):

    def __init__(
        self,
        output_dim,
        feature_dim,
        seq_len
    ):

        super(TCNDecoder, self).__init__()

        self.seq_len = seq_len

        self.output_dim = output_dim

        self.fc = nn.Sequential(

            nn.Linear(feature_dim, 256),

            nn.ReLU(),

            nn.Dropout(0.1),

            nn.Linear(256, seq_len * output_dim)
        )

        self._init_weights()

    # =====================================================
    # INITIALIZATION
    # =====================================================
    def _init_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Linear):

                nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # =====================================================
    # FORWARD
    # =====================================================
    def forward(self, z):

        x = self.fc(z)

        x = x.view(
            z.size(0),
            self.seq_len,
            self.output_dim
        )

        x = torch.clamp(x, -10.0, 10.0)

        x = torch.nan_to_num(
            x,
            nan=0.0,
            posinf=10.0,
            neginf=-10.0
        )

        return x


# =========================================================
# MAIN NETWORK
# =========================================================
class NetworkTemporalTCN(nn.Module):

    def __init__(
        self,
        view,
        input_size,
        feature_dim,
        high_feature_dim,
        device,
        seq_len=200
    ):

        super(NetworkTemporalTCN, self).__init__()

        self.view = view

        self.seq_len = seq_len

        # -------------------------------------------------
        # ENCODERS
        # -------------------------------------------------
        self.encoders = nn.ModuleList([

            TCNEncoder(
                input_size[v],
                feature_dim,
                seq_len
            ).to(device)

            for v in range(view)
        ])

        # -------------------------------------------------
        # DECODERS
        # -------------------------------------------------
        self.decoders = nn.ModuleList([

            TCNDecoder(
                input_size[v],
                feature_dim,
                seq_len
            ).to(device)

            for v in range(view)
        ])

        # -------------------------------------------------
        # FUSION MODULE
        # -------------------------------------------------
        self.feature_fusion_module = nn.Sequential(

            nn.Linear(view * feature_dim, 256),

            nn.BatchNorm1d(256),

            nn.ReLU(),

            nn.Dropout(0.1),

            nn.Linear(256, 128),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, high_feature_dim)
        )
        # -------------------------------------------------
        # COMMON INFORMATION MODULE
        # -------------------------------------------------
        self.common_information_module = nn.Sequential(

            nn.Linear(feature_dim, high_feature_dim),

            nn.BatchNorm1d(high_feature_dim)
        )

        self._init_weights()

    # =====================================================
    # INITIALIZATION
    # =====================================================
    def _init_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Linear):

                nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # =====================================================
    # FORWARD
    # =====================================================
    def forward(self, xs):

        rs = []

        xrs = []

        zs = []

        # -------------------------------------------------
        # PROCESS EACH VIEW
        # -------------------------------------------------
        for v in range(self.view):

            # Encoder
            z = self.encoders[v](xs[v])

            # Decoder
            xr = self.decoders[v](z)

            # Common representation
            r = self.common_information_module(z)

            # Stability only
            z = torch.clamp(z, -10.0, 10.0)

            r = torch.clamp(r, -10.0, 10.0)

            zs.append(z)

            rs.append(r)

            xrs.append(xr)

        # -------------------------------------------------
        # FUSION
        # -------------------------------------------------
        fusion_input = torch.cat(zs, dim=1)

        H = self.feature_fusion_module(fusion_input)

        # ONLY FINAL NORMALIZATION
        H = F.normalize(H, dim=1, eps=1e-8)

        H = torch.clamp(H, -10.0, 10.0)

        H = torch.nan_to_num(
            H,
            nan=0.0,
            posinf=10.0,
            neginf=-10.0
        )
        
        return xrs, zs, rs, H
