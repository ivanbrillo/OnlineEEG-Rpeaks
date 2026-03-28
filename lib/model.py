import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention for 1D."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w


class ECABlock(nn.Module):
    """Efficient Channel Attention (Wang et al., 2020)."""
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = w.squeeze(-1).unsqueeze(1)
        w = self.sigmoid(self.conv(w))
        w = w.squeeze(1).unsqueeze(-1)
        return x * w


class SEInception(nn.Module):
    """4 parallel Conv1d branches (inception-style) followed by an SE block."""
    def __init__(self, in_channels, out_channels, kernel_size, se_reduction=4):
        super().__init__()
        assert out_channels % 4 == 0, "out_channels must be divisible by 4"
        branch_channels = out_channels // 4
        offsets = [-4, -2, 2, 4]

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels,
                branch_channels,
                kernel_size=max(1, kernel_size + offset),
                padding=max(1, kernel_size + offset) // 2,
            )
            for offset in offsets
        ])
        self.se = SEBlock(out_channels, reduction=se_reduction)

    def forward(self, x):
        out = torch.cat([conv(x) for conv in self.convs], dim=1)
        return self.se(out)


class MultiScaleConv1d(nn.Module):
    """4 parallel Conv1d at varied kernel sizes, outputs concatenated."""
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        assert out_channels % 4 == 0, "out_channels must be divisible by 4"
        branch_channels = out_channels // 4
        offsets = [-4, -2, 2, 4]
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels,
                branch_channels,
                kernel_size=max(1, kernel_size + offset),
                padding=max(1, kernel_size + offset) // 2,
            )
            for offset in offsets
        ])

    def forward(self, x):
        return torch.cat([conv(x) for conv in self.convs], dim=1)


class SelectiveKernelConv1d(nn.Module):
    """SKNet-style adaptive multi-scale convolution (Li et al., 2019)."""
    def __init__(self, in_channels, out_channels, kernel_size, r=4, L=8):
        super().__init__()
        offsets = [-4, -2, 2, 4]
        self.n_branches = len(offsets)

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels,
                    kernel_size=max(1, kernel_size + offset),
                    padding=max(1, kernel_size + offset) // 2,
                    bias=False
                ),
                nn.ReLU(inplace=True)
            )
            for offset in offsets
        ])

        self.pool = nn.AdaptiveAvgPool1d(1)
        mid = max(out_channels // r, L)
        self.fc_reduce = nn.Linear(out_channels, mid, bias=False)
        self.fcs = nn.ModuleList([
            nn.Linear(mid, out_channels, bias=False)
            for _ in offsets
        ])
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        feats = [conv(x) for conv in self.convs]
        fused = sum(feats)
        s = self.pool(fused).squeeze(-1)
        z = F.elu(self.fc_reduce(s))
        weights = torch.stack([fc(z) for fc in self.fcs], dim=0)
        weights = self.softmax(weights)
        out = sum(w.unsqueeze(-1) * f for w, f in zip(weights, feats))
        return out


class SpatialDropout1d(nn.Module):
    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout2d(drop_rate)

    def forward(self, x):
        x = x.unsqueeze(dim=-1)
        x = self.dropout(x)
        x = x.squeeze(dim=-1)
        return x


class ResCNNBlock(nn.Module):
    def __init__(self, filters, ker, drop_rate):
        super().__init__()
        self.manual_padding = False
        padding = ker // 2

        if ker == 2:
            self.manual_padding = True
            padding = 0

        self.dropout = SpatialDropout1d(drop_rate)
        self.norm1 = nn.BatchNorm1d(filters, eps=1e-3)
        self.conv1 = nn.Conv1d(filters, filters, ker, padding=padding)
        self.norm2 = nn.BatchNorm1d(filters, eps=1e-3)
        self.conv2 = nn.Conv1d(filters, filters, ker, padding=padding)

    def forward(self, x):
        y = self.norm1(x)
        y = F.relu(y)
        y = self.dropout(y)
        if self.manual_padding:
            y = F.pad(y, (0, 1), "constant", 0)
        y = self.conv1(y)

        y = self.norm2(y)
        y = F.relu(y)
        y = self.dropout(y)
        if self.manual_padding:
            y = F.pad(y, (0, 1), "constant", 0)
        y = self.conv2(y)

        return x + y


class ResCNNStack(nn.Module):
    def __init__(self, kernel_sizes, filters, drop_rate):
        super().__init__()
        members = []
        for ker in kernel_sizes:
            members.append(ResCNNBlock(filters, ker, drop_rate))
        self.members = nn.ModuleList(members)

    def forward(self, x):
        for member in self.members:
            x = member(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self, input_channels, filters, kernel_sizes, in_samples, conv_type):
        super().__init__()
        convs = []
        pools = []
        elus = []
        self.paddings = []
        for in_channels, out_channels, kernel_size in zip(
            [input_channels] + filters[:-1], filters, kernel_sizes
        ):
            if conv_type == "default":
                convs.append(
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
                )
            elif conv_type == "multiscale":
                convs.append(MultiScaleConv1d(in_channels, out_channels, kernel_size))
            elif conv_type == "SK":
                convs.append(SelectiveKernelConv1d(in_channels, out_channels, kernel_size))
            elif conv_type == "InceptionSE":
                convs.append(SEInception(in_channels, out_channels, kernel_size))
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")

            padding = in_samples % 2
            self.paddings.append(padding)
            pools.append(nn.MaxPool1d(2, padding=0))
            elus.append(nn.ELU(inplace=True))
            in_samples = (in_samples + padding) // 2

        self.convs = nn.ModuleList(convs)
        self.pools = nn.ModuleList(pools)
        self.elus = nn.ModuleList(elus)

    def forward(self, x):
        skips = []
        for conv, pool, padding, elu in zip(self.convs, self.pools, self.paddings, self.elus):
            x = elu(conv(x))
            skips.append(x)
            if padding != 0:
                x = F.pad(x, (0, padding), "constant", -1e10)
            x = pool(x)
        return x, skips


class Decoder(nn.Module):
    def __init__(self, input_channels, filters, kernel_sizes, out_samples, skip_type, skip_concat):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.skip_concat = skip_concat
        self.skip_type = skip_type

        self.crops = []
        current_samples = out_samples
        for i, _ in enumerate(filters):
            padding = current_samples % 2
            current_samples = (current_samples + padding) // 2
            if padding == 1:
                self.crops.append(len(filters) - 1 - i)

        convs = []
        elus = []
        skip_se = []
        for in_channels, out_channels, kernel_size in zip(
            [input_channels] + filters[:-1], filters, kernel_sizes
        ):
            conv_in_channels = in_channels + out_channels if skip_concat else in_channels
            convs.append(
                nn.Conv1d(conv_in_channels, out_channels, kernel_size, padding=kernel_size // 2)
            )
            elus.append(nn.ELU(inplace=True))
            skip_se.append(SEBlock(out_channels) if skip_type == "SE" else ECABlock(out_channels))

        self.convs = nn.ModuleList(convs)
        self.elus = nn.ModuleList(elus)
        self.skip_se = nn.ModuleList(skip_se)

    def forward(self, x, skip_connections):
        for i, (conv, elu) in enumerate(zip(self.convs, self.elus)):
            x = self.upsample(x)
            if i in self.crops:
                x = x[:, :, :-1]

            if skip_connections is not None and i < len(skip_connections):
                skip = skip_connections[-(i + 1)]
                if self.skip_type != "default":
                    skip = self.skip_se[i](skip)

                if self.skip_concat:
                    if skip.shape[-1] != x.shape[-1]:
                        skip = skip[:, :, : x.shape[-1]]
                    x = torch.cat([x, skip], dim=1)
                    x = elu(conv(x))
                else:
                    x = elu(conv(x))
                    x = x + skip
            else:
                x = elu(conv(x))

        return x


class SeizureTransformerImproved(nn.Module):
    def __init__(
        self,
        in_channels=128,
        in_samples=5000,
        dim_feedforward=512,
        num_layers=8,
        num_heads=4,
        drop_rate=0.1,
        skip_type="SE",
        conv_type="default",
        skip_concat=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.in_samples = in_samples
        self.drop_rate = drop_rate

        self.filters = [8, 16, 32, 64, 128]
        self.kernel_sizes = [25, 23, 19, 15, 11]
        self.res_cnn_kernels = [3, 3, 3, 3, 3, 3, 3]

        self.encoder = Encoder(
            input_channels=self.in_channels,
            filters=self.filters,
            kernel_sizes=self.kernel_sizes,
            in_samples=self.in_samples,
            conv_type=conv_type
        )

        self.res_cnn_stack = ResCNNStack(
            kernel_sizes=self.res_cnn_kernels,
            filters=self.filters[-1],
            drop_rate=self.drop_rate,
        )

        self.position_encoding = PositionalEncoding(d_model=128)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            norm_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers,
        )

        self.decoder_d = Decoder(
            input_channels=128,
            filters=self.filters[::-1],
            kernel_sizes=self.kernel_sizes[::-1],
            out_samples=in_samples,
            skip_type=skip_type,
            skip_concat=skip_concat,
        )

        self.conv_d = nn.Conv1d(in_channels=self.filters[0], out_channels=1, kernel_size=1)

    def forward(self, x, logits=True):
        x = x.permute(0, 2, 1)
        assert x.ndim == 3
        assert x.shape[1:] == (self.in_channels, self.in_samples)

        x, skips = self.encoder(x)
        res_x = self.res_cnn_stack(x)

        x = res_x.permute(2, 0, 1)
        x = self.position_encoding(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0)
        x = x + res_x

        detection = self.decoder_d(x, skips)
        detection = self.conv_d(detection)
        return detection
