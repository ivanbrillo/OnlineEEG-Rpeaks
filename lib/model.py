import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., 2018) for 1D.
    Recalibrates channel importance via global context without suppressing
    any temporal positions — critical for dense regression tasks."""
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
        w = self.fc(self.pool(x))   # (B, C, 1) channel weights
        return x * w


class ECABlock(nn.Module):
    """Efficient Channel Attention (Wang et al., 2020).
    Avoids the SE bottleneck by using a 1D conv over the channel dimension.
    Adaptive kernel size based on channel count."""
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1          # nearest odd number
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, L)
        w = self.pool(x)                    # (B, C, 1)
        w = w.squeeze(-1).unsqueeze(1)      # (B, 1, C)
        w = self.sigmoid(self.conv(w))      # (B, 1, C)
        w = w.squeeze(1).unsqueeze(-1)      # (B, C, 1)
        return x * w


class SeizureTransformerImproved(nn.Module):

    def __init__(
        self,
        in_channels=128,
        in_samples=5000,
        dim_feedforward=512,
        num_layers=8,
        num_heads=4,
        drop_rate=0.1,
        skip_type="SE",  # SE or ECA
        conv_type="default",
        skip_concat=False,  # True for summing else False
        norm_type="batch",  # "batch" | "instance" | "group"
    ):
        super().__init__()

        self.in_channels = in_channels
        self.in_samples = in_samples
        self.drop_rate = drop_rate

        # Parameters from EQTransformer repository
        self.filters = [
            8,
            16,
            32,
            64,
            128,
          #  256,
          #  512
        ]  # Number of filters for the convolutions
        self.kernel_sizes = [25, 23, 19, 15, 11]  # Kernel sizes for the convolutions
        self.res_cnn_kernels = [3, 3, 3, 3, 3, 3, 3]

        # Encoder stack
        self.encoder = Encoder(
            input_channels=self.in_channels,
            filters=self.filters,
            kernel_sizes=self.kernel_sizes,
            in_samples=self.in_samples,
            conv_type = conv_type
        )

        # Res CNN Stack
        self.res_cnn_stack = ResCNNStack(
            kernel_sizes=self.res_cnn_kernels,
            filters=self.filters[-1],
            drop_rate=self.drop_rate,
            norm_type=norm_type,
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

        # Detection decoder and final Conv
        self.decoder_d = Decoder(
            input_channels=128,
            filters=self.filters[::-1],
            kernel_sizes=self.kernel_sizes[::-1],
            out_samples=in_samples,
            skip_type = skip_type,
            skip_concat=skip_concat,
        )
        
        self.conv_d = nn.Conv1d(in_channels=self.filters[0], out_channels=1, kernel_size=1)  #1, padding=5)    

    def forward(self, x, logits=True, return_features=False):
        x = x.permute(0, 2, 1)
        assert x.ndim == 3
        assert x.shape[1:] == (self.in_channels, self.in_samples)

        x, skips = self.encoder(x)
        res_x = self.res_cnn_stack(x)

        x = res_x.permute(2, 0, 1)
        x = self.position_encoding(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0)
        bottleneck = x + res_x                       # (B, 128, T)

        detection = self.decoder_d(bottleneck, skips)
        detection = self.conv_d(detection)

        if return_features:
            return detection, bottleneck
        return detection



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
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class SEInception(nn.Module):
    """4 parallel Conv1d branches (inception-style) followed by an SE block.
    
    Kernel offsets: [-4, -2, +2, +4] relative to the requested kernel_size.
    Branch outputs are concatenated then channel-wise re-calibrated by the SE block.
    """
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

        # SE block operates on the full concatenated feature map (out_channels)
        self.se = SEBlock(out_channels, reduction=se_reduction)

    def forward(self, x):
        out = torch.cat([conv(x) for conv in self.convs], dim=1)  # (B, out_channels, L)
        return self.se(out)                                         # channel recalibration



class MultiScaleConv1d(nn.Module):
    """4 parallel Conv1d at kernel_size +/- 1 and +/- 3, outputs concatenated."""
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        assert out_channels % 4 == 0, "out_channels must be divisible by 4"
        branch_channels = out_channels // 4
        offsets = [-4, -2, 2, 4]
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels,
                branch_channels,
                kernel_size=max(1, kernel_size + offset),  # ensure kernel >= 1
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

        # (1) Split Phase: Conv1d -> BatchNorm1d -> ReLU 
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels,
                    kernel_size=max(1, kernel_size + offset),
                    padding=max(1, kernel_size + offset) // 2,
                    bias=False # Bias is redundant before BatchNorm
                ),
                nn.ReLU(inplace=True)
            )
            for offset in offsets
        ])

        # Compact attention router
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Determine the reduced dimension based on ratio 'r' and minimum 'L' 
        mid = max(out_channels // r, L)
        
        # (2 & 3) Fuse Phase components 
        self.fc_reduce = nn.Linear(out_channels, mid, bias=False)
        
        self.fcs = nn.ModuleList([
            nn.Linear(mid, out_channels, bias=False)
            for _ in offsets
        ])
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        # (1) Multi-scale feature extraction
        feats = [conv(x) for conv in self.convs]     # list of (B, C, L)

        # (2) Fused descriptor
        fused = sum(feats)                           # (B, C, L)

        # (3) Compact channel attention
        s = self.pool(fused).squeeze(-1)             # (B, C)
        z = F.elu(self.fc_reduce(s))# (B, mid)

        # Per-branch attention weights
        weights = torch.stack(
            [fc(z) for fc in self.fcs], dim=0        # (n_branches, B, C)
        )
        weights = self.softmax(weights)              # (n_branches, B, C)

        # (4) Adaptive weighted sum
        out = sum(
            w.unsqueeze(-1) * f
            for w, f in zip(weights, feats)
        )                                            # (B, C, L)
        return out

class Encoder(nn.Module):
    """
    Encoder stack
    """
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
                    nn.Conv1d(
                        in_channels, out_channels, kernel_size, padding=kernel_size // 2
                    )
                )
            elif conv_type == "multiscale":
                convs.append(MultiScaleConv1d(in_channels, out_channels, kernel_size))
            elif conv_type == "SK":
                convs.append(SelectiveKernelConv1d(in_channels, out_channels, kernel_size))
            elif conv_type == "InceptionSE":
                convs.append(SEInception(in_channels, out_channels, kernel_size))
            else:
                raise Exception()
            # To be consistent with the behaviour in tensorflow,
            # padding needs to be added for odd numbers of input_samples
            padding = in_samples % 2

            # Padding for MaxPool1d needs to be handled manually to conform with tf padding
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
                # Only pad right, use -1e10 as negative infinity
                x = F.pad(x, (0, padding), "constant", -1e10)
            x = pool(x)

        return x, skips


class Decoder(nn.Module):
    def __init__(
        self,
        input_channels,
        filters,
        kernel_sizes,
        out_samples,
        skip_type,
        skip_concat,          # If True, concat skip; if False, element-wise add
    ):
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
                nn.Conv1d(
                    conv_in_channels, out_channels, kernel_size, padding=kernel_size // 2
                )
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
                    x = torch.cat([x, skip], dim=1)     # (B, 2C, T)
                    x = elu(conv(x))
                else:
                    x = elu(conv(x))
                    x = x + skip
            else:
                x = elu(conv(x))

        return x


def _make_norm(norm_type: str, filters: int):
    """Return a 1-D normalisation layer for `filters` channels.

    norm_type : "batch" | "instance" | "group"
        batch    — BatchNorm1d  (shares stats across the batch; original behaviour)
        instance — InstanceNorm1d  (per-sample normalisation; recommended for cross-subject)
        group    — GroupNorm with min(filters, 8) groups
    """
    if norm_type == "batch":
        return nn.BatchNorm1d(filters, eps=1e-3)
    elif norm_type == "instance":
        return nn.InstanceNorm1d(filters, eps=1e-3, affine=True)
    elif norm_type == "group":
        num_groups = min(filters, 8)
        return nn.GroupNorm(num_groups, filters, eps=1e-3)
    else:
        raise ValueError(f"Unknown norm_type '{norm_type}'. Choose 'batch', 'instance', or 'group'.")


class ResCNNStack(nn.Module):
    def __init__(self, kernel_sizes, filters, drop_rate, norm_type="instance"):
        super().__init__()

        members = []
        for ker in kernel_sizes:
            members.append(ResCNNBlock(filters, ker, drop_rate, norm_type=norm_type))
        self.members = nn.ModuleList(members)
    def forward(self, x):
        for member in self.members:
            x = member(x)
        return x


class ResCNNBlock(nn.Module):
    def __init__(self, filters, ker, drop_rate, norm_type="instance"):
        super().__init__()

        self.manual_padding = False
        padding = ker // 2

        if ker == 2:
            # Manual padding emulate the padding in tensorflow
            self.manual_padding = True
            padding = 0

        self.dropout = SpatialDropout1d(drop_rate)

        self.norm1 = _make_norm(norm_type, filters)
        self.conv1 = nn.Conv1d(filters, filters, ker, padding=padding)

        self.norm2 = _make_norm(norm_type, filters)
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


class SpatialDropout1d(nn.Module):
    def __init__(self, drop_rate):
        super().__init__()

        self.drop_rate = drop_rate
        self.dropout = nn.Dropout2d(drop_rate)
    def forward(self, x):
        x = x.unsqueeze(dim=-1)  # Add fake dimension
        x = self.dropout(x)
        x = x.squeeze(dim=-1)  # Remove fake dimension
        return x