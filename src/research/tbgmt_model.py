import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, dilation=dilation, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        r = self.res(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop(x)
        return F.relu(x + r)


class TBGMT(nn.Module):
    def __init__(
        self,
        input_channels,
        input_length,
        n_classes,
        d_model=128,
        tcn_channels=(64, 128),
        tcn_kernel_size=3,
        tcn_dilations=(1, 2, 4),
        gru_hidden=128,
        transformer_layers=2,
        transformer_heads=4,
        transformer_ff=512,
        dropout=0.3,
    ):
        super().__init__()
        self.input_length = input_length
        self.n_classes = n_classes

        tcn = []
        in_ch = input_channels
        for out_ch in tcn_channels:
            for d in tcn_dilations:
                tcn.append(TCNBlock(in_ch, out_ch, kernel_size=tcn_kernel_size, dilation=d, dropout=dropout))
                in_ch = out_ch
        self.tcn = nn.Sequential(*tcn)

        self.bigru = nn.GRU(
            input_size=in_ch,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.gru_drop = nn.Dropout(dropout)

        self.proj = nn.Linear(gru_hidden * 2, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model=d_model, max_len=max(512, input_length))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)

        self.fc1 = nn.Linear(d_model, 256)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 128)
        self.drop2 = nn.Dropout(0.4)
        self.out = nn.Linear(128, n_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x.permute(0, 2, 1)

        x, _ = self.bigru(x)
        x = self.gru_drop(x)

        x = self.proj(x)
        x = self.pos(x)
        x = self.transformer(x)

        x = torch.mean(x, dim=1)
        x = F.gelu(self.fc1(x))
        x = self.drop1(x)
        x = F.gelu(self.fc2(x))
        x = self.drop2(x)
        return self.out(x)
