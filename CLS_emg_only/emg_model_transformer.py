"""Transformer-based acoustic model for the 15-channel EMG corpus."""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


__all__ = ["EMGTransformer15"]


def _conv_block(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: Tuple[int, int] = (3, 3),
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (1, 1),
    dropout: float = 0.0,
) -> nn.Sequential:
    layers = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout2d(p=dropout))
    return nn.Sequential(*layers)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, dim: int, dropout: float = 0.1, max_len: int = 1000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, dim)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, C)
        length = x.size(1)
        x = x + self.pe[:, :length]
        return self.dropout(x)


class EMGTransformer15(nn.Module):
    """CNN front-end followed by a Transformer encoder for EMG recognition."""

    def __init__(
        self,
        num_classes: int,
        *,
        in_channels: int = 15,
        embed_dim: int = 256,
        num_heads: int = 4,
        ff_dim: int = 512,
        num_layers: int = 4,
        dropout: float = 0.1,
        every_frame: bool = True,
    ) -> None:
        super().__init__()
        self.every_frame = every_frame

        self.frontend = nn.Sequential(
            _conv_block(in_channels, 64),
            _conv_block(64, 64),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 36x36 -> 18x18
            _conv_block(64, 128, dropout=0.1),
            _conv_block(128, 128),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 18x18 -> 9x9
            _conv_block(128, 256, dropout=0.2),
            _conv_block(256, 256),
            nn.MaxPool2d(kernel_size=(2, 1)),  # reduce frequency, preserve time
            nn.Dropout2d(p=0.3),
        )

        self.temporal_proj = nn.Linear(256, embed_dim)
        self.positional = PositionalEncoding(embed_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

        self._reset_parameters()

    # ------------------------------------------------------------------
    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, F, T)
        x = self.frontend(x)
        x = x.mean(dim=2)  # average frequency axis -> (B, channels, time)
        x = x.transpose(1, 2)  # (B, time, channels)
        x = self.temporal_proj(x)
        x = self.positional(x)
        x = self.transformer(x)
        if self.every_frame:
            logits = self.classifier(x)
        else:
            logits = self.classifier(x[:, -1, :])
        return logits
