"""Residual convolutional policy-value network.

The policy head emits **unnormalized** logits over the full fixed action space
(no softmax inside the model); masking and softmax happen in the loss/metrics.
The value head predicts the normalized optimal remaining cost
(``-raw_moves / constant``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import EncodingConfig, ModelConfig


def _normalization(channels: int, kind: str) -> nn.Module:
    if kind == "batch_norm":
        return nn.BatchNorm2d(channels)
    if kind == "group_norm":
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"unsupported normalization {kind!r}")


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, normalization: str = "batch_norm") -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = _normalization(channels, normalization)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = _normalization(channels, normalization)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return torch.relu(x + residual)


class _PolicyResidualAdapter(nn.Module):
    """Policy-only residual capacity with an exact identity initialization."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(torch.relu(self.conv1(x)))


class PolicyValueNet(nn.Module):
    """Compact AlphaZero-style dual-head residual net."""

    def __init__(
        self, encoding_config: EncodingConfig, model_config: ModelConfig
    ) -> None:
        super().__init__()
        self.encoding_config = encoding_config
        self.model_config = model_config

        in_ch = encoding_config.num_board_channels
        c = model_config.channels
        normalization = model_config.normalization
        h, w = encoding_config.max_rows, encoding_config.max_cols
        self.policy_planes = encoding_config.policy_channels  # 4 * move_code_count

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1, bias=False),
            _normalization(c, normalization),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            *[_ResidualBlock(c, normalization)
              for _ in range(model_config.residual_blocks)]
        )

        # Optional policy-specific capacity. The zero-initialized final
        # convolution makes each block an exact identity when introduced into
        # a legacy checkpoint, and the value branch never observes it.
        self.policy_adapter = nn.Sequential(*[
            _PolicyResidualAdapter(c)
            for _ in range(model_config.policy_adapter_blocks)
        ])

        # Policy head: spatial conv producing 4 * move_code_count channels.
        self.policy_conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            _normalization(c, normalization),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, self.policy_planes, 1),
        )

        # Value head: 1x1 conv -> flatten -> + global features -> MLP -> scalar.
        self.value_conv = nn.Sequential(
            nn.Conv2d(c, 1, 1),
            nn.ReLU(inplace=True),
        )
        value_in = h * w + encoding_config.num_global_features
        self.value_mlp = nn.Sequential(
            nn.Linear(value_in, model_config.value_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(model_config.value_hidden_size, 1),
        )

    def forward(
        self, board: torch.Tensor, global_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(policy_logits [B, A], value [B])``."""
        x = self.stem(board)
        x = self.trunk(x)

        p = self.policy_conv(self.policy_adapter(x))  # [B, 4*M, H, W]
        logits = self._flatten_policy(p)

        v = self.value_conv(x).flatten(1)  # [B, H*W]
        v = torch.cat([v, global_features], dim=1)
        value = self.value_mlp(v).squeeze(-1)  # [B]
        return logits, value

    def _flatten_policy(self, p: torch.Tensor) -> torch.Tensor:
        """[B, 4*M, H, W] -> [B, A] in the documented index order.

        index = ((r * W + c) * 4 + dir) * M + move_code, with channel =
        dir * M + move_code. Splitting the channel into (dir, M) and permuting to
        [B, H, W, dir, M] then flattening reproduces the formula exactly.
        """
        b, ch, h, w = p.shape
        m = self.encoding_config.move_code_count
        p = p.view(b, 4, m, h, w)            # (dir, move_code, H, W)
        p = p.permute(0, 3, 4, 1, 2)         # (B, H, W, dir, move_code)
        return p.reshape(b, h * w * 4 * m)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
