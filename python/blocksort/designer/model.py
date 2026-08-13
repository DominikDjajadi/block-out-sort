"""Designer policy-value network.

Policy logits cover the fixed designer action space: index 0 is ``STOP`` (a scalar
head from globally pooled features) and the remaining indices are reverse-slide
logits produced spatially by a ``4 * max_distance`` conv, flattened with the same
``((r*W + c)*4 + dir)*M + (distance-1)`` ordering used by
:class:`~blocksort.designer.actions.DesignerActionSpace`. The value head predicts
the expected final designer reward (a scalar, unbounded).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from ..training.config import EncodingConfig
from ..training.model import _ResidualBlock
from .encoding import num_global_features


@dataclass(frozen=True)
class DesignerModelConfig:
    channels: int = 32
    residual_blocks: int = 2
    hidden_size: int = 64

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DesignerModelConfig":
        return cls(channels=int(data["channels"]),
                   residual_blocks=int(data["residual_blocks"]),
                   hidden_size=int(data["hidden_size"]))


class DesignerNet(nn.Module):
    def __init__(self, encoding_config: EncodingConfig,
                 model_config: DesignerModelConfig) -> None:
        super().__init__()
        self.encoding_config = encoding_config
        self.model_config = model_config

        in_ch = encoding_config.num_board_channels
        c = model_config.channels
        h, w = encoding_config.max_rows, encoding_config.max_cols
        self.max_distance = encoding_config.max_slide_distance
        self.reverse_planes = 4 * self.max_distance
        self.global_dim = num_global_features(encoding_config)

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            *[_ResidualBlock(c) for _ in range(model_config.residual_blocks)]
        )

        self.policy_conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, self.reverse_planes, 1),
        )

        # STOP logit + value share globally pooled trunk features + globals.
        head_in = c + self.global_dim
        self.stop_head = nn.Sequential(
            nn.Linear(head_in, model_config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(model_config.hidden_size, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(head_in, model_config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(model_config.hidden_size, 1),
        )

    def forward(self, board: torch.Tensor, global_features: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(policy_logits [B, A], value [B])`` with A = action-space size."""
        x = self.stem(board)
        x = self.trunk(x)

        reverse = self._flatten_reverse(self.policy_conv(x))  # [B, R*C*4*M]
        pooled = x.mean(dim=(2, 3))                           # [B, c]
        head_in = torch.cat([pooled, global_features], dim=1)
        stop_logit = self.stop_head(head_in)                 # [B, 1]
        logits = torch.cat([stop_logit, reverse], dim=1)     # [B, 1 + R*C*4*M]
        value = self.value_head(head_in).squeeze(-1)         # [B]
        return logits, value

    def _flatten_reverse(self, p: torch.Tensor) -> torch.Tensor:
        b, ch, h, w = p.shape
        m = self.max_distance
        p = p.view(b, 4, m, h, w)        # (dir, distance-1, H, W)
        p = p.permute(0, 3, 4, 1, 2)     # (B, H, W, dir, distance-1)
        return p.reshape(b, h * w * 4 * m)

    @property
    def action_space_size(self) -> int:
        h, w = self.encoding_config.max_rows, self.encoding_config.max_cols
        return 1 + h * w * self.reverse_planes
