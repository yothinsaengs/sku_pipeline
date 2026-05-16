from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean", pos_weights: torch.Tensor | None = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer("pos_weights", pos_weights.float() if pos_weights is not None else None)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(inputs, targets, reduction="none")
        p_t = inputs * targets + (1.0 - inputs) * (1.0 - targets)
        loss = bce * ((1.0 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss
        if self.pos_weights is not None:
            positive_mask = (targets > 0.5).float()
            weights = 1.0 + positive_mask * (self.pos_weights.view(1, -1) - 1.0)
            loss = loss * weights
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weights: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("pos_weights", pos_weights.float() if pos_weights is not None else None)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        if self.pos_weights is not None:
            positive_mask = (targets > 0.5).float()
            weights = 1.0 + positive_mask * (self.pos_weights.view(1, -1) - 1.0)
            loss = loss * weights
        return loss.mean()


def get_loss_fn(config: Dict[str, Any], pos_weights: torch.Tensor | None = None) -> nn.Module:
    loss_cfg = config.get("loss", {})
    loss_type = loss_cfg.get("type", "focal")
    if loss_type == "focal":
        focal_cfg = loss_cfg.get("focal", {})
        return FocalLoss(alpha=float(focal_cfg.get("alpha", 0.25)), gamma=float(focal_cfg.get("gamma", 2.0)), pos_weights=pos_weights)
    if loss_type == "bce":
        return WeightedBCELoss(pos_weights=pos_weights)
    raise ValueError(f"Unsupported loss type: {loss_type}")
