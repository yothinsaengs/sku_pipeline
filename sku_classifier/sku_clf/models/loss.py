from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(inputs, targets, reduction="none")
        p_t = inputs * targets + (1.0 - inputs) * (1.0 - targets)
        loss = bce * ((1.0 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def get_loss_fn(config: Dict[str, Any]) -> nn.Module:
    loss_cfg = config.get("loss", {})
    loss_type = loss_cfg.get("type", "focal")
    if loss_type == "focal":
        focal_cfg = loss_cfg.get("focal", {})
        return FocalLoss(alpha=float(focal_cfg.get("alpha", 0.25)), gamma=float(focal_cfg.get("gamma", 2.0)))
    if loss_type == "bce":
        return nn.BCELoss()
    raise ValueError(f"Unsupported loss type: {loss_type}")
