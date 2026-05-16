import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [B, C], targets: [B, C]
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        loss = bce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class BCELabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1, reduction='mean'):
        super(BCELabelSmoothing, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        # targets: 0 or 1
        # smoothed_targets = targets * (1 - ε) + ε / 2 ?
        # Actually, the requirement says: "ε applied to positive targets only"
        # "target_smooth = (1 - ε) for positive, 0.0 for negative"
        
        # We need to know which ones are positive. 
        # But wait, in sigmoid multi-label, targets are already [B, C].
        # If a sample is positive for class i, targets[i] = 1.
        # If it's negative for class i, targets[i] = 0.
        # If it's a global negative sample, all targets[i] = 0.
        
        smoothed_targets = targets * (1 - self.smoothing)
        return F.binary_cross_entropy(inputs, smoothed_targets, reduction=self.reduction)

def get_loss_fn(config: Dict[str, Any]) -> nn.Module:
    loss_type = config['loss']['type']
    if loss_type == 'focal':
        alpha = config['loss']['focal'].get('alpha', 0.25)
        gamma = config['loss']['focal'].get('gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    elif loss_type == 'bce':
        smoothing = config.get('label_smoothing', 0.1)
        return BCELabelSmoothing(smoothing=smoothing)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
