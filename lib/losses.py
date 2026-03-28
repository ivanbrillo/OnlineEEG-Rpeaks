import math
import torch.nn as nn
import torch


class WingLoss(nn.Module):
    def __init__(self, width=5, curvature=0.5, reduction='mean'):
        super(WingLoss, self).__init__()
        self.width = width
        self.curvature = curvature
        self.reduction = reduction

        # Constant C ensures the function is continuous at the transition point 'w'
        self.C = self.width - self.width * math.log(1 + self.width / self.curvature)

    def forward(self, prediction, target):
        diff = torch.abs(prediction - target)

        # Case 1: Small errors (logarithmic region) -> Loss = w * ln(1 + |x| / epsilon)
        wing_part = self.width * torch.log(1 + diff / self.curvature)

        # Case 2: Large errors (linear region like L1) -> Loss = |x| - C
        linear_part = diff - self.C

        # Select between the two cases based on the threshold 'width'
        loss = torch.where(diff < self.width, wing_part, linear_part)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
