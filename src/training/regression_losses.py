import torch
import torch.nn as nn


class MultiTaskRegressionLoss(nn.Module):
    def __init__(
        self, depth_weight=0.7, overflow_weight=0.3, flood_weight=5.0, pos_weight=None
    ):
        super().__init__()
        self.depth_weight = depth_weight
        self.overflow_weight = overflow_weight
        self.flood_weight = flood_weight
        self.pos_weight = pos_weight
        self.bce = nn.BCELoss(reduction="none")

    def forward(
        self, pred_depths, pred_overflow, true_depths, true_overflow, flood_mask
    ):
        depth_mse = (pred_depths - true_depths) ** 2
        weights = torch.where(flood_mask > 0.5, self.flood_weight, 1.0).unsqueeze(1)
        depth_loss = (depth_mse * weights).mean()

        pred_overflow = pred_overflow.squeeze(1)
        overflow_bce = self.bce(pred_overflow, true_overflow)

        if self.pos_weight is not None:
            class_weights = torch.where(true_overflow > 0.5, self.pos_weight, 1.0)
            overflow_loss = (overflow_bce * class_weights).mean()
        else:
            overflow_loss = overflow_bce.mean()

        total_loss = (
            self.depth_weight * depth_loss + self.overflow_weight * overflow_loss
        )
        return total_loss, depth_loss, overflow_loss
