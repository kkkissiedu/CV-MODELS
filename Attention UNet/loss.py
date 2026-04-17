# %% Imports
import torch
import torch.nn as nn
import torch.nn.functional as F


# %% HybridLoss
class HybridLoss(nn.Module):
    """
    Binary segmentation loss: BCEWithLogitsLoss + Dice loss.

        total = alpha * bce_loss + (1 - alpha) * dice_loss

    Receives raw logits — sigmoid is applied internally.
    pos_weight handles class imbalance (background >> crack pixels).

    Args:
        alpha     : weight of BCE vs Dice (0.5 = equal weight)
        pos_weight: scalar tensor from get_automated_weights()
        smooth    : Laplace smoothing in Dice denominator (prevents div/0)
    """

    def __init__(self,
                 alpha: float = 0.5,
                 pos_weight: torch.Tensor = None,
                 smooth: float = 1e-6):
        super(HybridLoss, self).__init__()

        self.alpha  = alpha
        self.smooth = smooth

        if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
            self.pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
        else:
            self.pos_weight = pos_weight

        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Move pos_weight to same device as inputs
        if self.pos_weight is not None:
            self.pos_weight          = self.pos_weight.to(inputs.device)
            self.bce_loss.pos_weight = self.pos_weight

        # BCE loss — operates on raw logits
        bce = self.bce_loss(inputs, targets.float())

        # Dice loss — convert logits to probabilities first
        probs = torch.sigmoid(inputs)
        true  = targets.float()

        # Flatten spatial dims: (B, 1, H, W) -> (B, 1, H*W)
        probs_flat = probs.view(probs.size(0), -1)
        true_flat  = true.view(true.size(0),  -1)

        intersection = torch.sum(probs_flat * true_flat, dim=1)
        cardinality  = torch.sum(probs_flat + true_flat, dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss  = (1.0 - dice_score).mean()

        return self.alpha * bce + (1.0 - self.alpha) * dice_loss
