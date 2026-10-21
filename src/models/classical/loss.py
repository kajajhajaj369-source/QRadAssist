import torch
from torch import nn

def masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute binary cross‑entropy with logits, masking out uncertain entries.

    Args:
        logits: Tensor of shape (batch, num_classes).
        targets: Same shape, containing 0/1 targets.
        mask: Same shape, containing 1.0 for valid entries and 0.0 for masked (uncertain).
    Returns:
        Scalar loss (mean over unmasked elements).
    """
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    masked = bce * mask
    # Avoid division by zero – if all entries are masked, return 0 loss.
    total = mask.sum()
    return masked.sum() / total.clamp_min(1.0)
