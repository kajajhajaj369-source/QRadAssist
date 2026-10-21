import numpy as np
from typing import Tuple, List
import torch
from torch import Tensor

from sklearn.metrics import roc_auc_score, average_precision_score

def compute_auc_ap(logits: Tensor, targets: Tensor, mask: Tensor) -> Tuple[dict, dict]:
    """Compute per‑class AUROC and Average Precision.

    Args:
        logits: (N, C) raw model outputs.
        targets: (N, C) binary ground‑truth labels.
        mask: (N, C) mask with 1.0 for valid entries, 0.0 for uncertain/ignored.
    Returns:
        (auroc_dict, ap_dict) mapping disease name to metric value (float) or ``None``
        when the metric cannot be computed due to lack of positive/negative samples.
    """
    # Apply mask
    valid_logits = logits[mask.bool()].cpu().numpy()
    valid_targets = targets[mask.bool()].cpu().numpy()
    # Need original per‑class arrays; we compute class‑wise using full tensors
    num_classes = logits.shape[1]
    auroc: dict = {}
    ap: dict = {}
    # Use the global TARGET_LABELS order for naming
    from preprocessing.labels import TARGET_LABELS
    for i, disease in enumerate(TARGET_LABELS):
        # Extract column i
        col_mask = mask[:, i].bool()
        if col_mask.sum() == 0:
            auroc[disease] = None
            ap[disease] = None
            continue
        y_true = targets[col_mask, i].cpu().numpy()
        y_score = logits[col_mask, i].cpu().numpy()
        # AUROC requires at least one positive and one negative sample
        if len(np.unique(y_true)) < 2:
            auroc[disease] = None
        else:
            try:
                auroc[disease] = float(roc_auc_score(y_true, y_score))
            except Exception:
                auroc[disease] = None
        # Average precision works with single‑class but returns 0.0 if no positives
        try:
            ap[disease] = float(average_precision_score(y_true, y_score))
        except Exception:
            ap[disease] = None
    return auroc, ap
