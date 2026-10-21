"""Focused unit tests for Track B QML experiment components.

Tests feature extraction shapes, head output shapes, gradient flow,
masked loss calculation, reproducibility, and evaluation metrics.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.qml.qml_heads import ClassicalLinearHead, SimulatedQuantumHead
from src.models.classical.loss import masked_bce_with_logits
from src.models.classical.metrics import compute_auc_ap
from src.models.classical.model import DenseNet121MultiLabel
from src.preprocessing.labels import TARGET_LABELS


def test_feature_extractor_output_shape():
    """Verify DenseNet-121 feature extractor emits (batch, 1024) embeddings."""
    model = DenseNet121MultiLabel(pretrained=False)
    dummy = torch.randn(2, 1, 224, 224)
    if dummy.shape[1] == 1:
        dummy = dummy.repeat(1, 3, 1, 1)

    with torch.no_grad():
        feat = model.backbone.features(dummy)
        feat = F.relu(feat, inplace=False)
        feat = F.adaptive_avg_pool2d(feat, (1, 1))
        feat = torch.flatten(feat, 1)

    assert feat.shape == (2, 1024), f"Expected shape (2, 1024), got {feat.shape}"


def test_classical_linear_head_shape_and_grad():
    """Verify ClassicalLinearHead output shape and gradient propagation."""
    head = ClassicalLinearHead(in_features=1024, num_classes=len(TARGET_LABELS))
    features = torch.randn(4, 1024)
    logits = head(features)

    assert logits.shape == (4, 3)
    loss = logits.sum()
    loss.backward()

    for name, param in head.named_parameters():
        assert param.grad is not None, f"Gradient missing for {name}"


def test_simulated_quantum_head_shape_and_grad():
    """Verify SimulatedQuantumHead output shape and gradient propagation."""
    head = SimulatedQuantumHead(in_features=1024, n_qubits=2, num_classes=len(TARGET_LABELS))
    features = torch.randn(4, 1024)
    logits = head(features)

    assert logits.shape == (4, 3)
    loss = logits.sum()
    loss.backward()

    for name, param in head.named_parameters():
        assert param.grad is not None, f"Gradient missing for {name}"
    # Verify the toy quantum layer params received gradient
    assert head.quantum.params.grad is not None


def test_masked_loss_with_uncertain_labels():
    """Verify masked BCE loss strictly masks out uncertain entries (mask=0)."""
    logits = torch.tensor([[2.0, -1.0, 3.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 1.0]])

    # Case 1: All masked as valid
    mask_all = torch.tensor([[1.0, 1.0, 1.0]])
    loss_all = masked_bce_with_logits(logits, targets, mask_all)

    # Case 2: Middle label is uncertain (masked out)
    mask_ignore = torch.tensor([[1.0, 0.0, 1.0]])
    loss_ignore = masked_bce_with_logits(logits, targets, mask_ignore)

    # Compute expected loss for labels 0 and 2 only
    bce_0 = F.binary_cross_entropy_with_logits(logits[:, 0:1], targets[:, 0:1])
    bce_2 = F.binary_cross_entropy_with_logits(logits[:, 2:3], targets[:, 2:3])
    expected = (bce_0 + bce_2) / 2.0

    assert torch.isclose(loss_ignore, expected), f"Expected {expected}, got {loss_ignore}"
    assert loss_all != loss_ignore


def test_heads_reproducibility():
    """Verify deterministic output for both heads under fixed seeds."""
    torch.manual_seed(42)
    head_c1 = ClassicalLinearHead(1024, 3)
    features = torch.randn(2, 1024)
    out_c1 = head_c1(features)

    torch.manual_seed(42)
    head_c2 = ClassicalLinearHead(1024, 3)
    out_c2 = head_c2(features)
    assert torch.allclose(out_c1, out_c2)

    torch.manual_seed(42)
    head_q1 = SimulatedQuantumHead(1024, 2, 3)
    out_q1 = head_q1(features)

    torch.manual_seed(42)
    head_q2 = SimulatedQuantumHead(1024, 2, 3)
    out_q2 = head_q2(features)
    assert torch.allclose(out_q1, out_q2)


def test_evaluation_metrics_computation():
    """Verify compute_auc_ap produces valid AUROC and AP when both classes exist."""
    # Synthetic logits, targets, masks
    logits = torch.tensor([
        [2.0, -1.0, 1.5],
        [-2.0, 1.0, -1.5],
        [1.0, -2.0, 0.5],
        [-1.0, 2.0, -0.5],
    ])
    targets = torch.tensor([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ])
    masks = torch.ones_like(targets)

    auroc, ap = compute_auc_ap(logits, targets, masks)
    for disease in TARGET_LABELS:
        assert auroc[disease] is not None
        assert 0.0 <= auroc[disease] <= 1.0
        assert ap[disease] is not None
        assert 0.0 <= ap[disease] <= 1.0
