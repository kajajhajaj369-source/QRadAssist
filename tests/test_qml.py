import torch
import pytest
from src.models.classical.qml import HybridQuantumClassifier
from preprocessing.labels import TARGET_LABELS


def test_hybrid_qml_forward_shape():
    model = HybridQuantumClassifier(n_qubits=2)
    dummy = torch.randn(1, 1, 224, 224)
    logits = model(dummy)
    assert logits.shape == (1, len(TARGET_LABELS))


def test_hybrid_qml_forward_three_channels():
    model = HybridQuantumClassifier(n_qubits=2)
    dummy = torch.randn(1, 3, 224, 224)
    logits = model(dummy)
    assert logits.shape == (1, len(TARGET_LABELS))


def test_hybrid_qml_gradient_flow():
    model = HybridQuantumClassifier(n_qubits=2)
    dummy = torch.randn(1, 1, 224, 224)
    logits = model(dummy)
    loss = logits.sum()
    loss.backward()
    # All parameters should have non‑None grads
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"


def test_hybrid_qml_deterministic_output():
    torch.manual_seed(0)
    model = HybridQuantumClassifier(n_qubits=2)
    dummy = torch.randn(1, 1, 224, 224)
    out1 = model(dummy)
    out2 = model(dummy)
    assert torch.allclose(out1, out2), "Outputs differ on repeated forward passes"
