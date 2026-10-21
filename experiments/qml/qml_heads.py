"""Track B Classification Heads: Classical vs. Simulated Quantum Head.

Both heads consume frozen 1024-dimensional DenseNet-121 feature vectors
and output raw logits of shape (batch, 3) for multi-label prediction.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from src.models.classical.qml import SimpleQuantumLayer
from src.preprocessing.labels import TARGET_LABELS


class ClassicalLinearHead(nn.Module):
    """Standard classical linear multi-label classification head."""

    def __init__(self, in_features: int = 1024, num_classes: int = len(TARGET_LABELS)):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, in_features).
        Returns:
            Logits of shape (batch, num_classes).
        """
        return self.classifier(x)


class SimulatedQuantumHead(nn.Module):
    """Classically simulated quantum-inspired classification head.

    Uses a linear angle encoder to map 1024 features down to n_qubits rotation
    angles, applies the existing SimpleQuantumLayer parameterized rotation unitary,
    and maps the resulting 2^n_qubits statevector to target logits.
    """

    def __init__(
        self,
        in_features: int = 1024,
        n_qubits: int = 2,
        num_classes: int = len(TARGET_LABELS),
    ):
        super().__init__()
        self.in_features = in_features
        self.n_qubits = n_qubits
        self.num_classes = num_classes

        # Map high-dimensional feature embedding down to rotation angles
        self.angle_encoder = nn.Linear(in_features, n_qubits)
        # Existing toy quantum layer simulating single-qubit rotations
        self.quantum = SimpleQuantumLayer(n_qubits=n_qubits)
        # Final linear classifier from statevector dimension (2^n_qubits) to logits
        self.classifier = nn.Linear(2 ** n_qubits, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, in_features).
        Returns:
            Logits of shape (batch, num_classes).
        """
        angles = self.angle_encoder(x)
        quantum_state = self.quantum(angles)  # shape: (batch, 2**n_qubits)
        logits = self.classifier(quantum_state)
        return logits
