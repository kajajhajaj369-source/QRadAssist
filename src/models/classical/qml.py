import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

from preprocessing.labels import TARGET_LABELS


class SimpleQuantumLayer(nn.Module):
    """A toy "quantum" layer that mimics a parametrized unitary.

    The implementation uses basic trigonometric functions to simulate a rotation
    on a 2‑qubit state vector. It is **not** a true quantum simulation – it is a
    deterministic, fully‑classical computation that preserves the API shape of a
    quantum circuit for experimental purposes.
    """

    def __init__(self, n_qubits: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        # Number of parameters = n_qubits (one rotation angle per qubit)
        self.params = nn.Parameter(torch.randn(n_qubits))

    def forward(self, x: Tensor) -> Tensor:
        """Apply a simple rotation‑based transformation.

        Args:
            x: Input tensor of shape (batch, n_qubits). Values are interpreted as
               rotation angles (radians).
        Returns:
            Tensor of shape (batch, 2**n_qubits) representing a flattened state
            vector after applying the toy unitary.
        """
        batch = x.size(0)
        # Compute rotation matrices for each qubit
        rot_mats = []
        for i in range(self.n_qubits):
            theta = x[:, i] + self.params[i]
            c, s = torch.cos(theta), torch.sin(theta)
            # 2x2 rotation matrix for a single qubit
            rot = torch.stack([torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)], dim=-2)
            rot_mats.append(rot)
        # Initialise state as |0...0>
        state = torch.zeros(batch, 2 ** self.n_qubits, device=x.device)
        state[:, 0] = 1.0
        # Apply Kronecker products of rotation matrices sequentially
        for rot in rot_mats:
            # Expand state to (batch, 2, dim/2) for matmul
            dim = state.size(1)
            state = state.view(batch, 2, dim // 2)
            state = torch.einsum('bik,bkj->bij', rot, state)
            state = state.reshape(batch, dim)
        return state


class HybridQuantumClassifier(nn.Module):
    """Hybrid classifier: classical feature reduction + toy quantum layer.

    * Classical encoder reduces the high‑dimensional image to ``n_qubits`` real
      values (interpreted as rotation angles).
    * ``SimpleQuantumLayer`` produces a flattened state vector.
    * A final linear head maps the state vector to the target disease logits.

    The model respects the multi‑label setup used elsewhere (order given by
    ``TARGET_LABELS``) and returns raw logits compatible with the existing loss
    function.
    """

    def __init__(self, n_qubits: int = 2, pretrained: bool = False):
        super().__init__()
        self.n_qubits = n_qubits
        # Simple encoder: flatten + linear to n_qubits values
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1 * 224 * 224, n_qubits),  # assumes input (1,224,224)
        )
        self.quantum = SimpleQuantumLayer(n_qubits=n_qubits)
        self.classifier = nn.Linear(2 ** n_qubits, len(TARGET_LABELS))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass returning logits of shape (batch, num_classes)."""
        # ``x`` is expected to be (B, C, H, W). The classical encoder expects a
        # single‑channel image; if ``C`` == 3 we repeat to 1 channel as in the
        # DenseNet wrapper.
        if x.shape[1] == 3:
            x = x[:, 0:1, :, :]  # drop extra channels (placeholder handling)
        batch = x.size(0)
        angles = self.encoder(x)
        quantum_state = self.quantum(angles)  # (B, 2**n_qubits)
        logits = self.classifier(quantum_state)
        return logits
