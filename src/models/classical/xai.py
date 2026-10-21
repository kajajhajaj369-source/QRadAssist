import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Any

from .model import DenseNet121MultiLabel


def _register_hooks(
    model: DenseNet121MultiLabel,
) -> Tuple[list[Tensor], list[Tensor], torch.utils.hooks.RemovableHandle, Any]:
    """Register forward and backward hooks on the last convolutional layer of the model.

    Captures the activation feature map and gradient of the target layer. Uses a tensor
    backward hook on the layer output to avoid in-place ReLU view conflicts in DenseNet-121
    and eliminate deprecated module backward-hook usage.
    """
    features: list[Tensor] = []
    grads: list[Tensor] = []
    tensor_hook_handles: list[Any] = []

    def forward_hook(module, input, output):
        features.append(output)
        h = output.register_hook(lambda grad: grads.append(grad))
        tensor_hook_handles.append(h)

    # In torchvision's DenseNet, the final conv feature map is produced by the last
    # layer (norm5) in model.backbone.features.
    layer = model.backbone.features[-1]
    handle_fwd = layer.register_forward_hook(forward_hook)

    class _CombinedBwdHandle:
        def remove(self):
            for th in tensor_hook_handles:
                try:
                    th.remove()
                except Exception:
                    pass

    return features, grads, handle_fwd, _CombinedBwdHandle()


def gradcam(
    model: DenseNet121MultiLabel,
    input_tensor: Tensor,
    target_class: int,
    device: str | torch.device | None = None,
) -> Tensor:
    """Compute a Grad‑CAM heatmap for a single target class.

    Parameters
    ----------
    model: DenseNet121MultiLabel
        The trained model (will be moved to the appropriate device internally).
    input_tensor: torch.Tensor
        Input image tensor of shape (1, C, H, W). Batch size must be 1.
    target_class: int
        Index of the disease class (as ordered in ``preprocessing.labels.TARGET_LABELS``).
    device: optional str or torch.device
        Device on which to run the computation. If ``None`` auto‑selects CUDA if available.

    Returns
    -------
    torch.Tensor
        2‑D heatmap of shape (H, W) with values in ``[0, 1]``.
    """
    if input_tensor.ndim != 4 or input_tensor.size(0) != 1:
        raise ValueError("input_tensor must be shape (1, C, H, W)")

    # Determine device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    num_classes = model.backbone.classifier.out_features
    if not (0 <= target_class < num_classes):
        raise ValueError(f"target_class {target_class} out of range (0‑{num_classes-1})")

    model.eval()
    # Register hooks to capture activations and gradients
    features, grads, h_fwd, h_bwd = _register_hooks(model)

    # Forward pass
    logits = model(input_tensor)  # shape (1, C)
    score = logits[0, target_class]

    # Backward pass – gradient of the selected class score
    model.zero_grad()
    score.backward()

    # Retrieve stored tensors
    activations = features[0].detach()  # (1, K, H', W')
    gradients = grads[0].detach()      # (1, K, H', W')

    # Remove hooks
    h_fwd.remove()
    h_bwd.remove()

    # Global average pooling of gradients
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, K, 1, 1)
    cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (1, 1, H', W')
    cam = F.relu(cam)

    # Upsample to the original image size
    cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
    cam = cam.squeeze()  # (H, W)

    # Normalize to [0, 1]
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = torch.zeros_like(cam)
    return cam.cpu()
