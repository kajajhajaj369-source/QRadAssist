import torch
from torch import nn
from torchvision import models
from preprocessing.labels import TARGET_LABELS

class DenseNet121MultiLabel(nn.Module):
    """DenseNet-121 backbone with a linear head for multi-label disease prediction.

    Returns raw logits (no sigmoid) suitable for ``BCEWithLogitsLoss``.
    """

    def __init__(self, pretrained: bool = True, num_classes: int = len(TARGET_LABELS)) -> None:
        super().__init__()
        # Load DenseNet-121; use default weights if pretrained=True
        self.backbone = models.densenet121(weights="DEFAULT" if pretrained else None)
        # Replace classifier head
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept single-channel (grayscale) images by replicating to 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)
