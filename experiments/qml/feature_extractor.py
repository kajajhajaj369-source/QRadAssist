"""Feature extraction and caching utility using frozen DenseNet-121.

Extracts 1024-dimensional feature vectors from trained checkpoint
experiments/full_gpu/best_full_model.pt and caches them to disk.
"""

import hashlib
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.models.classical.dataset import CheXpertDataset
from src.models.classical.model import DenseNet121MultiLabel
from src.preprocessing.transforms import DeterministicPreprocessor, PreprocessConfig


EXPECTED_FULL_MODEL_SHA256 = "4b74019200acc0d0e7a9cdc7d5e005a6fb31ad55a482e947759808fd9726315a"


def compute_sha256(filepath: Path) -> str:
    """Compute the SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_frozen_feature_extractor(
    checkpoint_path: Path,
    device: torch.device,
) -> DenseNet121MultiLabel:
    """Load and freeze DenseNet-121 feature extractor.

    Verifies checkpoint SHA-256 before loading. Sets eval() mode and disables
    gradient calculation.
    """
    actual_hash = compute_sha256(checkpoint_path)
    if actual_hash != EXPECTED_FULL_MODEL_SHA256:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch!\nExpected: {EXPECTED_FULL_MODEL_SHA256}\nActual:   {actual_hash}"
        )

    model = DenseNet121MultiLabel(pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    model.to(device)
    return model


def extract_features(
    model: DenseNet121MultiLabel,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Extract 1024-dimensional pooled features from frozen DenseNet-121 backbone.

    Args:
        model: Frozen DenseNet121MultiLabel model.
        dataloader: DataLoader yielding dict with 'image', 'target', 'mask'.
        device: CUDA or CPU device.

    Returns:
        Dict containing:
            'features': (N, 1024) tensor of pooled feature embeddings.
            'targets': (N, 3) tensor of multi-label targets.
            'masks': (N, 3) tensor of loss masks.
    """
    all_features = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = batch["target"]
            masks = batch["mask"]

            # Match model input channel handling (grayscale -> 3 channels)
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)

            # DenseNet feature extraction: features -> relu -> adaptive avg pool -> flatten
            feat = model.backbone.features(images)
            feat = F.relu(feat, inplace=False)
            feat = F.adaptive_avg_pool2d(feat, (1, 1))
            feat = torch.flatten(feat, 1)  # shape: (batch_size, 1024)

            all_features.append(feat.cpu())
            all_targets.append(targets)
            all_masks.append(masks)

    return {
        "features": torch.cat(all_features, dim=0),
        "targets": torch.cat(all_targets, dim=0),
        "masks": torch.cat(all_masks, dim=0),
    }


def get_or_extract_features(
    manifest_path: Path,
    checkpoint_path: Path,
    cache_path: Path,
    max_samples: Optional[int] = None,
    batch_size: int = 16,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Load cached features or extract and cache them if not present."""
    if cache_path.exists():
        print(f"Loading cached features from {cache_path}...")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    print(f"Extracting features from {manifest_path} (max_samples={max_samples})...")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_frozen_feature_extractor(checkpoint_path, device)
    preprocessor = DeterministicPreprocessor(PreprocessConfig())
    dataset = CheXpertDataset(manifest_path, preprocessor, policy="u_ignore")

    if max_samples is not None and max_samples < len(dataset):
        dataset = Subset(dataset, list(range(max_samples)))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    t0 = time.time()
    data = extract_features(model, loader, device)
    elapsed = time.time() - t0

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache_path)
    print(
        f"Extracted {len(data['features'])} samples in {elapsed:.2f}s. Saved to {cache_path}."
    )
    return data
