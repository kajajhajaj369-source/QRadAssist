"""Configurable image preprocessing and transformation components (Phase 2A).

Adheres to Phase 1D dataset protocol:
1. Loading JPEG images with format verification.
2. Deterministic resizing to configurable dimensions (default: 224x224).
3. Grayscale handling (single-channel chest X-ray representation).
4. Conversion to normalized float numpy array.
5. Strict separation between deterministic validation preprocessing and training augmentations.
6. Non-aggressive transformations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

PathLike = Union[str, Path]


@dataclass(frozen=True)
class PreprocessConfig:
    image_size: tuple[int, int] = (224, 224)  # (width, height)
    grayscale: bool = True
    normalize: bool = False  # False by default; requires explicit mean and std when enabled
    mean: float | None = None  # Must be explicitly calculated from training cohort
    std: float | None = None   # Must be explicitly calculated from training cohort
    channels_first: bool = True  # (C, H, W) vs (H, W, C)


class DeterministicPreprocessor:
    """Deterministic, reproducible preprocessor for validation and baseline inference."""

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()

    def load_image(self, path: PathLike) -> Image.Image:
        """Load an image from disk and verify it is a valid JPEG/raster image."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Image file not found: {p}")
        try:
            img = Image.open(p)
            img.verify()  # Check for corruption
            # verify() leaves file pointer in undefined state, reopen cleanly
            img = Image.open(p)
            img.load()
            return img
        except Exception as exc:
            raise ValueError(f"Failed to load valid image from {p}: {exc}") from exc

    def transform(self, image: Image.Image | PathLike) -> np.ndarray:
        """Apply deterministic preprocessing pipeline.

        Returns:
            np.ndarray of shape (1, H, W) or (H, W, 1) in float32.
        """
        if isinstance(image, (str, Path)):
            img = self.load_image(image)
        else:
            img = image

        # 1. Grayscale handling
        target_mode = "L" if self.config.grayscale else "RGB"
        if img.mode != target_mode:
            img = img.convert(target_mode)

        # 2. Deterministic resizing
        if img.size != self.config.image_size:
            img = img.resize(self.config.image_size, resample=Image.Resampling.BILINEAR)

        # 3. Convert to float array [0.0, 1.0]
        arr = np.array(img, dtype=np.float32) / 255.0

        # 4. Optional Standardization
        # Dataset normalization constants must be explicitly established from training data.
        if self.config.normalize:
            if self.config.mean is None or self.config.std is None:
                raise ValueError(
                    "Standardization (normalize=True) requires explicit 'mean' and 'std' statistics. "
                    "Do not assume or hard-code normalization statistics; calculate them in a controlled step."
                )
            arr = (arr - self.config.mean) / self.config.std

        # 5. Channel formatting
        if self.config.grayscale:
            if self.config.channels_first:
                arr = arr[np.newaxis, :, :]  # (1, H, W)
            else:
                arr = arr[:, :, np.newaxis]  # (H, W, 1)
        else:
            if self.config.channels_first and arr.ndim == 3:
                arr = np.transpose(arr, (2, 0, 1))  # (3, H, W)

        return arr

    def __call__(self, image: Image.Image | PathLike) -> np.ndarray:
        return self.transform(image)


@dataclass(frozen=True)
class AugmentationConfig:
    """Provisional training augmentation configuration.

    Kept strictly separate from deterministic validation preprocessing.
    """
    enable_horizontal_flip: bool = True
    horizontal_flip_prob: float = 0.5
    # Aggressive transforms (color jitter, elastic deformation, extreme rotation) are avoided.


class TrainingAugmentor:
    """Optional, provisional image augmentor for training experiments only."""

    def __init__(self, config: AugmentationConfig | None = None) -> None:
        self.config = config or AugmentationConfig()

    def augment(self, img: Image.Image, seed: int | None = None) -> Image.Image:
        """Apply mild spatial augmentations."""
        if not self.config.enable_horizontal_flip:
            return img

        rng = np.random.default_rng(seed)
        if rng.random() < self.config.horizontal_flip_prob:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        return img
