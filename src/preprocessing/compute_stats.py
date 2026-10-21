"""Compute exact dataset normalization statistics from the training split (Phase 2C).

Adheres strictly to the Phase 2C specification:
1. Source: ONLY data/splits/train_manifest.csv (172,033 training frontal images).
2. Validation data excluded: official validation and internal development partition are NEVER touched.
3. Preprocessing applied before statistic computation:
   - Grayscale conversion (mode 'L')
   - Deterministic resizing to (224, 224) via bilinear interpolation
   - Float32 conversion and scaling to [0.0, 1.0]
4. Streaming numerical accumulation (float64) to prevent high memory usage.
5. Emits exact mean, variance, and standard deviation over all pixels.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import PROJECT_ROOT, SPLITS_DIR

DEFAULT_TRAIN_MANIFEST = SPLITS_DIR / "train_manifest.csv"
DEFAULT_OUTPUT_STATS = SPLITS_DIR / "training_statistics.json"


def compute_training_statistics(
    manifest_path: Path = DEFAULT_TRAIN_MANIFEST,
    output_path: Path | None = DEFAULT_OUTPUT_STATS,
    image_size: tuple[int, int] = (224, 224),
    max_images: int | None = None,
    log_interval: int = 20000,
) -> dict[str, Any]:
    if "val" in manifest_path.stem.lower() or "valid" in manifest_path.stem.lower():
        raise ValueError(
            f"Attempted to calculate training statistics on validation manifest: {manifest_path}. "
            "Normalization statistics must be derived strictly from the training partition."
        )

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Training manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    if max_images is not None and max_images > 0:
        rows = reader[:max_images]
    else:
        rows = reader

    num_images = len(rows)
    print(f"Calculating normalization statistics over {num_images:,} images from {manifest_path}...")
    print(f"Pipeline: Grayscale ('L') -> Resize {image_size} Bilinear -> Scale [0.0, 1.0] -> float64 accumulator")

    total_pixels = 0
    sum_x = np.float64(0.0)
    sum_x2 = np.float64(0.0)

    t_start = time.time()
    t_last = t_start

    for idx, row in enumerate(rows, start=1):
        rel_path = row["image_path"]
        img_path = PROJECT_ROOT / rel_path

        # Verify image format and view
        if row.get("view") != "Frontal":
            raise ValueError(f"Encountered non-frontal image in training statistics: {rel_path}")

        # Strictly enforce exclusion of any validation images or validation split samples
        if row.get("split") == "val" or "/valid/" in rel_path.replace("\\", "/"):
            raise ValueError(
                f"Encountered validation image/sample in training statistics calculation: {rel_path}. "
                "Normalization statistics must be derived strictly from training data only."
            )

        img = Image.open(img_path).convert("L")
        if img.size != image_size:
            img = img.resize(image_size, resample=Image.Resampling.BILINEAR)

        arr = np.array(img, dtype=np.float32) / 255.0

        sum_x += np.sum(arr, dtype=np.float64)
        sum_x2 += np.sum(arr ** 2, dtype=np.float64)
        total_pixels += arr.size

        if idx % log_interval == 0 or idx == num_images:
            elapsed = time.time() - t_start
            speed = idx / elapsed
            print(f"Processed {idx:,}/{num_images:,} images ({idx/num_images*100:.1f}%) in {elapsed:.1f}s ({speed:.1f} img/s)...")

    mean = float(sum_x / total_pixels)
    var = float((sum_x2 / total_pixels) - (mean ** 2))
    std = float(np.sqrt(max(var, 0.0)))

    results = {
        "dataset": "CheXpert-v1.0-small",
        "source_manifest": str(manifest_path),
        "total_images_processed": num_images,
        "is_full_training_partition": (num_images == len(reader)),
        "total_pixels_processed": int(total_pixels),
        "image_size": list(image_size),
        "interpolation": "bilinear",
        "color_mode": "grayscale_L",
        "pixel_range": [0.0, 1.0],
        "mean": mean,
        "variance": var,
        "std": std,
        "elapsed_seconds": round(time.time() - t_start, 2),
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved statistics to {output_path}")

    return results


def main() -> int:
    results = compute_training_statistics()
    print("\n--- Exact Training Statistics ---")
    print(f"Total Images: {results['total_images_processed']:,}")
    print(f"Total Pixels: {results['total_pixels_processed']:,}")
    print(f"Mean:         {results['mean']:.6f}")
    print(f"Std:          {results['std']:.6f}")
    print(f"Elapsed:      {results['elapsed_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
