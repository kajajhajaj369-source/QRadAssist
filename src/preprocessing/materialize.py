"""Materialize reproducible dataset splits and development cohort manifests (Phase 2B).

Adheres to Phase 1D/2A dataset protocols:
1. Operates on verified CheXpert frontal images from train_cheXbert.csv.
2. Partitions patients deterministically with seed=42 (90% train, 10% internal development/val).
3. Selects a patient-aware development cohort of ~15,000 frontal images.
4. Programmatically enforces zero patient overlap and zero lateral images.
5. Writes clean CSV manifests to data/splits/ without copying image files.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import PROJECT_ROOT, RAW_DATA_DIR, SPLITS_DIR

from preprocessing.manifest import (
    DEFAULT_IMAGE_ROOT,
    SampleManifest,
    filter_manifest_by_view,
    parse_row_to_sample,
    save_manifest_csv,
)
from preprocessing.splitting import (
    select_development_cohort,
    split_patients,
    verify_zero_patient_overlap,
)

DEFAULT_TRAIN_CSV = RAW_DATA_DIR / "train_cheXbert.csv"


def build_split_summary(samples: list[SampleManifest]) -> dict[str, Any]:
    """Compute summary statistics for a sample collection."""
    unique_patients = {s.patient_id for s in samples}
    unique_studies = {(s.patient_id, s.study_id) for s in samples}
    views = Counter(s.view for s in samples)
    projections = Counter(s.projection for s in samples)

    label_stats: dict[str, dict[str, int]] = {}
    for target in ("Pneumonia", "Atelectasis", "Pleural Effusion"):
        attr = target.lower().replace(" ", "_") + "_raw"
        counts = Counter(getattr(s, attr) for s in samples)
        label_stats[target] = dict(counts)

    return {
        "image_count": len(samples),
        "patient_count": len(unique_patients),
        "study_count": len(unique_studies),
        "views": dict(views),
        "projections": dict(projections),
        "labels": label_stats,
    }


def materialize_splits(
    raw_csv_path: Path = DEFAULT_TRAIN_CSV,
    splits_dir: Path = SPLITS_DIR,
    val_fraction: float = 0.10,
    dev_cohort_size: int = 15000,
    seed: int = 42,
    check_images_exist: bool = True,
) -> dict[str, Any]:
    """Generate and materialize patient-level splits and development cohort manifests.

    Args:
        raw_csv_path: Path to train_cheXbert.csv.
        splits_dir: Target directory for manifest CSV files (data/splits/).
        val_fraction: Fraction of unique patients for internal development partition (0.10).
        dev_cohort_size: Target image count for development cohort (~15,000).
        seed: Random seed for deterministic patient shuffling (42).
        check_images_exist: If True, asserts all referenced image files exist on disk.

    Returns:
        Dictionary with manifest paths and comprehensive partition metrics.
    """
    if not raw_csv_path.is_file():
        raise FileNotFoundError(f"Source raw CSV not found: {raw_csv_path}")

    splits_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse all raw CSV rows into SampleManifest records
    with raw_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_samples = [parse_row_to_sample(row, check_file_exists=False) for row in reader]

    # 2. Filter to Frontal views only (AP and PA)
    frontal_samples = filter_manifest_by_view(all_samples, allowed_views=("Frontal",))

    # Guardrail: Check that zero lateral images are present in frontal set
    lateral_in_frontal = [s for s in frontal_samples if s.view != "Frontal"]
    if lateral_in_frontal:
        raise ValueError(f"Found non-frontal images after filtering: {len(lateral_in_frontal)}")

    # 3. Patient-level train / internal-development split
    train_samples, val_samples = split_patients(
        frontal_samples,
        val_fraction=val_fraction,
        seed=seed,
    )

    # Guardrail: Enforce zero patient overlap between train and val
    verify_zero_patient_overlap(train_samples, val_samples)

    # 4. Select development cohort from the training partition
    dev_cohort_samples = select_development_cohort(
        train_samples,
        target_count=dev_cohort_size,
        seed=seed,
        allowed_views=("Frontal",),
    )

    # Guardrail: Enforce zero patient overlap between dev cohort and internal val partition
    verify_zero_patient_overlap(dev_cohort_samples, val_samples)

    # Guardrail: Check that all images in dev cohort are strictly frontal
    lateral_in_dev = [s for s in dev_cohort_samples if s.view != "Frontal"]
    if lateral_in_dev:
        raise ValueError(f"Found lateral images in development cohort: {len(lateral_in_dev)}")

    # Guardrail: Verify all referenced files exist if requested
    if check_images_exist:
        for sample in frontal_samples:
            full_path = PROJECT_ROOT / sample.image_path
            if not full_path.is_file():
                raise FileNotFoundError(f"Referenced image not found on disk: {full_path}")

    # 5. Materialize manifests to CSV
    full_split_path = splits_dir / "train_val_split.csv"
    train_path = splits_dir / "train_manifest.csv"
    val_path = splits_dir / "val_manifest.csv"
    dev_cohort_path = splits_dir / "dev_cohort_manifest.csv"

    # Combine train and val with split field populated
    full_frontal_split = train_samples + val_samples
    save_manifest_csv(full_frontal_split, full_split_path)
    save_manifest_csv(train_samples, train_path)
    save_manifest_csv(val_samples, val_path)
    save_manifest_csv(dev_cohort_samples, dev_cohort_path)

    # 6. Build and return summary report
    summary = {
        "raw_total_images": len(all_samples),
        "frontal_total_images": len(frontal_samples),
        "train": build_split_summary(train_samples),
        "val": build_split_summary(val_samples),
        "dev_cohort": build_split_summary(dev_cohort_samples),
        "manifest_paths": {
            "train_val_split": str(full_split_path),
            "train_manifest": str(train_path),
            "val_manifest": str(val_path),
            "dev_cohort_manifest": str(dev_cohort_path),
        },
    }
    return summary


def main() -> int:
    print("Materializing CheXpert patient-level splits and development cohort...")
    summary = materialize_splits()
    print("\n--- Materialization Complete ---")
    print(f"Total Frontal Images: {summary['frontal_total_images']:,}")
    print(f"Train Partition:      {summary['train']['image_count']:,} images across {summary['train']['patient_count']:,} patients")
    print(f"Internal Val Split:   {summary['val']['image_count']:,} images across {summary['val']['patient_count']:,} patients")
    print(f"Development Cohort:   {summary['dev_cohort']['image_count']:,} images across {summary['dev_cohort']['patient_count']:,} patients")
    print("\nManifests saved to:")
    for name, path in summary["manifest_paths"].items():
        print(f"  - {name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
