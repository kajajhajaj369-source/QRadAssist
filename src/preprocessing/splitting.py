"""Patient-level dataset splitting and development cohort selection (Phase 2A).

Adheres to Phase 1D dataset protocol:
1. Patient-level grouping: operates strictly on patient IDs (no image-level random splits).
2. Zero data leakage: programmatically asserts train/val patient sets are disjoint.
3. Reproducibility: deterministic shuffling with configurable random seed.
4. Development cohort selection: patient-aware selection of approximately 10k-20k images.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Sequence

from preprocessing.manifest import SampleManifest


class PatientLeakageError(Exception):
    """Raised when patient IDs overlap across dataset splits."""


def verify_zero_patient_overlap(
    train_samples: Sequence[SampleManifest],
    val_samples: Sequence[SampleManifest],
) -> None:
    """Assert that train and validation sample sets share zero patient IDs."""
    train_patients = {s.patient_id for s in train_samples}
    val_patients = {s.patient_id for s in val_samples}
    overlap = train_patients & val_patients
    if overlap:
        raise PatientLeakageError(
            f"Patient leakage detected! {len(overlap)} patient(s) found in both train and val: {sorted(list(overlap))[:10]}"
        )


def split_patients(
    samples: Sequence[SampleManifest],
    val_fraction: float = 0.10,
    seed: int = 42,
) -> tuple[list[SampleManifest], list[SampleManifest]]:
    """Partition samples into train and internal-validation sets by patient ID.

    Args:
        samples: List of SampleManifest records.
        val_fraction: Fraction of unique patients to assign to validation (default: 0.10).
        seed: Random seed for deterministic shuffling.

    Returns:
        (train_samples, val_samples) with sample.split populated.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be strictly between 0 and 1, got {val_fraction}")

    patient_to_samples: dict[str, list[SampleManifest]] = defaultdict(list)
    for sample in samples:
        patient_to_samples[sample.patient_id].append(sample)

    unique_patients = sorted(patient_to_samples.keys())
    if not unique_patients:
        return [], []

    rng = random.Random(seed)
    shuffled_patients = list(unique_patients)
    rng.shuffle(shuffled_patients)

    num_val_patients = max(1, int(round(len(shuffled_patients) * val_fraction)))
    num_train_patients = len(shuffled_patients) - num_val_patients
    train_patient_set = set(shuffled_patients[:num_train_patients])
    val_patient_set = set(shuffled_patients[num_train_patients:])

    train_samples: list[SampleManifest] = []
    val_samples: list[SampleManifest] = []

    for sample in samples:
        # Create shallow copy or update split field
        s_copy = SampleManifest(
            sample_id=sample.sample_id,
            image_path=sample.image_path,
            raw_csv_path=sample.raw_csv_path,
            patient_id=sample.patient_id,
            study_id=sample.study_id,
            view=sample.view,
            projection=sample.projection,
            pneumonia_raw=sample.pneumonia_raw,
            atelectasis_raw=sample.atelectasis_raw,
            pleural_effusion_raw=sample.pleural_effusion_raw,
            split="val" if sample.patient_id in val_patient_set else "train",
        )
        if sample.patient_id in val_patient_set:
            val_samples.append(s_copy)
        else:
            train_samples.append(s_copy)

    verify_zero_patient_overlap(train_samples, val_samples)
    return train_samples, val_samples


def select_development_cohort(
    samples: Sequence[SampleManifest],
    target_count: int = 15000,
    seed: int = 42,
    allowed_views: Sequence[str] | None = ("Frontal",),
) -> list[SampleManifest]:
    """Select a patient-aware development cohort of approximately target_count images.

    Operates strictly on frontal images for the initial baseline (or specified allowed_views).
    It will never reintroduce lateral images even if an unfiltered manifest is passed.
    Patient-level grouping remains intact (whole patients within the view constraint are selected).
    Does NOT modify or duplicate raw image files.

    Args:
        samples: Sequence of SampleManifest records.
        target_count: Approximate number of images desired (e.g. 10,000 - 20,000).
        seed: Random seed for deterministic patient shuffling.
        allowed_views: Tuple of allowed views (default: ('Frontal',)). Excludes lateral views.

    Returns:
        List of SampleManifest records comprising the development cohort.
    """
    if target_count <= 0:
        raise ValueError(f"target_count must be positive, got {target_count}")

    # Enforce view filtering: only retain allowed views (Frontal for initial baseline)
    if allowed_views is not None:
        view_set = set(allowed_views)
        filtered_samples = [s for s in samples if s.view in view_set]
    else:
        filtered_samples = list(samples)

    if not filtered_samples:
        return []

    patient_to_samples: dict[str, list[SampleManifest]] = defaultdict(list)
    for sample in filtered_samples:
        patient_to_samples[sample.patient_id].append(sample)

    unique_patients = sorted(patient_to_samples.keys())
    rng = random.Random(seed)
    shuffled_patients = list(unique_patients)
    rng.shuffle(shuffled_patients)

    selected_samples: list[SampleManifest] = []
    for pid in shuffled_patients:
        selected_samples.extend(patient_to_samples[pid])
        if len(selected_samples) >= target_count:
            break

    return selected_samples

