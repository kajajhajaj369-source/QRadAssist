"""Image validation, manifest construction, and preprocessing pipeline (Phase 2A).

This module defines the data preparation and deterministic preprocessing pipeline
adhering strictly to the Phase 1D dataset protocol (docs/research/dataset-protocol.md).
"""

from .labels import (
    SUPPORTED_POLICIES,
    TARGET_LABELS,
    classify_raw_label,
    encode_labels,
)
from .compute_stats import compute_training_statistics
from .manifest import (
    DEFAULT_IMAGE_ROOT,
    SampleManifest,
    filter_manifest_by_view,
    load_manifest_csv,
    parse_patient_id,
    parse_row_to_sample,
    parse_study_id,
    resolve_local_image_path,
    save_manifest_csv,
)
from .materialize import materialize_splits
from .splitting import (
    PatientLeakageError,
    select_development_cohort,
    split_patients,
    verify_zero_patient_overlap,
)
from .transforms import (
    AugmentationConfig,
    DeterministicPreprocessor,
    PreprocessConfig,
    TrainingAugmentor,
)

__all__ = [
    # Labels
    "TARGET_LABELS",
    "SUPPORTED_POLICIES",
    "classify_raw_label",
    "encode_labels",
    # Manifest
    "DEFAULT_IMAGE_ROOT",
    "SampleManifest",
    "resolve_local_image_path",
    "parse_patient_id",
    "parse_study_id",
    "parse_row_to_sample",
    "filter_manifest_by_view",
    "save_manifest_csv",
    "load_manifest_csv",
    # Splitting & Materialization
    "PatientLeakageError",
    "verify_zero_patient_overlap",
    "split_patients",
    "select_development_cohort",
    "materialize_splits",
    "compute_training_statistics",
    # Transforms
    "PreprocessConfig",
    "DeterministicPreprocessor",
    "AugmentationConfig",
    "TrainingAugmentor",
]
