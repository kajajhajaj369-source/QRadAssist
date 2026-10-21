# Preprocessing Pipeline Specification (Phase 2A)

This document specifies the technical design, configuration, and operational behavior of the reproducible preprocessing and data-preparation pipeline implemented in **Phase 2A** of the Q-RadAssist project under `src/preprocessing/`.

All components adhere strictly to the authoritative experimental protocol established in [dataset-protocol.md](file:///e:/College/QRadAssist/docs/research/dataset-protocol.md).

---

## 1. Overview & Phase 2A Boundaries

The goal of Phase 2A is the **implementation and unit testing** of the modular preprocessing components without executing them over the full dataset:
- **No full dataset processing:** The 191,027 frontal images are not processed or resized in this phase.
- **No development cohort generation:** The 10k–20k development cohort selection logic is implemented and verified, but not saved or materialized to disk.
- **No raw data modification:** All files under `data/raw/` remain strictly immutable and read-only.
- **No model training:** No neural network training or inference is performed.

---

## 2. Pipeline Stages & Architectural Flow

The pipeline is organized into modular, independently testable stages:

```
[Raw CSV Row] 
      │
      ├──> 1. Path Resolution (Maps 'CheXpert-v1.0/' -> 'data/raw/chexpert-small/')
      │
      ├──> 2. Patient / Study ID Parsing (Regex extraction of patient and study numbers)
      │
      ├──> 3. View Filtering (Retains Frontal AP/PA; excludes Lateral non-destructively)
      │
      ├──> 4. Raw Label Preservation (Captures exact raw strings: '1.0', '0.0', '-1.0', '')
      │
      ▼
[SampleManifest Record]
      │
      ├──> 5. Patient-Level Splitting (Grouped by patient_id; asserts zero leakage)
      │
      ├──> 6. Development Cohort Selection (Patient-aware ~10k-20k selection)
      │
      ├──> 7. Label Policy Encoding (U-Ignore baseline: loss mask = 0 for uncertain)
      │
      ▼
[Deterministic Image Preprocessor]
      │
      ├──> JPEG Loading & Corruption Verification
      ├──> Grayscale ('L' Mode) Conversion
      ├──> Deterministic Bilinear Resizing (Configurable, default 224x224)
      ├──> Float32 Conversion & Empirical Normalization
      └──> Channel Formatting (1, H, W)
```

---

## 3. Path Resolution & View Filtering

### 3.1 Path Resolution (`resolve_local_image_path`)
The pipeline maps Stanford-style path strings to local filesystem paths:
- `CheXpert-v1.0/` &rarr; `data/raw/chexpert-small/`
- `CheXpert-v1.0-small/` &rarr; `data/raw/chexpert-small/`

Backslashes are normalized to forward slashes. Paths that do not match the expected prefixes or have empty relative targets fail loudly with explicit `ValueError` exceptions.

### 3.2 View Filtering (`filter_manifest_by_view`)
- **Initial Baseline Scope:** Frontal projections only (`AP` and `PA`).
- **Lateral Exclusions:** Radiographs where `Frontal/Lateral == "Lateral"` (14.50% of the raw training images) are filtered out of the manifest.
- **Non-destructive:** Excluded lateral images are never deleted or moved; they remain intact on disk for future multi-view fusion architectures.

---

## 4. Patient and Study Identification

Patient and study boundaries are parsed directly from path values:
- `patient_id`: extracted via regex `patient(\d+)` (e.g., `"00042"`).
- `study_id`: extracted via regex `study(\d+)` (e.g., `"2"`).
- `sample_id`: a deterministic, unique identifier combining patient, study, and filename stem:
  $$\text{sample\_id} = \text{p}\{\text{patient\_id}\}\_\text{s}\{\text{study\_id}\}\_\{\text{stem}\}$$
  *(e.g., `p00042_s2_view1_frontal`)*.

---

## 5. Label Semantics & Uncertainty Handling

### 5.1 Raw Value Preservation
To prevent silent data corruption or irrevocable label conversions, manifests retain the exact raw strings:
- `"1.0"`: Positive
- `"0.0"`: Negative
- `"-1.0"`: Uncertain
- `""` (empty string): Unmentioned / Blank

### 5.2 Phase 1D Baseline Policy: U-Ignore (`encode_labels`)
During training loss calculation, labels are converted into numerical target tensors and loss masks:

| Raw Cell Value | Interpretation | Baseline Target ($y$) | Baseline Loss Mask ($m$) | Loss Formulation |
| :--- | :--- | :---: | :---: | :--- |
| `"1.0"` | Positive finding | `1.0` | `1.0` | Active in loss calculation |
| `"0.0"` | Negative finding | `0.0` | `1.0` | Active in loss calculation |
| `""` (blank) | Unmentioned | `0.0` | `1.0` | Active in loss calculation (Negative reporting bias) |
| `"-1.0"` | Diagnostic uncertainty | `0.0` | **`0.0`** | **Masked out ($w=0$) from loss calculation** |

### 5.3 Alternative Policy Support
The pipeline also provides explicit support for alternative uncertainty policies to enable planned ablation experiments:
- **`u_zeros`:** Maps `"-1.0"` &rarr; target `0.0`, mask `1.0`.
- **`u_ones`:** Maps `"-1.0"` &rarr; target `1.0`, mask `1.0`.

---

## 6. Patient-Level Splitting & Leakage Prevention

### 6.1 Patient Grouping (`split_patients`)
To guarantee that images from the same patient never appear in both training and validation splits:
1. All unique `patient_id` values are extracted and sorted for deterministic order.
2. Patient IDs are shuffled using a configured random seed (`seed=42`).
3. Patients are partitioned into train and validation sets (default validation fraction: 10%).
4. Each sample inherits the split assignment (`sample.split = "train"` or `"val"`) based on its `patient_id`.

### 6.2 Programmatic Leakage Guardrail (`verify_zero_patient_overlap`)
Every split execution enforces a hard constraint:
$$\text{len}(\text{set}(\text{Train Patients}) \cap \text{set}(\text{Val Patients})) == 0$$
If any patient overlap is detected, the function raises `PatientLeakageError` immediately.

---

## 7. Development Cohort Support

### 7.1 Methodological Design (`select_development_cohort`)
- **Target Size:** Configurable, targeting approximately 10,000 to 20,000 frontal images.
- **Frontal-Only Enforcement:** Operates strictly on the frontal-image manifest (default: `allowed_views=("Frontal",)`). Lateral images are never reintroduced into the development cohort even if an unfiltered manifest is passed.
- **Patient-Aware Accumulation:** The algorithm shuffles patients deterministically and accumulates whole patients (including all studies and frontal views belonging to each patient). It terminates as soon as the accumulated sample count reaches or exceeds `target_count`.
- **Integrity Guarantee:** Patients are never partially included; every frontal study/view of an included patient is retained (patient-level grouping intact).
- **Status in Phase 2A:** The selection algorithm is fully implemented and unit-tested, but has **not** been executed to generate or save files.

---

## 8. Deterministic Image Preprocessing

### 8.1 Preprocessing Pipeline (`DeterministicPreprocessor`)
Designed for reproducible baseline evaluation and validation:

| Operation | Configuration / Method | Rationale |
| :--- | :--- | :--- |
| **Image Loading** | `PIL.Image.open` + `Image.verify` | Verifies raster integrity; rejects truncated files |
| **Color Mode** | Convert to `"L"` (Grayscale) | Single-channel chest X-ray representation |
| **Resizing** | `Image.Resampling.BILINEAR` | Deterministic, anti-aliased spatial reduction |
| **Target Dimensions** | Default: `(224, 224)` | Standard CNN input size; configurable for high-res |
| **Array Conversion** | NumPy `float32`, scaled to `[0.0, 1.0]` | Standard numerical range for neural networks |
| **Normalization / Standardization** | Configurable (`normalize=False` default; requires explicit `mean` & `std`) | **No hard-coded/assumed statistics.** Pixel intensities are scaled to `[0.0, 1.0]`. Standardization statistics were explicitly established in Phase 2C from the training partition (mean: 0.506422, std: 0.286777; see [preprocessing-statistics.md](file:///e:/College/QRadAssist/docs/research/preprocessing-statistics.md)). |
| **Channel Layout** | `(1, H, W)` (`channels_first=True`) | Standard PyTorch tensor convention |

### 8.2 Separation of Training Augmentation (`TrainingAugmentor`)
- Preprocessing for validation and inference is **strictly deterministic**.
- Training augmentations (e.g., mild horizontal flipping) are implemented in an independent class, disabled by default, and kept completely separate from validation preprocessing.
- Aggressive spatial transformations (heavy warping, severe rotations, color jitter) are avoided to preserve subtle pathological opacity cues.

---

## 9. Manifest Specification & Schema

Generated manifests are written to CSV without duplicating raw image files.

### 9.1 Column Schema
1. `sample_id`: Unique identifier (`p00001_s1_view1_frontal`).
2. `image_path`: Path to image relative to project root (`data/raw/chexpert-small/train/...`).
3. `raw_csv_path`: Original path string from CheXpert CSV.
4. `patient_id`: Extracted patient identifier (`00001`).
5. `study_id`: Extracted study identifier (`1`).
6. `view`: View orientation (`Frontal`, `Lateral`).
7. `projection`: View projection (`AP`, `PA`, or empty).
8. `pneumonia_raw`: Original raw string (`1.0`, `0.0`, `-1.0`, or empty).
9. `atelectasis_raw`: Original raw string (`1.0`, `0.0`, `-1.0`, or empty).
10. `pleural_effusion_raw`: Original raw string (`1.0`, `0.0`, `-1.0`, or empty).
11. `split`: Assigned partition (`train`, `val`, or empty).

---

## 10. Raw Data Protection & File Organization

- **Immutability of `data/raw/`:** No raw CSVs or raw image directories are edited, moved, or deleted.
- **Output Locations:**
  - Manifests and split definitions: `data/splits/`
  - Cached preprocessed tensors or metadata: `data/processed/`

---

## 11. Provisional Decisions & Next Steps

The following decisions remain provisional and will be evaluated empirically during Phase 2B (baseline model development):
1. **Input Resolution:** Baseline uses 224&times;224; higher resolutions (320&times;320, 512&times;512) will be compared for fine interstitial pneumonia opacities.
2. **Augmentation Policy:** Horizontal flip is provisionally available; its impact on unilateral pathologies (e.g. unilateral pleural effusion) will be validated.
3. **Uncertainty Ablation:** U-Ignore remains the default; empirical comparisons with condition-specific U-Ones (Atelectasis) and U-Zeros (Pleural Effusion) will be tested on the internal validation split.

---

## 12. Verification & Test Coverage

All modules in `src/preprocessing/` are tested in [test_preprocessing.py](file:///e:/College/QRadAssist/tests/test_preprocessing.py):
- Path resolution and error handling (valid/invalid prefixes)
- Patient and study ID parsing
- Frontal view filtering
- Raw label preservation and uncertainty policy encoding (`u_ignore`, `u_zeros`, `u_ones`)
- Patient-level splitting with programmatic assertion of zero patient leakage
- Deterministic reproducibility across identical seeds
- Patient-aware development cohort selection
- Manifest serialization and round-trip CSV deserialization
- Deterministic image transformations on synthetic image fixtures
- Total test suite: **34/34 tests passing cleanly** across the repository.
