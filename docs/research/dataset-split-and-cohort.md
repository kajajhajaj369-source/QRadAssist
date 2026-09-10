# Dataset Split and Development Cohort Specification (Phase 2B)

This document records the materialization of the patient-level dataset splits and initial development cohort generated in **Phase 2B** of the Q-RadAssist project.

All manifests are stored in machine-readable CSV format under `data/splits/`.
Raw image files under `data/raw/` remain strictly immutable and were not copied or duplicated.

---

## 1. Source Dataset and Frontal-Only Scope

- **Labels Source:** `data/raw/train_cheXbert.csv` (223,414 total raw training rows).
- **Images Source:** `data/raw/chexpert-small/train/`.
- **Frontal-Only Scope:** 
  - In accordance with the Phase 1D protocol, the initial baseline cohort includes **only Frontal images** (`AP` and `PA` projections).
  - Out of 223,414 raw entries, **191,027 frontal images** were retained.
  - All **32,387 lateral images** were excluded non-destructively; raw image files remain intact on disk for future multi-view experiments.
  - Zero lateral images are present in any generated manifest.

---

## 2. Patient-Level Splitting Methodology

1. **Patient-Level Grouping:** Splitting operates strictly on unique patient identifiers (`patient_id`). No individual image or study is split across partitions.
2. **Deterministic Reproducibility:** Unique patient IDs are sorted and shuffled using `seed=42`.
3. **Partition Proportions:**
   - **Training Partition (90% of patients):** 58,081 patients, 168,949 studies, **172,033 images**.
   - **Internal Development Partition (10% of patients):** 6,453 patients, 18,676 studies, **18,994 images**.
4. **Programmatic Leakage Verification:** Verified with `verify_zero_patient_overlap`:
   $$\text{len}(\text{Train Patients} \cap \text{Internal Val Patients}) = 0$$

---

## 3. Development Cohort Materialization

To support rapid pipeline verification, debugging, and baseline training before launching compute-heavy jobs on 191k images:
- **Target Size:** Approximately 15,000 frontal images.
- **Actual Size Selected:** **15,000 images** across **5,084 patients** and **14,732 studies**.
- **Subset Integrity:** The development cohort was selected strictly from the **training partition**.
  - Overlap with Training Partition: **100%** (5,084 / 5,084 patients).
  - Overlap with Internal Development/Val Partition: **0 (Zero)**.
- **Patient Grouping Integrity:** Whole patients were accumulated; all frontal studies and views of every selected patient are present in the cohort (no partial patient splitting).

---

## 4. Disease Prevalence in Development Cohort

The development cohort preserves the natural multi-label clinical prevalence of the full frontal dataset without artificial balancing, undersampling, or oversampling:

| Target Disease | Category | Development Cohort Count ($N=15,000$) | Dev % | Full Frontal Cohort % ($N=191,027$) |
| :--- | :--- | ---: | ---: | ---: |
| **Pneumonia** | Positive (`1.0`) | 272 | 1.81% | 1.96% |
| | Negative (`0.0`) | 185 | 1.23% | 1.12% |
| | Uncertain (`-1.0`) | 1,358 | 9.05% | 8.69% |
| | Blank / Unmentioned | 13,185 | 87.90% | 88.23% |
| **Atelectasis** | Positive (`1.0`) | 2,391 | 15.94% | 15.80% |
| | Negative (`0.0`) | 31 | 0.21% | 0.28% |
| | Uncertain (`-1.0`) | 2,471 | 16.47% | 15.94% |
| | Blank / Unmentioned | 10,107 | 67.38% | 67.98% |
| **Pleural Effusion** | Positive (`1.0`) | 6,460 | 43.07% | 41.73% |
| | Negative (`0.0`) | 1,938 | 12.92% | 13.50% |
| | Uncertain (`-1.0`) | 501 | 3.34% | 3.32% |
| | Blank / Unmentioned | 6,101 | 40.67% | 41.44% |

### Why No Class Balancing Was Applied
1. **Multi-label Co-occurrence:** Conditions frequently co-occur on the same radiograph. Discarding negatives for Pneumonia would discard legitimate positive findings for Pleural Effusion and Atelectasis.
2. **Patient Anatomical Diversity:** Oversampling Pneumonia would duplicate images of the same patients, inducing severe memorization rather than generalization.
3. **Clinical Realism:** The model must learn calibrated probabilities reflecting true clinical prevalence. Class imbalance is handled during training via loss weighting (e.g. positive class weighting in BCE loss) rather than dataset distortion.

---

## 5. Label and Uncertainty Handling

- **Raw Preservation:** All manifests preserve the exact raw strings (`1.0`, `0.0`, `-1.0`, `""`) in columns `pneumonia_raw`, `atelectasis_raw`, and `pleural_effusion_raw`.
- **Phase 1D Baseline Policy (U-Ignore):** In training and loss calculation, uncertain labels (`-1.0`) are masked out with loss weight 0.0, avoiding gradient corruption from guessed labels.

---

## 6. Official Validation Set Protection

- The official held-out CheXpert validation benchmark (`data/raw/chexpert-small/valid.csv`, 234 images across `patient64541`–`patient64740`) was **completely excluded** from all split and cohort generation.
- **Zero Patient Overlap:** Programmatically verified that zero patients or paths from `valid.csv` appear in `train_manifest.csv`, `val_manifest.csv`, `dev_cohort_manifest.csv`, or `train_val_split.csv`.
- The official validation set remains strictly reserved for final benchmark evaluation of trained models.

---

## 7. Materialized Manifest Files (`data/splits/`)

| File Name | Row Count | Unique Patients | Unique Studies | Description |
| :--- | ---: | ---: | ---: | :--- |
| `train_val_split.csv` | 191,027 | 64,534 | 187,625 | Full frontal cohort with `split` column (`train` / `val`) |
| `train_manifest.csv` | 172,033 | 58,081 | 168,949 | Training partition (90% of frontal patients) |
| `val_manifest.csv` | 18,994 | 6,453 | 18,676 | Internal development partition (10% of frontal patients) |
| `dev_cohort_manifest.csv` | 15,000 | 5,084 | 14,732 | Initial development cohort (subset of training partition) |

*Storage note:* No raw image files were copied or duplicated. Manifests reference local image paths relative to project root.

---

## 8. Reproducibility Procedure

The entire materialization can be reproduced deterministically by executing:
```bash
python src/preprocessing/materialize.py
```
Or programmatically:
```python
from preprocessing.materialize import materialize_splits
summary = materialize_splits(seed=42)
```
Automated integrity is verified via `tests/test_materialized_splits.py`.
