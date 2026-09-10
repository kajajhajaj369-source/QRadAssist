# Dataset Integrity Verification (Phase 1C)

This document records the verification results for the local CheXpert dataset and metadata alignment conducted in **Phase 1C** of the Q-RadAssist project.

All counts and assertions in this document were computed using the verification script:
`src/evaluation/dataset_integrity.py`

This check examined:
1. `data/raw/train_cheXbert.csv` (training metadata with CheXbert-derived labels)
2. `data/raw/chexpert-small/` (local directory tree of extracted images)
3. `data/raw/chexpert-small/valid.csv` (official validation split metadata)

No CSV files were modified, no images were preprocessed or transformed, no subsets were written to disk, and no models were trained.

---

## 1. Path Resolution Mapping

The Stanford-provided `train_cheXbert.csv` file references images with a root prefix that differs from the local directory layout:

| Source | Example Path Format |
| --- | --- |
| `train_cheXbert.csv` | `CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg` |
| Local filesystem | `data/raw/chexpert-small/train/patient00001/study1/view1_frontal.jpg` |
| `valid.csv` | `CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg` |
| Local filesystem | `data/raw/chexpert-small/valid/patient64541/study1/view1_frontal.jpg` |

### Prefix Mapping Rules
- `CheXpert-v1.0/` &rarr; `data/raw/chexpert-small/`
- `CheXpert-v1.0-small/` &rarr; `data/raw/chexpert-small/`

Both forward slashes (`/`) and Windows backslashes (`\`) are normalized to forward slashes before prefix matching.

---

## 2. Training Split Verification (`train_cheXbert.csv`)

### 2.1 File and Path Integrity

| Metric | Result | Note |
| --- | ---: | --- |
| Total CSV rows | **223,414** | Exact match with Phase 1B inspection |
| Unique `Path` values | **223,414** | One row per image path |
| Duplicate `Path` keys | **0** | No duplicated rows or paths |
| Unresolvable paths | **0** | Every row matches a valid prefix |
| **Referenced images found** | **223,414** | **100.0%** of referenced image files exist locally |
| **Referenced images missing** | **0** | **0 missing files** |

### 2.2 Patient and Study Distribution

Patient IDs and Study IDs were extracted by parsing directory paths (`patient(\d+)` and `study(\d+)`):

| Identifier Level | Unique Count | Ratio / Notes |
| --- | ---: | --- |
| Unique Patients | **64,540** | Range: `patient00001` through `patient64540` (~3.46 images/patient) |
| Unique Patient–Study Pairs | **187,641** | ~1.19 images/study (~2.91 studies/patient) |
| Total Images | **223,414** | Multiple views per study present |

---

## 3. Training Label Distributions (Target Conditions)

Observation values in `train_cheXbert.csv` for the three target labels:
- `1.0`: Positive
- `0.0`: Negative
- `-1.0`: Uncertain
- Blank (empty string): Unmentioned / Blank

### 3.1 Breakdown by Condition

#### Pneumonia
| Classification | Count | Percentage |
| --- | ---: | ---: |
| Positive (`1.0`) | 4,856 | 2.17% |
| Negative (`0.0`) | 3,314 | 1.48% |
| Uncertain (`-1.0`) | 19,386 | 8.68% |
| Blank / Unmentioned | 195,858 | 87.67% |
| **Total** | **223,414** | **100.00%** |

#### Atelectasis
| Classification | Count | Percentage |
| --- | ---: | ---: |
| Positive (`1.0`) | 33,862 | 15.16% |
| Negative (`0.0`) | 730 | 0.33% |
| Uncertain (`-1.0`) | 34,416 | 15.40% |
| Blank / Unmentioned | 154,406 | 69.11% |
| **Total** | **223,414** | **100.00%** |

#### Pleural Effusion
| Classification | Count | Percentage |
| --- | ---: | ---: |
| Positive (`1.0`) | 89,314 | 39.98% |
| Negative (`0.0`) | 36,278 | 16.24% |
| Uncertain (`-1.0`) | 7,634 | 3.42% |
| Blank / Unmentioned | 90,188 | 40.37% |
| **Total** | **223,414** | **100.00%** |

---

## 4. Positive-Label Co-occurrence (Training Split, `1.0` Only)

When restricting solely to explicit positive indications (`1.0`), the co-occurrence matrix for the three target labels is:

| Positive Combination | Row Count | % of All Rows |
| --- | ---: | ---: |
| *(None positive)* | 114,225 | 51.13% |
| Pleural Effusion only | 70,839 | 31.71% |
| Atelectasis + Pleural Effusion | 17,015 | 7.62% |
| Atelectasis only | 16,479 | 7.38% |
| Pneumonia only | 3,201 | 1.43% |
| Pleural Effusion + Pneumonia | 1,287 | 0.58% |
| Atelectasis + Pneumonia | 195 | 0.09% |
| Atelectasis + Pleural Effusion + Pneumonia | 173 | 0.08% |
| **Total** | **223,414** | **100.00%** |

### Key Multi-Label Insights
1. **Multi-Label Overlap:** **18,670 rows** (8.36%) exhibit 2 or 3 concurrent positive labels.
2. **Triple Positives:** **173 rows** are simultaneously positive for Atelectasis, Pleural Effusion, and Pneumonia.
3. **Clinical Feasibility:** The labels are not mutually exclusive and confirm the multi-label nature of the task.

---

## 5. Validation Split Verification (`valid.csv`)

The official validation split (`data/raw/chexpert-small/valid.csv`) was evaluated against local image files:

| Metric | Result | Note |
| --- | ---: | --- |
| Total CSV rows | **234** | Official CheXpert validation set size |
| Unique `Path` values | **234** | One row per image |
| Duplicate `Path` keys | **0** | No duplicates |
| Unresolvable paths | **0** | Prefix `CheXpert-v1.0-small/` resolved correctly |
| **Referenced images found** | **234** | **100.0%** of validation images present locally |
| **Referenced images missing** | **0** | **0 missing files** |
| Unique Patients | **200** | Range: `patient64541` through `patient64740` |
| Unique Patient–Study Pairs | **200** | 1 study per patient in validation set |

---

## 6. Train / Validation Patient Overlap and Leakage Assessment

A critical risk in chest X-ray modeling is patient-level data leakage (having images of the same patient in both training and validation sets).

- **Training Patient ID set:** 64,540 patients (`patient00001` &ndash; `patient64540`)
- **Validation Patient ID set:** 200 patients (`patient64541` &ndash; `patient64740`)
- **Intersection (Overlapping Patient IDs):** **0 (Zero)**

There is zero patient overlap between the training labels CSV and the validation set.

---

## 7. Failure Modes & Loud Failure Enforcement

The verification script `src/evaluation/dataset_integrity.py` enforces loud failures:
1. If any referenced training or validation image file does not exist on disk, the script exits with status code `1` and lists missing file paths on `sys.stderr`.
2. If any patient ID is present in both training and validation sets, the script exits with status code `1` and reports patient leakage.
3. If any CSV `Path` contains an unresolvable prefix or missing relative path, the script raises an `IntegrityError` and exits with status code `1`.
4. If required target label columns are missing, the script records an error and exits with status code `1`.

In the current environment, the verification script executed cleanly with:
```
RESULT: PASS
All referenced training and validation images were found.
No train/validation patient ID overlap was detected.
Exit code: 0
```

---

## 8. Summary and Phase 1C Completion

All Phase 1C verification objectives have been satisfied:
- All 223,414 training images referenced in `train_cheXbert.csv` exist locally in `data/raw/chexpert-small/train/`.
- All 234 validation images referenced in `valid.csv` exist locally in `data/raw/chexpert-small/valid/`.
- Path prefixes are cleanly mapped and validated.
- Zero patient leakage exists between train and validation splits.
- Multi-label distributions and co-occurrences are documented.
- Automated unit tests (`tests/test_dataset_integrity.py`) pass alongside the existing test suite (18/18 tests passing).

**Phase 1C boundary:** No image preprocessing, no training subset creation, and no model training have been initiated.
