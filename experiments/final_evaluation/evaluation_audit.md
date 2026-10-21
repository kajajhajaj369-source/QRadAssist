# Q-RadAssist — Phase 2 Evaluation Audit Report

**Audit Timestamp (UTC):** 2026-10-21T14:58:00Z  
**Target Directory:** `experiments/final_evaluation/`  
**Audited Checkpoint:** `experiments/full_gpu/best_full_model.pt`  
**Production Checkpoint:** `src/models/classical/best_model.pt`  

---

## 1. Checkpoint Identity & Integrity Verification

1. **Evaluated Checkpoint (`experiments/full_gpu/best_full_model.pt`):**
   - **File Size:** 28,446,149 bytes (28.45 MB)
   - **SHA-256 Checksum:** `4b74019200acc0d0e7a9cdc7d5e005a6fb31ad55a482e947759808fd9726315a`
   - **Weight Verification:** Exactly 727 tensor parameters; all values are finite (zero NaNs, zero Infs).
   - **Confirmation:** Matches the expected hash exactly.

2. **Production Checkpoint (`src/models/classical/best_model.pt`):**
   - **File Size:** 28,446,149 bytes (28.45 MB)
   - **SHA-256 Checksum:** `392b65c9130717bfb19053f5e638ef31bb9fcddc40e169399dbc89cb15b9d25a`
   - **Confirmation:** Confirmed 100% identical to baseline; **completely untouched and unpromoted**.

---

## 2. Recomputation, Agreement & Artifact Structure Audit

### 2.1 Artifact Structure Observation
* In `experiments/final_evaluation/test_metrics.json`, the deliverable contains comprehensive summary metadata, per-class metrics, confusion matrices, and macro metrics, but **does not embed the 202 individual per-sample prediction dictionaries** (which were retained in the memory namespace of the evaluation run).
* To audit all reported numbers without assuming validity or modifying files, an independent model forward pass was executed using `best_full_model.pt` on all 202 frontal test images.

### 2.2 Independent Recomputation Findings

All metrics were recomputed from raw logits and compared against `test_metrics.json` and `test_metrics.csv`:

| Pathology / Metric | `test_metrics.json` | `test_metrics.csv` | Independent Recomputation | Agreement Status |
| :--- | :---: | :---: | :---: | :---: |
| **Pneumonia AUROC** | `0.8182989690721649` | `0.8183` | `0.8182989690721649` | **EXACT MATCH** |
| **Pneumonia AP** | `0.15883664326216607` | `0.1588` | `0.15883664326216607` | **EXACT MATCH** |
| **Pneumonia TP/FP/TN/FN (0.5)** | `0 / 0 / 194 / 8` | `0 / 0 / 194 / 8` | `0 / 0 / 194 / 8` | **EXACT MATCH** |
| **Atelectasis AUROC** | `0.8075590551181103` | `0.8076` | `0.8075590551181103` | **EXACT MATCH** |
| **Atelectasis AP** | `0.6959610748018188` | `0.6960` | `0.6959610748018188` | **EXACT MATCH** |
| **Atelectasis TP/FP/TN/FN (0.5)** | `9 / 1 / 126 / 66` | `9 / 1 / 126 / 66` | `9 / 1 / 126 / 66` | **EXACT MATCH** |
| **Pleural Effusion AUROC** | `0.9358016304347827` | `0.9358` | `0.9358016304347827` | **EXACT MATCH** |
| **Pleural Effusion AP** | `0.8754394442599822` | `0.8754` | `0.8754394442599822` | **EXACT MATCH** |
| **Pleural Effusion TP/FP/TN/FN (0.5)** | `37 / 6 / 132 / 27` | `37 / 6 / 132 / 27` | `37 / 6 / 132 / 27` | **EXACT MATCH** |
| **Macro Average AUROC** | `0.8538865515416859` | `0.8539` | `0.8538865515416859` | **EXACT MATCH** |
| **Macro Average AP** | `0.5767457207746557` | `0.5767` | `0.5767457207746557` | **EXACT MATCH** |

**Conclusion:** All numerical outputs in `test_metrics.json` and `test_metrics.csv` match independent evaluation down to machine precision.

---

## 3. Patient Count, Study Structure & Multi-Image Verification

### 3.1 Patient vs. Image Count Discrepancy (202 Images from 200 Patients)
Inspection of all 202 frontal records in `data/raw/chexpert-small/valid.csv` confirms:
* **Unique Patients:** Exactly 200 distinct patient IDs (`patient64541` through `patient64740`).
* **Patients with 1 Frontal Image:** 198 patients.
* **Patients with 2 Frontal Images:** Exactly 2 patients:
  1. `patient64547`: `study1/view1_frontal.jpg` (PA) and `study1/view2_frontal.jpg` (PA)
  2. `patient64581`: `study1/view1_frontal.jpg` (PA) and `study1/view2_frontal.jpg` (PA)
* **Clinical Explanation:** Both patients had repeat PA exposures acquired during the same study encounter (`study1`). Under CheXpert's report-level labeling, both views in each patient inherit identical labels (all negative for the 3 target conditions).

### 3.2 Exact Patient-Level Label Counts (Denominator = 200 Patients)

Because the two multi-image patients are negative for all target conditions, the patient-level positive counts are identical to the image-level counts, while negative counts decrease by 2:

| Pathology | Image-Level Positives (N=202) | Patient-Level Positives (N=200) | Patient-Level Negatives (N=200) | Patient-Level Prevalence |
| :--- | :---: | :---: | :---: | :---: |
| **Pneumonia** | 8 | **8** | **192** | 4.00% |
| **Atelectasis** | 75 | **75** | **125** | 37.50% |
| **Pleural Effusion** | 64 | **64** | **136** | 32.00% |

---

## 4. Comprehensive Patient-Level Separation Audit

To guarantee zero leakage without relying on boundary heuristics (min/max IDs), an exhaustive element-by-element set intersection was computed across all 249,110 image records:

### 4.1 Extraction Methods
1. **Method A (Manifest Column):** Read integer values from `patient_id` column in `data/splits/train_manifest.csv` and `data/splits/val_manifest.csv`.
2. **Method B (Filesystem Path Regex):** Applied regex `patient(\d+)` to `image_path` in manifests and `Path` in `valid.csv`.

### 4.2 Set Intersection Results

| Partition Pair | Comparison Method | Set 1 Size | Set 2 Size | Overlap Count | Leakage Detected |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Train vs. Test** | `patient_id` column vs. Test Path | 58,081 | 200 | **0** | **NO** |
| **Train vs. Test** | Path regex vs. Test Path | 58,081 | 200 | **0** | **NO** |
| **Val vs. Test** | `patient_id` column vs. Test Path | 6,453 | 200 | **0** | **NO** |
| **Val vs. Test** | Path regex vs. Test Path | 6,453 | 200 | **0** | **NO** |
| **Train vs. Val** | `patient_id` column | 58,081 | 6,453 | **0** | **NO** |

**Conclusion:** Zero patient overlap confirmed by exhaustive set intersection across all materialized manifests and the raw test split.

---

## 5. Statistical Uncertainty: Patient-Clustered Bootstrap Confidence Intervals

Because 202 images originate from 200 patients (clustered observations) and Pneumonia has only 8 positive cases, standard asymptotic normal approximations (Wald intervals) are invalid.

We implemented a **Patient-Clustered Non-Parametric Bootstrap** ($B = 2,000$ resamples with replacement at the patient level, seed=42):
* In each iteration, 200 patient IDs are sampled with replacement.
* All constituent frontal images for sampled patients are gathered.
* Percentile 95% Confidence Intervals are derived from the empirical 2.5th and 97.5th percentiles of successful replications.

### 5.1 Confidence Interval Results

| Pathology | Metric | Point Estimate | 95% Clustered Bootstrap CI | Valid Replications | Method Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Pneumonia** | AUROC | **0.8183** | **[0.7109, 0.9188]** | 1,999 / 2,000 | 1 sample had zero positives; broad CI reflects small sample ($N^+=8$) |
| | Average Precision | **0.1588** | **[0.0501, 0.4452]** | 1,999 / 2,000 | Precision is constrained by low prevalence (4%) |
| **Atelectasis** | AUROC | **0.8076** | **[0.7433, 0.8662]** | 2,000 / 2,000 | Tightly bounded discriminative performance |
| | Average Precision | **0.6960** | **[0.5828, 0.8067]** | 2,000 / 2,000 | Strong precision-recall balance |
| **Pleural Effusion** | AUROC | **0.9358** | **[0.9020, 0.9646]** | 2,000 / 2,000 | Excellent discrimination ($>0.90$ across lower bound) |
| | Average Precision | **0.8754** | **[0.8009, 0.9334]** | 2,000 / 2,000 | Robust performance across all resamples |

---

## 6. Review of Clinical Validity & Independence Claims in `evaluation_report.md`

An audit of the claims in [evaluation_report.md](file:///E:/College/QRadAssist/experiments/final_evaluation/evaluation_report.md) was performed:

1. **Dataset Independence:**
   - *Verification:* The report correctly clarifies that `data/raw/chexpert-small/valid.csv` is an **internal held-out benchmark** from the Stanford CheXpert distribution (patients 64,541–64,740), **not an external multi-center clinical validation cohort** from an outside healthcare system.
   - *Assessment:* Accurately represented; no overclaiming of generalizability to external health systems.

2. **Threshold 0.5 Status & Pneumonia Sensitivity Critical Finding:**
   - *Exploratory Status:* Section 2.2 explicitly flags threshold 0.5 with a warning banner indicating it is an untuned, exploratory baseline.
   - *Critical Clinical Warning:* At threshold 0.5, **Pneumonia Sensitivity is 0.0000 (0 / 8 detected, 8 False Negatives)**.
   - *Root Cause:* Because Pneumonia training prevalence is low (~2.17%) and U-Ignore masking was applied, the model outputs raw uncalibrated probabilities below 0.5 for all positive pneumonia cases in this test benchmark.
   - *Clinical Takeaway:* The model possesses strong ranking ability (AUROC 0.8183, 95% CI [0.7109, 0.9188]), but **cannot be used clinically with an arbitrary 0.5 decision threshold**. Operating thresholds must be calibrated on validation data (e.g. at 80% or 90% target sensitivity) prior to clinical deployment.

---

## 7. Regression Test Suite Execution

The existing regression test suites were executed to verify system-wide stability:

```bash
.venv_cuda\Scripts\python.exe -m pytest tests/test_training_recovery.py tests/test_phase2_modules.py tests/test_api.py -v --tb=short
```

* **Outcome:** **27 passed, 1 skipped, 0 failed**
* **Exit Code:** `0`

```bash
.venv_cuda\Scripts\python.exe -m pytest tests/test_materialized_splits.py -v --tb=short
```

* **Outcome:** **10 passed, 0 failed**
* **Exit Code:** `0`
