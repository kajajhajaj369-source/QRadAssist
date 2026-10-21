# Q-RadAssist — Final Model Evaluation Report

**Evaluation Timestamp (UTC):** 2026-10-21T14:47:55.605608+00:00  
**Model Architecture:** DenseNet-121 Multi-Label (`DenseNet121MultiLabel`)  
**Evaluated Checkpoint:** `experiments/full_gpu/best_full_model.pt`  
**Checkpoint SHA-256:** `4b74019200acc0d0e7a9cdc7d5e005a6fb31ad55a482e947759808fd9726315a`  
**Evaluation Device:** `cuda`  

---

## 1. Test Split Identity, Provenance & Leakage Verification

### 1.1 Dataset Identification
- **Designated Held-Out Benchmark:** `data/raw/chexpert-small/valid.csv` (Official Stanford CheXpert radiologist-consensus validation benchmark).
- **Non-existence of `data/splits/test_manifest.csv`:** Verified that `data/splits/test_manifest.csv` does not exist on disk and was never part of the dataset splits. It was only referenced as a temporary fixture path in unit test suites (`tests/test_phase2_modules.py`).
- **Total Benchmark Samples:** 234 radiographs across 200 distinct patients (`patient64541` through `patient64740`).
- **Primary Frontal Evaluation Cohort (Eligible):** 202 frontal images (169 AP, 33 PA) across 200 patients.
- **Excluded Lateral Cohort:** 32 lateral images excluded under the project's frontal-only specification (`config/settings.yaml` and `docs/research/dataset-protocol.md`).

### 1.2 Patient-Level Separation Verification
Patient identifiers were extracted from raw paths and manifests and numerically verified:
- **Training partition (`data/splits/train_manifest.csv`):** 58,081 patients (IDs 1 through 64,540, 172,033 images).
- **Internal development partition (`data/splits/val_manifest.csv`):** 6,453 patients (IDs 27 through 64,529, 18,994 images).
- **Held-out test benchmark (`data/raw/chexpert-small/valid.csv`):** 200 patients (IDs 64,541 through 64,740, 234 images).
- **Patient Overlap Verification Result:**
  - $\text{len}(\text{Train Patients} \cap \text{Test Patients}) = 0$
  - $\text{len}(\text{Val Patients} \cap \text{Test Patients}) = 0$
  - $\text{len}(\text{Train Patients} \cap \text{Val Patients}) = 0$
  - **Status:** **PASS — Strict Zero Patient Leakage Confirmed**.

---

## 2. Primary Evaluation: Eligible Frontal Test Cohort (N = 202)

In strict accordance with the project protocol (`docs/research/dataset-protocol.md` Sections 7.2 & 8.A), the primary evaluation cohort comprises all 202 frontal radiographs. All 202 images loaded and executed inference without failure (0 inference failures).

### 2.1 Threshold-Independent Performance (AUROC & Average Precision)

| Target Pathology | Total Samples | Positive (Prev %) | Negative | AUROC | Average Precision (AP) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pneumonia** | 202 | 8 (3.96%) | 194 | **0.8183** | **0.1588** |
| **Atelectasis** | 202 | 75 (37.13%) | 127 | **0.8076** | **0.6960** |
| **Pleural Effusion** | 202 | 64 (31.68%) | 138 | **0.9358** | **0.8754** |
| **MACRO AVERAGE** | **202** | — | — | **0.8539** | **0.5767** |

> **Note on Metric Validity:** All target classes contain at least one positive and one negative sample; zero AUROC metrics are undefined. Undefined AUROCs would be reported as `UNDEFINED` (never 1.0).

---

### 2.2 Exploratory Threshold Analysis (Threshold = 0.5)

> [!WARNING]
> Decision threshold 0.5 is reported **strictly as an exploratory diagnostic reference**. In adherence to project guidelines, decision thresholds were **never tuned on this held-out test split**.

| Pathology | TP | FP | TN | FN | Sensitivity (Recall) | Specificity | PPV (Precision) | NPV | Accuracy | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pneumonia** | 0 | 0 | 194 | 8 | 0.0000 | 1.0000 | 0.0000 | 0.9604 | 0.9604 | 0.0000 |
| **Atelectasis** | 9 | 1 | 126 | 66 | 0.1200 | 0.9921 | 0.9000 | 0.6562 | 0.6683 | 0.2118 |
| **Pleural Effusion** | 37 | 6 | 132 | 27 | 0.5781 | 0.9565 | 0.8605 | 0.8302 | 0.8366 | 0.6916 |

---

## 3. Secondary Reference: All Views Cohort (N = 234)

For complete transparency, all 234 images in `valid.csv` (including 32 lateral views) were also evaluated:

| Target Pathology | Total Samples | Positive (Prev %) | AUROC | Average Precision (AP) |
| :--- | :---: | :---: | :---: | :---: |
| **Pneumonia** | 234 | 8 (3.42%) | **0.7721** | **0.1303** |
| **Atelectasis** | 234 | 80 (34.19%) | **0.7888** | **0.6635** |
| **Pleural Effusion** | 234 | 67 (28.63%) | **0.9039** | **0.8224** |
| **MACRO AVERAGE** | **234** | — | **0.8216** | **0.5388** |

---

## 4. Comparison: Internal Validation vs. Held-Out Test Set

| Metric / Pathology | Internal Validation (Epoch 3, N=18,994) | Held-Out Test Frontal (N=202) | Difference (Test - Val) |
| :--- | :---: | :---: | :---: |
| **Pneumonia AUROC** | 0.7869 | 0.8183 | +0.0314 |
| **Atelectasis AUROC** | 0.7136 | 0.8076 | +0.0940 |
| **Pleural Effusion AUROC** | 0.8898 | 0.9358 | +0.0460 |
| **Mean AUROC** | **0.7968** | **0.8539** | **+0.0571** |

---

## 5. Methodological Limitations & Integrity Assurance

1. **Pneumonia Sample Size:** The held-out benchmark contains only 8 positive pneumonia cases. Each single positive misclassification moves sensitivity by 12.5%. While the AUROC demonstrates strong discriminative ranking (0.8183), the confidence interval is broad.
2. **Consensus vs. Weak Labels:** Internal validation labels originate from CheXbert NLP extraction over clinical reports, whereas the held-out test split ground truth was established by consensus across 3 board-certified radiologists with zero ambiguous/uncertain labels.
3. **No Test-Set Tuning:** The held-out test split was never exposed during training, loss calculation, checkpoint selection, learning rate decay, or threshold calibration.
4. **Production Model Isolation:** The production checkpoint `src/models/classical/best_model.pt` was not modified or overwritten.
