# Dataset Protocol Design (Phase 1D)

This document establishes the experimental dataset protocol for subsequent phases of the Q-RadAssist project. It synthesizes the empirical evidence collected across Phases 1B, 1C, and 1D from:

- `data/raw/train_cheXbert.csv` (CheXbert-derived multi-label annotations)
- `data/raw/chexpert-small/train.csv` (CheXpert rule-based annotations)
- `data/raw/chexpert-small/train/` (223,414 extracted local images)
- `data/raw/chexpert-small/valid.csv` (Official validation set metadata)
- `data/raw/chexpert-small/valid/` (234 extracted local validation images)

All counts and figures were computed using [dataset_protocol_analysis.py](file:///e:/College/QRadAssist/src/evaluation/dataset_protocol_analysis.py) and [dataset_integrity.py](file:///e:/College/QRadAssist/src/evaluation/dataset_integrity.py).

---

## 1. Observed Dataset Structure

### 1.1 Path and Metadata Alignment
Comparing the normalized `Path` columns of `data/raw/train_cheXbert.csv` and `data/raw/chexpert-small/train.csv`:
- **Total Rows in `train_cheXbert.csv`:** 223,414
- **Total Rows in `chexpert-small/train.csv`:** 223,414
- **Path Equivalence:** After stripping root prefixes (`CheXpert-v1.0/` and `CheXpert-v1.0-small/`), both files reference the **exact same 223,414 images** in the **identical sequence** (0 path discrepancies, 100% set equality).

### 1.2 Target Disease Scope
The project boundary remains strictly focused on three core multi-label pathologies:
1. **Pneumonia**
2. **Atelectasis**
3. **Pleural Effusion**

COVID-19 remains completely excluded from the CheXpert training task as CheXpert predates the SARS-CoV-2 pandemic.

---

## 2. View Analysis

Radiographic projection strongly influences anatomy visualization, organ magnification, and pathological opacity presentation.

### 2.1 Training Set View Distribution (223,414 images)

| Dimension | Category | Count | Percentage |
| :--- | :--- | ---: | ---: |
| **View Orientation** | Frontal | 191,027 | 85.50% |
| | Lateral | 32,387 | 14.50% |
| **Projection (Frontal)** | AP (Anteroposterior) | 161,590 | 72.33% |
| | PA (Posteroanterior) | 29,420 | 13.17% |
| **Projection (Lateral/Other)** | Unspecified / Blank (all Lateral) | 32,387 | 14.50% |
| | LL (Latero-lateral) | 16 | 0.01% |
| | RL | 1 | <0.01% |

### 2.2 Study-Level View Combinations (187,641 studies)
Chest radiograph encounters (studies) frequently contain one or more views:

| Study View Composition | Study Count | % of All Studies | Clinical & Modeling Context |
| :--- | ---: | ---: | :--- |
| **Frontal views only** | 156,212 | 83.25% | Standard bedside or routine screening |
| **Both Frontal and Lateral views** | 31,413 | 16.74% | Full routine diagnostic examinations |
| **Lateral views only** | 16 | 0.01% | Isolated lateral exposures (negligible artifact) |
| **Multiple Frontal views** | 3,361 | 1.79% | Multiple exposures/angles in single encounter |

*Key Takeaway:* Over 83% of all clinical studies lack lateral views entirely. A model requiring lateral views would be unusable on >83% of encounters, whereas a frontal-view model can evaluate 99.99% of studies.

---

## 3. Patient / Study / Image Hierarchy

A single patient can present multiple clinical studies across time, and a study can contain multiple radiographic exposures.

### 3.1 Distribution Metrics

| Metric Level | Entities | Min | Max | Mean | Median | Dominant Modes |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **Studies per Patient** | 64,540 patients | 1 | 91 | 2.91 | 1 | 1 study: 33,564 pts (52.0%)<br>2 studies: 11,172 pts (17.3%)<br>3 studies: 5,921 pts (9.2%) |
| **Images per Patient** | 64,540 patients | 1 | 92 | 3.46 | 2 | 1 image: 22,581 pts (35.0%)<br>2 images: 17,547 pts (27.2%) |
| **Images per Study** | 187,641 studies | 1 | 3 | 1.19 | 1 | 1 image: 153,647 studies (81.9%)<br>2 images: 32,215 studies (17.2%)<br>3 images: 1,779 studies (0.9%) |

### 3.2 View Diversity Within Studies
- The same study **can and does** contain multiple views: 31,413 studies contain both Frontal and Lateral views; 3,361 studies contain multiple Frontal views (e.g., AP supine followed by erect AP, or expiratory views).
- In every case, multiple images in a study depict the same clinical acute or chronic episode.

---

## 4. Target-Label Analysis and Intra-Study Consistency

### 4.1 Label Consistency Within Studies
A critical architectural and data-engineering question is whether labels differ between images of the same study.

**Empirical Finding:**
- Studies with varying labels across constituent images for Pneumonia: **0**
- Studies with varying labels across constituent images for Atelectasis: **0**
- Studies with varying labels across constituent images for Pleural Effusion: **0**

**Root Cause:**
In CheXpert and CheXbert, annotations were extracted by NLP from free-text clinical radiology reports. Radiology reports are written at the **study (encounter) level**, summarizing all radiographic views acquired during that session. Consequently, every image associated with a study inherits identical study-level labels.

### 4.2 Label Extraction Source Comparison: CheXbert vs. Original CheXpert Rule-Based Labeler
Comparing label distributions between `train_cheXbert.csv` (BERT-based labeler) and `train.csv` (CheXpert rule-based labeler):

| Target | Category | `train_cheXbert.csv` | `chexpert-small/train.csv` | Difference / Shift |
| :--- | :--- | ---: | ---: | :--- |
| **Pneumonia** | Positive (`1.0`) | 4,856 (2.17%) | 6,039 (2.70%) | -1,183 (more conservative) |
| | Negative (`0.0`) | 3,314 (1.48%) | 2,799 (1.25%) | +515 |
| | Uncertain (`-1.0`) | 19,386 (8.68%) | 18,770 (8.40%) | +616 |
| | Blank / Unmentioned | 195,858 (87.67%) | 195,806 (87.64%) | +52 |
| **Atelectasis** | Positive (`1.0`) | 33,862 (15.16%) | 33,376 (14.94%) | +486 |
| | Negative (`0.0`) | 730 (0.33%) | 1,328 (0.59%) | -598 |
| | Uncertain (`-1.0`) | 34,416 (15.40%) | 33,739 (15.10%) | +677 |
| | Blank / Unmentioned | 154,406 (69.11%) | 154,971 (69.36%) | -565 |
| **Pleural Effusion** | Positive (`1.0`) | 89,314 (39.98%) | 86,187 (38.58%) | +3,127 |
| | Negative (`0.0`) | 36,278 (16.24%) | 35,396 (15.84%) | +882 |
| | Uncertain (`-1.0`) | 7,634 (3.42%) | 11,628 (5.20%) | -3,994 (CheXbert resolves ambiguity) |
| | Blank / Unmentioned | 90,188 (40.37%) | 90,203 (40.37%) | -15 |

*Conclusion:* Stanford's CheXbert model resolves significant ambiguity in Pleural Effusion (reducing uncertainty from 11,628 to 7,634) while maintaining strict precision on Pneumonia.

---

## 5. Uncertainty-Label Considerations (`-1.0`) and Blank Handling

### 5.1 Nature of Uncertain (`-1.0`) and Blank Labels
- **Uncertain (`-1.0`):** Indicates that the radiologist's report explicitly expressed diagnostic doubt (e.g., *"cannot exclude aspiration pneumonia"*, *"infiltrate vs. atelectasis"*, *"possible minimal blunting"*).
- **Blank / Unmentioned:** Indicates that the pathology was not referenced in the report. In routine radiology reporting, radiologists typically omit mentioning an abnormality if the lungs are clear of it (implicit negative), but an omission is not an explicit radiologist declaration of absence.

### 5.2 Comparative Analysis of Uncertainty Handling Strategies

| Strategy | Definition | Advantages | Disadvantages / Scientific Risks |
| :--- | :--- | :--- | :--- |
| **U-Ignore (Masking)** | Exclude uncertain rows from binary cross-entropy loss calculation for that specific task (loss weight = 0). | - Does not inject synthetic label noise.<br>- Models genuine diagnostic ambiguity.<br>- Mathematically principled under partial labels. | - Discards substantial supervision (e.g., 15.4% of Atelectasis rows).<br>- Potential selection bias if uncertainty correlates with disease severity. |
| **U-Zeros (Map to 0.0)** | Treat all `-1.0` as negative (`0.0`). | - Simple binary formulation.<br>- Preserves 100% of sample size.<br>- High specificity. | - Increases false negatives.<br>- For Pneumonia, converts 19,386 ambiguous cases (many clinical infections) into negative ground truth. |
| **U-Ones (Map to 1.0)** | Treat all `-1.0` as positive (`1.0`). | - High sensitivity.<br>- Established by Irvin et al. (2019) to yield high Atelectasis AUROC. | - Severe precision penalty for Pneumonia (inflates positive count 5x from 4,856 to 24,242).<br>- Biases model toward over-calling abnormalities. |
| **U-SelfTrained / Soft Labels** | Map `-1.0` to continuous target $p \in (0, 1)$ (e.g., $p=0.5$). | - Reflects epistemic probability.<br>- Avoids hard misclassification penalties. | - Non-standard cross-entropy formulation.<br>- Can pull model calibration towards the decision boundary. |
| **Multi-Class (3-way)** | Train separate output logits for Negative, Uncertain, and Positive. | - Preserves full granular output of report extraction. | - Severe evaluation mismatch: official validation set has **zero** uncertain labels (strictly binary 0/1). |

### 5.3 Handling of Blank / Unmentioned Cells
In CheXpert benchmark conventions (Irvin et al., 2019; Rajpurkar et al., 2020), blank entries are conventionally treated as **negative (`0.0`)** for the primary abnormalities under the clinical convention that chest radiograph reporting operates under negative reporting bias (abnormalities are commented on; normal structures are omitted). 
*Protocol Requirement:* In our implementation, this mapping must be clearly codified and logged as an assumption rather than concealed as raw data truth.

---

## 6. Class Imbalance and Multi-Label Structure

### 6.1 Multi-Label Prevalence and Imbalance Ratios (Denominator: 223,414)
- **Pneumonia Positives:** 4,856 (**2.17%**) &rarr; Severe Imbalance (~1 : 45 positive-to-negative ratio)
- **Atelectasis Positives:** 33,862 (**15.16%**) &rarr; Moderate Imbalance (~1 : 5.6 ratio)
- **Pleural Effusion Positives:** 89,314 (**39.98%**) &rarr; Balanced / High Prevalence (~1 : 1.5 ratio)

### 6.2 Co-occurrence Structure
- **114,225 rows (51.13%)** are negative/blank for all three conditions.
- **18,670 rows (8.36%)** are positive for **two or three** conditions simultaneously:
  - Atelectasis + Pleural Effusion: 17,015
  - Pleural Effusion + Pneumonia: 1,287
  - Atelectasis + Pneumonia: 195
  - Atelectasis + Pleural Effusion + Pneumonia: 173

### 6.3 Methodological Constraints on Imbalance
1. **No Random Undersampling:** Discarding majority negatives to balance Pneumonia (2.17%) would eliminate >90% of data and artificially destroy the clinically valid prevalence of Pleural Effusion and Atelectasis.
2. **No Random Oversampling:** Duplicating Pneumonia images would over-replicate individual patient anatomies, leading directly to patient-level memorization and overfitting.
3. **Multi-Label Preservation:** Because conditions co-occur on the same physical image, modifying one class boundary alters the distribution of the other classes.

---

## 7. Official Validation Set Analysis

### 7.1 Key Characteristics (`data/raw/chexpert-small/valid.csv`)
- **Total Images:** 234
- **Unique Patients:** 200 (`patient64541` through `patient64740`)
- **Unique Studies:** 200 (strictly 1 study per patient)
- **View Distribution:** Frontal = 202 (86.32%), Lateral = 32 (13.68%)
- **Target Ground Truth Distributions:**
  - **Pneumonia:** `0.0`: 226 (96.58%), `1.0`: 8 (3.42%)
  - **Atelectasis:** `0.0`: 154 (65.81%), `1.0`: 80 (34.19%)
  - **Pleural Effusion:** `0.0`: 167 (71.37%), `1.0`: 67 (28.63%)
- **Zero Ambiguity:** There are **0 uncertain (`-1.0`)** and **0 blank** entries in the official validation set. It was manually annotated by a panel of board-certified radiologists who reached consensus binary labels.

### 7.2 Benchmark Role and Governance
- **Role within CheXpert:** The official validation set is an **official held-out CheXpert validation benchmark** within the CheXpert dataset. It is not an external benchmark from a different hospital or dataset, but rather the curated, radiologist-consensus evaluation cohort for CheXpert.
- **Frontal-Only Evaluation Cohort (202 images):** Because the initial model protocol is strictly **frontal-only**, the primary official validation evaluation cohort is explicitly defined as the **202 frontal validation images** (169 AP, 33 PA). The **32 lateral validation images** must be excluded from the frontal-only baseline evaluation and reserved for future multi-view/lateral experiments.
- **Strict Prohibition on Development Use:** The 234-image validation set (including its 202 frontal subset) is **not used for training, early stopping, hyperparameter tuning, or threshold tuning**.
- **Development vs. Final Evaluation Separation:** The **internal patient-level validation split** is used for all development decisions (early stopping, model checkpoint selection, hyperparameter sweeps, learning rate decay, and decision threshold calibration). The official held-out CheXpert validation benchmark is used **only for final/locked evaluation**.
- **Pneumonia Small-Sample Vulnerability:** The validation set contains only **8 positive pneumonia cases** (all 8 occur in frontal views). Tuning hyperparameters, checkpoints, or decision thresholds on this set would lead to extreme variance and severe overfitting.

---

## 8. Proposed Initial Experimental Protocol

Based on empirical evidence and clinical modeling constraints, the initial experimental protocol is defined across dimensions A through H:

### A. Image Views for First Classification Experiment
- **Protocol:** Restrict initial experiments exclusively to **Frontal views** (`Frontal/Lateral == "Frontal"`), encompassing both AP and PA projections.
- **Justification:** Frontal views constitute 85.50% of images and are present in 99.99% of all clinical encounters. Standard single-view 2D architectures (DenseNet-121, ResNet-50) cannot accept both frontal and lateral views simultaneously without introducing major unconditioned view artifacts.
- **Primary Official Validation Cohort (202 images):** In direct alignment with the frontal-only protocol, the primary official validation evaluation cohort is explicitly defined as the **202 frontal validation images**. The 32 lateral validation images are excluded from the frontal-only baseline evaluation and reserved for future multi-view/lateral experiments.
- **Lateral Views Disposition:** Set aside lateral views for future multi-view fusion architectures. Do not discard them from raw storage.

### B. Label Assignment Level
- **Protocol:** Labels are inherited at the **image level** from the corresponding study during training. During evaluation, image-level predictions are reported directly, and study-level predictions are obtained via maximum-probability pooling across all frontal views within the study.
- **Justification:** Deep networks ingest individual images. Because all views within any study share identical ground truth (0 inconsistencies), image-level supervision directly mirrors the clinical report while study-level pooling aligns with clinical evaluation standards.

### C. Uncertainty Label Handling
- **Protocol:** Adopt **U-Ignore (loss masking)** as the primary baseline default. In parallel, implement condition-specific mappings established in benchmark literature:
  - *Atelectasis:* U-Ones (`-1.0` &rarr; `1.0`) vs. U-Ignore
  - *Pleural Effusion:* U-Zeros (`-1.0` &rarr; `0.0`) vs. U-Ignore
  - *Pneumonia:* U-Ignore (never U-Ones, to protect the model from a 5x false-positive inflation)
- **Justification:** U-Ignore does not corrupt model gradients with guessed labels, providing a clean scientific baseline.

### D. Handling Class Imbalance
- **Protocol:** 
  1. *Training Loss:* Apply **Class-Weighted Binary Cross-Entropy Loss** with positive weights:
     $$w_c = \frac{N - N_c^+}{N_c^+}$$
     or implement Asymmetric Loss / Focal Loss to down-weight easy negative examples.
  2. *Data Sampling:* Maintain natural multi-label empirical prevalence; do not artificially undersample or oversample.
  3. *Evaluation Metrics:* Reject raw accuracy. Mandate threshold-independent and imbalance-robust metrics:
     - Area Under the ROC Curve (AUROC / ROC-AUC) per disease and Macro-AUROC
     - Area Under the Precision-Recall Curve (AUPRC / PR-AUC) per disease
     - F1-score and Sensitivity at 80% / 90% Specificity operating points

### E. Cohort Scale: Development Subset vs. Full Dataset
- **Protocol:** 
  1. Define a stratified **Development Subset** (e.g., 10,000 &ndash; 20,000 frontal images partitioned strictly at the patient level) for rapid pipeline debugging, architectural smoke testing, and hyperparameter tuning.
  2. Train final baseline and benchmark models on the **Full Frontal Training Cohort** (191,027 images).
- **Justification:** Training on 191k images requires significant computational runtime per epoch; a patient-partitioned dev subset accelerates iteration cycles without compromising scientific validity.

### F. Validation Strategy and Partitioning
- **Protocol:**
  1. **Internal Patient-Level Validation Split:** Derive an internal validation set (10% of training patients, ~6,454 patients, ~19,000 frontal images) sampled strictly by `patient_id` from `train_cheXbert.csv` with multi-label stratification.
  2. **Role of Internal Validation Split:** The internal patient-level validation split is used exclusively for all development decisions: model checkpointing, early stopping, hyperparameter tuning, learning-rate decay scheduling, and decision threshold calibration.
  3. **Official Held-Out CheXpert Validation Benchmark:** The official validation set (evaluated on its **202 frontal images** for the initial baseline) is an official held-out CheXpert validation benchmark. It is used **only for final/locked evaluation** after all training, tuning, and threshold calibration are finalized.
  4. **Strict Isolation:** The 234-image validation set is **not used for training, early stopping, hyperparameter tuning, or threshold tuning**.

### G. Prevention of Patient-Level Data Leakage
- **Protocol:**
  - All dataset splits (internal train vs. internal val vs. test) must be executed strictly on unique `patient_id` sets using patient-grouped splitting (`GroupShuffleSplit` / `GroupKFold`).
  - **Automated Guardrail:** Any split-generating script must programmatically assert:
    $$\text{len}(\text{set}(\text{Train Patients}) \cap \text{set}(\text{Val Patients})) == 0$$
    and fail loudly with exit code `1` if any overlap is detected.

### H. Decisions That Remain Provisional
The following protocol decisions are explicitly provisional and subject to empirical review following initial baseline results:
1. **Frontal-only restriction:** Provisional until a multi-view (Frontal + Lateral) fusion architecture can be benchmarked on the 31,413 multi-view studies.
2. **Uncertainty policy:** U-Ignore vs. condition-specific U-Ones/U-Zeros will be decided via ablation study on the internal validation split.
3. **Loss function:** Weighted BCE vs. Focal Loss vs. Asymmetric Loss will be compared empirically.
4. **Input resolution:** 224&times;224 initial resolution vs. 320&times;320 / 512&times;512 for fine-grained pneumonia opacity detection.

---

## 9. Risks and Limitations

1. **Extreme Pneumonia Scarcity in Validation:** The official validation set contains only 8 positive pneumonia cases. Metric variance (e.g., a single false positive or false negative shifts recall by 12.5%) is very high on this set. The internal validation split must serve as the primary statistical anchor.
2. **Weak Supervision Noise:** Labels originate from report NLP (CheXbert). While CheXbert achieves near-radiologist F1 on reports, the reports themselves reflect clinical radiologist variance, portable bedside imaging quality limitations, and missing clinical history.
3. **Negative Reporting Bias:** Unmentioned pathologies are assumed negative. In complex ICU encounters, minor atelectasis may be omitted by the reporting radiologist if acute pulmonary edema or severe pneumothorax is the clinical priority.
4. **Hardware Runtime:** Processing 191k frontal images requires rigorous caching, mixed-precision training, and efficient data loading pipelines.

---

## 10. Summary and Phase 1D Completion

Phase 1D establishes a comprehensive, empirically backed dataset protocol:
- Path and view alignment between `train_cheXbert.csv` and `chexpert-small/train.csv` is 100% verified.
- The hierarchy of 64,540 patients, 187,641 studies, and 223,414 images is mapped.
- Intra-study label consistency is 100% (0 discrepancies).
- An actionable, scientifically rigorous 8-point experimental protocol (A&ndash;H) is defined.

**Phase 1D Boundary Maintained:** No images were preprocessed, resized, or moved. No models were trained. No packages were installed.
