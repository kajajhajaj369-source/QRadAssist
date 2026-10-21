# Preprocessing Specification & Training Normalization Statistics (Phase 2C)

This document establishes the locked, reproducible preprocessing specification for the Q-RadAssist baseline and records the empirical dataset normalization statistics calculated strictly from the patient-level **training partition**.

---

## 1. Locked Preprocessing Specification

The baseline preprocessing choices are formally locked across configuration, transforms, and data loading:

| Attribute | Specification | Justification / Notes |
| :--- | :--- | :--- |
| **Input Domain** | Chest X-ray radiographs | Standard CheXpert-v1.0-small JPEG images |
| **Baseline View** | Frontal only (`AP` / `PA`) | 172,033 training frontal images. Lateral images (32,387 training) are excluded non-destructively for multi-view fusion in future phases |
| **Color Mode** | Grayscale (`PIL.Image.convert("L")`) | Single-channel representation preserving radiographic optical density |
| **Output Tensor Layout** | `(C, H, W)` | Standard PyTorch tensor format (`channels_first=True`) |
| **Channel Count** | 1 | Single-channel grayscale |
| **Spatial Resolution** | $224 \times 224$ pixels | Standard baseline CNN input dimension; configurable for high-resolution experiments |
| **Interpolation** | Bilinear (`Image.Resampling.BILINEAR`) | Deterministic anti-aliased spatial reduction |
| **Pixel Data Type** | `float32` | Standard single-precision neural network tensor element |
| **Initial Scaling** | $[0.0, 1.0]$ | Divided by $255.0$; zero baseline bias |
| **Normalization / Standardization** | Configurable (default: `enabled: false`) | Requires explicit `mean` and `std` when enabled; raises `ValueError` if missing. No hard-coded, undocumented community constants (e.g. ImageNet) are silently applied |
| **Data Augmentation** | Separate from preprocessing | Deterministic preprocessing for validation and test; training augmentation (`TrainingAugmentor`) is independent and mild (horizontal flip only) |

---

## 2. Statistical Source & Partitions

The normalization statistics were computed under strict dataset hygiene constraints:

- **Source Manifest:** `data/splits/train_manifest.csv`
- **Referenced Image Directory:** `data/raw/chexpert-small/train/`
- **Total Images Processed:** Exactly **172,033** frontal images (100.0% of the patient-level training partition).
- **Sampling Used:** **None.** An exact, exhaustive calculation was performed over all 172,033 images.
- **Random Seed:** Not applicable (exhaustive computation, fully deterministic).

### Strict Validation Data Exclusion
1. **Official CheXpert Validation Set (`data/raw/chexpert-small/valid/`, `valid.csv`):**
   - Strictly excluded.
   - **Rationale:** The official validation set (234 images, 202 frontal) serves as the held-out final benchmark. Incorporating official validation images into training normalization statistics would cause data snooping / transductive statistical leakage, compromising benchmark objectivity.
2. **Internal Validation Partition (`data/splits/val_manifest.csv`):**
   - Strictly excluded (18,994 frontal images across 6,523 patients).
   - **Rationale:** The internal validation split is reserved for hyperparameter tuning, model checkpoint selection, learning rate scheduling, and early stopping. It must remain strictly out-of-sample relative to all training statistics.
3. **Development Cohort Manifest (`data/splits/dev_cohort_manifest.csv`):**
   - Not used as the primary source of truth. The full training partition (172,033 images) was used to ensure maximal statistical fidelity across the population.
4. **Lateral Images:**
   - Strictly excluded (all 32,387 lateral images filtered out).

---

## 3. Exact Calculation Procedure

The calculation was executed using `src/preprocessing/compute_stats.py`:

1. **Iteration:** Rows of `data/splits/train_manifest.csv` were read sequentially via streaming (O(1) memory overhead).
2. **Verification:**
   - Rejection if `view != "Frontal"`
   - Rejection if `split == "val"` or path contains `/valid/`
3. **Image Loading & Transform:**
   - Opened using PIL and converted to grayscale mode `"L"`.
   - Bilinearly resized to target baseline resolution: $(224, 224)$.
   - Converted to NumPy `float32` array and scaled to $[0.0, 1.0]$ via division by $255.0$.
4. **Accumulation:**
   - Accumulated pixel sum $\sum x$ in `numpy.float64`
   - Accumulated sum of squared pixels $\sum x^2$ in `numpy.float64`
   - Total pixel count: $N_{\text{total}} = 172,033 \times 224 \times 224 = 8,631,927,808$ pixels.
5. **Exact Moments:**
   $$\mu = \frac{\sum x}{N_{\text{total}}}$$
   $$\sigma^2 = \frac{\sum x^2}{N_{\text{total}}} - \mu^2$$
   $$\sigma = \sqrt{\sigma^2}$$

---

## 4. Calculated Numerical Results

The exact statistics over all 8,631,927,808 pixels across the 172,033 frontal training images are:

| Metric | Exact Value | Rounded (6 decimals) |
| :--- | :--- | :--- |
| **Total Images** | `172,033` | 172,033 |
| **Total Pixels** | `8,631,927,808` | 8.632 billion |
| **Mean ($\mu$)** | `0.5064217991464778` | **0.506422** |
| **Variance ($\sigma^2$)** | `0.08224078420292391` | **0.082241** |
| **Standard Deviation ($\sigma$)** | `0.2867765405379664` | **0.286777** |
| **Wall Clock Duration** | `1159.28` seconds (~19.3 min) | ~148.4 img/s |

### Numerical Sanity Checks
- **Mean Sensibility:** $\mu \approx 0.5064$ falls well within the sensible $[0.0, 1.0]$ interval, reflecting the typical balance between dark lung fields, radiopaque mediastinal/cardiac structures, and neutral thoracic boundaries.
- **Standard Deviation Sensibility:** $\sigma \approx 0.2868 > 0.0$, consistent with standard chest radiographic dynamic range and contrast spread.
- **Determinism:** Accumulation order and bilinear interpolation on identical inputs yield reproducible values down to machine floating-point precision.

---

## 5. Artifacts & Configuration Integration

1. **Statistics Record:** Saved to [data/splits/training_statistics.json](file:///e:/College/QRadAssist/data/splits/training_statistics.json).
2. **Project Settings:** Documented in [config/settings.yaml](file:///e:/College/QRadAssist/config/settings.yaml) under `preprocessing.normalization`.
3. **Preprocessor Implementation:** `DeterministicPreprocessor` in `src/preprocessing/transforms.py` accepts `mean` and `std` explicitly when `normalize=True`, and raises a clear `ValueError` if standardization is activated without statistics.

---

## 6. Limitations & Methodological Constraints

1. **Resolution Specificity:** These statistics are computed post-resizing at $224 \times 224$ with bilinear interpolation. Resizing introduces subtle anti-aliasing filtering. If future experiments scale to higher resolutions (e.g., $320 \times 320$ or $512 \times 512$), statistics should be re-verified or recalculated at the target resolution.
2. **Frontal Projection Invariance:** The statistics combine AP (anteroposterior) and PA (posteroanterior) projections. Because both are frontal views of thoracic anatomy, their pixel intensity distributions are mutually compatible for single-channel baseline modeling.
3. **No Patient Weighting:** Each pixel is weighted equally across all images; patients with multiple frontal studies contribute proportionally to their image count in the training split. Because patient-level splitting guarantees no patient leakage into validation or test sets, this does not compromise split independence.
