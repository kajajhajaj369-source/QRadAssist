# Dataset inspection (Phase 1B)

Inspection of the local metadata file only:

`data/raw/train_cheXbert.csv`

No image files were inspected. The CSV was not modified. Counts below were
computed by reading every row of this file.

The filename is `train_cheXbert.csv`. The file itself does not state which
labeler produced the observation columns. These results describe **this CSV**,
not a verified comparison with other CheXpert label versions.

## 1. Number of rows

**223,414** data rows (plus one header row).

## 2. Column names

In file order:

1. `Path`
2. `Sex`
3. `Age`
4. `Frontal/Lateral`
5. `AP/PA`
6. `Enlarged Cardiomediastinum`
7. `Cardiomegaly`
8. `Lung Opacity`
9. `Lung Lesion`
10. `Edema`
11. `Consolidation`
12. `Pneumonia`
13. `Atelectasis`
14. `Pneumothorax`
15. `Pleural Effusion`
16. `Pleural Other`
17. `Fracture`
18. `Support Devices`
19. `No Finding`

## 3. Image / path identifier

**`Path`.** Every row has a non-empty path. All 223,414 paths end in `.jpg`.

Example:

`CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg`

This is a relative CheXpert-style path. Presence of the path in the CSV does
**not** mean the corresponding image file is present locally.

## 4. Target label columns

| Target | CSV column |
| --- | --- |
| Pneumonia | `Pneumonia` |
| Atelectasis | `Atelectasis` |
| Pleural Effusion | `Pleural Effusion` |

Exact header match. No other columns were used as proxies.

## 5. Unique values in the three label columns

For each of `Pneumonia`, `Atelectasis`, and `Pleural Effusion`, the only values
observed were:

| Raw CSV value | Interpretation used in this report |
| --- | --- |
| `1.0` | positive |
| `0.0` | negative |
| `-1.0` | uncertain |
| empty string | blank / unmentioned |

No other strings or numbers appeared in these three columns.

## 6. Label counts and percentages

Denominator: 223,414 rows.

### Pneumonia

| Category | Count | Percentage |
| --- | ---: | ---: |
| positive (`1.0`) | 4,856 | 2.17% |
| negative (`0.0`) | 3,314 | 1.48% |
| uncertain (`-1.0`) | 19,386 | 8.68% |
| blank / unmentioned | 195,858 | 87.67% |

### Atelectasis

| Category | Count | Percentage |
| --- | ---: | ---: |
| positive (`1.0`) | 33,862 | 15.16% |
| negative (`0.0`) | 730 | 0.33% |
| uncertain (`-1.0`) | 34,416 | 15.40% |
| blank / unmentioned | 154,406 | 69.11% |

### Pleural Effusion

| Category | Count | Percentage |
| --- | ---: | ---: |
| positive (`1.0`) | 89,314 | 39.98% |
| negative (`0.0`) | 36,278 | 16.24% |
| uncertain (`-1.0`) | 7,634 | 3.42% |
| blank / unmentioned | 90,188 | 40.37% |

Percentages are rounded to two decimal places; counts are exact.

## 7. Joint positive status of the three targets

A target is counted as positive only when the cell is `1.0`. Uncertain,
negative, and blank are **not** treated as positive.

| Set | Count | Percentage |
| --- | ---: | ---: |
| At least one of the three labels is positive | 109,189 | 48.87% |
| Two or three of the three labels are positive | 18,670 | 8.36% |
| None of the three labels is positive | 114,225 | 51.13% |

These three groups partition the 223,414 rows.

## 8. Patient and study identifiers

There are **no** separate `patient` or `study` columns.

Identifiers appear **inside `Path`**, with this pattern on every row:

`CheXpert-v1.0/train/patientNNNNN/studyK/...`

Observed from paths:

| Identifier | How it appears | Unique values in this CSV |
| --- | --- | ---: |
| Patient | directory `patient` + digits (e.g. `patient00001`) | 64,540 |
| Study | directory `study` + digits under that patient (e.g. `study1`) | 187,641 patient–study pairs |

Because there are more rows (223,414) than patient–study pairs (187,641), and
more studies than patients (64,540), the CSV contains multiple images per
patient and, in many cases, multiple views or images per study.

No patient identifier exists except via `Path`. If splits are built later,
they would need to parse this path (or an equivalent derived field). That is
not implemented here.

## 9. Duplicate image / path entries

**No duplicate `Path` values.** Unique paths = 223,414 = number of rows.

## 10. Multi-label co-occurrence (positive = `1.0` only)

Positive-label combinations:

| Positive targets on the row | Count |
| --- | ---: |
| none | 114,225 |
| Pleural Effusion only | 70,839 |
| Atelectasis + Pleural Effusion | 17,015 |
| Atelectasis only | 16,479 |
| Pneumonia only | 3,201 |
| Pleural Effusion + Pneumonia | 1,287 |
| Atelectasis + Pneumonia | 195 |
| Atelectasis + Pleural Effusion + Pneumonia | 173 |

**18,670** rows have two or three simultaneous positive target labels,
including **173** rows positive for all three.

From this CSV, the three labels are **not** mutually exclusive. They can form a
multi-label task in the narrow sense that positive findings co-occur on the
same rows. This does not evaluate clinical independence, label accuracy, or
how blank/uncertain values should be trained.

## 11. Obvious data-quality notes

Observed in this file (not an image-quality audit):

- **Metadata only.** Image files were not present in this inspection. Paths
  point at `CheXpert-v1.0/train/...`; that tree was not downloaded for this
  phase.
- **Label imbalance is large.** Pneumonia positives are 2.17% of rows.
  Atelectasis has very few explicit negatives (730; 0.33%) compared with
  positives and uncertain values.
- **Blank is the majority code** for Pneumonia and Atelectasis. Blank is not
  the same as `0.0` in this file; the two must not be collapsed without an
  explicit later rule.
- **`Sex`:** `Female` 90,777; `Male` 132,636; `Unknown` 1.
- **`Frontal/Lateral`:** Frontal 191,027; Lateral 32,387.
- **`AP/PA`:** AP 161,590; PA 29,420; empty 32,387; `LL` 16; `RL` 1.
  The number of empty `AP/PA` cells equals the number of Lateral rows. `LL`
  and `RL` are not AP/PA values; they are rare and unexplained by the CSV
  alone.
- **`Age`:** integers from 0 to 90+ (observed range 0–110). Extremes are
  reported, not judged as errors.
- **No ragged rows.** Every data row has 19 fields.

## Short interpretation

This CSV is a 223,414-row, one-row-per-path table with CheXpert-style columns.
The three core targets are present as their own columns and use the four-way
coding `1.0` / `0.0` / `-1.0` / blank.

Pleural effusion is commonly coded positive; pneumonia positives are uncommon;
atelectasis positives are intermediate but explicit negatives are rare. Positive
labels co-occur often enough that a multi-label formulation is consistent with
the table. Patient and study identity can be recovered from `Path`, so
image-level random splits would risk patient-level leakage if used later.

What this inspection does **not** decide: uncertainty mapping, whether blank
means negative, which views to keep, train/validation/test splits, or any
filtered training subset. No training dataset was created.
