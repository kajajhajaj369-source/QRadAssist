"""CheXpert dataset protocol analysis and verification (Phase 1D).

Performs read-only inspection of metadata to support experimental protocol design:
- Compares train_cheXbert.csv and train.csv path alignment.
- Analyzes frontal vs. lateral view distributions and study-level view combinations.
- Computes patient/study/image hierarchical relationships.
- Verifies target-label consistency across views within studies.
- Analyzes official validation set structure and ground truth distributions.

Does not preprocess images, write datasets, train models, or modify any files.
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CHEXBERT_CSV = PROJECT_ROOT / "data" / "raw" / "train_cheXbert.csv"
TRAIN_ORIGINAL_CSV = PROJECT_ROOT / "data" / "raw" / "chexpert-small" / "train.csv"
VALID_CSV = PROJECT_ROOT / "data" / "raw" / "chexpert-small" / "valid.csv"

PATIENT_RE = re.compile(r"patient(\d+)", re.IGNORECASE)
STUDY_RE = re.compile(r"study(\d+)", re.IGNORECASE)

TARGET_COLUMNS = ("Pneumonia", "Atelectasis", "Pleural Effusion")


def normalize_chexpert_path(raw: str) -> str:
    """Strip root dataset prefix and normalize backslashes."""
    p = (raw or "").strip().replace("\\", "/")
    for prefix in ("CheXpert-v1.0-small/", "CheXpert-v1.0/"):
        if p.startswith(prefix):
            return p[len(prefix) :]
    return p


@dataclass
class ProtocolAnalysisResults:
    paths_match: bool = False
    total_train_rows: int = 0
    frontal_count: int = 0
    lateral_count: int = 0
    ap_count: int = 0
    pa_count: int = 0
    total_studies: int = 0
    only_frontal_studies: int = 0
    only_lateral_studies: int = 0
    both_views_studies: int = 0
    multiple_frontal_studies: int = 0
    total_patients: int = 0
    study_label_inconsistencies: dict[str, int] = field(default_factory=dict)
    valid_rows: int = 0
    valid_patients: int = 0
    valid_studies: int = 0
    valid_frontal: int = 0
    valid_lateral: int = 0
    valid_label_counts: dict[str, dict[str, int]] = field(default_factory=dict)


def run_protocol_analysis(
    train_chexbert_csv: Path = TRAIN_CHEXBERT_CSV,
    train_orig_csv: Path = TRAIN_ORIGINAL_CSV,
    valid_csv: Path = VALID_CSV,
) -> ProtocolAnalysisResults:
    results = ProtocolAnalysisResults()

    # 1. Compare train_cheXbert vs train.csv paths
    with train_chexbert_csv.open(encoding="utf-8") as f:
        chexbert_paths = [normalize_chexpert_path(row["Path"]) for row in csv.DictReader(f)]

    with train_orig_csv.open(encoding="utf-8") as f:
        orig_paths = [normalize_chexpert_path(row["Path"]) for row in csv.DictReader(f)]

    results.paths_match = (chexbert_paths == orig_paths)
    results.total_train_rows = len(chexbert_paths)

    # 2. View, Hierarchy, and Study Consistency Analysis
    study_to_views: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    study_to_frontal_count: Counter[tuple[str, str]] = Counter()
    patient_studies: dict[str, set[str]] = defaultdict(set)
    study_labels: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    with train_chexbert_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            norm_p = normalize_chexpert_path(row["Path"])
            patient = PATIENT_RE.search(norm_p).group(1)
            study = STUDY_RE.search(norm_p).group(1)
            fl = row.get("Frontal/Lateral", "")
            appa = row.get("AP/PA", "")
            ps_key = (patient, study)

            patient_studies[patient].add(study)
            study_to_views[ps_key].append((fl, appa))

            if fl == "Frontal":
                results.frontal_count += 1
                study_to_frontal_count[ps_key] += 1
            elif fl == "Lateral":
                results.lateral_count += 1

            if appa == "AP":
                results.ap_count += 1
            elif appa == "PA":
                results.pa_count += 1

            for target in TARGET_COLUMNS:
                study_labels[ps_key][target].add(row.get(target, ""))

    results.total_patients = len(patient_studies)
    results.total_studies = len(study_to_views)

    results.only_frontal_studies = sum(
        1 for ps, views in study_to_views.items() if all(fl == "Frontal" for fl, _ in views)
    )
    results.only_lateral_studies = sum(
        1 for ps, views in study_to_views.items() if all(fl == "Lateral" for fl, _ in views)
    )
    results.both_views_studies = sum(
        1
        for ps, views in study_to_views.items()
        if any(fl == "Frontal" for fl, _ in views) and any(fl == "Lateral" for fl, _ in views)
    )
    results.multiple_frontal_studies = sum(
        1 for ps, count in study_to_frontal_count.items() if count > 1
    )

    for target in TARGET_COLUMNS:
        inconsistencies = sum(1 for ps, t_dict in study_labels.items() if len(t_dict[target]) > 1)
        results.study_label_inconsistencies[target] = inconsistencies

    # 3. Validation set analysis
    with valid_csv.open(encoding="utf-8") as f:
        valid_rows = list(csv.DictReader(f))

    results.valid_rows = len(valid_rows)
    v_patients = set()
    v_studies = set()
    v_labels: dict[str, Counter[str]] = {t: Counter() for t in TARGET_COLUMNS}

    for r in valid_rows:
        norm_p = normalize_chexpert_path(r["Path"])
        patient = PATIENT_RE.search(norm_p).group(1)
        study = STUDY_RE.search(norm_p).group(1)
        v_patients.add(patient)
        v_studies.add((patient, study))

        fl = r.get("Frontal/Lateral", "")
        if fl == "Frontal":
            results.valid_frontal += 1
        elif fl == "Lateral":
            results.valid_lateral += 1

        for target in TARGET_COLUMNS:
            v_labels[target][r.get(target, "")] += 1

    results.valid_patients = len(v_patients)
    results.valid_studies = len(v_studies)
    results.valid_label_counts = {t: dict(c) for t, c in v_labels.items()}

    return results


def print_report(res: ProtocolAnalysisResults) -> None:
    lines = [
        "=================================================================",
        "Q-RadAssist Dataset Protocol Analysis (Phase 1D)",
        "=================================================================",
        f"1. Path Match between train_cheXbert.csv and train.csv: {res.paths_match}",
        f"   Total Training Rows: {res.total_train_rows:,}",
        "",
        "2. Training Set View Breakdown:",
        f"   - Frontal Views: {res.frontal_count:,} ({res.frontal_count/res.total_train_rows*100:.2f}%)",
        f"   - Lateral Views: {res.lateral_count:,} ({res.lateral_count/res.total_train_rows*100:.2f}%)",
        f"   - AP Projections: {res.ap_count:,} ({res.ap_count/res.total_train_rows*100:.2f}%)",
        f"   - PA Projections: {res.pa_count:,} ({res.pa_count/res.total_train_rows*100:.2f}%)",
        "",
        "3. Study-Level View Structure:",
        f"   - Total Studies: {res.total_studies:,}",
        f"   - Studies with ONLY Frontal views: {res.only_frontal_studies:,} ({res.only_frontal_studies/res.total_studies*100:.2f}%)",
        f"   - Studies with BOTH Frontal & Lateral: {res.both_views_studies:,} ({res.both_views_studies/res.total_studies*100:.2f}%)",
        f"   - Studies with ONLY Lateral views: {res.only_lateral_studies:,} ({res.only_lateral_studies/res.total_studies*100:.2f}%)",
        f"   - Studies with Multiple Frontal views: {res.multiple_frontal_studies:,} ({res.multiple_frontal_studies/res.total_studies*100:.2f}%)",
        "",
        "4. Patient / Study Hierarchy:",
        f"   - Total Unique Patients: {res.total_patients:,}",
        f"   - Mean Studies per Patient: {res.total_studies / res.total_patients:.2f}",
        f"   - Mean Images per Study: {res.total_train_rows / res.total_studies:.2f}",
        "",
        "5. Intra-Study Label Inconsistencies (Multi-View Studies):",
    ]
    for target, cnt in res.study_label_inconsistencies.items():
        lines.append(f"   - {target}: {cnt} studies with varying labels")

    lines.extend(
        [
            "",
            "6. Official Validation Set Structure:",
            f"   - Total Rows: {res.valid_rows:,}",
            f"   - Unique Patients: {res.valid_patients:,}",
            f"   - Unique Studies: {res.valid_studies:,}",
            f"   - Frontal Views: {res.valid_frontal:,} ({res.valid_frontal/res.valid_rows*100:.2f}%)",
            f"   - Lateral Views: {res.valid_lateral:,} ({res.valid_lateral/res.valid_rows*100:.2f}%)",
            "   - Target Label Counts (All binary ground truth):",
        ]
    )
    for target, counts in res.valid_label_counts.items():
        lines.append(f"     * {target}: {counts}")

    lines.append("=================================================================")
    sys.stdout.write("\n".join(lines) + "\n")


def main() -> int:
    results = run_protocol_analysis()
    print_report(results)
    return 0 if results.paths_match else 1


if __name__ == "__main__":
    sys.exit(main())
