"""CheXpert metadata vs local image integrity check (Phase 1C).

Maps CSV Path prefixes to data/raw/chexpert-small/ and verifies that
referenced images exist. Does not train, preprocess, or write a subset.

Exit status is non-zero if paths cannot be resolved, images are missing,
or a patient ID appears in both the training and validation CSVs.
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "chexpert-small"
TRAIN_LABELS_CSV = PROJECT_ROOT / "data" / "raw" / "train_cheXbert.csv"
VALID_CSV = IMAGE_ROOT / "valid.csv"

# Longer prefix first: CheXpert-v1.0-small/ must not be treated as CheXpert-v1.0/.
CSV_PATH_PREFIXES = (
    "CheXpert-v1.0-small/",
    "CheXpert-v1.0/",
)

TARGET_COLUMNS = ("Pneumonia", "Atelectasis", "Pleural Effusion")
PATIENT_RE = re.compile(r"patient(\d+)", re.IGNORECASE)
STUDY_RE = re.compile(r"study(\d+)", re.IGNORECASE)

MAX_EXAMPLES = 20


class IntegrityError(Exception):
    """Dataset integrity check failed."""


def resolve_csv_path(csv_path: str, image_root: Path = IMAGE_ROOT) -> Path:
    """Convert a CheXpert CSV Path value to a local filesystem path."""
    raw = (csv_path or "").strip().replace("\\", "/")
    if not raw:
        raise IntegrityError("Encountered an empty Path value in a CSV row.")
    for prefix in CSV_PATH_PREFIXES:
        if raw.startswith(prefix):
            relative = raw[len(prefix) :]
            if not relative:
                raise IntegrityError(f"Path has a recognized prefix but no relative file: {csv_path!r}")
            return image_root / relative
    raise IntegrityError(
        "Cannot resolve image path; expected prefix "
        f"{CSV_PATH_PREFIXES} but got {csv_path!r}."
    )


def classify_label(raw: str | None) -> str:
    if raw is None or not str(raw).strip():
        return "blank"
    text = str(raw).strip()
    try:
        value = float(text)
    except ValueError as exc:
        raise IntegrityError(f"Unexpected non-numeric label value: {raw!r}") from exc
    if value == 1.0:
        return "positive"
    if value == 0.0:
        return "negative"
    if value == -1.0:
        return "uncertain"
    raise IntegrityError(f"Unexpected numeric label value: {raw!r}")


def _patient_id(path_value: str) -> str:
    match = PATIENT_RE.search(path_value.replace("\\", "/"))
    if not match:
        raise IntegrityError(f"Cannot extract patient ID from Path: {path_value!r}")
    return match.group(1)


def _study_id(path_value: str) -> str:
    match = STUDY_RE.search(path_value.replace("\\", "/"))
    if not match:
        raise IntegrityError(f"Cannot extract study ID from Path: {path_value!r}")
    return match.group(1)


@dataclass
class SplitIntegrity:
    name: str
    csv_path: Path
    row_count: int = 0
    existing_images: int = 0
    missing_images: int = 0
    unresolvable_paths: int = 0
    duplicate_path_keys: int = 0
    unique_paths: int = 0
    unique_patients: int = 0
    unique_studies: int = 0
    missing_examples: list[str] = field(default_factory=list)
    unresolvable_examples: list[str] = field(default_factory=list)
    duplicate_examples: list[str] = field(default_factory=list)
    patient_ids: set[str] = field(default_factory=set)
    label_counts: dict[str, Counter] = field(default_factory=dict)
    cooccurrence: Counter = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)


def _inspect_split(
    name: str,
    csv_path: Path,
    *,
    collect_labels: bool,
) -> SplitIntegrity:
    result = SplitIntegrity(name=name, csv_path=csv_path)
    if not csv_path.is_file():
        result.errors.append(f"CSV file does not exist: {csv_path}")
        return result

    path_counts: Counter[str] = Counter()
    patients: set[str] = set()
    studies: set[tuple[str, str]] = set()
    if collect_labels:
        result.label_counts = {column: Counter() for column in TARGET_COLUMNS}

    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "Path" not in reader.fieldnames:
            result.errors.append(f"{csv_path} has no Path column.")
            return result
        if collect_labels:
            missing_cols = [column for column in TARGET_COLUMNS if column not in reader.fieldnames]
            if missing_cols:
                result.errors.append(f"{csv_path} is missing target columns: {missing_cols}")
                return result

        for row in reader:
            result.row_count += 1
            csv_rel = row.get("Path", "")
            path_counts[csv_rel] += 1
            try:
                patient = _patient_id(csv_rel)
                study = _study_id(csv_rel)
                local_path = resolve_csv_path(csv_rel)
            except IntegrityError as exc:
                result.unresolvable_paths += 1
                if len(result.unresolvable_examples) < MAX_EXAMPLES:
                    result.unresolvable_examples.append(f"{csv_rel!r}: {exc}")
                continue

            patients.add(patient)
            studies.add((patient, study))
            if local_path.is_file():
                result.existing_images += 1
            else:
                result.missing_images += 1
                if len(result.missing_examples) < MAX_EXAMPLES:
                    result.missing_examples.append(str(local_path))

            if collect_labels:
                positives: list[str] = []
                for column in TARGET_COLUMNS:
                    try:
                        category = classify_label(row.get(column))
                    except IntegrityError as exc:
                        result.errors.append(str(exc))
                        category = "invalid"
                    result.label_counts[column][category] += 1
                    if category == "positive":
                        positives.append(column)
                result.cooccurrence[frozenset(positives)] += 1

    duplicates = {key: count for key, count in path_counts.items() if count > 1}
    result.duplicate_path_keys = len(duplicates)
    result.duplicate_examples = [f"{key} (x{count})" for key, count in list(duplicates.items())[:MAX_EXAMPLES]]
    result.unique_paths = len(path_counts)
    result.unique_patients = len(patients)
    result.unique_studies = len(studies)
    result.patient_ids = patients
    return result


def _format_label_table(counts: Counter, total: int) -> str:
    lines = []
    for key in ("positive", "negative", "uncertain", "blank"):
        n = counts.get(key, 0)
        pct = (100.0 * n / total) if total else 0.0
        lines.append(f"    {key:12s} {n:8d}  ({pct:6.2f}%)")
    extras = {key: n for key, n in counts.items() if key not in {"positive", "negative", "uncertain", "blank"}}
    for key, n in extras.items():
        lines.append(f"    {key:12s} {n:8d}")
    return "\n".join(lines)


def _format_cooccurrence(counter: Counter) -> str:
    lines = []
    for key, count in sorted(counter.items(), key=lambda item: (-item[1], tuple(sorted(item[0])))):
        label = ", ".join(sorted(key)) if key else "(none positive)"
        lines.append(f"    {count:8d}  {label}")
    return "\n".join(lines)


def build_report(train: SplitIntegrity, valid: SplitIntegrity) -> tuple[str, list[str]]:
    overlap = sorted(train.patient_ids & valid.patient_ids)
    failures: list[str] = []
    failures.extend(train.errors)
    failures.extend(valid.errors)

    if train.unresolvable_paths:
        failures.append(f"Training: {train.unresolvable_paths} Path values could not be resolved.")
    if valid.unresolvable_paths:
        failures.append(f"Validation: {valid.unresolvable_paths} Path values could not be resolved.")
    if train.missing_images:
        failures.append(f"Training: {train.missing_images} referenced image files are missing.")
    if valid.missing_images:
        failures.append(f"Validation: {valid.missing_images} referenced image files are missing.")
    if overlap:
        failures.append(
            f"Patient leakage: {len(overlap)} patient ID(s) appear in both training and validation."
        )

    lines = [
        "Q-RadAssist dataset integrity verification (Phase 1C)",
        f"Image root: {IMAGE_ROOT}",
        f"Training labels CSV: {train.csv_path}",
        f"Validation CSV: {valid.csv_path}",
        "",
        "--- Training (train_cheXbert.csv) ---",
        f"CSV rows:              {train.row_count}",
        f"Unique Path values:    {train.unique_paths}",
        f"Duplicate Path keys:   {train.duplicate_path_keys}",
        f"Images found:          {train.existing_images}",
        f"Images missing:        {train.missing_images}",
        f"Unresolvable paths:    {train.unresolvable_paths}",
        f"Unique patients:       {train.unique_patients}",
        f"Unique patient-study:  {train.unique_studies}",
        "",
        "Target labels (training):",
    ]
    for column in TARGET_COLUMNS:
        lines.append(f"  {column}")
        lines.append(_format_label_table(train.label_counts.get(column, Counter()), train.row_count))
    lines.extend(
        [
            "",
            "Positive-label co-occurrence (training, 1.0 only):",
            _format_cooccurrence(train.cooccurrence),
            "",
            "--- Validation (valid.csv) ---",
            f"CSV rows:              {valid.row_count}",
            f"Unique Path values:    {valid.unique_paths}",
            f"Duplicate Path keys:   {valid.duplicate_path_keys}",
            f"Images found:          {valid.existing_images}",
            f"Images missing:        {valid.missing_images}",
            f"Unresolvable paths:    {valid.unresolvable_paths}",
            f"Unique patients:       {valid.unique_patients}",
            f"Unique patient-study:  {valid.unique_studies}",
            "",
            "--- Train / validation patient overlap ---",
            f"Overlapping patient IDs: {len(overlap)}",
        ]
    )
    if overlap[:MAX_EXAMPLES]:
        lines.append("Examples: " + ", ".join(f"patient{pid}" for pid in overlap[:MAX_EXAMPLES]))
    if train.missing_examples:
        lines.extend(["", "Missing training image examples:"] + [f"  {p}" for p in train.missing_examples])
    if valid.missing_examples:
        lines.extend(["", "Missing validation image examples:"] + [f"  {p}" for p in valid.missing_examples])
    if train.unresolvable_examples:
        lines.extend(["", "Unresolvable training Path examples:"] + [f"  {p}" for p in train.unresolvable_examples])
    if valid.unresolvable_examples:
        lines.extend(["", "Unresolvable validation Path examples:"] + [f"  {p}" for p in valid.unresolvable_examples])
    if train.duplicate_examples:
        lines.extend(["", "Duplicate training Path examples:"] + [f"  {p}" for p in train.duplicate_examples])
    if valid.duplicate_examples:
        lines.extend(["", "Duplicate validation Path examples:"] + [f"  {p}" for p in valid.duplicate_examples])

    other: list[str] = []
    if train.duplicate_path_keys:
        other.append(f"Training CSV has {train.duplicate_path_keys} duplicated Path value(s).")
    if valid.duplicate_path_keys:
        other.append(f"Validation CSV has {valid.duplicate_path_keys} duplicated Path value(s).")
    if not IMAGE_ROOT.is_dir():
        other.append(f"Image root directory does not exist: {IMAGE_ROOT}")
        failures.append(other[-1])

    lines.extend(["", "--- Other notes ---"])
    if other:
        lines.extend(f"- {item}" for item in other)
    else:
        lines.append("- No additional CSV-level integrity problems were recorded by this check.")

    if failures:
        lines.extend(["", "RESULT: FAIL"] + [f"- {item}" for item in failures])
    else:
        lines.extend(
            [
                "",
                "RESULT: PASS",
                "All referenced training and validation images were found.",
                "No train/validation patient ID overlap was detected.",
            ]
        )
    return "\n".join(lines) + "\n", failures


def run() -> int:
    train = _inspect_split("train", TRAIN_LABELS_CSV, collect_labels=True)
    valid = _inspect_split("valid", VALID_CSV, collect_labels=False)
    report, failures = build_report(train, valid)
    sys.stdout.write(report)
    if failures:
        sys.stderr.write("\nIntegrity check failed.\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
