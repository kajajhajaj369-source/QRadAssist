"""Dataset manifest representation and path/view parsing (Phase 2A).

Implements:
1. Path resolution: maps 'CheXpert-v1.0/' and 'CheXpert-v1.0-small/' to 'data/raw/chexpert-small/'.
2. View filtering: retains Frontal views (AP/PA), excludes Lateral without deleting.
3. Patient & Study ID parsing: extracts patient_id and study_id.
4. Raw label preservation: stores exact strings for Pneumonia, Atelectasis, Pleural Effusion.
5. Manifest serialization: reads/writes clean CSV manifests without duplicating image files.
"""

from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from paths import PROJECT_ROOT, RAW_DATA_DIR
from preprocessing.labels import TARGET_LABELS

DEFAULT_IMAGE_ROOT = RAW_DATA_DIR / "chexpert-small"

CSV_PATH_PREFIXES = (
    "CheXpert-v1.0-small/",
    "CheXpert-v1.0/",
)

PATIENT_RE = re.compile(r"patient(\d+)", re.IGNORECASE)
STUDY_RE = re.compile(r"study(\d+)", re.IGNORECASE)


@dataclass
class SampleManifest:
    sample_id: str
    image_path: str  # Relative to PROJECT_ROOT or absolute
    raw_csv_path: str
    patient_id: str
    study_id: str
    view: str  # Frontal, Lateral
    projection: str  # AP, PA, LL, RL, or empty
    pneumonia_raw: str
    atelectasis_raw: str
    pleural_effusion_raw: str
    split: str | None = None

    @property
    def raw_labels(self) -> dict[str, str]:
        return {
            "Pneumonia": self.pneumonia_raw,
            "Atelectasis": self.atelectasis_raw,
            "Pleural Effusion": self.pleural_effusion_raw,
        }

    def to_dict(self) -> dict[str, str]:
        return {
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "raw_csv_path": self.raw_csv_path,
            "patient_id": self.patient_id,
            "study_id": self.study_id,
            "view": self.view,
            "projection": self.projection,
            "pneumonia_raw": self.pneumonia_raw,
            "atelectasis_raw": self.atelectasis_raw,
            "pleural_effusion_raw": self.pleural_effusion_raw,
            "split": self.split or "",
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> SampleManifest:
        return cls(
            sample_id=data["sample_id"],
            image_path=data["image_path"],
            raw_csv_path=data["raw_csv_path"],
            patient_id=data["patient_id"],
            study_id=data["study_id"],
            view=data.get("view", ""),
            projection=data.get("projection", ""),
            pneumonia_raw=data.get("pneumonia_raw", ""),
            atelectasis_raw=data.get("atelectasis_raw", ""),
            pleural_effusion_raw=data.get("pleural_effusion_raw", ""),
            split=data.get("split") or None,
        )


def resolve_local_image_path(raw_csv_path: str, image_root: Path = DEFAULT_IMAGE_ROOT) -> Path:
    """Map a CheXpert CSV Path prefix to the local image directory."""
    raw = (raw_csv_path or "").strip().replace("\\", "/")
    if not raw:
        raise ValueError("Encountered an empty Path value.")
    for prefix in CSV_PATH_PREFIXES:
        if raw.startswith(prefix):
            rel = raw[len(prefix) :]
            if not rel:
                raise ValueError(f"Path has prefix but no relative target: {raw_csv_path!r}")
            return (image_root / rel).resolve()
    raise ValueError(
        f"Cannot resolve image path; expected prefix {CSV_PATH_PREFIXES} but got {raw_csv_path!r}"
    )


def parse_patient_id(path_str: str) -> str:
    match = PATIENT_RE.search(path_str.replace("\\", "/"))
    if not match:
        raise ValueError(f"Cannot parse patient ID from path: {path_str!r}")
    return match.group(1)


def parse_study_id(path_str: str) -> str:
    match = STUDY_RE.search(path_str.replace("\\", "/"))
    if not match:
        raise ValueError(f"Cannot parse study ID from path: {path_str!r}")
    return match.group(1)


def generate_sample_id(patient_id: str, study_id: str, local_path: Path) -> str:
    """Deterministic sample identifier combining patient, study, and filename stem."""
    stem = local_path.stem
    return f"p{patient_id}_s{study_id}_{stem}"


def parse_row_to_sample(
    row: dict[str, str],
    image_root: Path = DEFAULT_IMAGE_ROOT,
    check_file_exists: bool = False,
) -> SampleManifest:
    """Parse a single raw CSV row into a SampleManifest record."""
    raw_path = row.get("Path", "").strip()
    if not raw_path:
        raise ValueError("Row is missing 'Path' column or has empty path.")

    local_path = resolve_local_image_path(raw_path, image_root=image_root)
    if check_file_exists and not local_path.is_file():
        raise FileNotFoundError(f"Referenced image does not exist: {local_path}")

    patient_id = parse_patient_id(raw_path)
    study_id = parse_study_id(raw_path)
    view = row.get("Frontal/Lateral", "").strip()
    projection = row.get("AP/PA", "").strip()
    sample_id = generate_sample_id(patient_id, study_id, local_path)

    # Relative path from project root for portability, if possible
    try:
        rel_image_path = local_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        rel_image_path = local_path.as_posix()

    return SampleManifest(
        sample_id=sample_id,
        image_path=rel_image_path,
        raw_csv_path=raw_path,
        patient_id=patient_id,
        study_id=study_id,
        view=view,
        projection=projection,
        pneumonia_raw=str(row.get("Pneumonia", "") or "").strip(),
        atelectasis_raw=str(row.get("Atelectasis", "") or "").strip(),
        pleural_effusion_raw=str(row.get("Pleural Effusion", "") or "").strip(),
    )


def filter_manifest_by_view(
    samples: Sequence[SampleManifest],
    allowed_views: Sequence[str] = ("Frontal",),
) -> list[SampleManifest]:
    """Filter manifest samples by view type (e.g. Frontal only). Excluded samples are not deleted."""
    allowed_set = set(allowed_views)
    return [s for s in samples if s.view in allowed_set]


def save_manifest_csv(samples: Sequence[SampleManifest], output_path: Path) -> None:
    """Write samples to a clean CSV manifest file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "image_path",
        "raw_csv_path",
        "patient_id",
        "study_id",
        "view",
        "projection",
        "pneumonia_raw",
        "atelectasis_raw",
        "pleural_effusion_raw",
        "split",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample.to_dict())


def load_manifest_csv(input_path: Path) -> list[SampleManifest]:
    """Read samples from a CSV manifest file."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {input_path}")
    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [SampleManifest.from_dict(row) for row in reader]
