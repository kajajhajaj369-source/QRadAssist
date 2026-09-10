import csv
import re
import pytest
from pathlib import Path

from paths import PROJECT_ROOT, RAW_DATA_DIR, SPLITS_DIR

PATIENT_RE = re.compile(r"^\d+$")
STUDY_RE = re.compile(r"^\d+$")

TRAIN_SPLIT_FILE = SPLITS_DIR / "train_manifest.csv"
VAL_SPLIT_FILE = SPLITS_DIR / "val_manifest.csv"
DEV_COHORT_FILE = SPLITS_DIR / "dev_cohort_manifest.csv"
FULL_SPLIT_FILE = SPLITS_DIR / "train_val_split.csv"
OFFICIAL_VALID_CSV = RAW_DATA_DIR / "chexpert-small" / "valid.csv"


def load_manifest_rows(path: Path) -> list[dict[str, str]]:
    assert path.is_file(), f"Manifest file missing: {path}"
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_materialized_manifest_files_exist():
    assert FULL_SPLIT_FILE.is_file()
    assert TRAIN_SPLIT_FILE.is_file()
    assert VAL_SPLIT_FILE.is_file()
    assert DEV_COHORT_FILE.is_file()


def test_zero_patient_overlap_between_train_and_val():
    train_rows = load_manifest_rows(TRAIN_SPLIT_FILE)
    val_rows = load_manifest_rows(VAL_SPLIT_FILE)

    train_patients = {r["patient_id"] for r in train_rows}
    val_patients = {r["patient_id"] for r in val_rows}

    overlap = train_patients & val_patients
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping patients between train and val"
    assert len(train_patients) == 58081
    assert len(val_patients) == 6453


def test_zero_patient_overlap_between_dev_cohort_and_val():
    dev_rows = load_manifest_rows(DEV_COHORT_FILE)
    val_rows = load_manifest_rows(VAL_SPLIT_FILE)

    dev_patients = {r["patient_id"] for r in dev_rows}
    val_patients = {r["patient_id"] for r in val_rows}

    overlap = dev_patients & val_patients
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping patients between dev cohort and val"


def test_zero_lateral_images_in_all_splits():
    for manifest_path in (TRAIN_SPLIT_FILE, VAL_SPLIT_FILE, DEV_COHORT_FILE, FULL_SPLIT_FILE):
        rows = load_manifest_rows(manifest_path)
        views = {r["view"] for r in rows}
        assert views == {"Frontal"}, f"Non-frontal view detected in {manifest_path}: {views}"


def test_development_cohort_size_and_patient_grouping():
    dev_rows = load_manifest_rows(DEV_COHORT_FILE)
    train_rows = load_manifest_rows(TRAIN_SPLIT_FILE)

    # Size must be in approximate 10,000 - 20,000 range (target ~15,000)
    assert 10000 <= len(dev_rows) <= 20000
    assert len(dev_rows) == 15000

    dev_patients = {r["patient_id"] for r in dev_rows}
    train_patients = {r["patient_id"] for r in train_rows}
    assert dev_patients.issubset(train_patients)

    # Verify patient-level grouping integrity: for every patient in dev_cohort,
    # all of their frontal images from train_manifest must be present in dev_cohort
    train_patient_images = {}
    for r in train_rows:
        pid = r["patient_id"]
        train_patient_images.setdefault(pid, []).append(r["sample_id"])

    dev_patient_images = {}
    for r in dev_rows:
        pid = r["patient_id"]
        dev_patient_images.setdefault(pid, []).append(r["sample_id"])

    for pid in dev_patients:
        assert set(dev_patient_images[pid]) == set(train_patient_images[pid]), (
            f"Patient {pid} was only partially included in dev cohort!"
        )


def test_no_duplicate_image_paths_in_manifests():
    for manifest_path in (TRAIN_SPLIT_FILE, VAL_SPLIT_FILE, DEV_COHORT_FILE):
        rows = load_manifest_rows(manifest_path)
        paths = [r["image_path"] for r in rows]
        assert len(paths) == len(set(paths)), f"Duplicate paths found in {manifest_path}"


def test_patient_and_study_ids_valid():
    dev_rows = load_manifest_rows(DEV_COHORT_FILE)
    for r in dev_rows:
        assert PATIENT_RE.match(r["patient_id"]), f"Invalid patient ID: {r['patient_id']}"
        assert STUDY_RE.match(r["study_id"]), f"Invalid study ID: {r['study_id']}"


def test_target_label_columns_and_valid_values():
    dev_rows = load_manifest_rows(DEV_COHORT_FILE)
    valid_values = {"1.0", "0.0", "-1.0", ""}
    for r in dev_rows:
        assert r["pneumonia_raw"] in valid_values
        assert r["atelectasis_raw"] in valid_values
        assert r["pleural_effusion_raw"] in valid_values


def test_official_validation_set_strictly_excluded():
    with OFFICIAL_VALID_CSV.open("r", encoding="utf-8") as f:
        official_val_rows = list(csv.DictReader(f))

    # Official validation patients are patient64541 through patient64740
    official_val_paths = {r["Path"].strip().replace("\\", "/") for r in official_val_rows}
    official_val_patients = {
        re.search(r"patient(\d+)", p).group(1) for p in official_val_paths if re.search(r"patient(\d+)", p)
    }

    full_rows = load_manifest_rows(FULL_SPLIT_FILE)
    full_paths = {r["raw_csv_path"].strip().replace("\\", "/") for r in full_rows}
    full_patients = {r["patient_id"] for r in full_rows}

    assert len(official_val_paths & full_paths) == 0
    assert len(official_val_patients & full_patients) == 0


def test_reproducibility_with_seed():
    # Verify that dev cohort manifest matches exact deterministic expectation for seed=42
    dev_rows = load_manifest_rows(DEV_COHORT_FILE)
    assert dev_rows[0]["sample_id"] == "p47802_s1_view1_frontal"
    assert dev_rows[0]["patient_id"] == "47802"
    assert len(dev_rows) == 15000

