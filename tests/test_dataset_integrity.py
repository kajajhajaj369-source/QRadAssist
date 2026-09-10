import pytest
from pathlib import Path
from collections import Counter

from evaluation.dataset_integrity import (
    IMAGE_ROOT,
    VALID_CSV,
    IntegrityError,
    SplitIntegrity,
    _patient_id,
    _study_id,
    _inspect_split,
    build_report,
    classify_label,
    resolve_csv_path,
)


def test_resolve_csv_path_valid():
    p1 = resolve_csv_path("CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg")
    assert p1 == IMAGE_ROOT / "train/patient00001/study1/view1_frontal.jpg"

    p2 = resolve_csv_path("CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg")
    assert p2 == IMAGE_ROOT / "valid/patient64541/study1/view1_frontal.jpg"

    # Normalized backslashes
    p3 = resolve_csv_path("CheXpert-v1.0\\train\\patient00001\\study1\\view1_frontal.jpg")
    assert p3 == IMAGE_ROOT / "train/patient00001/study1/view1_frontal.jpg"


def test_resolve_csv_path_invalid():
    with pytest.raises(IntegrityError, match="empty Path"):
        resolve_csv_path("")

    with pytest.raises(IntegrityError, match="Cannot resolve image path"):
        resolve_csv_path("other_prefix/train/patient00001/study1/view1_frontal.jpg")


def test_classify_label():
    assert classify_label("1.0") == "positive"
    assert classify_label("1") == "positive"
    assert classify_label("0.0") == "negative"
    assert classify_label("0") == "negative"
    assert classify_label("-1.0") == "uncertain"
    assert classify_label("-1") == "uncertain"
    assert classify_label("") == "blank"
    assert classify_label("   ") == "blank"
    assert classify_label(None) == "blank"

    with pytest.raises(IntegrityError, match="Unexpected numeric label value"):
        classify_label("2.0")

    with pytest.raises(IntegrityError, match="Unexpected non-numeric label value"):
        classify_label("positive")


def test_extract_patient_and_study_ids():
    path = "CheXpert-v1.0/train/patient00042/study3/view1_frontal.jpg"
    assert _patient_id(path) == "00042"
    assert _study_id(path) == "3"

    with pytest.raises(IntegrityError, match="Cannot extract patient ID"):
        _patient_id("unknown/path/without/id.jpg")

    with pytest.raises(IntegrityError, match="Cannot extract study ID"):
        _study_id("patient00042/without_study/view1.jpg")


def test_build_report_detects_leakage_and_missing_files():
    train_split = SplitIntegrity(
        name="train",
        csv_path=Path("dummy_train.csv"),
        row_count=10,
        existing_images=9,
        missing_images=1,
        patient_ids={"00001", "00002"},
        missing_examples=["dummy_missing.jpg"],
    )
    valid_split = SplitIntegrity(
        name="valid",
        csv_path=Path("dummy_valid.csv"),
        row_count=2,
        existing_images=2,
        missing_images=0,
        patient_ids={"00002", "00003"},  # '00002' overlaps!
    )

    report, failures = build_report(train_split, valid_split)
    assert "RESULT: FAIL" in report
    assert any("Patient leakage" in f for f in failures)
    assert any("missing" in f for f in failures)


def test_validation_split_integrity_on_real_csv():
    """Verifies that the validation dataset in data/raw/chexpert-small passes integrity."""
    valid_split = _inspect_split("valid", VALID_CSV, collect_labels=False)
    assert valid_split.row_count == 234
    assert valid_split.existing_images == 234
    assert valid_split.missing_images == 0
    assert valid_split.unresolvable_paths == 0
    assert valid_split.unique_patients == 200
    assert valid_split.unique_studies == 200
    assert len(valid_split.errors) == 0
