import pytest
import numpy as np
from pathlib import Path
from PIL import Image

from preprocessing import (
    TARGET_LABELS,
    SUPPORTED_POLICIES,
    SampleManifest,
    resolve_local_image_path,
    parse_patient_id,
    parse_study_id,
    parse_row_to_sample,
    filter_manifest_by_view,
    save_manifest_csv,
    load_manifest_csv,
    split_patients,
    select_development_cohort,
    verify_zero_patient_overlap,
    PatientLeakageError,
    classify_raw_label,
    encode_labels,
    PreprocessConfig,
    DeterministicPreprocessor,
)
from paths import PROJECT_ROOT, RAW_DATA_DIR


# --- 1. Path Resolution Tests ---

def test_resolve_local_image_path_valid():
    p1 = resolve_local_image_path("CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg")
    expected = (RAW_DATA_DIR / "chexpert-small/train/patient00001/study1/view1_frontal.jpg").resolve()
    assert p1 == expected

    p2 = resolve_local_image_path("CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg")
    expected_v = (RAW_DATA_DIR / "chexpert-small/valid/patient64541/study1/view1_frontal.jpg").resolve()
    assert p2 == expected_v


def test_resolve_local_image_path_invalid():
    with pytest.raises(ValueError, match="empty Path"):
        resolve_local_image_path("")

    with pytest.raises(ValueError, match="Cannot resolve image path"):
        resolve_local_image_path("unrecognized_prefix/image.jpg")


# --- 2. Patient and Study ID Parsing ---

def test_patient_and_study_parsing():
    raw_path = "CheXpert-v1.0/train/patient12345/study6/view2_lateral.jpg"
    assert parse_patient_id(raw_path) == "12345"
    assert parse_study_id(raw_path) == "6"

    with pytest.raises(ValueError, match="Cannot parse patient ID"):
        parse_patient_id("no_patient/study1/view1.jpg")

    with pytest.raises(ValueError, match="Cannot parse study ID"):
        parse_study_id("patient00001/no_study/view1.jpg")


# --- 3. Row to Sample Parsing and Manifest Schema ---

def test_parse_row_to_sample():
    row = {
        "Path": "CheXpert-v1.0/train/patient00042/study2/view1_frontal.jpg",
        "Frontal/Lateral": "Frontal",
        "AP/PA": "AP",
        "Pneumonia": "1.0",
        "Atelectasis": "-1.0",
        "Pleural Effusion": "",
    }
    sample = parse_row_to_sample(row)
    assert sample.sample_id == "p00042_s2_view1_frontal"
    assert sample.patient_id == "00042"
    assert sample.study_id == "2"
    assert sample.view == "Frontal"
    assert sample.projection == "AP"
    assert sample.pneumonia_raw == "1.0"
    assert sample.atelectasis_raw == "-1.0"
    assert sample.pleural_effusion_raw == ""
    assert sample.split is None

    # Verify dictionary serialization and reconstruction
    d = sample.to_dict()
    assert d["sample_id"] == "p00042_s2_view1_frontal"
    reconstructed = SampleManifest.from_dict(d)
    assert reconstructed.sample_id == sample.sample_id
    assert reconstructed.raw_labels == sample.raw_labels


# --- 4. View Filtering Tests ---

def test_filter_manifest_by_view():
    samples = [
        SampleManifest("s1", "p1.jpg", "r1", "001", "1", "Frontal", "AP", "1.0", "", ""),
        SampleManifest("s2", "p2.jpg", "r2", "001", "1", "Lateral", "", "", "", ""),
        SampleManifest("s3", "p3.jpg", "r3", "002", "1", "Frontal", "PA", "", "1.0", ""),
    ]
    frontal_only = filter_manifest_by_view(samples, allowed_views=("Frontal",))
    assert len(frontal_only) == 2
    assert {s.sample_id for s in frontal_only} == {"s1", "s3"}
    # Verify lateral was excluded without mutating original list
    assert len(samples) == 3


# --- 5. Label Preservation and Policy Encoding Tests ---

def test_raw_label_classification():
    assert classify_raw_label("1.0") == "positive"
    assert classify_raw_label("0.0") == "negative"
    assert classify_raw_label("-1.0") == "uncertain"
    assert classify_raw_label("") == "blank"
    assert classify_raw_label(None) == "blank"

    with pytest.raises(ValueError):
        classify_raw_label("2.0")


def test_encode_labels_baseline_u_ignore():
    raw = {"Pneumonia": "1.0", "Atelectasis": "-1.0", "Pleural Effusion": ""}
    targets, masks = encode_labels(raw, policy="u_ignore")

    # Pneumonia (positive): target=1.0, mask=1.0
    assert targets["Pneumonia"] == 1.0
    assert masks["Pneumonia"] == 1.0

    # Atelectasis (uncertain): target=0.0, mask=0.0 (masked out from loss!)
    assert targets["Atelectasis"] == 0.0
    assert masks["Atelectasis"] == 0.0

    # Pleural Effusion (blank): assumed negative under negative reporting bias
    assert targets["Pleural Effusion"] == 0.0
    assert masks["Pleural Effusion"] == 1.0


def test_encode_labels_alternative_policies():
    raw = {"Pneumonia": "-1.0", "Atelectasis": "-1.0", "Pleural Effusion": "0.0"}

    # U-Zeros
    t_zero, m_zero = encode_labels(raw, policy="u_zeros")
    assert t_zero["Pneumonia"] == 0.0
    assert m_zero["Pneumonia"] == 1.0

    # U-Ones
    t_one, m_one = encode_labels(raw, policy="u_ones")
    assert t_one["Pneumonia"] == 1.0
    assert m_one["Pneumonia"] == 1.0

    with pytest.raises(ValueError, match="Unsupported uncertainty policy"):
        encode_labels(raw, policy="invalid_policy")


# --- 6. Patient-Level Splitting & Leakage Prevention ---

def create_synthetic_samples(num_patients: int = 20, images_per_patient: int = 2) -> list[SampleManifest]:
    samples = []
    for pid_int in range(1, num_patients + 1):
        pid_str = f"{pid_int:05d}"
        for img_idx in range(1, images_per_patient + 1):
            sid_str = "1"
            sample_id = f"p{pid_str}_s{sid_str}_v{img_idx}"
            samples.append(
                SampleManifest(
                    sample_id=sample_id,
                    image_path=f"data/raw/chexpert-small/train/patient{pid_str}/study1/view{img_idx}.jpg",
                    raw_csv_path=f"CheXpert-v1.0/train/patient{pid_str}/study1/view{img_idx}.jpg",
                    patient_id=pid_str,
                    study_id=sid_str,
                    view="Frontal",
                    projection="AP",
                    pneumonia_raw="0.0",
                    atelectasis_raw="1.0",
                    pleural_effusion_raw="",
                )
            )
    return samples


def test_patient_level_splitting_zero_overlap():
    samples = create_synthetic_samples(num_patients=20, images_per_patient=3)
    train_samples, val_samples = split_patients(samples, val_fraction=0.20, seed=42)

    assert len(train_samples) > 0
    assert len(val_samples) > 0
    assert len(train_samples) + len(val_samples) == len(samples)

    train_patients = {s.patient_id for s in train_samples}
    val_patients = {s.patient_id for s in val_samples}

    # Strict assertion: zero patient overlap
    assert len(train_patients & val_patients) == 0
    verify_zero_patient_overlap(train_samples, val_samples)


def test_patient_level_splitting_reproducibility():
    samples = create_synthetic_samples(num_patients=30, images_per_patient=2)
    train_1, val_1 = split_patients(samples, val_fraction=0.15, seed=123)
    train_2, val_2 = split_patients(samples, val_fraction=0.15, seed=123)

    assert [s.sample_id for s in train_1] == [s.sample_id for s in train_2]
    assert [s.sample_id for s in val_1] == [s.sample_id for s in val_2]

    # Different seed yields different split
    _, val_3 = split_patients(samples, val_fraction=0.15, seed=999)
    assert {s.patient_id for s in val_1} != {s.patient_id for s in val_3}


def test_leakage_detector_raises_error():
    samples_a = [SampleManifest("s1", "p1", "r1", "001", "1", "Frontal", "AP", "", "", "")]
    samples_b = [SampleManifest("s2", "p2", "r2", "001", "2", "Frontal", "PA", "", "", "")]  # same patient 001!

    with pytest.raises(PatientLeakageError, match="Patient leakage detected"):
        verify_zero_patient_overlap(samples_a, samples_b)


# --- 7. Development Cohort Selection ---

def test_select_development_cohort():
    samples = create_synthetic_samples(num_patients=50, images_per_patient=2)  # 100 images total
    cohort = select_development_cohort(samples, target_count=30, seed=42)

    assert len(cohort) >= 30
    assert len(cohort) <= 35  # patient-aware accumulation stops promptly

    # Verify whole patients are selected
    cohort_patients = {s.patient_id for s in cohort}
    for pid in cohort_patients:
        patient_images_in_orig = [s for s in samples if s.patient_id == pid]
        patient_images_in_cohort = [s for s in cohort if s.patient_id == pid]
        assert len(patient_images_in_orig) == len(patient_images_in_cohort)

    # Reproducibility
    cohort_again = select_development_cohort(samples, target_count=30, seed=42)
    assert [s.sample_id for s in cohort] == [s.sample_id for s in cohort_again]


def test_select_development_cohort_excludes_lateral_images():
    # Construct mixed cohort of frontal and lateral images
    mixed_samples = [
        SampleManifest("s1", "p1.jpg", "r1", "001", "1", "Frontal", "AP", "", "", ""),
        SampleManifest("s2", "p2.jpg", "r2", "001", "1", "Lateral", "", "", "", ""),  # Lateral for patient 001
        SampleManifest("s3", "p3.jpg", "r3", "002", "1", "Frontal", "PA", "", "", ""),
        SampleManifest("s4", "p4.jpg", "r4", "002", "1", "Lateral", "", "", "", ""),  # Lateral for patient 002
    ]
    cohort = select_development_cohort(mixed_samples, target_count=2, seed=42)
    # Lateral images must NOT be reintroduced
    assert all(s.view == "Frontal" for s in cohort)
    assert not any(s.view == "Lateral" for s in cohort)


# --- 8. Manifest File IO ---

def test_manifest_save_and_load(tmp_path: Path):
    samples = create_synthetic_samples(num_patients=5, images_per_patient=1)
    manifest_file = tmp_path / "test_manifest.csv"

    save_manifest_csv(samples, manifest_file)
    assert manifest_file.is_file()

    loaded = load_manifest_csv(manifest_file)
    assert len(loaded) == len(samples)
    for orig, res in zip(samples, loaded):
        assert orig.sample_id == res.sample_id
        assert orig.patient_id == res.patient_id
        assert orig.pneumonia_raw == res.pneumonia_raw


# --- 9. Deterministic Preprocessor Image Transforms ---

def test_deterministic_preprocessor_synthetic_image(tmp_path: Path):
    # Create synthetic test image
    img = Image.new("RGB", (300, 400), color=(128, 64, 32))
    img_path = tmp_path / "synthetic_xray.jpg"
    img.save(img_path, format="JPEG")

    # 1. Default config: normalize=False, scales to [0.0, 1.0] without unverified standardization
    config_default = PreprocessConfig(
        image_size=(224, 224),
        grayscale=True,
        normalize=False,
        channels_first=True,
    )
    preprocessor = DeterministicPreprocessor(config_default)

    arr = preprocessor.transform(img_path)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1, 224, 224)
    assert arr.dtype == np.float32
    assert arr.min() >= 0.0 and arr.max() <= 1.0

    # Test processing from PIL Image object gives identical result
    arr_pil = preprocessor(img)
    assert arr_pil.shape == (1, 224, 224)
    np.testing.assert_allclose(arr, arr_pil, rtol=1e-5)

    # 2. Configured standardization with explicit mean and std
    config_norm = PreprocessConfig(
        image_size=(224, 224),
        grayscale=True,
        normalize=True,
        mean=0.5,
        std=0.25,
        channels_first=True,
    )
    preprocessor_norm = DeterministicPreprocessor(config_norm)
    arr_norm = preprocessor_norm(img)
    expected = (arr - 0.5) / 0.25
    np.testing.assert_allclose(arr_norm, expected, rtol=1e-5)

    # 3. Standardization requested without explicit statistics raises ValueError
    config_missing = PreprocessConfig(normalize=True, mean=None, std=None)
    preprocessor_invalid = DeterministicPreprocessor(config_missing)
    with pytest.raises(ValueError, match="requires explicit 'mean' and 'std' statistics"):
        preprocessor_invalid(img)

