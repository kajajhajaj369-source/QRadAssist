import pytest
from pathlib import Path
from evaluation.dataset_protocol_analysis import (
    normalize_chexpert_path,
    ProtocolAnalysisResults,
)


def test_normalize_chexpert_path():
    assert normalize_chexpert_path("CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg") == "train/patient00001/study1/view1_frontal.jpg"
    assert normalize_chexpert_path("CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg") == "valid/patient64541/study1/view1_frontal.jpg"
    assert normalize_chexpert_path("CheXpert-v1.0\\train\\patient00001\\study1\\view1_frontal.jpg") == "train/patient00001/study1/view1_frontal.jpg"
    assert normalize_chexpert_path("plain/relative/path.jpg") == "plain/relative/path.jpg"
    assert normalize_chexpert_path("") == ""


def test_protocol_analysis_results_defaults():
    res = ProtocolAnalysisResults()
    assert res.paths_match is False
    assert res.total_train_rows == 0
    assert res.frontal_count == 0
    assert res.lateral_count == 0
    assert isinstance(res.study_label_inconsistencies, dict)
    assert isinstance(res.valid_label_counts, dict)
