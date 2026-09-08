from contracts import AnalysisFindings, DiseasePrediction, XAIAttachment


def test_single_prediction_validates():
    prediction = DiseasePrediction(disease="pneumonia", probability=0.87, positive=True)
    assert prediction.to_dict() == {
        "disease": "pneumonia",
        "probability": 0.87,
        "positive": True,
    }


def test_multi_label_findings_validate():
    findings = AnalysisFindings.from_dict(
        {
            "predictions": [
                {"disease": "pneumonia", "probability": 0.87, "positive": True},
                {"disease": "covid19", "probability": 0.12, "positive": False},
                {"disease": "pleural_effusion", "probability": 0.64, "positive": True},
                {"disease": "atelectasis", "probability": 0.08, "positive": False},
            ],
            "xai": {
                "method": "gradcam",
                "target": "pneumonia",
                "heatmap_available": True,
            },
        }
    )
    assert len(findings.predictions) == 4
    assert findings.predictions[0].positive is True
    assert len(findings.xai) == 1
    assert findings.xai[0].method == "gradcam"
    payload = findings.to_dict()
    assert payload["xai"]["heatmap_available"] is True


def test_xai_is_optional_and_method_agnostic():
    findings = AnalysisFindings(
        predictions=[
            DiseasePrediction(disease="atelectasis", probability=0.51, positive=True),
        ]
    )
    assert findings.xai == []
    findings_with_details = AnalysisFindings(
        predictions=findings.predictions,
        xai=[
            XAIAttachment(
                method="placeholder-method",
                target="atelectasis",
                heatmap_available=False,
                details={"note": "method not selected yet"},
            )
        ],
    )
    assert findings_with_details.xai[0].details["note"] == "method not selected yet"


def test_invalid_probability_is_rejected():
    try:
        DiseasePrediction(disease="pneumonia", probability=1.2, positive=True)
    except ValueError as exc:
        assert "probability" in str(exc)
    else:
        raise AssertionError("Expected invalid probability to raise ValueError.")


def test_duplicate_prediction_labels_are_rejected():
    try:
        AnalysisFindings.from_dict(
            {
                "predictions": [
                    {"disease": "pneumonia", "probability": 0.2, "positive": False},
                    {"disease": "pneumonia", "probability": 0.9, "positive": True},
                ]
            }
        )
    except ValueError as exc:
        assert "Duplicate disease ids" in str(exc)
    else:
        raise AssertionError("Expected duplicate prediction labels to raise ValueError.")
