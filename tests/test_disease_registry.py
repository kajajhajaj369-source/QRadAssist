from registry import DiseaseRegistry


INITIAL_IDS = ["pneumonia", "covid19", "pleural_effusion", "atelectasis"]


def test_diseases_json_can_be_loaded():
    registry = DiseaseRegistry.from_json()
    assert len(registry) == 4


def test_all_initial_diseases_exist():
    registry = DiseaseRegistry.from_json()
    for disease_id in INITIAL_IDS:
        assert registry.contains(disease_id)
        disease = registry.get(disease_id)
        assert disease.display_name
        assert disease.enabled is True


def test_disease_ids_are_unique():
    registry = DiseaseRegistry.from_json()
    ids = registry.ids()
    assert len(ids) == len(set(ids))


def test_duplicate_ids_are_rejected():
    try:
        DiseaseRegistry.from_records(
            [
                {"id": "pneumonia", "display_name": "Pneumonia", "enabled": True},
                {"id": "pneumonia", "display_name": "Duplicate", "enabled": False},
            ]
        )
    except ValueError as exc:
        assert "Duplicate disease ids" in str(exc)
    else:
        raise AssertionError("Expected duplicate ids to raise ValueError.")


def test_enabled_and_disabled_status():
    registry = DiseaseRegistry.from_records(
        [
            {"id": "pneumonia", "display_name": "Pneumonia", "enabled": True},
            {"id": "covid19", "display_name": "COVID-19", "enabled": False},
        ]
    )
    assert registry.is_enabled("pneumonia") is True
    assert registry.is_enabled("covid19") is False
    assert registry.ids(enabled_only=True) == ["pneumonia"]
    assert "covid19" in registry.ids()
    assert [disease.id for disease in registry.enabled()] == ["pneumonia"]


def test_registry_supports_additional_disease_without_new_application_logic():
    """Adding a label is a data change; the same registry API still applies."""
    records = [
        {"id": "pneumonia", "display_name": "Pneumonia", "enabled": True},
        {"id": "covid19", "display_name": "COVID-19", "enabled": True},
        {"id": "pleural_effusion", "display_name": "Pleural Effusion", "enabled": True},
        {"id": "atelectasis", "display_name": "Atelectasis", "enabled": True},
        {"id": "cardiomegaly", "display_name": "Cardiomegaly", "enabled": True},
    ]
    registry = DiseaseRegistry.from_records(records)

    assert registry.contains("cardiomegaly")
    assert registry.get("cardiomegaly").display_name == "Cardiomegaly"
    assert "cardiomegaly" in registry.ids(enabled_only=True)
    assert len(registry) == 5
