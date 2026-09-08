from settings import load_settings


def test_settings_file_loads():
    settings = load_settings()
    assert settings.project_name == "Q-RadAssist"
    assert settings.phase == 0
    assert settings.diseases_file.exists()
