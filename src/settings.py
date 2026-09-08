"""Load Phase 0 project settings from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from paths import DEFAULT_SETTINGS_FILE, PROJECT_ROOT


@dataclass(frozen=True)
class Settings:
    project_name: str
    phase: int
    diseases_file: Path
    raw: dict[str, Any]


def load_settings(path: Path | None = None) -> Settings:
    settings_path = path or DEFAULT_SETTINGS_FILE
    with settings_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError("settings.yaml must contain a mapping at the top level.")

    project = data.get("project") or {}
    config = data.get("config") or {}

    name = project.get("name")
    phase = project.get("phase")
    diseases_file = config.get("diseases_file")

    if not isinstance(name, str) or not name.strip():
        raise ValueError("settings.yaml must define project.name as a non-empty string.")
    if not isinstance(phase, int):
        raise ValueError("settings.yaml must define project.phase as an integer.")
    if not isinstance(diseases_file, str) or not diseases_file.strip():
        raise ValueError("settings.yaml must define config.diseases_file.")

    return Settings(
        project_name=name.strip(),
        phase=phase,
        diseases_file=(PROJECT_ROOT / diseases_file).resolve(),
        raw=data,
    )
